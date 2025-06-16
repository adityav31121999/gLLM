
#include "include/transformer.hpp"
#include <chrono>
#include <iostream>

#ifdef USE_CPU

/**
 * @brief run transformer for single prompt-response
 * @param prompt prompt to get single response
 */
void transformer::run() {
    if(currentTokenCount+promptCount >= FULL_CONTEXT) {
        throw std::runtime_error("TOKEN LIMIT REACHED AT FULL CONTEXT!");
    }
    int c = std::abs(currentTokenCount - (blockCount-1)*CONTEXT_WIN);
    // under local context
    if(c + promptCount <= CONTEXT_WIN) {
        // when first block, tokenEmbed is directly utilised
        if(blockCount > 1) {
            for(int k = 0; k < promptCount; k++) {
                for(int i = 0; i < EMBEDDING; i++) {
                    tokForBlock(c + k, i) = tokenEmbed(currentTokenCount + k, i);
                }
            }
        }
        for(int i = 0; i < x; i++) {
            for(int j = 0; j < y; j++) {
                for(int k = 0; k < promptCount; k++) {
                    for(int m = 0; m < EMBEDDING; m++) {
                        t[0].b[i][j].EV(c-1+k, m) = tokenEmbed(currentTokenCount + k, m);
                    }
                }
            }
        }
    }
    // if it goes over context window, increment to next block
    if(c + promptCount > CONTEXT_WIN) {
        int m1 = c + promptCount - CONTEXT_WIN;     // part of prompt in next block
        int m2 = CONTEXT_WIN - c;   // available space in this block
        // add prompt to EVs and tokforblock
        for(int i = 0; i < x; i++) {
            for(int j = 0; j < y; j++) {
                for(int k = 0; k < m2; k++) {
                    for(int m = 0; m < EMBEDDING; m++) {
                        t[0].b[i][j].EV(c-1+k, m) = tokenEmbed(currentTokenCount + k, m);
                    }
                }
            }
        }
        for(int i = 0; i < m2; i++) {
            for(int m = 0; m < EMBEDDING; m++) {
                tokForBlock(i, m) = tokenEmbed(currentTokenCount + i, m);
            }
        }
        computeKdotQs(promptCount, currentTokenCount, blockCount, isSelf, inTraining);
        // token limit reached for first block
        currentTokenCount += m1;
        // set vertical retention vectors
        for(int i = 0; i < x; i++) {
            for(int j = 0; j < y; j++) {
                for(int k = 0; k < CONTEXT_WIN; k++) {
                    for(int m = 0; m < EMBEDDING; m++) {
                        EVuse[i][j][k][m] = t[0].b[i][j].EV(k, m);
                    }
                }
            }
        }
        blockCount += 1;
        for(int i = 0; i < CONTEXT_WIN; i++) {
            for(int m = 0; m < EMBEDDING; m++) {
                tokForBlock(i,m) = tokenEmbed(currentTokenCount - CONTEXT_WIN + i, m);
            }
        }
        currentTokenCount += m2;
        blockCount += 1;
    }
    // caculate response
    int rCount = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    while (1) { // Removed int k, l; declaration, will use specific float variables for sums
        // forprop for EH and EV
        for(int i = 0; i < x; i++) {
            for(int j = 0; j < y; j++) {
                // block specific KdotQ
                if(blockCount == 1) {
                    std::vector<std::vector<float>> kdotq = t[0].b[i][j].KdotQ.make2dVector(t[0].b[i][j].KdotQ, CONTEXT_WIN, CONTEXT_WIN);
                    std::vector<std::vector<float>> tokenembed = tokenEmbed.make2dVector(tokenEmbed, CONTEXT_WIN, EMBEDDING);
                    computeKdotQ(kdotq, tokenembed, t[0].b[i][j].qkCache, currentTokenCount, promptCount, isSelf);
                }
                else {
                    std::vector<std::vector<float>> kdotq = t[0].b[i][j].KdotQ.make2dVector(t[0].b[i][j].KdotQ, CONTEXT_WIN, CONTEXT_WIN);
                    std::vector<std::vector<float>> tokforblock = t[0].tokForBlock.make2dVector(t[blockCount-1].tokForBlock, CONTEXT_WIN, EMBEDDING);
                    computeKdotQ(kdotq, tokforblock, EVuse[i][j], t[blockCount-1].b[i][j].qkCache, currentTokenCount, promptCount, blockCount, isSelf);
                }
                // number of tokens in context window of this block
                int count = std::abs(currentTokenCount - n * (blockCount-1));
                // calculate KdotQ and head
                // Ensure KdotQ is appropriately sized or LOTA handles 'count' correctly.
                mat head = LOTA(t[0].b[i][j].KdotQ, count, isSelf);

                // Reset dh and dv for the current head, similar to forward.cpp
                t[0].b[i][j].dh.assign(EMBEDDING, 0.0f);
                t[0].b[i][j].dv.assign(EMBEDDING, 0.0f);

                // get weighted sums
                for(int w_idx = 0; w_idx < count; w_idx++) { // w_idx is current token in block
                    float k_sum_val = 0.0f;
                    float l_sum_val = 0.0f;
                    int limit_z = isSelf ? (w_idx + 1) : count; // Corrected limit for self-attention
                    limit_z = std::min(limit_z, head.col); // Boundary check from forward.cpp

                    for(int z_idx = 0; z_idx < limit_z; z_idx++) {    
                        if (w_idx < head.row) k_sum_val += head(w_idx, z_idx);    // row sum
                        if (z_idx < head.row && w_idx < head.col) l_sum_val += head(z_idx, w_idx);    // column sum
                    }
                    for(int m = 0; m < EMBEDDING; m++) {
                        // ti*k, dh = weighted sums horizontal
                        float k_source_val = (blockCount == 1) ? tokenEmbed(w_idx, m) : tokForBlock(w_idx, m);
                        t[0].b[i][j].dh[m] += k_sum_val * k_source_val;
                        // ti*l, dv = weighted sums vertical
                        // EVuse[i][j] is [CONTEXT_WIN][EMBEDDING], w_idx is token index in current block
                        float l_source_val = (blockCount == 1) ? tokenEmbed(w_idx, m) : EVuse[i][j][w_idx][m];
                        t[0].b[i][j].dv[m] += l_sum_val * l_source_val;
                    }
                }
                t[0].b[i][j].dh = dot(t[0].b[i][j].dh, t[0].b[i][j].khCache);
                t[0].b[i][j].dv = dot(t[0].b[i][j].dv, t[0].b[i][j].qvCache);
                
                // get the required change from MLPs
                t[0].b[i][j].hor.input = t[0].b[i][j].EH + t[0].b[i][j].dh;
                
                t[0].b[i][j].ver.input.assign(EMBEDDING, 0.0f); // Reset ver.input
                for(int token_idx_in_block = 0; token_idx_in_block < count; ++token_idx_in_block) {
                    // Assumes getRow and += are defined for std::vector or your math types, or t[0].b[i][j].EV allows row access and addition
                    t[0].b[i][j].ver.input += getRow(t[0].b[i][j].EV, token_idx_in_block);
                }
                t[0].b[i][j].ver.input += t[0].b[i][j].dv; // Add delta_v

                t[0].b[i][j].hor.forward(this->d, this->l); // Use class members for EMBEDDING and MLP_LAYERS
                t[0].b[i][j].ver.forward(this->d, this->l);

                // AND gate for the final output
                t[0].b[i][j].EH += ReLU(t[0].b[i][j].hor.output);
                std::vector<float> relu_ver_output = ReLU(t[0].b[i][j].ver.output);
                for(int token_idx_in_block = 0; token_idx_in_block < count; ++token_idx_in_block) {
                    // Assumes t[0].b[i][j].EV(token_idx_in_block) returns a row object that supports += std::vector<float>
                    // Or implement with: for(int m=0; m<EMBEDDING; ++m) t[0].b[i][j].EV(token_idx_in_block, m) += relu_ver_output[m];
                    t[0].b[i][j].EV(token_idx_in_block) += relu_ver_output;
                }
            }
            otok += t[0].b[i][y-1].EH;
        }
        computeOutput(otok, embeddings, vocabsize, indexForToken);
        for(int i = 0; i < EMBEDDING; i++) {
            tokenEmbed(currentTokenCount, i) = embeddings(indexForToken, i);
        }
        mTokens[currentTokenCount] = tokens[indexForToken];
        std::cout << mTokens[currentTokenCount] << " ";
        currentTokenCount += 1;
        // check for local context
        if(currentTokenCount%CONTEXT_WIN == 0) {
            // set vertical context retention for next blocks
            for(int i = 0; i < x; i++) {
                for(int j = 0; j < y; j++) {
                    for(int k = 0; k < CONTEXT_WIN; k++) {
                        for(int m = 0; m < EMBEDDING; m++) {
                            EVuse[i][j][k][m] = t[0].b[i][j].EV(k, m);
                        }
                    }
                }
            }
            blockCount += 1;
        }
        // check for maximum token limit
        if(currentTokenCount == FULL_CONTEXT) {
            std::cout << "\nTOKEN LIMIT REACHED AT FULL CONTEXT!";
            break;
        }
        // check for terminating word '@#0'
        if(tokens[indexForToken] == TERMINATE) {
            break;
        }
        rCount += 1;
    }
}

#endif
