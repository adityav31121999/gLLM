
#include "include/transformer.hpp"
#include <chrono>
#include <iostream>

#ifdef USE_CPU

/**
 * @brief run transformer for sequence1's sequence2
 * @param sequence1 sequence1 to get sequence2
 */
void transformer::run() {
    if(currentTokenCount+sequence1Count >= FULL_CONTEXT) {
        throw std::runtime_error("TOKEN LIMIT REACHED AT FULL CONTEXT! SORRY -_-");
    }
    int c = std::abs(currentTokenCount - (blockCount-1)*CONTEXT_WIN);
    // under local context
    if(c + sequence1Count <= CONTEXT_WIN) {
        // when first block, tokenEmbed is directly utilised
        if(blockCount > 1) {
            for(int k = 0; k < sequence1Count; k++) {
                for(int i = 0; i < EMBEDDING; i++) {
                    tokForBlock(c + k, i) = tokenEmbed(currentTokenCount + k, i);
                }
            }
        }
        for(int i = 0; i < x; i++) {
            for(int j = 0; j < y; j++) {
                for(int k = 0; k < sequence1Count; k++) {
                    for(int m = 0; m < EMBEDDING; m++) {
                        blocks[0].b[i][j].EV(c-1+k, m) = tokenEmbed(currentTokenCount + k, m);
                    }
                }
            }
        }
    }
    // if it goes over context window, increment to next block
    if(c + sequence1Count > CONTEXT_WIN) {
        int m1 = c + sequence1Count - CONTEXT_WIN;     // part of sequence1 in next block
        int m2 = CONTEXT_WIN - c;   // available space in this block
        // add sequence1 to EVs and tokforblock
        for(int i = 0; i < x; i++) {
            for(int j = 0; j < y; j++) {
                for(int k = 0; k < m2; k++) {
                    for(int m = 0; m < EMBEDDING; m++) {
                        blocks[0].b[i][j].EV(c-1+k, m) = tokenEmbed(currentTokenCount + k, m);
                    }
                }
            }
        }
        for(int i = 0; i < m2; i++) {
            for(int m = 0; m < EMBEDDING; m++) {
                tokForBlock(i, m) = tokenEmbed(currentTokenCount + i, m);
            }
        }
        computeKdotQs(sequence1Count, currentTokenCount, blockCount, isSelf, inTraining);
        // token limit reached for first block
        currentTokenCount += m1;
        // set vertical retention vectors
        for(int i = 0; i < x; i++) {
            for(int j = 0; j < y; j++) {
                for(int k = 0; k < CONTEXT_WIN; k++) {
                    for(int m = 0; m < EMBEDDING; m++) {
                        EVuse[i][j][k][m] = blocks[0].b[i][j].EV(k, m);
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
    // caculate sequence2
    int rCount = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    while (1) { // Removed int k, l; declaration, will use specific float variables for sums
        // forprop for EH and EV
        for(int i = 0; i < x; i++) {
            for(int j = 0; j < y; j++) {
                // block specific KdotQ
                if(blockCount == 1) {
                    std::vector<std::vector<float>> kdotq = blocks[0].b[i][j].KdotQ.make2dVector(t[0].b[i][j].KdotQ, CONTEXT_WIN, CONTEXT_WIN);
                    std::vector<std::vector<float>> tokenembed = tokenEmbed.make2dVector(tokenEmbed, CONTEXT_WIN, EMBEDDING);
                    computeKdotQ(kdotq, tokenembed, blocks[0].b[i][j].qkCache, currentTokenCount, sequence1Count, isSelf);
                }
                else {
                    std::vector<std::vector<float>> kdotq = blocks[0].b[i][j].KdotQ.make2dVector(t[0].b[i][j].KdotQ, CONTEXT_WIN, CONTEXT_WIN);
                    std::vector<std::vector<float>> tokforblock = blocks[0].tokForBlock.make2dVector(t[blockCount-1].tokForBlock, CONTEXT_WIN, EMBEDDING);
                    computeKdotQ(kdotq, tokforblock, EVuse[i][j], blocks[blockCount-1].b[i][j].qkCache, currentTokenCount, sequence1Count, blockCount, isSelf);
                }
                // number of tokens in context window of this block
                int count = std::abs(currentTokenCount - n * (blockCount-1));
                // calculate KdotQ and head
                // Ensure KdotQ is appropriately sized or LOTA handles 'count' correctly.
                mat head = LOTA(t[0].b[i][j].KdotQ, count, isSelf);

                // Reset dh and dv for the current head, similar to forward.cpp
                blocks[0].b[i][j].dh.assign(EMBEDDING, 0.0f);
                blocks[0].b[i][j].dv.assign(EMBEDDING, 0.0f);

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
                        blocks[0].b[i][j].dh[m] += k_sum_val * k_source_val;
                        // ti*l, dv = weighted sums vertical
                        // EVuse[i][j] is [CONTEXT_WIN][EMBEDDING], w_idx is token index in current block
                        float l_source_val = (blockCount == 1) ? tokenEmbed(w_idx, m) : EVuse[i][j][w_idx][m];
                        blocks[0].b[i][j].dv[m] += l_sum_val * l_source_val;
                    }
                }
                blocks[0].b[i][j].dh = dot(t[0].b[i][j].dh, blocks[0].b[i][j].khCache);
                blocks[0].b[i][j].dv = dot(t[0].b[i][j].dv, blocks[0].b[i][j].qvCache);
                
                // get the required change from MLPs
                blocks[0].b[i][j].hor.input = blocks[0].b[i][j].EH + blocks[0].b[i][j].dh;
                
                blocks[0].b[i][j].ver.input.assign(EMBEDDING, 0.0f); // Reset ver.input
                for(int token_idx_in_block = 0; token_idx_in_block < count; ++token_idx_in_block) {
                    // Assumes getRow and += are defined for std::vector or your math types, or blocks[0].b[i][j].EV allows row access and addition
                    blocks[0].b[i][j].ver.input += getRow(t[0].b[i][j].EV, token_idx_in_block);
                }
                blocks[0].b[i][j].ver.input += blocks[0].b[i][j].dv; // Add delta_v

                blocks[0].b[i][j].hor.forward(d, l); // Use class members for EMBEDDING and MLP_LAYERS
                blocks[0].b[i][j].ver.forward(d, l);

                // AND gate for the final output
                blocks[0].b[i][j].EH += ReLU(t[0].b[i][j].hor.output);
                std::vector<float> relu_ver_output = ReLU(t[0].b[i][j].ver.output);
                for(int token_idx_in_block = 0; token_idx_in_block < count; ++token_idx_in_block) {
                    // Assumes blocks[0].b[i][j].EV(token_idx_in_block) returns a row object that supports += std::vector<float>
                    // Or implement with: for(int m=0; m<EMBEDDING; ++m) blocks[0].b[i][j].EV(token_idx_in_block, m) += relu_ver_output[m];
                    blocks[0].b[i][j].EV(token_idx_in_block) += relu_ver_output;
                }
            }
            otok += blocks[0].b[i][y-1].EH;
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
                            EVuse[i][j][k][m] = blocks[0].b[i][j].EV(k, m);
                        }
                    }
                }
            }
            blockCount += 1;
        }
        // check for maximum token limit
        if(currentTokenCount == FULL_CONTEXT) {
            std::cout << "\nTOKEN LIMIT REACHED AT FULL CONTEXT WHILE IN PROCESS! SORRY :(";
            break;
        }
        // check for terminating word '</s>'
        if(tokens[indexForToken] == TERMINATE) {
            break;
        }
        rCount += 1;
    }
}

#endif
