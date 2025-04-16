
#include "include/transformer.hpp"

/**
 * @brief Run transformer using model parameters of cache and MLPs for inference, use
 * cache QK' for calculation of KdotQ and then use caches QV' and KH' for EV and EH 
 * calculation.
 */
void transformer::run() {
    // set for inference
    inTraining = 0;
    while (1) {
        // take user prompt
        std::string userPrompt;
        std::cout << "ENTER PROMPT (LIMIT" << CONTEXT_WIN/4 << " TOKENS): ";
        std::cin >> userPrompt;
        promptCount = tokenise(userPrompt, mTokens, currentTokenCount) + 1;
        if(currentTokenCount+promptCount >= FULL_CONTEXT) {
            throw std::runtime_error("TOKEN LIMIT REACHED AT FULL CONTEXT!");
            break;
        }
        int c = std::abs(currentTokenCount - (blockCount-1)*CONTEXT_WIN);
        // under local context
        if(c + promptCount <= CONTEXT_WIN) {
            if(blockCount > 1) {
                for(int k = 0; k < promptCount; k++) {
                    tokForBlock[c + k] = tokenEmbed[currentTokenCount + k];
                }
            }
            for(int i = 0; i < x; i++) {
                for(int j = 0; j < y; j++) {
                    for(int k = 0; k < promptCount; k++) {
                        t[0].b[i][j].EV[c-1+k] = tokenEmbed[currentTokenCount + k];
                    }
                }
            }
        }
        // if it goes over context window increment to next block
        if(c + promptCount > CONTEXT_WIN) {
            int m1 = c + promptCount - CONTEXT_WIN;
            int m2 = CONTEXT_WIN - c;   // available token count
            if(blockCount == 1) {
                // fix the prompts
                for(int i = 0; i < x; i++) {
                    for(int j = 0; j < y; j++) {
                        for(int k = 0; k < m2; k++) {
                            t[0].b[i][j].EV[c-1+k] = tokenEmbed[currentTokenCount + k];
                        }
                    }
                }
                for(int i = 0; i < m2; i++) {
                    tokForBlock[i] = tokenEmbed[currentTokenCount + i];
                }
                currentTokenCount += m2;
                // set vertical retention vectors
                for(int i = 0; i < x; i++) {
                    for(int j = 0; j < y; j++) {
                        for(int k = 0; k < CONTEXT_WIN; k++) {
                            EVuse[i][j][k] = t[0].b[i][j].EV[k];
                        }
                    }
                }
                blockCount += 1;
                for(int i = 0; i < CONTEXT_WIN; i++) {
                    tokForBlock[i] = tokenEmbed[currentTokenCount - CONTEXT_WIN + i];
                }
            }
            blockCount += 1;
        }
        for(int i = 0; i < promptCount; i++) { 
            // get token embeddings from 'embeddings' vector in transformer class
            // tokenEmbed[currentTokenCount+i] = embeddings[];
            // getEmbeddings(mToken[currentTokenCount+i], tokenEmbed[currentTokenCount+i]);
        }
        // caculate response
        while (1) {
            int k, l;   // for row and column sum
            // forprop for EH and EV
            for(int i = 0; i < x; i++) {
                for(int j = 0; j < y; j++) {
                    // block specific KdotQ
                    if(blockCount == 1) {
                        computeKdotQ(t[0].b[i][j].KdotQ, tokenEmbed, t[0].b[i][j].qkCache, currentTokenCount, promptCount, isSelf);
                    }
                    else {
                        computeKdotQ(t[0].b[i][j].KdotQ, tokForBlock, EVuse[i][j], t[0].b[i][j].qkCache, currentTokenCount, promptCount,blockCount, isSelf);
                    }
                    // number of tokens in context window of this block
                    int count = std::abs(currentTokenCount - n * (blockCount-1));
                    // calculate KdotQ and head
                    std::vector<std::vector<float>> head(count, std::vector<float>(count, 0.0f));
                    head = LOTA(t[0].b[i][j].KdotQ, count, isSelf);
                    // get weighted sums
                    for(int w = 0; w < count; w++) {
                        k = 0;
                        l = 0;
                        for(int z = 0; z < (isSelf ? w : count); w++) {    
                            k += head[w][z];    // row sum
                            l += head[z][w];    // column sum
                        }
                        // ti*k, dh = weighted sums horizontal
                        t[0].b[i][j].dh = t[0].b[i][j].dh + (k * ((blockCount == 1) ? tokenEmbed[i] : tokForBlock[i]));
                        // ti*l, dv = weighted sums vertical
                        t[0].b[i][j].dv = t[0].b[i][j].dv + (l * ((blockCount == 1) ? tokenEmbed[i] : EVuse[i][j][count]));
                    }
                    t[0].b[i][j].dh = dot(t[0].b[i][j].dh, t[0].b[i][j].khCache);
                    t[0].b[i][j].dv = dot(t[0].b[i][j].dv, t[0].b[i][j].qvCache);
                    // get the required change from MLPs
                    t[0].b[i][j].hor.input = t[0].b[i][j].EH + t[0].b[i][j].dh;
                    t[0].b[i][j].ver.input = EVuse[i][j][currentTokenCount] + t[0].b[i][j].dv;
                    t[0].b[i][j].hor.forward(d, l);
                    t[0].b[i][j].ver.forward(d, l);
                    // AND gate for the final output
                    t[0].b[i][j].EH += ReLU(t[0].b[i][j].hor.output);
                    t[0].b[i][j].EV[i] += ReLU(t[0].b[i][j].ver.output);
                }
                t[0].EH += t[0].b[i][y-1].EH;
            }
            computeOutput(t[0].EH, embeddings, vocabsize, indexForToken);
            tokenEmbed[currentTokenCount] = embeddings[indexForToken];
            mTokens[currentTokenCount] = tokens[indexForToken];
            std::cout << mTokens[currentTokenCount-1] << " ";
            currentTokenCount += 1;
            // check for local context
            if(currentTokenCount%CONTEXT_WIN == 0) {
                // set vertical context retention for next blocks
                for(int i = 0; i < x; i++) {
                    for(int j = 0; j < y; j++) {
                        for(int k = 0; k < CONTEXT_WIN; k++) {
                            EVuse[i][j][k] = t[0].b[i][j].EV[k];
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
        }
        std::cout << std::endl;
        // redo
    }
}
