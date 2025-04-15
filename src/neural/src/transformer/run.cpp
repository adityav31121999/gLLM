
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
        // compute kdotq
        for(int i = 0; i < promptCount; i++) {
            // get prompt and push it and its embedding to tokens and tokenEmbed vectors
        }
        computeKdotQs(promptCount, currentTokenCount, blockCount, isSelf, inTraining);
        // caculate response
        while (1) {
            // forprop for EH and EV
            for(int i = 0; i < x; i++) {
                for(int j = 0; j < y; j++) {
                    // for first block
                    if(blockCount == 1) {
                        int k, l;
                        // calculate KdotQ and head
                        computeKdotQ(t[0].b[i][j].KdotQ, tokenEmbed, t[0].b[i][j].qkCache, currentTokenCount, promptCount, isSelf);
                        std::vector<std::vector<float>> head(currentTokenCount, std::vector<float>(currentTokenCount, 0.0f));
                        head = LOTA(t[0].b[i][j].KdotQ, currentTokenCount, isSelf);
                        // get weighted sums
                        for(int w = 0; w < currentTokenCount; w++) {
                            k = 0;
                            l = 0;
                            for(int z = 0; z < (isSelf ? w : currentTokenCount); w++) {    
                                k += head[w][z];    // row sum
                                l += head[z][w];    // column sum
                            }
                            // ti*k, dh = weighted sums horizontal
                            t[0].b[i][j].dh = t[0].b[i][j].dh + (k * tokenEmbed[i]);
                            // ti*l, dv = weighted sums vertical
                            t[0].b[i][j].dv = t[0].b[i][j].dv + (l * tokenEmbed[i]);
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
                        EVuse[i][j][currentTokenCount] += ReLU(t[0].b[i][j].ver.output);
                    }
                    // for next local context with same block for inference
                    else if(blockCount > 1) {
                        int k = 0, l = 0;
                        // number of tokens in context window of this block
                        int count = std::abs(currentTokenCount - n * (blockCount-1));
                        // calculate KdotQ and head
                        computeKdotQ(t[0].b[i][j].KdotQ, tokForBlock, EVuse[i][j], t[0].b[i][j].qkCache, currentTokenCount, promptCount,blockCount, isSelf);
                        std::vector<std::vector<float>> head(currentTokenCount, std::vector<float>(currentTokenCount, 0.0f));
                        head = LOTA(t[0].b[i][j].KdotQ, currentTokenCount, isSelf);
                        // get weighted sums
                        for(int w = 0; w < count; w++) {
                            k = 0;
                            l = 0;
                            for(int z = 0; z < (isSelf ? w : count); w++) {    
                                k += head[w][z];    // row sum
                                l += head[z][w];    // column sum
                            }
                            // ti*k, dh = weighted sums horizontal
                            t[0].b[i][j].dh = t[0].b[i][j].dh + (k * tokenEmbed[i]);
                            // ti*l, dv = weighted sums vertical
                            t[0].b[i][j].dv = t[0].b[i][j].dv + (l * tokenEmbed[i]);
                        }
                        t[0].b[i][j].dh = dot(t[0].b[i][j].dh, t[0].b[i][j].khCache);
                        t[0].b[i][j].dv = dot(t[0].b[i][j].dv, t[0].b[i][j].qvCache);
                        // get the required change from MLPs
                        t[0].b[i][j].hor.input = t[0].b[i][j].EH + t[0].b[i][j].dh;
                        t[0].b[i][j].ver.input = EVuse[i][j][count] + t[0].b[i][j].dv;
                        t[0].b[i][j].hor.forward(d, l);
                        t[0].b[i][j].ver.forward(d, l);
                        // AND gate for the final output
                        t[0].b[i][j].EH += ReLU(t[0].b[i][j].hor.output);
                        EVuse[i][j][count] += ReLU(t[0].b[i][j].ver.output);
                    }
                }
                t[0].EH += t[0].b[i][y-1].EH;
            }
            computeOutput(t[0].EH, tokenEmbed, vocabsize, indexForToken);
            std::cout << tokens[currentTokenCount-1] << " ";
            tokenEmbed[currentTokenCount] = tokenEmbed[indexForToken];
            tokens[currentTokenCount] = tokens[indexForToken];
            currentTokenCount += 1;
            if(currentTokenCount%CONTEXT_WIN == 0) {
                blockCount += 1;
            }
            // check for terminating word '@#0'
            if(tokens[indexForToken] == TERMINATE) {
                break;
            }
        }
        // redo
    }
}
