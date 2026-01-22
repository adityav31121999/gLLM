#ifdef USE_CPU
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <algorithm>
#include "include/transformer.hpp"

/**
 * @brief test function for transformer
 * @param [in] prompt prompt embeddings
 * @param [out] rString expected response vector
 */
void transformer::test(std::vector<std::vector<float>> &prompt, std::vector<std::string> &rString)
{
    // check for full context
    if(currentTokenCount + prompt.size() + rString.size() >= FULL_CONTEXT) {
        throw std::runtime_error("test: TOKEN LIMIT REACHED AT FULL CONTEXT! FURTHER PROCESS CANNOT TAKE PLACE -_-");
    }

    // prompt should not be empty
    if(prompt.empty()) {
        throw std::runtime_error("test: Prompt cannot be empty.");
    }

    std::cout << "--- Starting Test ---" << std::endl;
    std::cout << "Prompt Size: " << prompt.size() << " | Expected Response Size: " << rString.size() << std::endl;

    // 1. Process Prompt (sequence1)
    // Populate tokenEmbed with prompt + positional
    for(size_t i = 0; i < prompt.size(); ++i) {
        tokenEmbed(currentTokenCount + i) = prompt[i] + positionalEmbeddings(currentTokenCount + i, EMBEDDING);
    }

    int sequence1Count = prompt.size();
    int c = std::abs(currentTokenCount - (blockCount - 1) * CONTEXT_WIN);
    
    // under local context
    if(c + sequence1Count <= CONTEXT_WIN) {
        for(int i = 0; i < x; i++) {
            for(int j = 0; j < y; j++) {
                for(int k = 0; k < sequence1Count; k++) {
                    for(int m = 0; m < EMBEDDING; m++) {
                        blocks[0].b[i][j].EV(c + k, m) = tokenEmbed(currentTokenCount + k, m);
                    }
                }
            }
        }
        if(blockCount > 1) {
            for(int k = 0; k < sequence1Count; k++) {
                for(int i = 0; i < EMBEDDING; i++) {
                    tokForBlock(c + k, i) = tokenEmbed(currentTokenCount + k, i);
                }
            }
        }
        computeKdotQs(sequence1Count, currentTokenCount, blockCount, isSelf, inTraining);
        currentTokenCount += sequence1Count;
    }
    else {
        int m1 = c + sequence1Count - CONTEXT_WIN;
        int m2 = CONTEXT_WIN - c;
        
        // Part 1
        for(int i = 0; i < x; i++) {
            for(int j = 0; j < y; j++) {
                for(int k = 0; k < m2; k++) {
                    for(int m = 0; m < EMBEDDING; m++) {
                        blocks[0].b[i][j].EV(c + k, m) = tokenEmbed(currentTokenCount + k, m);
                    }
                }
            }
        }
        if(blockCount > 1) {
             for(int i = 0; i < m2; i++) {
                for(int m = 0; m < EMBEDDING; m++) {
                    tokForBlock(c + i, m) = tokenEmbed(currentTokenCount + i, m);
                }
            }
        }
        computeKdotQs(m2, currentTokenCount, blockCount, isSelf, inTraining);
        currentTokenCount += m2;

        // Transition
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
        
        // Part 2
        for(int i = 0; i < x; i++) {
            for(int j = 0; j < y; j++) {
                for(int k = 0; k < m1; k++) {
                    for(int m = 0; m < EMBEDDING; m++) {
                        blocks[0].b[i][j].EV(k, m) = tokenEmbed(currentTokenCount + k, m);
                    }
                }
            }
        }
        computeKdotQs(m1, currentTokenCount, blockCount, isSelf, inTraining);
        currentTokenCount += m1;
    }

    std::vector<std::string> generated_response;
    
    // 2. Generation Loop
    for(size_t r = 0; r < rString.size(); ++r) {
        if(currentTokenCount >= FULL_CONTEXT) {
            std::cout << "\nTOKEN LIMIT REACHED AT FULL CONTEXT!" << std::endl;
            break;
        }

        std::vector<float> otok(EMBEDDING, 0.0f);

        for(int i = 0; i < x; i++) {
            for(int j = 0; j < y; j++) {
                if(blockCount == 1) {
                    std::vector<std::vector<float>> kdotq = blocks[0].b[i][j].KdotQ.make2dVector(blocks[0].b[i][j].KdotQ, CONTEXT_WIN, CONTEXT_WIN);
                    std::vector<std::vector<float>> tokenembed = tokenEmbed.make2dVector(tokenEmbed, CONTEXT_WIN, EMBEDDING);
                    computeKdotQ(kdotq, tokenembed, blocks[0].b[i][j].qkCache, currentTokenCount, sequence1Count, isSelf);
                }
                else {
                    std::vector<std::vector<float>> kdotq = blocks[0].b[i][j].KdotQ.make2dVector(blocks[0].b[i][j].KdotQ, CONTEXT_WIN, CONTEXT_WIN);
                    std::vector<std::vector<float>> tokforblock = blocks[0].tokForBlock.make2dVector(blocks[blockCount-1].tokForBlock, CONTEXT_WIN, EMBEDDING);
                    computeKdotQ(kdotq, tokforblock, EVuse[i][j], blocks[blockCount-1].b[i][j].qkCache, currentTokenCount, sequence1Count, blockCount, isSelf);
                }

                int count = std::abs(currentTokenCount - (blockCount-1) * CONTEXT_WIN);
                mat head = LOTA(blocks[0].b[i][j].KdotQ, count, isSelf);

                blocks[0].b[i][j].h.assign(EMBEDDING, 0.0f);
                blocks[0].b[i][j].v.assign(EMBEDDING, 0.0f);

                for(int w_idx = 0; w_idx < count; w_idx++) {
                    float k_sum_val = 0.0f;
                    float l_sum_val = 0.0f;
                    int limit_z = isSelf ? (w_idx + 1) : count;
                    limit_z = std::min(limit_z, head.col);

                    for(int z_idx = 0; z_idx < limit_z; z_idx++) {    
                        if (w_idx < head.row) k_sum_val += head(w_idx, z_idx);
                        if (z_idx < head.row && w_idx < head.col) l_sum_val += head(z_idx, w_idx);
                    }
                    for(int m = 0; m < EMBEDDING; m++) {
                        float k_source_val = (blockCount == 1) ? tokenEmbed(w_idx, m) : tokForBlock(w_idx, m);
                        blocks[0].b[i][j].h[m] += k_sum_val * k_source_val;
                        float l_source_val = (blockCount == 1) ? tokenEmbed(w_idx, m) : EVuse[i][j][w_idx][m];
                        blocks[0].b[i][j].v[m] += l_sum_val * l_source_val;
                    }
                }
                blocks[0].b[i][j].h = dot(blocks[0].b[i][j].h, blocks[0].b[i][j].khCache);
                blocks[0].b[i][j].v = dot(blocks[0].b[i][j].v, blocks[0].b[i][j].qvCache);
                
                blocks[0].b[i][j].hor.input = blocks[0].b[i][j].EH + blocks[0].b[i][j].h;
                blocks[0].b[i][j].ver.input.assign(EMBEDDING, 0.0f);
                for(int token_idx_in_block = 0; token_idx_in_block < count; ++token_idx_in_block) {
                     blocks[0].b[i][j].ver.input += getRow(blocks[0].b[i][j].EV, token_idx_in_block);
                }
                blocks[0].b[i][j].ver.input += blocks[0].b[i][j].v;

                blocks[0].b[i][j].hor.forward();
                blocks[0].b[i][j].ver.forward();

                blocks[0].b[i][j].EH += ReLU(blocks[0].b[i][j].hor.output);
                std::vector<float> relu_ver_output = ReLU(blocks[0].b[i][j].ver.output);
                for(int token_idx_in_block = 0; token_idx_in_block < count; ++token_idx_in_block) {
                    blocks[0].b[i][j].EV(token_idx_in_block) += relu_ver_output;
                }
            }
            for(int k=0; k<EMBEDDING; ++k) otok[k] += blocks[0].b[i][y-1].EH[k];
        }

        computePrediction();
        
        std::string gen_token = tokens[indexForToken];
        generated_response.push_back(gen_token);
        std::cout << gen_token << " " << std::flush;

        for(int i = 0; i < EMBEDDING; i++) {
            tokenEmbed(currentTokenCount, i) = embeddings(indexForToken, i);
        }
        if(currentTokenCount < mTokens.size()) mTokens[currentTokenCount] = gen_token;
        else mTokens.push_back(gen_token);

        currentTokenCount += 1;

        if(currentTokenCount % CONTEXT_WIN == 0) {
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
        }

        if(gen_token == "</s>") break;
    }

    std::cout << "\n\n--- Test Results ---" << std::endl;
    std::cout << "Expected: ";
    for(const auto& s : rString) std::cout << s << " ";
    std::cout << "\nGenerated: ";
    for(const auto& s : generated_response) std::cout << s << " ";
    std::cout << std::endl;

    int correct = 0;
    size_t min_len = std::min(rString.size(), generated_response.size());
    for(size_t i=0; i<min_len; ++i) {
        if(rString[i] == generated_response[i]) correct++;
    }
    float acc = rString.empty() ? 0.0f : (float)correct / rString.size() * 100.0f;
    std::cout << "Accuracy: " << std::fixed << std::setprecision(2) << acc << "%" << std::endl;
}

#endif // USE_CPU
