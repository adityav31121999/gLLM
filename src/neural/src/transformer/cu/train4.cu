#ifdef USE_CU
#include "include/transformer.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <string>
#include <cmath>

// --- CUDA Error Checking Macro ---
#define CUDA_CHECK(call)                                                     \
do {                                                                         \
    cudaError_t err = call;                                                  \
    if (err != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n",                 \
                __FILE__, __LINE__, cudaGetErrorString(err));                \
        throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(err)));    \
    }                                                                        \
} while (0)


/**
 * @brief (CUDA) Train the transformer for sequence1 and sequence2
 * @param sequence1 sequence1 token embeddings (on host)
 * @param sequence2 sequence2 token embeddings (on host)
 * @param rString tokens of sequence2 (on host)
 */
void transformer::cuTrainContext(std::vector<std::vector<float>>& sequence1, std::vector<std::vector<float>>& sequence2,
    std::vector<std::string>& rString)
{
    // --- Basic validation ---
    if (sequence1.empty()) {
        throw std::runtime_error("cuTrainContext(sequence1-sequence2): Initial sequence1 cannot be empty.");
    }
    if (sequence2.empty() || sequence2.size() != rString.size()) {
        throw std::runtime_error("cuTrainContext(sequence1-sequence2): sequence2 embeddings/strings mismatch or empty.");
    }
    if ((!sequence1.empty() && sequence1[0].size() != static_cast<size_t>(d)) || (!sequence2.empty() && sequence2[0].size() != static_cast<size_t>(d))) {
        throw std::runtime_error("cuTrainContext(sequence1-sequence2): Embedding dimension mismatch.");
    }
    if (currentTokenCount + sequence1.size() + sequence2.size() > FULL_CONTEXT) {
        throw std::runtime_error("cuTrainContext(sequence1-sequence2): Adding sequence1 and sequence2 exceeds FULL_CONTEXT limit.");
    }

    // --- Variable Initialization ---
    float current_error = 0.0f;
    float prev_error = 0.0f;
    float initialLearning = learning;
    bool blockShifted = 0;
    int initial_epochs = epochs;
    int effective_context_size = 0;
    int resCount = 0;

    // --- Device Buffer Allocation & H->D Transfer ---
    size_t totalTokenEmbedFloats = static_cast<size_t>(m) * CONTEXT_WIN * d;
    size_t tokenEmbedBytes = totalTokenEmbedFloats * sizeof(float);
    size_t deEmbeddingsBytes = static_cast<size_t>(vocabsize) * x * d * sizeof(float);
    size_t singleTokenBytes = static_cast<size_t>(d) * sizeof(float);
    size_t projection_matrix_bytes = static_cast<size_t>(CONTEXT_WIN) * EMBEDDING * sizeof(float);
    size_t KQmatbytes = static_cast<size_t>(CONTEXT_WIN) * CONTEXT_WIN * sizeof(float);
    size_t embedding_bytes_loc = static_cast<size_t>(EMBEDDING) * CONTEXT_WIN * sizeof(float);
    size_t predBytes = static_cast<size_t>(vocabsize) * sizeof(float);

    float *d_deEmbeddings, *d_tokenEmbed, *d_expected_token;
    float *d_Q, *d_K, *d_mQ, *d_mK, *d_tok, *d_pEV;
    float *d_otok_buffer, *d_predictions;
    int* d_result_index;

    CUDA_CHECK(cudaMalloc(&d_deEmbeddings, deEmbeddingsBytes));
    CUDA_CHECK(cudaMemcpy(d_deEmbeddings, deEmbeddings.mapped_data, deEmbeddingsBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_tokenEmbed, tokenEmbedBytes));
    CUDA_CHECK(cudaMalloc(&d_expected_token, singleTokenBytes));
    CUDA_CHECK(cudaMalloc(&d_Q, KQmatbytes));
    CUDA_CHECK(cudaMalloc(&d_K, KQmatbytes));
    CUDA_CHECK(cudaMalloc(&d_mQ, projection_matrix_bytes));
    CUDA_CHECK(cudaMalloc(&d_mK, projection_matrix_bytes));
    CUDA_CHECK(cudaMalloc(&d_tok, embedding_bytes_loc));
    CUDA_CHECK(cudaMalloc(&d_pEV, embedding_bytes_loc));
    CUDA_CHECK(cudaMalloc(&d_otok_buffer, static_cast<size_t>(x) * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_predictions, predBytes));
    CUDA_CHECK(cudaMalloc(&d_result_index, sizeof(int)));

    try {
        otok.clear(); otok.resize(d * x, 0.0f);
        pred.clear(); pred.resize(vocabsize, 0.0f);
        oneHotEncode.clear(); oneHotEncode.resize(vocabsize, 0.0f);

        // start training from first
        if(currentTokenCount == 0) {
            // set tokenEmbed
            blockCount = 1;
            for(int i = 0; i < sequence1.size(); i++) {
                tokenEmbed.addRow(sequence1[i] + positionalEmbeddings(i, d), i);
                // prepare EVs
                for(int m1 = 0; m1 < x; m1++) {
                    for(int m2 = 0; m2 < y; m2++) {
                        blocks[0].b[m1][m2].EV.addRow(sequence1[i], i);
                    }
                }
            }
            currentTokenCount = sequence1.size();
            effective_context_size = currentTokenCount;
        }
        // continue training in first block
        else if(currentTokenCount > 0 && currentTokenCount + sequence1.size() <= CONTEXT_WIN) {
            blockCount = 1;
            // add sequence1 tokens from currentTokenCount
            if(currentTokenCount + sequence1.size() < CONTEXT_WIN) {
                for(int i = 0; i < sequence1.size(); i++) {
                    int actual_row_in_ev = (currentTokenCount + i) % CONTEXT_WIN;
                    tokenEmbed.addRow(sequence1[i] + positionalEmbeddings(currentTokenCount + i, d), currentTokenCount + i);
                    // prepare EVs
                    for(int m1 = 0; m1 < x; m1++) {
                        for(int m2 = 0; m2 < y; m2++) {
                            std::vector<float> v(EMBEDDING, 0.0f);
                            blocks[0].b[m1][m2].EV.addRow(sequence1[i], actual_row_in_ev);
                        }
                    }
                }
                currentTokenCount += sequence1.size();
                effective_context_size = currentTokenCount;
            }
            // add tokens to fill context and then perform partial forprop to add 
            else if (currentTokenCount + sequence1.size() == CONTEXT_WIN) {
                int promptCount = 0;
                for(int i = 0; i < sequence1.size(); i++) {
                    int actual_row_in_ev = (currentTokenCount + i) % CONTEXT_WIN;
                    tokenEmbed.addRow(sequence1[i] + positionalEmbeddings(currentTokenCount + i, d), currentTokenCount + i);
                    // prepare EVs
                    for(int m1 = 0; m1 < x; m1++) {
                        for(int m2 = 0; m2 < y; m2++) {
                            std::vector<float> v(EMBEDDING, 0.0f);
                            blocks[0].b[m1][m2].EV.addRow(sequence1[i], actual_row_in_ev);
                        }
                    }
                }
                currentTokenCount += sequence1.size();
                effective_context_size = currentTokenCount;
                cuForward_ev(blockCount, effective_context_size, promptCount);
                tokenEmbed.addRow(sequence1[sequence1.size() - 1] + positionalEmbeddings(currentTokenCount, d), currentTokenCount);
                // prepare EVs
                for(int m1 = 0; m1 < x; m1++) {
                    for(int m2 = 0; m2 < y; m2++) {
                        std::vector<float> v(EMBEDDING, 0.0f);
                        blocks[1].b[m1][m2].EV.addRow(sequence1[sequence1.size() - 1], 0);
                    }
                }
                // shift to next block
                effective_context_size = 1;
                blockCount = 2;
            }
        }
        // add tokens to fill context and then perform partial forprop to add to EV, then add to next block
        else if (currentTokenCount < CONTEXT_WIN && currentTokenCount + sequence1.size() > CONTEXT_WIN) {
            int promptCount = 0;
            int dif = currentTokenCount + sequence1.size() - CONTEXT_WIN;
            int dif1 = sequence1.size() - dif;
            for(int i = 0; i < dif; i++) {
                int actual_row_in_ev = (currentTokenCount + i) % CONTEXT_WIN;
                tokenEmbed.addRow(sequence1[i] + positionalEmbeddings(currentTokenCount + i, d), currentTokenCount + i);
                for(int m1 = 0; m1 < x; m1++) {
                    for(int m2 = 0; m2 < y; m2++) {
                        std::vector<float> v(EMBEDDING, 0.0f);
                        blocks[0].b[m1][m2].EV.addRow(sequence1[i], actual_row_in_ev);
                    }
                }
            }
            currentTokenCount += dif;
            effective_context_size += dif;
            cuForward_ev(blockCount, effective_context_size, promptCount);
            // shift to next block
            blockCount = 2;
            for(int i = 0; i < dif1; i++) {
                tokenEmbed.addRow(sequence1[dif + i] + positionalEmbeddings(currentTokenCount + i, d), currentTokenCount + i);
                // prepare EVs
                for(int m1 = 0; m1 < x; m1++) {
                    for(int m2 = 0; m2 < y; m2++) {
                        std::vector<float> v(EMBEDDING, 0.0f);
                        blocks[1].b[m1][m2].EV.addRow(sequence1[i], i);
                    }
                }
            }
            effective_context_size = dif1;
            currentTokenCount += dif1;
            blockCount += 1;
        }
        // training in non-first blocks
        else {
            effective_context_size = currentTokenCount % CONTEXT_WIN;
            // add sequence1 tokens from currentTokenCount: 8192 % 1024 = 0
            if(effective_context_size == 0 && effective_context_size + sequence1.size() < CONTEXT_WIN) {
                for(int i = 0; i < sequence1.size(); i++) {
                    tokenEmbed.addRow(sequence1[i] + positionalEmbeddings(currentTokenCount + i, d), currentTokenCount + i);
                    // prepare EVs
                    for(int m1 = 0; m1 < x; m1++) {
                        for(int m2 = 0; m2 < y; m2++) {
                            blocks[blockCount - 1].b[m1][m2].EV.addRow(sequence1[i], i);
                        }
                    }
                }
                currentTokenCount += sequence1.size();
                effective_context_size = sequence1.size();
            }
            // 8194 % 1024 = 2
            else if(effective_context_size > 0 && effective_context_size + sequence1.size() < CONTEXT_WIN) {
                for(int i = 0; i < sequence1.size(); i++) {
                    tokenEmbed.addRow(sequence1[i] + positionalEmbeddings(currentTokenCount + i, d), currentTokenCount + i);
                    // prepare EVs
                    for(int m1 = 0; m1 < x; m1++) {
                        for(int m2 = 0; m2 < y; m2++) {
                            blocks[blockCount - 1].b[m1][m2].EV.addRow(sequence1[i], i);
                        }
                    }
                }
                currentTokenCount += sequence1.size();
                effective_context_size += sequence1.size();
            }
            // add tokens to fill context and then perform partial forprop to add  to EV
            else if (effective_context_size > 0 && effective_context_size + sequence1.size() == CONTEXT_WIN) {
                int promptCount = 0;
                for(int i = 0; i < sequence1.size(); i++) {
                    tokenEmbed.addRow(sequence1[i] + positionalEmbeddings(currentTokenCount + i, d), currentTokenCount + i);
                    // prepare EVs
                    for(int m1 = 0; m1 < x; m1++) {
                        for(int m2 = 0; m2 < y; m2++) {
                            blocks[blockCount - 1].b[m1][m2].EV.addRow(sequence1[i], i);
                        }
                    }
                }
                currentTokenCount += sequence1.size();
                effective_context_size += sequence1.size();
                cuForward_ev(blockCount, effective_context_size, promptCount);
                // shift to next block
                tokenEmbed.addRow(sequence1[sequence1.size() - 1] + positionalEmbeddings(currentTokenCount + 1, d), currentTokenCount);
                for(int m1 = 0; m1 < x; m1++) {
                    for(int m2 = 0; m2 < y; m2++) {
                        std::vector<float> v(EMBEDDING, 0.0f);
                        blocks[blockCount].b[m1][m2].EV.addRow(sequence1[sequence1.size() - 1], 0);
                    }
                }
                effective_context_size = 1;
                blockCount += 1;
            }
            // add tokens to fill context and then perform partial forprop to add to EV, then add to next block
            else if (effective_context_size + sequence1.size() > CONTEXT_WIN) {
                int promptCount = 0;
                int dif = currentTokenCount + sequence1.size() - CONTEXT_WIN;
                int dif1 = sequence1.size() - dif;
                for(int i = 0; i < dif; i++) {
                    tokenEmbed.addRow(sequence1[i] + positionalEmbeddings(currentTokenCount + i, d), currentTokenCount + i);
                    // prepare EVs
                    for(int m1 = 0; m1 < x; m1++) {
                        for(int m2 = 0; m2 < y; m2++) {
                            blocks[blockCount - 1].b[m1][m2].EV.addRow(sequence1[i] + positionalEmbeddings(currentTokenCount + i, d), i);
                        }
                    }
                }
                currentTokenCount += dif;
                effective_context_size += dif;
                cuForward_ev(blockCount, effective_context_size, promptCount);
                // shift to next block
                for(int i = 0; i < dif1; i++) {
                    tokenEmbed.addRow(sequence1[dif + i], currentTokenCount + i);
                    // prepare EVs
                    for(int m1 = 0; m1 < x; m1++) {
                        for(int m2 = 0; m2 < y; m2++) {
                            blocks[blockCount].b[m1][m2].EV.addRow(sequence1[i] + positionalEmbeddings(currentTokenCount + i, d), i);
                        }
                    }
                }
                currentTokenCount += dif1;
                effective_context_size = dif1;
                blockCount += 1;
            }
        }

        sequence1Count = sequence1.size();
        std::cout << "Prediction | Index | Entropy LOSS | del | e^Loss | EPOCHS | Learning Rate" << std::endl;

            // --- Train for sequence2 ---
        for (size_t i = 0; i < sequence2.size(); ++i) {
            if (currentTokenCount >= FULL_CONTEXT) {
                std::cerr << "Warning: cuTrainContext(seq1,seq2) reached FULL_CONTEXT limit. Stopping early." << std::endl;
                break;
            }

            std::vector<float>& expected_vec = sequence2[i];
            std::string& expected_str = rString[i];
            int current_block_idx = blockCount;

            int j = 0;
            std::fill(oneHotEncode.begin(), oneHotEncode.end(), 0.0f);
            oneHotEncode[indexVec[i]] = 1.0f;

            std::cout << "Training token " << i+1 << "/" << sequence2.size() << ": '" << expected_str << "'" << " at " << indexVec[i] << std::endl;
            std::cout << "current block: " << current_block_idx << " | current token count: " << currentTokenCount << " | eff. context size: " << effective_context_size <<std::endl;

            while (j < epochs) {
                    embedPlusPos = tokenEmbed + positional;
                    size_t currentBytes = static_cast<size_t>(EMBEDDING) * effective_context_size * sizeof(float);
                    CUDA_CHECK(cudaMemcpy(d_tok, embedPlusPos.mapped_data, currentBytes, cudaMemcpyHostToDevice));

                // --- K/Q Calculation ---
                dim3 threadsPerBlock(16, 16);
                dim3 numBlocks((CONTEXT_WIN + 15) / 16, (effective_context_size + 15) / 16);

                if (current_block_idx == 1) {
                    for (int layer_idx = 0; layer_idx < x; ++layer_idx) {
                        for (int parallel_idx = 0; parallel_idx < y; ++parallel_idx) {
                            auto& qMat = blocks[0].b[layer_idx][parallel_idx].MQ;
                            auto& kMat = blocks[0].b[layer_idx][parallel_idx].MK;

                            CUDA_CHECK(cudaMemcpy(d_mQ, qMat.mapped_data, projection_matrix_bytes, cudaMemcpyHostToDevice));
                            computeKQall<<<numBlocks, threadsPerBlock>>>(d_tok, d_mQ, d_Q, effective_context_size, EMBEDDING, CONTEXT_WIN);
                            CUDA_CHECK(cudaGetLastError());
                            CUDA_CHECK(cudaMemcpy(blocks[0].b[layer_idx][parallel_idx].Q.mapped_data, d_Q, KQmatbytes, cudaMemcpyDeviceToHost));

                            CUDA_CHECK(cudaMemcpy(d_mK, kMat.mapped_data, projection_matrix_bytes, cudaMemcpyHostToDevice));
                            computeKQall<<<numBlocks, threadsPerBlock>>>(d_tok, d_mK, d_K, effective_context_size, EMBEDDING, CONTEXT_WIN);
                            CUDA_CHECK(cudaMemcpy(blocks[0].b[layer_idx][parallel_idx].K.mapped_data, d_K, KQmatbytes, cudaMemcpyDeviceToHost));
                            CUDA_CHECK(cudaGetLastError());
                        }
                    }
                } else { // Subsequent blocks
                    size_t fromHereInTokenEmbed = static_cast<size_t>((CONTEXT_WIN) * (blockCount - 1) - 1) * d;
                    const float* host_src_ptr = embedPlusPos.mapped_data + fromHereInTokenEmbed;
                    CUDA_CHECK(cudaMemcpy(d_tok, host_src_ptr, currentBytes, cudaMemcpyHostToDevice));

                    for (int layer_idx = 0; layer_idx < x; ++layer_idx) {
                        for (int parallel_idx = 0; parallel_idx < y; ++parallel_idx) {
                            auto& qMat = blocks[blockCount - 1].b[layer_idx][parallel_idx].MQ;
                            auto& kMat = blocks[blockCount - 1].b[layer_idx][parallel_idx].MK;
                            auto& prevEV = blocks[blockCount - 2].b[layer_idx][parallel_idx].EV;
                            CUDA_CHECK(cudaMemcpy(d_pEV, prevEV.mapped_data, currentBytes, cudaMemcpyHostToDevice));

                            CUDA_CHECK(cudaMemcpy(d_mQ, qMat.mapped_data, projection_matrix_bytes, cudaMemcpyHostToDevice));
                            computeKQall<<<numBlocks, threadsPerBlock>>>(d_pEV, d_mQ, d_Q, effective_context_size, EMBEDDING, CONTEXT_WIN);
                            CUDA_CHECK(cudaMemcpy(blocks[blockCount - 1].b[layer_idx][parallel_idx].Q.mapped_data, d_Q, KQmatbytes, cudaMemcpyDeviceToHost));

                            CUDA_CHECK(cudaMemcpy(d_mK, kMat.mapped_data, projection_matrix_bytes, cudaMemcpyHostToDevice));
                            computeKQall<<<numBlocks, threadsPerBlock>>>(d_tok, d_mK, d_K, effective_context_size, EMBEDDING, CONTEXT_WIN);
                            CUDA_CHECK(cudaMemcpy(blocks[blockCount - 1].b[layer_idx][parallel_idx].K.mapped_data, d_K, KQmatbytes, cudaMemcpyDeviceToHost));
                        }
                    }
                }
                CUDA_CHECK(cudaDeviceSynchronize());

                // --- Forward Pass ---
                int current_prompt_count_in_block = effective_context_size % CONTEXT_WIN == 0 ? CONTEXT_WIN : effective_context_size % CONTEXT_WIN;
                cuForward(current_block_idx, effective_context_size, current_prompt_count_in_block);

                // --- Get EH output from all layers ---
                std::fill(otok.begin(), otok.end(), 0.0f);
                if (y > 0) {
                    for (int layer_idx = 0; layer_idx < x; ++layer_idx) {
                        for (int k = 0; k < d; ++k) {
                            otok[(layer_idx * d) + k] = blocks[blockCount-1].b[layer_idx][y - 1].EH[k];
                        }
                    }
                }
                for(size_t k_dim = 0; k_dim < static_cast<size_t>(x*d); k_dim++) {
                    if (std::isnan(otok[k_dim])) { otok[k_dim] = 0.0f; }
                    else if (std::isinf(otok[k_dim])) { otok[k_dim] = std::copysign((std::numeric_limits<float>::max)(), otok[k_dim]); }
                }

                // --- Prediction with Scores ---
                CUDA_CHECK(cudaMemcpy(d_otok_buffer, otok.data(), otok.size() * sizeof(float), cudaMemcpyHostToDevice));
                kernelComputePredictionWithScores<<<1, 1>>>(d_otok_buffer, d_deEmbeddings, d_predictions, d_result_index, d * x, vocabsize);
                CUDA_CHECK(cudaGetLastError());

                int host_indexForToken = -1;
                CUDA_CHECK(cudaMemcpy(&host_indexForToken, d_result_index, sizeof(int), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(pred.data(), d_predictions, predBytes, cudaMemcpyDeviceToHost));
                indexForToken = host_indexForToken;
                pred = softmax(pred);

                // --- Error Calculation & Logging ---
                current_error = - std::log(pred[indexVec[i]] + 1e-15f);
                float del = current_error - prev_error;
                std::string predicted_token_str = (indexForToken >= 0 && indexForToken < static_cast<unsigned int>(tokens.size()))
                                                ? tokens[indexForToken] : "INVALID_INDEX";

                std::cout << predicted_token_str << " ( " << indexForToken << " ) \t: "
                        << current_error << " | " << del << " | "
                        << std::exp(current_error) << " | " << j+1 << " | " << learning << std::endl;

                if (predicted_token_str == expected_str && predicted_token_str != "INVALID_INDEX") {
                    std::cout << "Token '" << expected_str << "' predicted correctly after " << j+1 << " epochs." << std::endl;
                    if(predicted_token_str != "</s>")
                        std::cout << "              -------------- To Next Token --------------              " << std::endl;
                    break;
                }
                if(j == epochs - 1) {
                    std::cout << "Reached maximum epochs. Increasing Epochs by " << EPOCHS/2 << "." << std::endl;
                    epochs += EPOCHS/2;
                }

                // --- Backward Pass ---
                std::vector<float> gradEH(d * x, 0.0f);
                cuUpdateDeEmbeddings(deEmbeddings, pred, oneHotEncode, learning, lambda_L1, lambda_L2, gradEH);

                std::vector<std::vector<float>> targets_for_heads(x, std::vector<float>(EMBEDDING, 0.0f));
                for(int head_idx = 0; head_idx < x; ++head_idx) {
                    for(int eidx = 0; eidx < EMBEDDING; ++eidx) {
                        float gradient = learning * (gradEH[(head_idx * EMBEDDING) + eidx] + (lambda_L1 * embeddings(indexForToken, eidx)) + (2.0f * lambda_L2 * embeddings(indexForToken, eidx)));
                        if (fabs(gradient) >= MAX_GRAD_CLIP) gradient = std::copysign(MAX_GRAD_CLIP, gradient);
                        targets_for_heads[head_idx][eidx] = otok[(head_idx * EMBEDDING) + eidx] - gradient;
                    }
                }

                cuBackwardContext(targets_for_heads, current_block_idx);
                cuUpdateEmbeddings(embeddings, blocks[blockCount-1].gradToken, learning, lambda_L1, lambda_L2, 12454);
                cuUpdateEmbeddings(tokenEmbed, blocks[blockCount-1].gradToken, learning, lambda_L1, lambda_L2, effective_context_size);

                totalLearning += learning;
                prev_error = current_error;
                totalBCELoss += current_error;
                totalBCEPerplexity += std::exp(current_error);
                j++;
            }
            // --- Update Host State ---
            trainCount++;
            epochCount += j;
            resCount++;
            tokenEmbed.addRow(sequence2[i], currentTokenCount);
            currentTokenCount++;
            effective_context_size++;

            if(currentTokenCount > 0 && currentTokenCount % CONTEXT_WIN == 0) {
                blockCount += 1;
                blockShifted = true;
                effective_context_size = 0; // Reset for new block
                std::cout << "----> Going to Next block in model -> " << blockCount - 1 << " to " << blockCount << std::endl;
            } else {
                blockShifted = false;
            }
        }
        
        learning = initialLearning;
        epochs = initial_epochs;
    }
    catch (const std::exception& e) {
        std::cerr << "Standard Exception in clTrain(sentence): " << e.what() << std::endl;
        epochs = initial_epochs;
        throw;
    }

    // --- Free temporary device memory ---
    CUDA_CHECK(cudaFree(d_deEmbeddings));
    CUDA_CHECK(cudaFree(d_tokenEmbed));
    CUDA_CHECK(cudaFree(d_expected_token));
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_mQ));
    CUDA_CHECK(cudaFree(d_mK));
    CUDA_CHECK(cudaFree(d_tok));
    CUDA_CHECK(cudaFree(d_pEV));
    CUDA_CHECK(cudaFree(d_otok_buffer));
    CUDA_CHECK(cudaFree(d_predictions));
    CUDA_CHECK(cudaFree(d_result_index));
}

#endif