#ifdef USE_CU
#include "include/transformer.hpp"
#include <maths.hpp>
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
 * @brief (CUDA) Train the transformer on sentences or paragraphs
 * @param sentence token embedding of sentence (on host)
 * @param rString sentence tokens (on host)
 */
void transformer::cuTrainContext(std::vector<std::vector<float>>& sentence, std::vector<std::string>& rString)
{
    // --- Basic validation ---
    if(sentence.size() + currentTokenCount > FULL_CONTEXT) {
        throw std::runtime_error("cuTrainContext(sentence): Previous tokens and sentence will exceed the FULL CONTEXT '-'");
    }
    if (sentence.size() > FULL_CONTEXT) {
        throw std::runtime_error("cuTrainContext(sentence): Sentence size (" + std::to_string(sentence.size()) + ") exceeds FULL_CONTEXT (" + std::to_string(FULL_CONTEXT) + ").");
    }
    if (sentence.empty() || sentence.size() != rString.size()) {
        throw std::runtime_error("cuTrainContext(sentence): Sentence embeddings/strings mismatch or empty.");
    }
    if (!sentence.empty() && sentence[0].size() != static_cast<size_t>(d)) {
        throw std::runtime_error("cuTrainContext(sentence): Sentence embedding dimension mismatch.");
    }

    // --- Variable Initialization ---
    float initial_learning_rate = learning;
    float current_error = 0.0f;
    float prev_error = 0.0f;
    int initial_epochs = epochs;
    bool blockShifted = false;
    int effective_context_size = 0;

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

        // --- set all tokens to tokenEmbed ---
        std::fill(tokenEmbed.mapped_data, tokenEmbed.mapped_data + totalTokenEmbedFloats, 0.0f);
        std::fill(positional.mapped_data, positional.mapped_data + totalTokenEmbedFloats, 0.0f);
        std::fill(embedPlusPos.mapped_data, embedPlusPos.mapped_data + totalTokenEmbedFloats, 0.0f);
        for(int i = 0; i < rString.size(); i++) {
            tokenEmbed.addRow(sentence[i], i);
            positional.addRow(positionalEmbeddings(i, d), i);
            if(i + 1 % CONTEXT_WIN == 0 && i + 1 < rString.size()) {
                tokenEmbed.addRow(sentence[i], i + 1);
                positional.addRow(positionalEmbeddings(i, d), i + 1);
                i++;
            }
        }
        effective_context_size += 1;
        currentTokenCount += 1;
        sequence1Count = 1, blockCount = 1;
        std::cout << "Predicted (Index;) | CE Loss | del (cur - pre) | e^Loss | Epochs | Learning Rate" << std::endl;

        // --- Train for each subsequent token in the sentence (i=1 to N-1) ---
        for (size_t i = 1; i < sentence.size(); ++i) {
            if (currentTokenCount >= FULL_CONTEXT) {
                std::cerr << "Warning: cuTrainContext(sentence) reached FULL_CONTEXT limit. Stopping early." << std::endl;
                break;
            }

            std::vector<float>& expected_vec = sentence[i];
            std::string& expected_str = rString[i];
            int current_block_idx = blockCount;

            int j = 0;
            std::fill(oneHotEncode.begin(), oneHotEncode.end(), 0.0f);
            oneHotEncode[indexVec[i]] = 1.0f;

            std::cout << "Training token " << i+1 << "/" << sentence.size() << ": '" << expected_str << "'" << " at " << indexVec[i] << std::endl;
            std::cout << "current block: " << current_block_idx << " | current token count: " << currentTokenCount << " | eff. context size: " << effective_context_size <<std::endl;

            while (j < epochs) {
                embedPlusPos = tokenEmbed + positional;
                size_t currentBytes = static_cast<size_t>(EMBEDDING) * effective_context_size * sizeof(float);

                // --- K/Q Calculation ---
                dim3 threadsPerBlock(16, 16);
                dim3 numBlocks((CONTEXT_WIN + 15) / 16, (effective_context_size + 15) / 16);

                if (current_block_idx == 1) {
                    CUDA_CHECK(cudaMemcpy(d_tok, embedPlusPos.mapped_data, currentBytes, cudaMemcpyHostToDevice));
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
                            CUDA_CHECK(cudaGetLastError());
                            CUDA_CHECK(cudaMemcpy(blocks[blockCount - 1].b[layer_idx][parallel_idx].Q.mapped_data, d_Q, KQmatbytes, cudaMemcpyDeviceToHost));

                            CUDA_CHECK(cudaMemcpy(d_mK, kMat.mapped_data, projection_matrix_bytes, cudaMemcpyHostToDevice));
                            computeKQall<<<numBlocks, threadsPerBlock>>>(d_tok, d_mK, d_K, effective_context_size, EMBEDDING, CONTEXT_WIN);
                            CUDA_CHECK(cudaMemcpy(blocks[blockCount - 1].b[layer_idx][parallel_idx].K.mapped_data, d_K, KQmatbytes, cudaMemcpyDeviceToHost));
                            CUDA_CHECK(cudaGetLastError());
                        }
                    }
                }
                CUDA_CHECK(cudaDeviceSynchronize());

                // --- Forward Pass ---
                cuForward(current_block_idx, effective_context_size, sequence1Count);

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
                    else if (std::isinf(otok[k_dim])) { 
                        otok[k_dim] = std::copysign((std::numeric_limits<float>::max)(), otok[k_dim]); 
                    }
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
                    learning = initial_learning_rate;
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
                        float gradient = learning * (gradEH[(head_idx * EMBEDDING) + eidx]
                                                  + (lambda_L1 * embeddings(indexForToken, eidx))
                                                  + (2.0f * lambda_L2 * embeddings(indexForToken, eidx)));
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
            currentTokenCount++;
            effective_context_size++;

            if(currentTokenCount > 0 && currentTokenCount % CONTEXT_WIN == 0) {
                blockCount += 1;
                blockShifted = true;
                tokenEmbed.addRow(sentence[i], currentTokenCount - 1);
                positional.addRow(positionalEmbeddings(currentTokenCount - 1, d), currentTokenCount - 1);
                std::cout << "----> Going to Next block in model -> " << blockCount - 1 << " to " << blockCount << std::endl;
            } else {
                blockShifted = false;
            }
        }
        learning = initial_learning_rate;
    } catch (const std::runtime_error& e) {
        std::cerr << "Standard Exception in cuTrainContext(sentence): " << e.what() << std::endl;
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