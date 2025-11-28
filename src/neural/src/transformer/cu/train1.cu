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
 * @brief (CUDA) Train the transformer on sentences or paragraphs
 * @param sentence token embedding of sentence (on host)
 * @param rString sentence tokens (on host)
 */
void transformer::cuTrain(std::vector<std::vector<float>>& sentence, std::vector<std::string>& rString)
{
    // --- Basic validation ---
    if(sentence.size() + currentTokenCount > FULL_CONTEXT) {
        throw std::runtime_error("cuTrain(sentence): Previous tokens and sentence will exceed the FULL CONTEXT '-'");
    }
    if (sentence.size() > FULL_CONTEXT) {
        throw std::runtime_error("cuTrain(sentence): Sentence size (" + std::to_string(sentence.size()) + ") exceeds FULL_CONTEXT (" + std::to_string(FULL_CONTEXT) + ").");
    }
    if (sentence.empty() || sentence.size() != rString.size()) {
        std::cout << "sentence.size(): " << sentence.size() << ", rString.size(): " << rString.size() << std::endl;
        throw std::runtime_error("cuTrain(sentence): Sentence embeddings/strings mismatch or empty.");
    }
    if (!sentence.empty() && sentence[0].size() != static_cast<size_t>(d)) {
        throw std::runtime_error("cuTrain(sentence): Sentence embedding dimension mismatch. Expected " + std::to_string(d) + ", got " + std::to_string(sentence[0].size()));
    }

    // --- Variable Initialization ---
    float initial_learning_rate = learning; // Store initial learning rate
    float current_error = 0.0f;
    float prev_error = 0.0f;
    int initial_epochs = epochs;
    bool blockShifted = 0;
    int effective_context_size = 0;

    // --- Device Buffer Allocation & H->D Transfer ---
    size_t totalTokenEmbedFloats = static_cast<size_t>(m) * CONTEXT_WIN * d;
    size_t tokenEmbedBytes = totalTokenEmbedFloats * sizeof(float);
    size_t embeddingsBytes = static_cast<size_t>(vocabsize) * d * sizeof(float);
    size_t singleTokenBytes = static_cast<size_t>(d) * sizeof(float);
    size_t projection_matrix_bytes = static_cast<size_t>(CONTEXT_WIN) * EMBEDDING * sizeof(float);
    size_t KQmatbytes = static_cast<size_t>(CONTEXT_WIN) * CONTEXT_WIN * sizeof(float);
    size_t embedding_bytes_loc = static_cast<size_t>(EMBEDDING) * CONTEXT_WIN * sizeof(float);

    float *d_embeddings, *d_tokenEmbed, *d_expected_token, *d_pEV;
    float *d_Q, *d_K, *d_mQ, *d_mK, *d_tok_cl;
    int* d_result_index;
    
    CUDA_CHECK(cudaMalloc(&d_embeddings, embeddingsBytes));
    CUDA_CHECK(cudaMemcpy(d_embeddings, embeddings.mapped_data, embeddingsBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_tokenEmbed, tokenEmbedBytes));
    CUDA_CHECK(cudaMalloc(&d_expected_token, singleTokenBytes));
    CUDA_CHECK(cudaMalloc(&d_Q, KQmatbytes));
    CUDA_CHECK(cudaMalloc(&d_K, KQmatbytes));
    CUDA_CHECK(cudaMalloc(&d_mQ, projection_matrix_bytes));
    CUDA_CHECK(cudaMalloc(&d_mK, projection_matrix_bytes));
    CUDA_CHECK(cudaMalloc(&d_tok_cl, embedding_bytes_loc));
    CUDA_CHECK(cudaMalloc(&d_pEV, embedding_bytes_loc));
    CUDA_CHECK(cudaMalloc(&d_result_index, sizeof(int)));

    try {
        otok.clear(); otok.resize(d, 0.0f);

        // --- set all tokens to tokenEmbed ---
        std::fill(tokenEmbed.mapped_data, tokenEmbed.mapped_data + totalTokenEmbedFloats, 0.0f);
        std::fill(positional.mapped_data, positional.mapped_data + totalTokenEmbedFloats, 0.0f);
        std::fill(embedPlusPos.mapped_data, embedPlusPos.mapped_data + totalTokenEmbedFloats, 0.0f);
        for(int i = 0; i < rString.size(); i++) {
            tokenEmbed.addRow(sentence[i], i);
            positional.addRow(positionalEmbeddings(i, d), i);
            // add the last token of local context to first token of next local
            if(i + 1 % CONTEXT_WIN == 0 && i + 1 < rString.size()) {
                tokenEmbed.addRow(sentence[i], i + 1);
                positional.addRow(positionalEmbeddings(i, d), i + 1);
                i++;
            }
        }
        effective_context_size += 1;
        currentTokenCount += 1;
        sequence1Count = 1, blockCount = 1;
        std::cout << "Predicted (Index) | CE Loss | del (cur - pre) | e^Loss | Epochs | Learning Rate" << std::endl;

        // --- Train for each subsequent token in the sentence (i=1 to N-1) ---
        for (size_t i = 1; i < sentence.size(); ++i) {
            if (currentTokenCount >= FULL_CONTEXT) {
                std::cerr << "Warning: cuTrain(sentence) reached FULL_CONTEXT limit ("
                          << currentTokenCount << "). Stopping training early at sentence index " << i << "." << std::endl;
                break;
            }
            int current_block_idx = blockCount;
            if (current_block_idx <= 0 || current_block_idx > m) {
                throw std::out_of_range("cuTrain(sentence): Calculated current_block_idx (" + std::to_string(current_block_idx) + ") is out of range [1, " + std::to_string(m) + "].");
            }

            // Target token for this iteration
            std::vector<float>& expected_vec = sentence[i];
            std::string& expected_str = rString[i];
            CUDA_CHECK(cudaMemcpy(d_expected_token, expected_vec.data(), singleTokenBytes, cudaMemcpyHostToDevice));

            // --- Training Loop for token i ---
            int j = 0;

            std::cout << "Training token " << i+1 << "/" << sentence.size() << ": '" << expected_str << "'" << " at " << indexVec[i] << std::endl;
            std::cout << "current block: " << current_block_idx << " | current token count: " << currentTokenCount << " | eff. context size: " << effective_context_size <<std::endl;

            while (j < epochs) {
                embedPlusPos = tokenEmbed + positional;
                size_t currentBytes = static_cast<size_t>(EMBEDDING) * effective_context_size * sizeof(float);

                // --- K/Q Calculation ---
                dim3 threadsPerBlock(16, 16);
                dim3 numBlocks((CONTEXT_WIN + 15) / 16, (effective_context_size + 15) / 16);

                if (current_block_idx == 1) {
                    CUDA_CHECK(cudaMemcpy(d_tok_cl, embedPlusPos.mapped_data, currentBytes, cudaMemcpyHostToDevice));
                    for (int layer_idx = 0; layer_idx < x; ++layer_idx) {
                        for (int parallel_idx = 0; parallel_idx < y; ++parallel_idx) {
                            auto& qMat = blocks[0].b[layer_idx][parallel_idx].MQ;
                            auto& kMat = blocks[0].b[layer_idx][parallel_idx].MK;

                            // Queries
                            CUDA_CHECK(cudaMemcpy(d_mQ, qMat.mapped_data, projection_matrix_bytes, cudaMemcpyHostToDevice));
                            computeKQall<<<numBlocks, threadsPerBlock>>>(d_tok_cl, d_mQ, d_Q, effective_context_size, EMBEDDING, CONTEXT_WIN);
                            CUDA_CHECK(cudaGetLastError());
                            CUDA_CHECK(cudaMemcpy(blocks[0].b[layer_idx][parallel_idx].Q.mapped_data, d_Q, KQmatbytes, cudaMemcpyDeviceToHost));

                            // Keys
                            CUDA_CHECK(cudaMemcpy(d_mK, kMat.mapped_data, projection_matrix_bytes, cudaMemcpyHostToDevice));
                            computeKQall<<<numBlocks, threadsPerBlock>>>(d_tok_cl, d_mK, d_K, effective_context_size, EMBEDDING, CONTEXT_WIN);
                            CUDA_CHECK(cudaGetLastError());
                            CUDA_CHECK(cudaMemcpy(blocks[0].b[layer_idx][parallel_idx].K.mapped_data, d_K, KQmatbytes, cudaMemcpyDeviceToHost));
                        }
                    }
                } else {
                    size_t fromHereInTokenEmbed = static_cast<size_t>((CONTEXT_WIN) * (blockCount - 1) - 1) * d;
                    const float* host_src_ptr = embedPlusPos.mapped_data + fromHereInTokenEmbed;
                    CUDA_CHECK(cudaMemcpy(d_tok_cl, host_src_ptr, currentBytes, cudaMemcpyHostToDevice));
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
                            computeKQall<<<numBlocks, threadsPerBlock>>>(d_tok_cl, d_mK, d_K, effective_context_size, EMBEDDING, CONTEXT_WIN);
                            CUDA_CHECK(cudaGetLastError());
                            CUDA_CHECK(cudaMemcpy(blocks[blockCount - 1].b[layer_idx][parallel_idx].K.mapped_data, d_K, KQmatbytes, cudaMemcpyDeviceToHost));
                        }
                    }
                }
                CUDA_CHECK(cudaDeviceSynchronize());

                // --- Forward Pass ---
                cuForward(current_block_idx, effective_context_size, sequence1Count);

                // --- Get EH output & Prediction ---
                std::fill(otok.begin(), otok.end(), 0.0f);
                if (y > 0) {
                    for (int j_layer = 0; j_layer < x; ++j_layer) {
                        for (int k = 0; k < d; ++k) {
                            otok[k] += blocks[blockCount-1].b[j_layer][y - 1].EH[k];
                        }
                    }
                    for(size_t k_dim = 0; k_dim < static_cast<size_t>(d); k_dim++) {
                        if (std::isnan(otok[k_dim])) { otok[k_dim] = 0.0f; }
                        else if (std::isinf(otok[k_dim])) { otok[k_dim] = std::copysign((std::numeric_limits<float>::max)(), otok[k_dim]); }
                    }
                } else {
                    std::cerr << "Warning: cuForward called with y = 0 columns. Cannot accumulate EH." << std::endl;
                }

                float* d_otok_buffer;
                CUDA_CHECK(cudaMalloc(&d_otok_buffer, singleTokenBytes));
                CUDA_CHECK(cudaMemcpy(d_otok_buffer, otok.data(), singleTokenBytes, cudaMemcpyHostToDevice));
                
                kernelComputePrediction<<<1, 1>>>(d_otok_buffer, d_embeddings, d_result_index, d, vocabsize);
                CUDA_CHECK(cudaGetLastError());

                int host_indexForToken = -1;
                CUDA_CHECK(cudaMemcpy(&host_indexForToken, d_result_index, sizeof(int), cudaMemcpyDeviceToHost));
                indexForToken = host_indexForToken;

                CUDA_CHECK(cudaFree(d_otok_buffer));

                current_error = binaryCrossEntropy(expected_vec, otok);
                float del = current_error - prev_error;
                std::string predicted_token_str = (indexForToken >= 0 && indexForToken < static_cast<unsigned long long>(tokens.size()))
                                                    ? tokens[indexForToken] : "INVALID_INDEX";

                std::cout << predicted_token_str << " ( " << indexForToken << " ) \t: "
                        << current_error << " | " << del << " | "
                        << std::exp(current_error) << " | " << j+1 << " | " << learning << std::endl;

                if (predicted_token_str == expected_str && predicted_token_str != "INVALID_INDEX") {
                    std::cout << "Token '" << expected_str << "' predicted correctly after " 
                            << j+1 << " epochs. Moving to next token." << std::endl;
                    learning = initial_learning_rate;
                    if(predicted_token_str != "</s>")
                        std::cout << "              -------------- To Next Token --------------              " << std::endl;
                    break;
                }

                cuBackward(expected_vec, current_block_idx); // Backward pass
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
                tokenEmbed.addRow(sentence[i], currentTokenCount - 1); // repeat last token to new block
                positional.addRow(positionalEmbeddings(currentTokenCount - 1, d), currentTokenCount - 1);
                std::cout << "----> Going to Next block in model -> " << blockCount - 1 << " to " << blockCount << std::endl;
            } else {
                blockShifted = false;
            }
        }
    }
    catch (const std::runtime_error& e) {
        std::cerr << "Standard Exception in cuTrain(sentence): " << e.what() << std::endl;
        epochs = initial_epochs;
        throw;
    }

    // --- Free temporary device memory ---
    CUDA_CHECK(cudaFree(d_embeddings));
    CUDA_CHECK(cudaFree(d_tokenEmbed));
    CUDA_CHECK(cudaFree(d_expected_token));
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_mQ));
    CUDA_CHECK(cudaFree(d_mK));
    CUDA_CHECK(cudaFree(d_tok_cl));
    CUDA_CHECK(cudaFree(d_pEV));
    CUDA_CHECK(cudaFree(d_result_index));
}

#endif  // USE_CU