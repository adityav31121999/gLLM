#ifdef USE_CUDA

#include "include/transformer.hpp"
#include "include/block.hpp"
#include "include/attention.hpp" // Include necessary headers
#include "include/mlp.hpp"
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
    int initial_token_count = currentTokenCount; // Store initial count
    bool blockShifted = false;
    int effective_context_size = 0;
    int start = 0;

    // --- Device Buffer Allocation & H->D Transfer ---
    size_t totalTokenEmbedFloats = static_cast<size_t>(m) * CONTEXT_WIN * d;
    size_t tokenEmbedBytes = totalTokenEmbedFloats * sizeof(float);
    size_t embeddingsBytes = static_cast<size_t>(vocabsize) * d * sizeof(float);
    size_t singleTokenBytes = static_cast<size_t>(d) * sizeof(float);
    size_t projection_matrix_bytes = static_cast<size_t>(CONTEXT_WIN) * EMBEDDING * sizeof(float);
    size_t KQmatbytes = static_cast<size_t>(CONTEXT_WIN) * CONTEXT_WIN * sizeof(float);
    size_t embedding_bytes_loc = static_cast<size_t>(EMBEDDING) * CONTEXT_WIN * sizeof(float);

    float *d_embeddings, *d_tokenEmbed, *d_expected_token;
    float *d_Q, *d_K, *d_mQ, *d_mK, *d_tok;
    int* d_result_index;

    CUDA_CHECK(cudaMalloc(&d_embeddings, embeddingsBytes));
    CUDA_CHECK(cudaMemcpy(d_embeddings, embeddings.mapped_data, embeddingsBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_tokenEmbed, tokenEmbedBytes));
    CUDA_CHECK(cudaMalloc(&d_expected_token, singleTokenBytes));
    CUDA_CHECK(cudaMalloc(&d_Q, KQmatbytes));
    CUDA_CHECK(cudaMalloc(&d_K, KQmatbytes));
    CUDA_CHECK(cudaMalloc(&d_mQ, projection_matrix_bytes));
    CUDA_CHECK(cudaMalloc(&d_mK, projection_matrix_bytes));
    CUDA_CHECK(cudaMalloc(&d_tok, embedding_bytes_loc));
    CUDA_CHECK(cudaMalloc(&d_result_index, sizeof(int)));

    otok.clear(); otok.resize(d, 0.0f);

    // --- Context Initialization ---
    std::vector<float> flat_host_tokenEmbed;
    if (currentTokenCount == 0) {
        blockCount = 1;
        tokenEmbed.addRow(sentence[0], 0);
        effective_context_size = 1;
        currentTokenCount += 1;
        start = 1;
    } else if (currentTokenCount > 0 && currentTokenCount < CONTEXT_WIN) {
        blockCount = 1;
        effective_context_size = currentTokenCount;
        flat_host_tokenEmbed.assign(tokenEmbed.mapped_data, tokenEmbed.mapped_data + static_cast<size_t>(currentTokenCount) * d);
        start = 0;
    } else {
        effective_context_size = currentTokenCount % CONTEXT_WIN;
        if (effective_context_size == 0 && currentTokenCount > 0) { // exactly at a boundary
            effective_context_size = CONTEXT_WIN;
            blockCount = currentTokenCount / CONTEXT_WIN;
        } else {
            blockCount = (currentTokenCount / CONTEXT_WIN) + 1;
        }
        flat_host_tokenEmbed.assign(tokenEmbed.mapped_data, tokenEmbed.mapped_data + static_cast<size_t>(currentTokenCount) * d);
        start = 0;
    }

    if (!flat_host_tokenEmbed.empty()) {
        CUDA_CHECK(cudaMemcpy(d_tokenEmbed, flat_host_tokenEmbed.data(), flat_host_tokenEmbed.size() * sizeof(float), cudaMemcpyHostToDevice));
    }

    sequence1Count = 1;
    std::cout << "Prediction | Index | Entropy LOSS | del | e^Loss | EPOCHS | Learning Rate" << std::endl;

    // --- Train for each subsequent token in the sentence ---
    for (size_t i = start; i < sentence.size(); ++i) {
        if (currentTokenCount >= FULL_CONTEXT) {
            std::cerr << "Warning: cuTrain(sentence) reached FULL_CONTEXT limit ("
                      << currentTokenCount << "). Stopping training early at sentence index " << i << "." << std::endl;
            break;
        }

        // Target token for this iteration
        std::vector<float>& expected_vec = sentence[i];
        std::string& expected_str = rString[i];
        CUDA_CHECK(cudaMemcpy(d_expected_token, expected_vec.data(), singleTokenBytes, cudaMemcpyHostToDevice));

        int current_block_idx = blockCount;
        if (current_block_idx <= 0 || current_block_idx > m) {
            throw std::out_of_range("cuTrain(sentence): Calculated current_block_idx (" + std::to_string(current_block_idx) + ") is out of range [1, " + std::to_string(m) + "].");
        }

        // --- Training Loop for token i ---
        int j = 0;

        std::cout << "Training token " << i+1 << "/" << sentence.size() << ": '" << expected_str << "'" << " at " << indexVec[i] << std::endl;
        std::cout << "current block: " << current_block_idx << " | current token count: " << currentTokenCount << " | eff. context size: " << effective_context_size <<std::endl;
        size_t currentBytes = static_cast<size_t>(EMBEDDING) * effective_context_size * sizeof(float);
        CUDA_CHECK(cudaMemcpy(d_tok, tokenEmbed.mapped_data, currentBytes, cudaMemcpyHostToDevice));

        while (j < epochs) {
            current_error = 0.0f;

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
                        CUDA_CHECK(cudaMemcpy(blocks[0].b[layer_idx][parallel_idx].Q.mapped_data, d_Q, KQmatbytes, cudaMemcpyDeviceToHost));

                        CUDA_CHECK(cudaMemcpy(d_mK, kMat.mapped_data, projection_matrix_bytes, cudaMemcpyHostToDevice));
                        computeKQall<<<numBlocks, threadsPerBlock>>>(d_tok, d_mK, d_K, effective_context_size, EMBEDDING, CONTEXT_WIN);
                        CUDA_CHECK(cudaMemcpy(blocks[0].b[layer_idx][parallel_idx].K.mapped_data, d_K, KQmatbytes, cudaMemcpyDeviceToHost));
                    }
                }
            } else {
                // Logic for subsequent blocks (requires passing previous block's EV)
                // This part is complex and needs careful implementation of EV passing.
                // For now, we focus on the single-block case which is the most common for this function.
                throw std::runtime_error("cuTrain for block > 1 not fully implemented in this refactor.");
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            // --- Forward Pass ---
            cuForward(current_block_idx, effective_context_size, sequence1Count);

            // --- Get EH output ---
            if (y > 0) {
                for (int j_layer = 0; j_layer < x; ++j_layer) {
                    for (int k = 0; k < d; ++k) {
                        otok[k] += blocks[blockCount-1].b[j_layer][y - 1].EH[k];
                    }
                }
            } else {
                std::cerr << "Warning: cuForward called with y = 0 columns. Cannot accumulate EH." << std::endl;
            }

            // --- Prediction ---
            float* d_otok_buffer;
            CUDA_CHECK(cudaMalloc(&d_otok_buffer, singleTokenBytes));
            CUDA_CHECK(cudaMemcpy(d_otok_buffer, otok.data(), singleTokenBytes, cudaMemcpyHostToDevice));
            
            // Launch the prediction kernel with a single thread to find the best token index
            kernelComputePrediction<<<1, 1>>>(d_otok_buffer, d_embeddings, d_result_index, d, vocabsize);
            CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors

            // Copy the resulting index from the device back to the host
            int host_indexForToken = -1;
            CUDA_CHECK(cudaMemcpy(&host_indexForToken, d_result_index, sizeof(int), cudaMemcpyDeviceToHost));
            indexForToken = host_indexForToken;

            CUDA_CHECK(cudaFree(d_otok_buffer));

            // --- Error Calculation & Logging ---
            std::vector<float> expv = sigmoid(expected_vec);
            current_error = binaryCrossEntropy(expv, otok);
            std::string predicted_token_str = (host_indexForToken >= 0 && host_indexForToken < static_cast<unsigned long long>(tokens.size()))
                                                  ? tokens[host_indexForToken] : "INVALID_INDEX";
            std::cout << predicted_token_str << "\t: " << indexForToken << " | "
                      << current_error << " | " << current_error - prev_error << " | "
                      << std::exp(current_error) << " | " << j+1 << " | " << learning << std::endl;

            if (predicted_token_str == expected_str && predicted_token_str != "INVALID_INDEX") {
                std::cout << "Token '" << expected_str << "' predicted correctly after " 
                          << j+1 << " epochs. Moving to next token." << std::endl;
                learning *= (current_error < prev_error) ? 0.95 : 1.05;
                if(predicted_token_str != "</s>")
                    std::cout << "              -------------- To Next Token --------------              " << std::endl;
                break;
            }
            if(j == epochs - 1) {
                std::cout << "Reached maximum epochs (" << epochs << ") for current token without correct prediction." << std::endl;
                std::cout << "Increasing Epochs by 15." << std::endl;
                epochs += 15;
            }

            // --- Learning Rate & Backward Pass ---
            if(current_error < prev_error) {
                if(j <= 6) learning *= 1.2;
                else if (j % 5 == 0) learning *= (1.025 + (j/6)*0.25);
            } else {
                if(j <= 6) learning *= 0.9;
                else if (j % 5 == 0) learning *= (1.0f - (j/6)*0.02);
            }

            cuBackward(expected_vec, current_block_idx);
            totalLearning += learning;
            prev_error = current_error;
            totalBCELoss += current_error;
            totalBCEPerplexity += std::exp(current_error);
            j++;
        }

        // --- Update Host State ---
        trainCount++;
        epochCount += j;
        totalLearning += learning;
        prev_error = current_error;
        totalBCELoss += current_error;
        totalBCEPerplexity += std::exp(current_error);

        if (tokenEmbed.mapped_data && static_cast<size_t>(currentTokenCount) < tokenEmbed.row && tokenEmbed.col == static_cast<size_t>(d)) {
            tokenEmbed.addRow(expected_vec, currentTokenCount);
        }

        currentTokenCount++;
        effective_context_size++;
        if(currentTokenCount > 0 && currentTokenCount % CONTEXT_WIN == 0) {
            blockCount += 1;
            blockShifted = true;
            std::cout << "----> Going to Next block in model -> " << blockCount - 1 << " to " << blockCount << std::endl;
        } else {
            blockShifted = false;
        }
    }

    // --- Free temporary device memory ---
    CUDA_CHECK(cudaFree(d_embeddings));
    CUDA_CHECK(cudaFree(d_tokenEmbed));
    CUDA_CHECK(cudaFree(d_expected_token));
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_mQ));
    CUDA_CHECK(cudaFree(d_mK));
    CUDA_CHECK(cudaFree(d_tok));
    CUDA_CHECK(cudaFree(d_result_index));
}

#endif  // USE_CUDA