#ifdef USE_CUDA
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
    int initial_token_count = currentTokenCount;
    bool blockShifted = false;
    int effective_context_size = 0;
    int start = 0;

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

    otok.clear(); otok.resize(d * x, 0.0f);
    pred.clear(); pred.resize(vocabsize, 0.0f);
    oneHotEncode.clear(); oneHotEncode.resize(vocabsize, 0.0f);

    // --- Context Initialization ---
    if (currentTokenCount == 0) {
        blockCount = 1;
        tokenEmbed.addRow(sentence[0], 0);
        effective_context_size = 1;
        currentTokenCount += 1;
        start = 1;
    } else if (currentTokenCount > 0 && currentTokenCount < CONTEXT_WIN) {
        blockCount = 1;
        effective_context_size = currentTokenCount;
        start = 0;
    } else {
        effective_context_size = currentTokenCount % CONTEXT_WIN;
        if (effective_context_size == 0 && currentTokenCount > 0) {
            effective_context_size = CONTEXT_WIN;
            blockCount = currentTokenCount / CONTEXT_WIN;
        } else {
            blockCount = (currentTokenCount / CONTEXT_WIN) + 1;
        }
        start = 0;
    }

    sequence1Count = 1;
    std::cout << "Prediction | Index | Entropy LOSS | del | e^Loss | EPOCHS | Learning Rate" << std::endl;

    // --- Train for each subsequent token in the sentence ---
    for (size_t i = start; i < sentence.size(); ++i) {
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
            } else { // Subsequent blocks
                size_t fromHereInTokenEmbed = static_cast<size_t>(CONTEXT_WIN) * (blockCount - 1) * d * sizeof(float);
                const float* host_src_ptr = tokenEmbed.mapped_data + (fromHereInTokenEmbed / sizeof(float));
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

            // --- Prediction with Scores ---
            CUDA_CHECK(cudaMemcpy(d_otok_buffer, otok.data(), otok.size() * sizeof(float), cudaMemcpyHostToDevice));
            kernelComputePredictionWithScores<<<1, 1>>>(d_otok_buffer, d_deEmbeddings, d_predictions, d_result_index, d * x, vocabsize);
            CUDA_CHECK(cudaGetLastError());

            int host_indexForToken = -1;
            CUDA_CHECK(cudaMemcpy(&host_indexForToken, d_result_index, sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(pred.data(), d_predictions, predBytes, cudaMemcpyDeviceToHost));
            indexForToken = host_indexForToken;

            // --- Error Calculation & Logging ---
            current_error = crossEntropy(oneHotEncode, pred);
            std::string predicted_token_str = (indexForToken >= 0 && indexForToken < static_cast<unsigned int>(tokens.size()))
                                              ? tokens[indexForToken] : "INVALID_INDEX";

            std::cout << predicted_token_str << "\t: " << indexForToken << " | "
                      << current_error << " | " << current_error - prev_error << " | "
                      << std::exp(current_error) << " | " << j+1 << " | " << learning << std::endl;

            if (predicted_token_str == expected_str && predicted_token_str != "INVALID_INDEX") {
                std::cout << "Token '" << expected_str << "' predicted correctly after " << j+1 << " epochs." << std::endl;
                break;
            }
            if(j == epochs - 1) {
                std::cout << "Reached maximum epochs. Increasing Epochs by 15." << std::endl;
                epochs += 15;
            }

            // --- Backward Pass ---
            learning *= (current_error > prev_error) ? 1.05f : 0.95f;
            std::vector<float> gradEH(d * x, 0.0f);
            cuUpdateDeEmbeddings(deEmbeddings, otok, pred, oneHotEncode, indexForToken, learning, lambda_L1, lambda_L2, gradEH);

            std::vector<std::vector<float>> targets_for_heads(x, std::vector<float>(EMBEDDING, 0.0f));
            for(int head_idx = 0; head_idx < x; ++head_idx) {
                for(int eidx = 0; eidx < EMBEDDING; ++eidx) {
                    float gradient = learning * (gradEH[(head_idx * EMBEDDING) + eidx]
                                              + (lambda_L1 * embeddings(indexForToken, eidx))
                                              + (lambda_L2 * embeddings(indexForToken, eidx)));
                    if (fabs(gradient) >= MAX_GRAD_CLIP) gradient = std::copysign(MAX_GRAD_CLIP, gradient);
                    targets_for_heads[head_idx][eidx] = otok[(head_idx * EMBEDDING) + eidx] - gradient;
                }
            }

            cuBackwardContext(targets_for_heads, current_block_idx);
            // cuUpdateEmbeddings(embeddings, blocks[blockCount-1].gradToken, learning, lambda_L1, lambda_L2, vocabsize);
            // cuUpdateEmbeddings(tokenEmbed, blocks[blockCount-1].gradToken, learning, lambda_L1, lambda_L2, effective_context_size);

            totalLearning += learning;
            prev_error = current_error;
            totalBCELoss += current_error;
            totalBCEPerplexity += std::exp(current_error);
            j++;
        }

        // --- Update Host State ---
        trainCount++;
        epochCount += j;
        tokenEmbed.addRow(expected_vec, currentTokenCount);
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