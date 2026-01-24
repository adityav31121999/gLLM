#ifdef USE_CU
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
 * @brief (CUDA) Train the transformer for next token prediction (single token training)
 * @param expected expected token embedding (on host)
 * @param expString expected token string (on host)
 */
void transformer::cuTrain(std::vector<float>& expected, std::string& expString)
{
    if (expected.size() != static_cast<size_t>(d)) throw std::runtime_error("cuTrain(single): Expected vector size mismatch.");
    if (currentTokenCount >= FULL_CONTEXT) throw std::runtime_error("cuTrain(single): FULL_CONTEXT limit reached.");

    // all token embeddings of prompt and response will be stored here
    float* d_tokenEmbed;    CUDA_CHECK(cudaMalloc(&d_tokenEmbed, m * CONTEXT_WIN * d * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_tokenEmbed, tokenEmbed.mapped_data, m * CONTEXT_WIN * d * sizeof(float), cudaMemcpyHostToDevice));
    // all embeddings will be stored here
    float* d_embeddings;    CUDA_CHECK(cudaMalloc(&d_embeddings, vocabsize * d * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_embeddings, embeddings.mapped_data, vocabsize * d * sizeof(float), cudaMemcpyHostToDevice));
    // expected and output vector
    float* d_expected;      CUDA_CHECK(cudaMalloc(&d_expected, expected.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_expected, expected.data(), expected.size() * sizeof(float), cudaMemcpyHostToDevice));
    
    // K/Q buffers
    size_t projection_matrix_bytes = static_cast<size_t>(CONTEXT_WIN) * EMBEDDING * sizeof(float);
    size_t KQmatbytes = static_cast<size_t>(CONTEXT_WIN) * CONTEXT_WIN * sizeof(float);
    size_t embedding_bytes_loc = static_cast<size_t>(EMBEDDING) * CONTEXT_WIN * sizeof(float);
    float *d_Q, *d_K, *d_mQ, *d_mK, *d_tok_cl, *d_pEV;
    CUDA_CHECK(cudaMalloc(&d_Q, KQmatbytes));
    CUDA_CHECK(cudaMalloc(&d_K, KQmatbytes));
    CUDA_CHECK(cudaMalloc(&d_mQ, projection_matrix_bytes));
    CUDA_CHECK(cudaMalloc(&d_mK, projection_matrix_bytes));
    CUDA_CHECK(cudaMalloc(&d_tok_cl, embedding_bytes_loc));
    CUDA_CHECK(cudaMalloc(&d_pEV, embedding_bytes_loc));
    
    int* d_result_index;
    CUDA_CHECK(cudaMalloc(&d_result_index, sizeof(int)));

    float prev_Error = 0.0f;        // previous iterations error
    sequence1Count = 1;
    int effective_context_size = currentTokenCount;
    blockCount = (currentTokenCount == 0) ? 1 : ((currentTokenCount) / CONTEXT_WIN) + 1;

    int i = 0;                      // epoch counter
    float current_error = 1.0f;     // Initialize error high
    int host_indexForToken = -1;    // index obtained from training for response

    try {
        while (i <= epochs) 
        {
            // --- K/Q Calculation ---
            embedPlusPos = tokenEmbed + positional;
            size_t currentBytes = static_cast<size_t>(EMBEDDING) * effective_context_size * sizeof(float);
            dim3 threadsPerBlock(16, 16);
            dim3 numBlocks((CONTEXT_WIN + 15) / 16, (effective_context_size + 15) / 16);

            if (blockCount == 1) {
                CUDA_CHECK(cudaMemcpy(d_tok_cl, embedPlusPos.mapped_data, currentBytes, cudaMemcpyHostToDevice));
                for (int layer_idx = 0; layer_idx < x; ++layer_idx) {
                    for (int parallel_idx = 0; parallel_idx < y; ++parallel_idx) {
                        auto& qMat = blocks[0].b[layer_idx][parallel_idx].MQ;
                        auto& kMat = blocks[0].b[layer_idx][parallel_idx].MK;
                        CUDA_CHECK(cudaMemcpy(d_mQ, qMat.mapped_data, projection_matrix_bytes, cudaMemcpyHostToDevice));
                        computeKQall<<<numBlocks, threadsPerBlock>>>(d_tok_cl, d_mQ, d_Q, effective_context_size, EMBEDDING, CONTEXT_WIN);
                        CUDA_CHECK(cudaMemcpy(blocks[0].b[layer_idx][parallel_idx].Q.mapped_data, d_Q, KQmatbytes, cudaMemcpyDeviceToHost));
                        CUDA_CHECK(cudaMemcpy(d_mK, kMat.mapped_data, projection_matrix_bytes, cudaMemcpyHostToDevice));
                        computeKQall<<<numBlocks, threadsPerBlock>>>(d_tok_cl, d_mK, d_K, effective_context_size, EMBEDDING, CONTEXT_WIN);
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
                        CUDA_CHECK(cudaMemcpy(blocks[blockCount - 1].b[layer_idx][parallel_idx].Q.mapped_data, d_Q, KQmatbytes, cudaMemcpyDeviceToHost));
                        CUDA_CHECK(cudaMemcpy(d_mK, kMat.mapped_data, projection_matrix_bytes, cudaMemcpyHostToDevice));
                        computeKQall<<<numBlocks, threadsPerBlock>>>(d_tok_cl, d_mK, d_K, effective_context_size, EMBEDDING, CONTEXT_WIN);
                        CUDA_CHECK(cudaMemcpy(blocks[blockCount - 1].b[layer_idx][parallel_idx].K.mapped_data, d_K, KQmatbytes, cudaMemcpyDeviceToHost));
                    }
                }
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            // --- Forward Pass ---
            cuForward(blockCount, effective_context_size, sequence1Count);

            // --- Accumulate Output ---
            std::fill(otok.begin(), otok.end(), 0.0f);
            if (y > 0) {
                for (int j_layer = 0; j_layer < x; ++j_layer) {
                    for (int k = 0; k < d; ++k) {
                        otok[k] += blocks[blockCount-1].b[j_layer][y - 1].EH[k];
                    }
                }
            }
            for(size_t k_dim = 0; k_dim < static_cast<size_t>(d); k_dim++) {
                if (std::isnan(otok[k_dim])) otok[k_dim] = 0.0f;
                else if (std::isinf(otok[k_dim])) otok[k_dim] = std::copysign(std::numeric_limits<float>::max(), otok[k_dim]);
            }

            // --- Prediction ---
            float* d_otok_buffer;
            CUDA_CHECK(cudaMalloc(&d_otok_buffer, d * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_otok_buffer, otok.data(), d * sizeof(float), cudaMemcpyHostToDevice));
            kernelComputePrediction<<<1, 1>>>(d_otok_buffer, d_embeddings, d_result_index, d, vocabsize);
            CUDA_CHECK(cudaMemcpy(&host_indexForToken, d_result_index, sizeof(int), cudaMemcpyDeviceToHost));
            indexForToken = host_indexForToken;
            CUDA_CHECK(cudaFree(d_otok_buffer));

            current_error = binaryCrossEntropy(otok, expected);

            // --- Convergence Check ---
            if(host_indexForToken >= 0 && host_indexForToken < tokens.size() && tokens[host_indexForToken] == expString) {
                CUDA_CHECK(cudaMemcpy(d_tokenEmbed + currentTokenCount * d, expected.data(), d * sizeof(float), cudaMemcpyHostToDevice));
                break;
            }

            // Increase epochs if error is persistent
            if(current_error > 0.01 && i == epochs) {
                epochs += 10;
            }

            // Learning Rate Update
            if(i > 0) {
                if(current_error <= prev_Error) {
                    if(i <= 6) learning *= 1.05;
                    else if (i % 6 == 0) learning *= (1 + (i/6)*0.05);
                } else {
                    if(i <= 6) learning *= 0.95;
                    else if (i % 6 == 0) learning *= (1 - (i/6)*0.05);
                }
            }

            cuBackward(expected, blockCount);
            prev_Error = current_error;
            i++;
        }

        // Update host counters
        trainCount++;
        epochCount += i;
        error += current_error; // Add final error
        totalLearning += learning;
        
        // Update host tokenEmbed
        int previousTokenCount = currentTokenCount;
        currentTokenCount += 1;
        if (tokenEmbed.mapped_data && previousTokenCount < tokenEmbed.row) {
             float* dest = tokenEmbed.mapped_data + (previousTokenCount * d);
             memcpy(dest, expected.data(), d * sizeof(float));
        }

        blockCount = (currentTokenCount == 0) ? 1 : ((currentTokenCount) / CONTEXT_WIN) + 1;
        sequence1Count = currentTokenCount % CONTEXT_WIN;
        if (sequence1Count == 0 && currentTokenCount > 0) sequence1Count = CONTEXT_WIN;
    }
    catch (...) {
        CUDA_CHECK(cudaFree(d_embeddings));
        CUDA_CHECK(cudaFree(d_tokenEmbed));
        CUDA_CHECK(cudaFree(d_expected));
        CUDA_CHECK(cudaFree(d_Q));
        CUDA_CHECK(cudaFree(d_K));
        CUDA_CHECK(cudaFree(d_mQ));
        CUDA_CHECK(cudaFree(d_mK));
        CUDA_CHECK(cudaFree(d_tok_cl));
        CUDA_CHECK(cudaFree(d_pEV));
        CUDA_CHECK(cudaFree(d_result_index));
        throw;
    }

    // Free temporary device memory
    CUDA_CHECK(cudaFree(d_expected));
    CUDA_CHECK(cudaFree(d_embeddings));
    CUDA_CHECK(cudaFree(d_tokenEmbed));
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_mQ));
    CUDA_CHECK(cudaFree(d_mK));
    CUDA_CHECK(cudaFree(d_tok_cl));
    CUDA_CHECK(cudaFree(d_pEV));
    CUDA_CHECK(cudaFree(d_result_index));
}

/**
 * @brief (CUDA) Train the transformer for next token prediction (single token training)
 * @param expected expected token embedding (on host)
 * @param expString expected token string (on host)
 */
void transformer::cuTrainContext(std::vector<float>& expected, std::string& expString)
{
    if (expected.size() != static_cast<size_t>(d)) throw std::runtime_error("cuTrainContext(single): Expected vector size mismatch.");
    if (currentTokenCount >= FULL_CONTEXT) throw std::runtime_error("cuTrainContext(single): FULL_CONTEXT limit reached.");

    // all token embeddings of prompt and response will be stored here
    float* d_tokenEmbed;    CUDA_CHECK(cudaMalloc(&d_tokenEmbed, m * CONTEXT_WIN * d * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_tokenEmbed, tokenEmbed.mapped_data, m * CONTEXT_WIN * d * sizeof(float), cudaMemcpyHostToDevice));
    // deEmbeddings
    float* d_deEmbeddings;  CUDA_CHECK(cudaMalloc(&d_deEmbeddings, vocabsize * x * d * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_deEmbeddings, deEmbeddings.mapped_data, vocabsize * x * d * sizeof(float), cudaMemcpyHostToDevice));
    // all embeddings will be stored here
    float* d_embeddings;    CUDA_CHECK(cudaMalloc(&d_embeddings, vocabsize * d * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_embeddings, embeddings.mapped_data, vocabsize * d * sizeof(float), cudaMemcpyHostToDevice));
    // expected and output vector
    float* d_expected;      CUDA_CHECK(cudaMalloc(&d_expected, expected.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_expected, expected.data(), expected.size() * sizeof(float), cudaMemcpyHostToDevice));
    
    // K/Q buffers
    size_t projection_matrix_bytes = static_cast<size_t>(CONTEXT_WIN) * EMBEDDING * sizeof(float);
    size_t KQmatbytes = static_cast<size_t>(CONTEXT_WIN) * CONTEXT_WIN * sizeof(float);
    size_t embedding_bytes_loc = static_cast<size_t>(EMBEDDING) * CONTEXT_WIN * sizeof(float);
    float *d_Q, *d_K, *d_mQ, *d_mK, *d_tok_cl, *d_pEV;
    CUDA_CHECK(cudaMalloc(&d_Q, KQmatbytes));
    CUDA_CHECK(cudaMalloc(&d_K, KQmatbytes));
    CUDA_CHECK(cudaMalloc(&d_mQ, projection_matrix_bytes));
    CUDA_CHECK(cudaMalloc(&d_mK, projection_matrix_bytes));
    CUDA_CHECK(cudaMalloc(&d_tok_cl, embedding_bytes_loc));
    CUDA_CHECK(cudaMalloc(&d_pEV, embedding_bytes_loc));

    float* d_predictions; CUDA_CHECK(cudaMalloc(&d_predictions, vocabsize * sizeof(float)));
    int* d_result_index; CUDA_CHECK(cudaMalloc(&d_result_index, sizeof(int)));

    float prev_Error = 0.0f;        // previous iterations error
    sequence1Count = 1;
    int effective_context_size = currentTokenCount;
    blockCount = (currentTokenCount == 0) ? 1 : ((currentTokenCount) / CONTEXT_WIN) + 1;

    int i = 0;                      // epoch counter
    float current_error = 1.0f;     // Initialize error high
    int host_indexForToken = -1;    // index obtained from training for response
    
    std::fill(oneHotEncode.begin(), oneHotEncode.end(), 0.0f);
    // Find index for expString
    int targetIndex = -1;
    for(size_t k=0; k<tokens.size(); ++k){
        if(tokens[k] == expString) {
            targetIndex = k;
            break;
        }
    }
    if(targetIndex != -1) oneHotEncode[targetIndex] = 1.0f;

    try {
        while (i <= epochs) 
        {
            // --- K/Q Calculation ---
            embedPlusPos = tokenEmbed + positional;
            size_t currentBytes = static_cast<size_t>(EMBEDDING) * effective_context_size * sizeof(float);
            dim3 threadsPerBlock(16, 16);
            dim3 numBlocks((CONTEXT_WIN + 15) / 16, (effective_context_size + 15) / 16);

            if (blockCount == 1) {
                CUDA_CHECK(cudaMemcpy(d_tok_cl, embedPlusPos.mapped_data, currentBytes, cudaMemcpyHostToDevice));
                for (int layer_idx = 0; layer_idx < x; ++layer_idx) {
                    for (int parallel_idx = 0; parallel_idx < y; ++parallel_idx) {
                        auto& qMat = blocks[0].b[layer_idx][parallel_idx].MQ;
                        auto& kMat = blocks[0].b[layer_idx][parallel_idx].MK;
                        CUDA_CHECK(cudaMemcpy(d_mQ, qMat.mapped_data, projection_matrix_bytes, cudaMemcpyHostToDevice));
                        computeKQall<<<numBlocks, threadsPerBlock>>>(d_tok_cl, d_mQ, d_Q, effective_context_size, EMBEDDING, CONTEXT_WIN);
                        CUDA_CHECK(cudaMemcpy(blocks[0].b[layer_idx][parallel_idx].Q.mapped_data, d_Q, KQmatbytes, cudaMemcpyDeviceToHost));
                        CUDA_CHECK(cudaMemcpy(d_mK, kMat.mapped_data, projection_matrix_bytes, cudaMemcpyHostToDevice));
                        computeKQall<<<numBlocks, threadsPerBlock>>>(d_tok_cl, d_mK, d_K, effective_context_size, EMBEDDING, CONTEXT_WIN);
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
                        CUDA_CHECK(cudaMemcpy(blocks[blockCount - 1].b[layer_idx][parallel_idx].Q.mapped_data, d_Q, KQmatbytes, cudaMemcpyDeviceToHost));
                        CUDA_CHECK(cudaMemcpy(d_mK, kMat.mapped_data, projection_matrix_bytes, cudaMemcpyHostToDevice));
                        computeKQall<<<numBlocks, threadsPerBlock>>>(d_tok_cl, d_mK, d_K, effective_context_size, EMBEDDING, CONTEXT_WIN);
                        CUDA_CHECK(cudaMemcpy(blocks[blockCount - 1].b[layer_idx][parallel_idx].K.mapped_data, d_K, KQmatbytes, cudaMemcpyDeviceToHost));
                    }
                }
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            // --- Forward Pass ---
            cuForward(blockCount, effective_context_size, sequence1Count);

            // --- Accumulate Output ---
            std::fill(otok.begin(), otok.end(), 0.0f);
            if (y > 0) {
                for (int layer_idx = 0; layer_idx < x; ++layer_idx) {
                    for (int k = 0; k < d; ++k) {
                        otok[(layer_idx * d) + k] = blocks[blockCount-1].b[layer_idx][y - 1].EH[k];
                    }
                }
            }
            for(size_t k_dim = 0; k_dim < static_cast<size_t>(x*d); k_dim++) {
                if (std::isnan(otok[k_dim])) otok[k_dim] = 0.0f;
                else if (std::isinf(otok[k_dim])) otok[k_dim] = std::copysign(std::numeric_limits<float>::max(), otok[k_dim]);
            }

            // --- Prediction ---
            float* d_otok_buffer;
            CUDA_CHECK(cudaMalloc(&d_otok_buffer, x * d * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_otok_buffer, otok.data(), x * d * sizeof(float), cudaMemcpyHostToDevice));
            kernelComputePredictionWithScores<<<1, 1>>>(d_otok_buffer, d_deEmbeddings, d_predictions, d_result_index, d * x, vocabsize);
            CUDA_CHECK(cudaMemcpy(&host_indexForToken, d_result_index, sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(pred.data(), d_predictions, vocabsize * sizeof(float), cudaMemcpyDeviceToHost));
            indexForToken = host_indexForToken;
            pred = softmax(pred);
            CUDA_CHECK(cudaFree(d_otok_buffer));

            current_error = - std::log(pred[targetIndex] + 1e-15f);

            if(tokens[this->indexForToken] == expString) {
                std::cout << "indexForToken: " << this->indexForToken << " | host_indexForToken: " << host_indexForToken << " | Epoch Count: " << epochCount << " | Current Token Count " << currentTokenCount << std::endl;
                std::cout << "Computed token is -> " << tokens[host_indexForToken] << " | with CE error " << current_error << std::endl;
                CUDA_CHECK(cudaMemcpy(d_tokenEmbed + currentTokenCount * d, expected.data(), d * sizeof(float), cudaMemcpyHostToDevice)); // H->D
                if(expString == "</s>")
                    std::cout << "--------------------To next LINE------------->>>>>>>>>>>" << std::endl;
                else
                    std::cout << "--------------------To next token------------->>>>>>>>>>>" << std::endl;
                break;
            }
            else if(i == epochs - 1) {
                std::cout << "Computed token is -> " << tokens[host_indexForToken] << " | with CE error " << current_error << std::endl;
                std::cout << "Increasing Epoch Count by 10 '-'" << std::endl;
                epochs += 10;
            }

            if(i > 0) {
                if(current_error <= prev_Error) {
                    if(i <= 6)   
                        learning *= 1.05;
                    else if (i % 6 == 0)
                        learning *= (1 + (i/6)*0.05);
                }
                else {
                    if(i <= 6)   
                        learning *= 0.95;
                    else if (i % 6 == 0)
                        learning *= (1 - (i/6)*0.05);
                }
            }
            prev_Error = current_error;
            
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

            cuBackwardContext(targets_for_heads, blockCount);
            cuUpdateEmbeddings(embeddings, blocks[blockCount-1].gradToken, learning, lambda_L1, lambda_L2, 12454);
            cuUpdateEmbeddings(tokenEmbed, blocks[blockCount-1].gradToken, learning, lambda_L1, lambda_L2, effective_context_size);

            i++;
        }

        // Update host counters
        trainCount++;
        epochCount += i;
        error += current_error; // Add final error
        totalLearning += learning;
        
        int previousTokenCount = currentTokenCount;
        currentTokenCount += 1;
        if (tokenEmbed.mapped_data && previousTokenCount < tokenEmbed.row) {
             float* dest = tokenEmbed.mapped_data + (previousTokenCount * d);
             memcpy(dest, expected.data(), d * sizeof(float));
        }

        blockCount = (currentTokenCount == 0) ? 1 : ((currentTokenCount) / CONTEXT_WIN) + 1;
        sequence1Count = currentTokenCount % CONTEXT_WIN;
        if (sequence1Count == 0 && currentTokenCount > 0) sequence1Count = CONTEXT_WIN;
    }
    catch (...) {
        CUDA_CHECK(cudaFree(d_embeddings));
        CUDA_CHECK(cudaFree(d_deEmbeddings));
        CUDA_CHECK(cudaFree(d_tokenEmbed));
        CUDA_CHECK(cudaFree(d_expected));
        CUDA_CHECK(cudaFree(d_Q));
        CUDA_CHECK(cudaFree(d_K));
        CUDA_CHECK(cudaFree(d_mQ));
        CUDA_CHECK(cudaFree(d_mK));
        CUDA_CHECK(cudaFree(d_tok_cl));
        CUDA_CHECK(cudaFree(d_pEV));
        CUDA_CHECK(cudaFree(d_predictions));
        CUDA_CHECK(cudaFree(d_result_index));
        throw;
    }

    // Free temporary device memory
    CUDA_CHECK(cudaFree(d_expected));
    CUDA_CHECK(cudaFree(d_embeddings));
    CUDA_CHECK(cudaFree(d_deEmbeddings));
    CUDA_CHECK(cudaFree(d_tokenEmbed));
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_mQ));
    CUDA_CHECK(cudaFree(d_mK));
    CUDA_CHECK(cudaFree(d_tok_cl));
    CUDA_CHECK(cudaFree(d_pEV));
    CUDA_CHECK(cudaFree(d_predictions));
    CUDA_CHECK(cudaFree(d_result_index));
}

#endif