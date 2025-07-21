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
 * @brief (CUDA) Train the transformer for next token prediction (single token training)
 * @param promptCount number of tokens in the prompt
 * @param currentTokenCount number of tokens in the full context
 * @param blockCount current block in full context
 * @param expected expected token embedding (on host)
 * @param expString expected token string (on host)
 */
void transformer::cuTrain(int& promptCount, int& currentTokenCount, int& blockCount, std::vector<float>& expected,
    std::string& expString)
{
    // all token embeddings of prompt and response will be stored here
    float* d_tokenEmbed;    CUDA_CHECK(cudaMalloc(&d_tokenEmbed, m * CONTEXT_WIN * d * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_tokenEmbed, tokenEmbed.mapped_data, m * CONTEXT_WIN * d * sizeof(float), cudaMemcpyHostToDevice));
    // all embeddings will be stored here
    float* d_embeddings;    CUDA_CHECK(cudaMalloc(&d_embeddings, vocabsize * d * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_embeddings, embeddings.mapped_data, vocabsize * d * sizeof(float), cudaMemcpyHostToDevice));
    // expected and output vector
    float* d_expected;      CUDA_CHECK(cudaMalloc(&d_expected, expected.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_expected, expected.data(), expected.size() * sizeof(float), cudaMemcpyHostToDevice));
    float* d_otok_buffer;   CUDA_CHECK(cudaMalloc(&d_otok_buffer, d * sizeof(float)));

    float prev_Error = 0.0f;        // previous iterations error

    // Training logic for the first block
    if(blockCount == 1 && currentTokenCount <= CONTEXT_WIN) {
        // scaled dot product for all heads
        cuParallelKdotQs(promptCount, currentTokenCount, blockCount, d, isSelf, inTraining);
        // forward pass for all columns
        cuForward(blockCount, currentTokenCount, promptCount);
        int i = 0;                      // epoch counter
        float current_error = 1.0f;     // Initialize error high
        int host_indexForToken = -1;    // index obtained from training for response
        while (i <= epochs) 
        {
            float* d_block0_EH_ptr = nullptr;
            CUDA_CHECK(cudaMalloc(&d_block0_EH_ptr, d * sizeof(float))); // This will hold the output from cuForward
            CUDA_CHECK(cudaMemcpy(d_block0_EH_ptr, this->otok.data(), d * sizeof(float), cudaMemcpyHostToDevice)); // Copy transformer's otok (output of cuForward)

            // Compute output token embedding and find best token index on GPU
            cuComputeOutput(d_block0_EH_ptr, d_embeddings, vocabsize, host_indexForToken, d);
            std::vector<float> h_otok_buffer(d);
            CUDA_CHECK(cudaMemcpy(h_otok_buffer.data(), d_block0_EH_ptr, d * sizeof(float), cudaMemcpyDeviceToHost));    // Copy result back for error check
            current_error = crossEntropy(h_otok_buffer, expected);        // Compare against target response[i]
            // if(host_indexForToken >= 0 && static_cast<size_t>(indexForToken) < tokens.size() && tokens[host_indexForToken] == rString[i])
            if(tokens[this->indexForToken] == expString) {
                std::cout << "indexForToken: " << this->indexForToken << " | host_indexForToken: " << host_indexForToken << " | Epoch Count: " << epochCount << " | Current Token Count " << currentTokenCount << std::endl;
                std::cout << "Computed token is -> " << tokens[host_indexForToken] << " | with BCE error " << current_error << " | MAE Error " << MAE(h_otok_buffer, expected) << std::endl;
                CUDA_CHECK(cudaMemcpy(d_tokenEmbed + currentTokenCount * d, expected.data(), d * sizeof(float), cudaMemcpyHostToDevice)); // H->D
                if(expString == "<@#0>")
                    std::cout << "--------------------To next LINE------------->>>>>>>>>>>" << std::endl;
                else
                    std::cout << "--------------------To next token------------->>>>>>>>>>>" << std::endl;
                break;
            }
            else if(i == epochs - 1) {
                std::cout << "Computed token is -> " << tokens[host_indexForToken] << " | with BCE error " << current_error << " | MAE Error " << MAE(h_otok_buffer, expected) << std::endl;
                std::cout << "Increasing Epoch Count by 10 '-'" << std::endl;
                epochs += 10;
            }
            else {
                std::cout << "Computed token is -> " << tokens[host_indexForToken] << " | with BCE error " << current_error << " | MAE Error " << MAE(h_otok_buffer, expected) << std::endl;
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
            cuBackward(expected);
            cuForward(blockCount, currentTokenCount, promptCount);
            i++;
        }
        // Update host counters
        trainCount++;
        epochCount += i;
        error += current_error; // Add final error
        totalLearning += learning;
        currentTokenCount += 1;
        if(currentTokenCount == CONTEXT_WIN) {
            blockCount += 1;
        }
    }
    // Training logic for subsequent blocks
    else if(blockCount > 1 && currentTokenCount >= CONTEXT_WIN) 
    {
        cuParallelKdotQs(promptCount, currentTokenCount, blockCount, d, isSelf, inTraining);
        cuForward(blockCount, currentTokenCount, promptCount); // Operates on device data

        int i = 0;                      // epoch counter
        float current_error = 1.0f;     // Initialize error high
        int host_indexForToken = -1;    // index obtained from training for response

        while (i < epochs) 
        {
            prev_Error = current_error;
            float* d_current_block_EH_ptr; // Pointer to the current block's EH on device
            CUDA_CHECK(cudaMalloc(&d_current_block_EH_ptr, d * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_current_block_EH_ptr, this->otok.data(), d * sizeof(float), cudaMemcpyHostToDevice)); // Copy transformer's otok

            // Compute output token embedding and find best token index on GPU
            cuComputeOutput(d_current_block_EH_ptr, d_embeddings, vocabsize, host_indexForToken, d);
            std::vector<float> h_otok_buffer(d);
            CUDA_CHECK(cudaMemcpy(h_otok_buffer.data(), d_current_block_EH_ptr, d * sizeof(float), cudaMemcpyDeviceToHost));    // Copy result back for error check
            current_error = crossEntropy(h_otok_buffer, expected);        // Compare against target response[i]
            // if(host_indexForToken >= 0 && static_cast<size_t>(indexForToken) < tokens.size() && tokens[host_indexForToken] == rString[i])
            if(tokens[this->indexForToken] == expString) {
                std::cout << "indexForToken: " << this->indexForToken << " | host_indexForToken: " << host_indexForToken << " | Epoch Count: " << epochCount << " | Current Token Count " << currentTokenCount << std::endl;
                std::cout << "Computed token is -> " << tokens[host_indexForToken] << " | with BCE error " << current_error << " | MAE Error " << MAE(h_otok_buffer, expected) << std::endl;
                CUDA_CHECK(cudaMemcpy(d_tokenEmbed + currentTokenCount * d, expected.data(), d * sizeof(float), cudaMemcpyHostToDevice)); // H->D
                if(expString == "<@#0>")
                    std::cout << "--------------------To next LINE------------->>>>>>>>>>>" << std::endl;
                else
                    std::cout << "--------------------To next token------------->>>>>>>>>>>" << std::endl;
                break;
            }
            else if(i == epochs - 1) {
                std::cout << "Computed token is -> " << tokens[host_indexForToken] << " | with BCE error " << current_error << " | MAE Error " << MAE(h_otok_buffer, expected) << std::endl;
                std::cout << "Increasing Epoch Count by 10 '-'" << std::endl;
                epochs += 10;
            }
            else {
                std::cout << "Computed token is -> " << tokens[host_indexForToken] << " | with BCE error " << current_error << " | MAE Error " << MAE(h_otok_buffer, expected) << std::endl;
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
            cuBackward(expected, blockCount);
            cuForward(blockCount, currentTokenCount, promptCount);
            i++;
        }
        // Update host counters
        trainCount++;
        epochCount += i;
        error += current_error;
        totalLearning += learning;
        currentTokenCount += 1;
        if(currentTokenCount % CONTEXT_WIN == 0) {
            blockCount += 1;
        }
    }

    // Free temporary device memory
    CUDA_CHECK(cudaFree(d_expected));
    CUDA_CHECK(cudaFree(d_otok_buffer));
}


/**
 * @brief (CUDA) Train the transformer on sentences or paragraphs
 * @param sentence token embedding of sentence (on host)
 * @param rString sentence tokens (on host)
 */
void transformer::cuTrain(std::vector<std::vector<float>>& sentence, std::vector<std::string>& rString)
{
    // Basic validation
    if(sentence.size() > FULL_CONTEXT) {
        throw std::runtime_error("Sentence size should not exceed FULL_CONTEXT");
    }
    if(sentence.empty() || sentence.size() != rString.size()) {
        throw std::runtime_error("Sentence embeddings and sentence strings must be non-empty and have the same size.");
    }

    // Allocate temporary device buffers
    float* d_expected_token;
    CUDA_CHECK(cudaMalloc(&d_expected_token, d * sizeof(float)));
    float* d_otok_buffer; // Buffer for EH output
    CUDA_CHECK(cudaMalloc(&d_otok_buffer, d * sizeof(float)));
    int* d_indexForToken_ptr;
    CUDA_CHECK(cudaMalloc(&d_indexForToken_ptr, sizeof(int)));
    float* d_tokenEmbed; // Device context buffer
    CUDA_CHECK(cudaMalloc(&d_tokenEmbed, m * CONTEXT_WIN * d * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_tokenEmbed, tokenEmbed.mapped_data, m * CONTEXT_WIN * d * sizeof(float), cudaMemcpyHostToDevice));
    float* d_embeddings; // Device embedding table
    CUDA_CHECK(cudaMalloc(&d_embeddings, vocabsize * d * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_embeddings, embeddings.mapped_data, vocabsize * d * sizeof(float), cudaMemcpyHostToDevice)); // Should be embeddings.mapped_data

    size_t totalTokenEmbedFloats = static_cast<size_t>(m) * CONTEXT_WIN * d;
    size_t tokenEmbedBytes = totalTokenEmbedFloats * sizeof(float);
    size_t singleTokenBytes = static_cast<size_t>(d) * sizeof(float);
    size_t outputBytes = singleTokenBytes; // Size of h_otok_buffer

    float* dQ; float* dK; float* mQ; float* mK; float* tok;    CUDA_CHECK(cudaMalloc(&dQ, static_cast<size_t>(EMBEDDING) * sizeof(float))); // Buffer for one token embedding
    CUDA_CHECK(cudaMalloc(&dK, static_cast<size_t>(EMBEDDING) * sizeof(float))); // Buffer for one token embedding
    CUDA_CHECK(cudaMalloc(&mQ, static_cast<size_t>(MATHEIGHTS) * EMBEDDING * sizeof(float))); // MQ is MATHEIGHTS x EMBEDDING
    CUDA_CHECK(cudaMalloc(&mK, static_cast<size_t>(MATHEIGHTS) * EMBEDDING * sizeof(float))); // MK is MATHEIGHTS x EMBEDDING
    CUDA_CHECK(cudaMalloc(&tok, static_cast<size_t>(MATHEIGHTS) * sizeof(float))); // Output of matrix-vector product (MATHEIGHTS elements)

    // start taking attention score for first token and then perform trainin
    // otherwise starting from zero means trying to perform training without attention score
    this->blockCount = 1, this->promptCount = 1;
    float prev_Error = 0.0f;

    // Train for each subsequent token in the sentence, starting from the second token
    // first token is used to set kdotq
    for(int i = 1; i < sentence.size(); ++i) {
        // Copy the expected token embedding to device
        CUDA_CHECK(cudaMemcpy(d_expected_token, sentence[i].data(), d * sizeof(float), cudaMemcpyHostToDevice));
        int effective_context_size = currentTokenCount; // Size of context *before* adding token i

        cuParallelKdotQs(promptCount, effective_context_size, blockCount, d, isSelf, inTraining);
        cuForward(blockCount, effective_context_size, promptCount);

        int j = 0;
        float current_error = 1.0f;
        int host_indexForToken = -1;
        std::string& expected_str = rString[i];
        prev_Error = 0.0f;
        while (j < epochs) 
        {
            float* d_current_block_EH_ptr; // Pointer to current block's EH on device
            CUDA_CHECK(cudaMalloc(&d_current_block_EH_ptr, d * sizeof(float))); // EH is size d
            CUDA_CHECK(cudaMemcpy(d_current_block_EH_ptr, this->otok.data(), d * sizeof(float), cudaMemcpyHostToDevice)); // Copy transformer's otok

            cuComputeOutput(d_current_block_EH_ptr, d_embeddings, vocabsize, this->indexForToken, d);
            std::vector<float> h_otok_buffer(d);
            CUDA_CHECK(cudaMemcpy(h_otok_buffer.data(), d_current_block_EH_ptr, d * sizeof(float), cudaMemcpyDeviceToHost)); // Copy result back for error check

            current_error = crossEntropy(h_otok_buffer, sentence[i]); // Compare against target sentence[i]
            std::string predicted_token_str = (host_indexForToken >= 0 && host_indexForToken < static_cast<unsigned long long>(tokens.size()))
                                                  ? tokens[host_indexForToken] : "INVALID_INDEX";
            std::cout << "Computed token is -> " << predicted_token_str << " (index: " << host_indexForToken << ") | with BCE error " << current_error << " | MAE Error " << MAE(h_otok_buffer, sentence[i]) << std::endl;

            if (predicted_token_str == expected_str && predicted_token_str != "INVALID_INDEX") {
                size_t offset_bytes = static_cast<size_t>(effective_context_size) * d * sizeof(float);
                if (offset_bytes + outputBytes > tokenEmbedBytes) {
                    throw std::out_of_range("clTrain(prompt-response): Offset exceeds buffer bounds when writing converged response token.");
                }
                std::cout << "indexForToken: " << this->indexForToken << " | host_indexForToken: " << host_indexForToken << " | Epoch Count for this token: " << j << " | Current Token Count " << currentTokenCount << std::endl;
                CUDA_CHECK(cudaMemcpy(d_tokenEmbed + effective_context_size * d, expected_str.data(), d * sizeof(float), cudaMemcpyHostToDevice)); // Write expected_vec (target EH)
                if(predicted_token_str == "@#0"){
                    std::cout << "--------------->>>>>>>>>>>>> To next LINE >>>>>>>>>>>>>>>>-------------" << std::endl;
                }
                else {
                    std::cout << "--------------------- To next token ------------->>>>>>>>>>>>>>>>>" << std::endl;
                    totalLearning += learning;
                    break;
                }
            }
            else if (j == this->epochs - 1) {
                if (predicted_token_str != expected_str) {
                    std::cout << "Increasing Epoch Count by 10 '-'" << std::endl;
                    this->epochs += 10;
                }
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
            cuBackward(sentence[i], blockCount);
            // first block
            if(blockCount == 1 && (effective_context_size < CONTEXT_WIN)) 
            {
                for(int i_pa = 0; i_pa < x; i_pa++) {
                    for(int j_head = 0; j_head < y; j_head++) {
                        CUDA_CHECK(cudaMemcpy(mQ, t[0].b[i_pa][j_head].MQ.mapped_data, static_cast<size_t>(MATHEIGHTS * EMBEDDING) * sizeof(float), cudaMemcpyHostToDevice));
                        CUDA_CHECK(cudaMemcpy(mK, t[0].b[i_pa][j_head].MK.mapped_data, static_cast<size_t>(MATHEIGHTS * EMBEDDING) * sizeof(float), cudaMemcpyHostToDevice));
                        for(int k = 0; k < i; k++) {
                            // copy H -> D
                            CUDA_CHECK(cudaMemcpy(dQ, sentence[k].data(), EMBEDDING * sizeof(float), cudaMemcpyHostToDevice));
                            CUDA_CHECK(cudaMemcpy(dK, sentence[k].data(), EMBEDDING * sizeof(float), cudaMemcpyHostToDevice));
                            // make queries using compute KorQ: t[0].b[i][j].Q[currentTokenCount%CONTEXT_WIN] = prompt(i) * t[0].b[i][j].MQ
                            compute_single_kq_vector_kernel<<<1, 1>>>(tok, mQ, dQ, EMBEDDING, MATHEIGHTS);
                            // copy D -> H
                            CUDA_CHECK(cudaMemcpy(t[0].b[i_pa][j_head].Q.mapped_data + (currentTokenCount%CONTEXT_WIN + k)*MATHEIGHTS, tok, MATHEIGHTS * sizeof(float), cudaMemcpyDeviceToHost));
                            // make keys using compute KorQ: t[0].b[i][j].K[currentTokenCount%CONTEXT_WIN] = prompt(i) * t[0].b[i][j].MK
                            compute_single_kq_vector_kernel<<<1, 1>>>(tok, mK, dK, EMBEDDING, MATHEIGHTS);
                            // copy D -> H
                            CUDA_CHECK(cudaMemcpy(t[0].b[i_pa][j_head].K.mapped_data + (currentTokenCount%CONTEXT_WIN + k)*MATHEIGHTS, tok, MATHEIGHTS * sizeof(float), cudaMemcpyDeviceToHost));
                        }
                    }
                }
            }
            else if(blockCount > 1 && effective_context_size >= CONTEXT_WIN)
            {
                for(int i_pa = 0; i_pa < x; i_pa++) {
                    for(int j_head = 0; j_head < y; j_head++) {
                        CUDA_CHECK(cudaMemcpy(mQ, t[blockCount-1].b[i_pa][j_head].MQ.mapped_data, static_cast<size_t>(MATHEIGHTS * EMBEDDING) * sizeof(float), cudaMemcpyHostToDevice));
                        CUDA_CHECK(cudaMemcpy(mK, t[blockCount-1].b[i_pa][j_head].MK.mapped_data, static_cast<size_t>(MATHEIGHTS * EMBEDDING) * sizeof(float), cudaMemcpyHostToDevice));
                        for(int k = 0; k < i; k++) {
                            // copy H -> D
                            CUDA_CHECK(cudaMemcpy(dK, sentence[(blockCount-1)*CONTEXT_WIN + k].data(), EMBEDDING * sizeof(float), cudaMemcpyHostToDevice));
                            // make keys using compute KorQ: t[0].b[i][j].K[currentTokenCount%CONTEXT_WIN] = prompt(i) * t[0].b[i][j].MK
                            compute_single_kq_vector_kernel<<<1, 1>>>(tok, mK, dK, EMBEDDING, MATHEIGHTS);
                            // copy D -> H
                            CUDA_CHECK(cudaMemcpy(t[blockCount-1].b[i_pa][j_head].K.mapped_data + (currentTokenCount%CONTEXT_WIN + k)*MATHEIGHTS, tok, MATHEIGHTS * sizeof(float), cudaMemcpyDeviceToHost));
                        }
                        for(int k = 0; k < CONTEXT_WIN; k++) {
                            // copy H -> D
                            CUDA_CHECK(cudaMemcpy(dQ, t[blockCount-1].b[i_pa][j_head].EV(k).data(), EMBEDDING * sizeof(float), cudaMemcpyHostToDevice));
                            compute_single_kq_vector_kernel<<<1, 1>>>(tok, mQ, dQ, EMBEDDING, MATHEIGHTS); // tok is output, mQ matrix, dQ input
                            // copy D -> H
                            CUDA_CHECK(cudaMemcpy(t[blockCount-1].b[i_pa][j_head].Q.mapped_data + k*MATHEIGHTS, tok, MATHEIGHTS * sizeof(float), cudaMemcpyDeviceToHost));
                        }
                    }
                }
            }
            cuForward(blockCount, effective_context_size, promptCount);
            j++;
        }
        // Update host counters
        trainCount++;
        epochCount += j;
        error += current_error;
        currentTokenCount += 1;
        if(currentTokenCount % CONTEXT_WIN == 0) {
            blockCount += 1;
            promptCount = 0;
        }
    }

    // --- Free temporary device memory ---
    CUDA_CHECK(cudaFree(d_expected_token)); // Free the buffer for the expected token
    CUDA_CHECK(cudaFree(d_otok_buffer));    // Free the temporary output buffer
}

#endif