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
 * @brief (CUDA) Train the transformer for prompt and response
 * @param prompt prompt token embeddings (on host)
 * @param response response token embeddings (on host)
 * @param rString tokens of response (on host)
 */
void transformer::cuTrain(std::vector<std::vector<float>>& prompt, std::vector<std::vector<float>>& response,
    std::vector<std::string>& rString)
{
    // Basic validation
    if (prompt.empty()) {
        throw std::runtime_error("Initial prompt cannot be empty!");
    }
    if(prompt.size() > PROMPT_THRESHOLD && prompt.size() > CONTEXT_WIN) {
        throw std::runtime_error("Prompt size should not exceed CONTEXT_WIN!");
    }
    if (response.empty() || response.size() != rString.size()) {
        throw std::runtime_error("Response embeddings and response strings must be non-empty and have the same size!");
    }

    float* d_tokenEmbed;
    CUDA_CHECK(cudaMalloc(&d_tokenEmbed, m * CONTEXT_WIN * d * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_tokenEmbed, tokenEmbed.mapped_data, m * CONTEXT_WIN * d * sizeof(float), cudaMemcpyHostToDevice));
    float* d_embeddings;
    CUDA_CHECK(cudaMalloc(&d_embeddings, vocabsize * d * sizeof(float))); // Size should be vocabsize * d
    CUDA_CHECK(cudaMemcpy(d_embeddings, embeddings.mapped_data, vocabsize * d * sizeof(float), cudaMemcpyHostToDevice));
    float* d_expected_response_token; // Buffer for the target response token
    CUDA_CHECK(cudaMalloc(&d_expected_response_token, d * sizeof(float)));
    float* d_otok_buffer; // Buffer for EH output
    CUDA_CHECK(cudaMalloc(&d_otok_buffer, d * sizeof(float)));
    int* d_indexForToken_ptr;
    CUDA_CHECK(cudaMalloc(&d_indexForToken_ptr, sizeof(int)));

    // Process Prompt: Add prompt tokens to context (Host and Device)
    int current_prompt_size = prompt.size();
    int resCount = 0;
    std::cout << "Current Token Count: " << this->currentTokenCount << std::endl;
    std::cout << "Prompt Size: " << prompt.size() << std::endl;
    std::cout << "Response Size: " << response.size() << std::endl; 

    // int ctokcount = currentTokenCount;

    for(int p = 0; p < current_prompt_size; ++p) {
        if (currentTokenCount >= FULL_CONTEXT) {
            throw std::runtime_error("Cannot add prompt, FULL_CONTEXT limit reached.");
        }
        // Add to device context
        CUDA_CHECK(cudaMemcpy(d_tokenEmbed + currentTokenCount * d, prompt[p].data(), d * sizeof(float), cudaMemcpyHostToDevice));
        currentTokenCount++;
        // Update blockCount if necessary
        if (currentTokenCount > 0 && currentTokenCount % CONTEXT_WIN == 0) {
            blockCount += 1;
        }
    }

    promptCount = current_prompt_size;
    blockCount = (currentTokenCount == 0) ? 1 : ((currentTokenCount - 1) / CONTEXT_WIN) + 1;

    float* dQ; float* dK; float* mQ; float* mK; float* tok;
    // int threadsPerBlock = 256;
    CUDA_CHECK(cudaMalloc(&dQ, static_cast<size_t>(EMBEDDING) * sizeof(float))); // Buffer for one token embedding
    CUDA_CHECK(cudaMalloc(&dK, static_cast<size_t>(EMBEDDING) * sizeof(float))); // Buffer for one token embedding
    CUDA_CHECK(cudaMalloc(&mQ, static_cast<size_t>(MATHEIGHTS) * EMBEDDING * sizeof(float))); // MQ is MATHEIGHTS x EMBEDDING
    CUDA_CHECK(cudaMalloc(&mK, static_cast<size_t>(MATHEIGHTS) * EMBEDDING * sizeof(float))); // MK is MATHEIGHTS x EMBEDDING
    CUDA_CHECK(cudaMalloc(&tok, static_cast<size_t>(MATHEIGHTS) * sizeof(float))); // Output of matrix-vector product (MATHEIGHTS elements)

    // Train for Response: Predict each response token based on context (prompt + previous response tokens)
    for(int i = 0; i < response.size(); ++i) 
    {
        if (currentTokenCount >= FULL_CONTEXT) {
            throw std::runtime_error("Cannot add response, FULL_CONTEXT limit reached.");
        }
        int effective_context_size = currentTokenCount;
        int current_block_idx = 1;
        CUDA_CHECK(cudaMemcpy(d_expected_response_token, response[i].data(), d * sizeof(float), cudaMemcpyHostToDevice));

        // for prompt
        std::cout << "Queries and Keys calculation for prompts" << std::endl;
        if(blockCount == 1) {
            for(int i_pa = 0; i_pa < x; i_pa++) {
                for(int j_head = 0; j_head < y; j_head++) {
                    CUDA_CHECK(cudaMemcpy(mQ, t[0].b[i_pa][j_head].MQ.mapped_data, static_cast<size_t>(MATHEIGHTS * EMBEDDING) * sizeof(float), cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(mK, t[0].b[i_pa][j_head].MK.mapped_data, static_cast<size_t>(MATHEIGHTS * EMBEDDING) * sizeof(float), cudaMemcpyHostToDevice));
                    for(int k = 0; k < prompt.size(); k++) {
                        // copy H -> D
                        CUDA_CHECK(cudaMemcpy(dQ, prompt[k].data(), EMBEDDING * sizeof(float), cudaMemcpyHostToDevice));
                        CUDA_CHECK(cudaMemcpy(dK, prompt[k].data(), EMBEDDING * sizeof(float), cudaMemcpyHostToDevice));
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
        else {
            for(int i_pa = 0; i_pa < x; i_pa++) {
                for(int j_head = 0; j_head < y; j_head++) {
                    CUDA_CHECK(cudaMemcpy(mQ, t[blockCount-1].b[i_pa][j_head].MQ.mapped_data, static_cast<size_t>(MATHEIGHTS * EMBEDDING) * sizeof(float), cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(mK, t[blockCount-1].b[i_pa][j_head].MK.mapped_data, static_cast<size_t>(MATHEIGHTS * EMBEDDING) * sizeof(float), cudaMemcpyHostToDevice));
                    for(int k = 0; k < prompt.size(); k++) {
                        // copy H -> D
                        CUDA_CHECK(cudaMemcpy(dK, prompt[k].data(), EMBEDDING * sizeof(float), cudaMemcpyHostToDevice));
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

        cuForward(current_block_idx, effective_context_size, promptCount);

        int j = 0;
        float current_error = 1.0f;
        int host_indexForToken = this->indexForToken;

        while (j < epochs)
        {
            float* d_current_block_EH_ptr; // Pointer to current block's EH on device
            CUDA_CHECK(cudaMalloc(&d_current_block_EH_ptr, d * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_current_block_EH_ptr, this->otok.data(), d * sizeof(float), cudaMemcpyHostToDevice));       // Copy transformer's otok

            cuComputeOutput(d_current_block_EH_ptr, d_embeddings, vocabsize, host_indexForToken, d);
            std::vector<float> h_otok_buffer(d);
            CUDA_CHECK(cudaMemcpy(h_otok_buffer.data(), d_current_block_EH_ptr, d * sizeof(float), cudaMemcpyDeviceToHost));    // Copy result back for error check
            current_error = crossEntropy(h_otok_buffer, response[i]);        // Compare against target response[i]
            // if(host_indexForToken >= 0 && static_cast<size_t>(indexForToken) < tokens.size() && tokens[host_indexForToken] == rString[i])
            if(tokens[this->indexForToken] == rString[i]) {
                std::cout << "indexForToken: " << this->indexForToken << " | host_indexForToken: " << host_indexForToken << " | Epoch Count: " << epochCount << " | Current Token Count " << currentTokenCount << std::endl;
                std::cout << "Computed token is -> " << tokens[host_indexForToken] << " | with BCE error " << current_error << " | MAE Error " << MAE(h_otok_buffer, response[i]) << std::endl;
                CUDA_CHECK(cudaMemcpy(d_tokenEmbed + effective_context_size * d, response[i].data(), d * sizeof(float), cudaMemcpyHostToDevice)); // H->D
                if(rString[i] == "<@#0>")
                    std::cout << "--------------------To next LINE------------->>>>>>>>>>>" << std::endl;
                else
                    std::cout << "--------------------To next token------------->>>>>>>>>>>" << std::endl;
                break;
            }
            else if(j == epochs - 1) {
                std::cout << "Computed token is -> " << tokens[host_indexForToken] << " | with BCE error " << current_error << " | MAE Error " << MAE(h_otok_buffer, response[i]) << std::endl;
                std::cout << "Increasing Epoch Count by 10 '-'" << std::endl;
                epochs += 10;
            }
            else {
                std::cout << "Computed token is -> " << tokens[host_indexForToken] << " | with BCE error " << current_error << " | MAE Error " << MAE(h_otok_buffer, response[i]) << std::endl;
            }

            cuBackward(response[i], current_block_idx);
            cuForward(current_block_idx, effective_context_size, promptCount);
            j++;
        }

        // Update host counters and add the true response token to context for the next prediction
        trainCount++;
        epochCount += j;
        error += current_error;
        currentTokenCount += 1;
        resCount += 1;
        std::cout << "Queries and Keys for responses" << std::endl;
        if(resCount > 0) {
            if(blockCount == 1) {
                // same block
                for(int m = 0; m < x; m++) {
                    for(int n = 0; n < y; n++) {
                        CUDA_CHECK(cudaMemcpy(mQ, t[0].b[m][n].MQ.mapped_data, static_cast<size_t>(MATHEIGHTS*EMBEDDING) * sizeof(float), cudaMemcpyHostToDevice));
                        CUDA_CHECK(cudaMemcpy(mK, t[0].b[m][n].MK.mapped_data, static_cast<size_t>(MATHEIGHTS*EMBEDDING) * sizeof(float), cudaMemcpyHostToDevice));
                        for(int k = 0; k < resCount; k++) {
                            // copy H -> D
                            CUDA_CHECK(cudaMemcpy(dQ, response[k].data(), EMBEDDING * sizeof(float), cudaMemcpyHostToDevice));
                            CUDA_CHECK(cudaMemcpy(dK, response[k].data(), EMBEDDING * sizeof(float), cudaMemcpyHostToDevice));
                            // make queries using compute KorQ: t[0].b[i][j].Q[currentTokenCount%CONTEXT_WIN] = response(i) * t[0].b[i][j].MQ
                            compute_single_kq_vector_kernel<<<1, 1>>>(tok, mQ, dQ, EMBEDDING, MATHEIGHTS);
                            // copy D -> H
                            CUDA_CHECK(cudaMemcpy(t[0].b[m][n].Q.mapped_data + (currentTokenCount%CONTEXT_WIN + k)*MATHEIGHTS, tok, MATHEIGHTS * sizeof(float), cudaMemcpyDeviceToHost));
                            // make keys using compute KorQ: t[0].b[i][j].K[currentTokenCount%CONTEXT_WIN] = response(i) * t[0].b[i][j].MK
                            compute_single_kq_vector_kernel<<<1, 1>>>(tok, mK, dK, EMBEDDING, MATHEIGHTS);
                            // copy D -> H
                            CUDA_CHECK(cudaMemcpy(t[0].b[m][n].K.mapped_data + (currentTokenCount%CONTEXT_WIN + k)*MATHEIGHTS, tok, MATHEIGHTS * sizeof(float), cudaMemcpyDeviceToHost));
                        }
                    }
                }
            }
            else {
                // shift to next block
                for(int m = 0; m < x; m++) {
                    for(int n = 0; n < y; n++) {
                        CUDA_CHECK(cudaMemcpy(mQ, t[blockCount-1].b[m][n].MQ.mapped_data, static_cast<size_t>(MATHEIGHTS*EMBEDDING) * sizeof(float), cudaMemcpyHostToDevice));
                        CUDA_CHECK(cudaMemcpy(mK, t[blockCount-1].b[m][n].MK.mapped_data, static_cast<size_t>(MATHEIGHTS*EMBEDDING) * sizeof(float), cudaMemcpyHostToDevice));
                        for(int k = 0; k < resCount; k++) {
                            // H -> D
                            CUDA_CHECK(cudaMemcpy(dQ, response[k].data(), EMBEDDING * sizeof(float), cudaMemcpyHostToDevice)); // Assuming response[k] is the source
                            CUDA_CHECK(cudaMemcpy(dK, response[k].data(), EMBEDDING * sizeof(float), cudaMemcpyHostToDevice)); // Assuming response[k] is the source
                            // make queries using compute KorQ: t[0].b[i][j].Q[currentTokenCount%CONTEXT_WIN] = response(i) * t[0].b[i][j].MQ
                            compute_single_kq_vector_kernel<<<1, 1>>>(tok, mQ, dQ, EMBEDDING, MATHEIGHTS);
                            // copy D -> H
                            CUDA_CHECK(cudaMemcpy(t[blockCount-1].b[m][n].Q.mapped_data + (currentTokenCount%CONTEXT_WIN + k)*MATHEIGHTS, tok, MATHEIGHTS * sizeof(float), cudaMemcpyDeviceToHost));
                            // make keys using compute KorQ: t[0].b[i][j].K[currentTokenCount%CONTEXT_WIN] = response(i) * t[0].b[i][j].MK
                            compute_single_kq_vector_kernel<<<1, 1>>>(tok, mK, dK, EMBEDDING, MATHEIGHTS);
                            // copy D -> H
                            CUDA_CHECK(cudaMemcpy(t[blockCount-1].b[m][n].K.mapped_data + (currentTokenCount%CONTEXT_WIN + k)*MATHEIGHTS, tok, MATHEIGHTS * sizeof(float), cudaMemcpyDeviceToHost));
                        }
                    }
                }
            }
        }
        // copy response to d_tokenEmbed from response[i]
        CUDA_CHECK(cudaMemcpy(d_tokenEmbed + currentTokenCount * d, response[i].data(), d * sizeof(float), cudaMemcpyHostToDevice));
    }

    // --- Free temporary device memory ---
    CUDA_CHECK(cudaFree(d_expected_response_token));
    CUDA_CHECK(cudaFree(d_tokenEmbed));
    CUDA_CHECK(cudaFree(d_otok_buffer));
    CUDA_CHECK(cudaFree(dQ));
    CUDA_CHECK(cudaFree(dK));
    CUDA_CHECK(cudaFree(mQ));
    CUDA_CHECK(cudaFree(mK));
    CUDA_CHECK(cudaFree(tok));
    CUDA_CHECK(cudaFree(d_indexForToken_ptr));
}


/**
 * @brief (CUDA) Train transformers for continuous chats
 * @param prompts all prompts (on host)
 * @param responses token embeddings all responses to the prompts (on host)
 * @param rString tokens of responses (on host)
 */
void transformer::cuTrain(std::vector<std::vector<std::vector<float>>>& prompts, std::vector<std::vector<std::vector<float>>>& responses,
                        std::vector<std::vector<std::string>>& rString)
{
    // Basic validation for chat structure
    if(prompts.size() != responses.size() || responses.size() != rString.size()) {
        throw std::runtime_error("Rows of all the vectors must match");
    }

    // Calculate total tokens to check against FULL_CONTEXT
    size_t total_tokens = currentTokenCount; // Start with existing context
    for(size_t i = 0; i < prompts.size(); ++i) {
        if (prompts[i].empty()) {
             throw std::runtime_error("Chat training requires non-empty prompts.");
        }
        if (responses[i].empty() || responses[i].size() != rString[i].size()) {
             throw std::runtime_error("Chat training response embeddings and strings mismatch or are empty.");
        }
        total_tokens += prompts[i].size() + responses[i].size();
    }

    if(total_tokens > FULL_CONTEXT) {
        throw std::runtime_error("TOTAL TOKENS SHOULD NOT EXCEED THE FULL CONTEXT");
    }

    // Train for chat by calling the prompt-response trainer for each turn
    for(int i = 0; i < prompts.size(); ++i) {
        if (!prompts[i].empty() && !responses[i].empty()) {
            cuTrain(prompts[i], responses[i], rString[i]);
        }
        else {
            std::cerr << "Warning: Skipping empty prompt/response pair at index " << i << std::endl;
        }
    }
}
#endif