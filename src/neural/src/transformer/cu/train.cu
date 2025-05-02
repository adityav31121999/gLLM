
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
    // Device pointers (assuming d_tokenEmbed and d_embeddings are managed elsewhere or allocated here)
    float* d_tokenEmbed; // Device context buffer
    CUDA_CHECK(cudaMalloc(&d_tokenEmbed, m * CONTEXT_WIN * d * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_tokenEmbed, tokenEmbed.data(), m * CONTEXT_WIN * d * sizeof(float), cudaMemcpyHostToDevice));
    float* d_embeddings; // Device embedding table
    CUDA_CHECK(cudaMalloc(&d_embeddings, vocabsize * d * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_embeddings, tokenEmbed.data(), vocabsize * d * sizeof(float), cudaMemcpyHostToDevice));
    // Allocate necessary temporary device memory
    float* d_expected;
    CUDA_CHECK(cudaMalloc(&d_expected, expected.size() * sizeof(float)));
    float* d_otok_buffer; // Temporary buffer for EH output
    CUDA_CHECK(cudaMalloc(&d_otok_buffer, d * sizeof(float)));
    // Copy expected vector from Host to Device
    CUDA_CHECK(cudaMemcpy(d_expected, expected.data(), expected.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Training logic for the first block
    if(blockCount == 1 && currentTokenCount < CONTEXT_WIN) {
        cuParallelKdotQs(promptCount, currentTokenCount, blockCount, d, isSelf, inTraining); // d is embedding dim
        cuForward(blockCount, currentTokenCount, promptCount); // Operates on device data
        int i = 0;      // epoch counter
        float current_error = 1.0f; // Initialize error high
        int host_indexForToken = -1;
        while (i <= epochs) 
        {
            float* d_block0_EH_ptr = nullptr; // Hypothetical function to get device pointer
            CUDA_CHECK(cudaMalloc(&d_block0_EH_ptr, d * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_block0_EH_ptr, t[0].EH.data(), d * sizeof(float), cudaMemcpyHostToDevice)); // Copy EH from device block if needed

            // Compute output token embedding and find best token index on GPU
            cuComputeOutput(d_block0_EH_ptr, d_embeddings, vocabsize, this->indexForToken, d);
            std::vector<float> h_otok_buffer(d);
            CUDA_CHECK(cudaMemcpy(h_otok_buffer.data(), d_block0_EH_ptr, d * sizeof(float), cudaMemcpyDeviceToHost)); // Copy result back to host for error check
            current_error = errorofv(h_otok_buffer, expected);

            if((current_error < 0.01) || (host_indexForToken >= 0 && host_indexForToken < tokens.size() && tokens[host_indexForToken] == expString))
            {
                CUDA_CHECK(cudaMemcpy(d_tokenEmbed + currentTokenCount * d, d_block0_EH_ptr, d * sizeof(float), cudaMemcpyDeviceToDevice));
                break;
            }

            // Increase epochs if error is persistent
            if(current_error > 0.01 && i == epochs) {
                epochs += 10;
            }

            cuBackward(expected);
            cuForward(blockCount, currentTokenCount, promptCount);
            i++;
        }
        // Update host counters
        trainCount++;
        epochCount += i;
        error += current_error; // Add final error
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

        int i = 0;
        float current_error = 1.0f;
        int host_indexForToken = -1;

        while (i < epochs) 
        {
            float* d_current_block_EH_ptr; // Pointer to the current block's EH on device
            CUDA_CHECK(cudaMalloc(&d_current_block_EH_ptr, d * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_current_block_EH_ptr, t[blockCount-1].EH.data(), d * sizeof(float), cudaMemcpyHostToDevice));
            cuComputeOutput(d_current_block_EH_ptr, d_embeddings, vocabsize, this->indexForToken, d);

            std::vector<float> h_otok_buffer(d);
            CUDA_CHECK(cudaMemcpy(h_otok_buffer.data(), d_current_block_EH_ptr, d * sizeof(float), cudaMemcpyDeviceToHost)); // Copy result back for error check
            current_error = errorofv(h_otok_buffer, expected);

            if(current_error < 0.01 || (host_indexForToken >= 0 && host_indexForToken < tokens.size() && tokens[host_indexForToken] == expString)) 
            {
                CUDA_CHECK(cudaMemcpy(d_tokenEmbed + currentTokenCount * d, d_current_block_EH_ptr, d * sizeof(float), cudaMemcpyDeviceToDevice));
                break; // Exit training loop
            }

            // Increase epochs if error is persistent
            if(current_error > 0.01 && i == epochs -1)
            {
                epochs += 10;
            }

            cuBackward(expected, blockCount);

            cuForward(blockCount, currentTokenCount, promptCount);
            i++;
        }
        // Update host counters
        trainCount++;
        epochCount += i;
        error += current_error;
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
 * @brief (CUDA) Train the transformer on sentences
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
    float* d_otok_buffer; // Buffer for EH output if needed
    CUDA_CHECK(cudaMalloc(&d_otok_buffer, d * sizeof(float)));
    int* d_indexForToken_ptr;
    CUDA_CHECK(cudaMalloc(&d_indexForToken_ptr, sizeof(int)));
    float* d_tokenEmbed; // Device context buffer
    CUDA_CHECK(cudaMalloc(&d_tokenEmbed, m * CONTEXT_WIN * d * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_tokenEmbed, tokenEmbed.data(), m * CONTEXT_WIN * d * sizeof(float), cudaMemcpyHostToDevice));
    float* d_embeddings; // Device embedding table
    CUDA_CHECK(cudaMalloc(&d_embeddings, vocabsize * d * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_embeddings, tokenEmbed.data(), vocabsize * d * sizeof(float), cudaMemcpyHostToDevice));

    // Initialize state for sentence training
    promptCount = 1; // The first token acts as the initial prompt
    blockCount = 1;
    currentTokenCount = 1;

    // Train for each subsequent token in the sentence
    for(int i = 1; i < sentence.size(); ++i) {
        // Copy the expected token embedding to device
        CUDA_CHECK(cudaMemcpy(d_expected_token, sentence[i].data(), d * sizeof(float), cudaMemcpyHostToDevice));

        // Determine context size *before* adding token i
        int effective_context_size = currentTokenCount; // Size of context *before* adding token i

        // first block
        if(blockCount == 1 && (effective_context_size < CONTEXT_WIN)) 
        {
            cuParallelKdotQs(promptCount, effective_context_size, blockCount, d, isSelf, inTraining);
            cuForward(blockCount, effective_context_size, promptCount);

            int j = 0;
            float current_error = 1.0f;
            int host_indexForToken = -1;

            while (j <= epochs) 
            {
                float* d_block0_EH_ptr; // Pointer to block 0's EH on device
                CUDA_CHECK(cudaMalloc(&d_block0_EH_ptr, d * sizeof(float)));
                CUDA_CHECK(cudaMemcpy(d_block0_EH_ptr, t[0].EH.data(), d * sizeof(float), cudaMemcpyHostToDevice));
                cuComputeOutput(d_block0_EH_ptr, d_embeddings, vocabsize, this->indexForToken, d);

                std::vector<float> h_otok_buffer(d);
                CUDA_CHECK(cudaMemcpy(h_otok_buffer.data(), d_block0_EH_ptr, d * sizeof(float), cudaMemcpyDeviceToHost)); // Copy result back for error check
                current_error = errorofv(h_otok_buffer, sentence[i]); // Compare against target sentence[i]

                if((current_error < 0.01) || (host_indexForToken >= 0 && host_indexForToken < tokens.size() && tokens[host_indexForToken] == rString[i])) 
                {
                    CUDA_CHECK(cudaMemcpy(d_tokenEmbed + effective_context_size * d, sentence[i].data(), d * sizeof(float), cudaMemcpyHostToDevice)); // H->D (Copying target to context)
                    break;
                }

                if(current_error > 0.01 && j == epochs) {
                    epochs += 10;
                }
                cuBackward(sentence[i]);

                cuForward(blockCount, effective_context_size, promptCount);
                j++;
            }
            // Update host counters
            trainCount++;
            epochCount += j;
            error += current_error; // Add final error for this token
            currentTokenCount += 1; // Increment *after* processing token i
            if(currentTokenCount == CONTEXT_WIN) { // Move to next block if context window is full
                blockCount += 1;
                promptCount = 0; // Reset prompt count when moving to a new block? Or keep it relative? Assume reset for now.
            }
        }
        // next blocks
        else if(blockCount > 1 && effective_context_size >= CONTEXT_WIN)
        {
            cuParallelKdotQs(promptCount, effective_context_size, blockCount, d, isSelf, inTraining);

            cuForward(blockCount, effective_context_size, promptCount);

            int j = 0;
            float current_error = 1.0f;
            int host_indexForToken = -1;

            while (j < epochs) 
            {
                float* d_current_block_EH_ptr; // Pointer to current block's EH on device
                CUDA_CHECK(cudaMalloc(&d_current_block_EH_ptr, m * CONTEXT_WIN * d * sizeof(float)));
                CUDA_CHECK(cudaMemcpy(d_current_block_EH_ptr, t[blockCount-1].EH.data(), m * CONTEXT_WIN * d * sizeof(float), cudaMemcpyHostToDevice));

                cuComputeOutput(d_current_block_EH_ptr, d_embeddings, vocabsize, this->indexForToken, d);
                std::vector<float> h_otok_buffer(d);
                CUDA_CHECK(cudaMemcpy(h_otok_buffer.data(), d_current_block_EH_ptr, d * sizeof(float), cudaMemcpyDeviceToHost)); // Copy result back for error check
                current_error = errorofv(h_otok_buffer, sentence[i]); // Compare against target sentence[i]

                if(current_error < 0.01 || (host_indexForToken >= 0 && host_indexForToken < tokens.size() && tokens[host_indexForToken] == rString[i]))
                {
                    CUDA_CHECK(cudaMemcpy(d_tokenEmbed + effective_context_size * d, sentence[i].data(), d * sizeof(float), cudaMemcpyHostToDevice)); // H->D
                    break;
                }

                if(current_error > 0.01 && j == epochs - 1) {
                    epochs += 10;
                }
                cuBackward(sentence[i], blockCount);

                cuForward(blockCount, effective_context_size, promptCount);
                j++;
            }
            // Update host counters
            trainCount++;
            epochCount += j;
            error += current_error;
            currentTokenCount += 1;
            if(currentTokenCount % CONTEXT_WIN == 0) { // Move to next block if needed
                blockCount += 1;
                 promptCount = 0; // Reset prompt count?
            }
        }
    }

    // --- Free temporary device memory ---
    CUDA_CHECK(cudaFree(d_expected_token)); // Free the buffer for the expected token
    CUDA_CHECK(cudaFree(d_otok_buffer));    // Free the temporary output buffer
} // Note: d_tokenEmbed and d_embeddings might need freeing depending on overall memory management strategy


/**
 * @brief (CUDA) Train the transformer for prompt and response
 * @param prompt prompt token embeddings (on host)
 * @param response response token embeddings (on host)
 * @param rString tokens of response (on host)
 */
void transformer::cuTrain(std::vector<std::vector<float>>& prompt, std::vector<std::vector<float>>& response, std::vector<std::string>& rString)
{
    // Basic validation
    if (prompt.empty()) {
        throw std::runtime_error("Initial prompt cannot be empty!");
    }
    if(prompt.size() > PROMPT_THRESHOLD) {
         if (prompt.size() > CONTEXT_WIN) {
            throw std::runtime_error("Prompt size should not exceed CONTEXT_WIN!");
         }
    }
    if (response.empty() || response.size() != rString.size()) {
        throw std::runtime_error("Response embeddings and response strings must be non-empty and have the same size!");
    }

    // Allocate device buffers (context, embeddings, temporary)
    float* d_tokenEmbed;
    CUDA_CHECK(cudaMalloc(&d_tokenEmbed, m * CONTEXT_WIN * d * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_tokenEmbed, tokenEmbed.data(), m * CONTEXT_WIN * d * sizeof(float), cudaMemcpyHostToDevice));
    float* d_embeddings;
    CUDA_CHECK(cudaMalloc(&d_embeddings, m * CONTEXT_WIN * d * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_embeddings, embeddings.data(), vocabsize * d * sizeof(float), cudaMemcpyHostToDevice));
    float* d_expected_response_token; // Buffer for the target response token
    CUDA_CHECK(cudaMalloc(&d_expected_response_token, d * sizeof(float)));
    float* d_otok_buffer; // Buffer for EH output
    CUDA_CHECK(cudaMalloc(&d_otok_buffer, d * sizeof(float)));
    int* d_indexForToken_ptr;
    CUDA_CHECK(cudaMalloc(&d_indexForToken_ptr, sizeof(int)));

    // Process Prompt: Add prompt tokens to context (Host and Device)
    int current_prompt_size = prompt.size();
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

    // Train for Response: Predict each response token based on context (prompt + previous response tokens)
    for(int i = 0; i < response.size(); ++i) {
        if (currentTokenCount >= FULL_CONTEXT) {
            throw std::runtime_error("Cannot add response, FULL_CONTEXT limit reached.");
        }
        int effective_context_size = currentTokenCount; // Context size *before* adding response[i]
        int current_block_idx = (effective_context_size == 0) ? 1 : ((effective_context_size -1) / CONTEXT_WIN) + 1;
        // Copy target response token to device
        CUDA_CHECK(cudaMemcpy(d_expected_response_token, response[i].data(), d * sizeof(float), cudaMemcpyHostToDevice));
        cuParallelKdotQs(promptCount, effective_context_size, current_block_idx, d, isSelf, inTraining);

        cuForward(current_block_idx, effective_context_size, promptCount);

        int j = 0;
        float current_error = 1.0f;
        int host_indexForToken = -1;

        while (j < epochs) 
        {
            float* d_current_block_EH_ptr; // Pointer to current block's EH on device
            CUDA_CHECK(cudaMalloc(&d_current_block_EH_ptr, d * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_current_block_EH_ptr, t[blockCount-1].EH.data(), d * sizeof(float), cudaMemcpyHostToDevice));

            cuComputeOutput(d_current_block_EH_ptr, d_embeddings, vocabsize, this->indexForToken, d);
            std::vector<float> h_otok_buffer(d);
            CUDA_CHECK(cudaMemcpy(h_otok_buffer.data(), d_current_block_EH_ptr, d * sizeof(float), cudaMemcpyDeviceToHost)); // Copy result back for error check
            current_error = errorofv(h_otok_buffer, response[i]); // Compare against target response[i]

            if(current_error < 0.01 || (host_indexForToken >= 0 && host_indexForToken < tokens.size() && tokens[host_indexForToken] == rString[i]))
            {
                CUDA_CHECK(cudaMemcpy(d_tokenEmbed + effective_context_size * d, response[i].data(), d * sizeof(float), cudaMemcpyHostToDevice)); // H->D
                break;
            }

            if(current_error > 0.01 && j == epochs - 1) {
                epochs += 10;
            }
            
            cuBackward(response[i], current_block_idx);

            cuForward(current_block_idx, effective_context_size, promptCount);
            j++;
        }
        // Update host counters and add the true response token to context for the next prediction
        trainCount++;
        epochCount += j;
        error += current_error;
        currentTokenCount += 1; // Increment *after* processing response token i
        blockCount = (currentTokenCount == 0) ? 1 : ((currentTokenCount - 1) / CONTEXT_WIN) + 1;
        // Add true response token to device context (handled within the loop if break condition met, or needs adding here?)
        // Assuming the break condition handles adding the correct token to d_tokenEmbed
    }

     // --- Free temporary device memory ---
    CUDA_CHECK(cudaFree(d_expected_response_token));
    CUDA_CHECK(cudaFree(d_otok_buffer)); // May not be needed
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
