
#include "include/transformer.hpp"
#include "include/block.hpp"
#include "include/attention.hpp"
#include "include/mlp.hpp" // For errorofv, MSE
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
        fprintf(stderr, "CUDA Error in %s at line %d: %s (%d)\n",            \
                __FILE__, __LINE__, cudaGetErrorString(err), err);           \
        throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(err)));    \
    }                                                                        \
} while (0)


/**
 * @brief (CUDA) Internal: Validate one token prediction based on current context.
 * @param promptCount Number of tokens in the prompt. (Note: Might be redundant if cuForward uses contextSize)
 * @param currentTokenCount Number of tokens in the full context *before* this validation step.
 * @param blockCount Current block index (1-based) in the full context. (Note: Might not be needed if cuForward handles it based on currentTokenCount)
 * @param expected Expected token embedding (on host).
 * @param expString Expected token string (on host, currently unused in logic).
 * @param d_tokenEmbed_global Device pointer to the global token embedding buffer (managed by caller).
 */
void transformer::cuValidate(std::vector<float>& expected, std::string& expString)
{
    // --- Basic Validation ---
    if (expected.size() != static_cast<size_t>(d)) {
        throw std::runtime_error("cuValidate(single): Expected vector size mismatch.");
    }
    if (currentTokenCount >= FULL_CONTEXT) {
        std::cerr << "Warning: cuValidate(single) called with currentTokenCount (" << currentTokenCount
                  << ") at or exceeding FULL_CONTEXT (" << FULL_CONTEXT << ")." << std::endl;
    }
    // Device Pointers (Local to this step)
    float* d_embeddings = nullptr;
    float* d_otok_buffer = nullptr;
    int* d_indexForToken_ptr = nullptr;

    int host_indexForToken = -1;
    float current_error = 0.0f;
    float current_mse = 0.0f;

    try {
        // --- Device Memory Allocation & H->D Transfer (Local buffers) ---
        size_t embeddingsBytes = static_cast<size_t>(vocabsize) * d * sizeof(float);
        size_t singleTokenBytes = static_cast<size_t>(d) * sizeof(float);
        size_t indexBytes = sizeof(int);

        CUDA_CHECK(cudaMalloc(&d_embeddings, embeddingsBytes));
        CUDA_CHECK(cudaMalloc(&d_otok_buffer, singleTokenBytes)); // Buffer for EH output
        CUDA_CHECK(cudaMalloc(&d_indexForToken_ptr, indexBytes));
        std::vector<float> flat_embeddings = flatten(this->embeddings);
        CUDA_CHECK(cudaMemcpy(d_embeddings, flat_embeddings.data(), embeddingsBytes, cudaMemcpyHostToDevice));

        // Determine effective context size and block index
        int effective_context_size = currentTokenCount;
        int current_block_idx = (effective_context_size == 0) ? 1 : ((effective_context_size - 1) / CONTEXT_WIN) + 1;
        if (current_block_idx <= 0 || current_block_idx > m) {
            throw std::out_of_range("cuValidate(single): Calculated current_block_idx (" + std::to_string(current_block_idx) + ") is out of range [1, " + std::to_string(m) + "].");
        }

        // --- Forward Pass ---
        cuForward(current_block_idx, effective_context_size, promptCount);

        // --- Compute Prediction & Error ---
        cuComputeOutput(d_otok_buffer, d_embeddings, vocabsize, *d_indexForToken_ptr, d);
        CUDA_CHECK(cudaMemcpy(&host_indexForToken, d_indexForToken_ptr, indexBytes, cudaMemcpyDeviceToHost));

        std::vector<float> h_otok_buffer(d);
        CUDA_CHECK(cudaMemcpy(h_otok_buffer.data(), d_otok_buffer, singleTokenBytes, cudaMemcpyDeviceToHost));

        current_error = errorofv(h_otok_buffer, expected);
        current_mse = MSE(h_otok_buffer, expected);

        // NO BACKWARD PASS

        // Update Host State (Metrics Only)
        this->validationCount++;
        this->validationError += current_error;
        this->validationMSE += current_mse;
        this->indexForToken = host_indexForToken; // Store predicted index
    }
    catch (const std::exception& e) {
        std::cerr << "Exception in cuValidate(single): " << e.what() << std::endl;
        if (d_embeddings) cudaFree(d_embeddings);
        if (d_otok_buffer) cudaFree(d_otok_buffer);
        if (d_indexForToken_ptr) cudaFree(d_indexForToken_ptr);
        throw; // Re-throw
    }

    // --- Free Device Memory (Local buffers) ---
    CUDA_CHECK(cudaFree(d_embeddings));
    CUDA_CHECK(cudaFree(d_otok_buffer));
    CUDA_CHECK(cudaFree(d_indexForToken_ptr));
}


/**
 * @brief (CUDA) Validate the transformer on sentences.
 * @param sentence Token embeddings of the sentence (on host).
 * @param rString Sentence tokens (on host).
 */
void transformer::cuValidate(std::vector<std::vector<float>>& sentence, std::vector<std::string>& rString)
{
    // --- Basic validation ---
    if (sentence.size() > FULL_CONTEXT) {
        std::cerr << "Warning: cuValidate(sentence) size exceeds FULL_CONTEXT. Validation might be truncated." << std::endl;
    }
    if (sentence.empty() || sentence.size() != rString.size()) {
        throw std::runtime_error("cuValidate(sentence): Sentence embeddings/strings mismatch or empty.");
    }
    if (!sentence.empty() && sentence[0].size() != static_cast<size_t>(d)) {
        throw std::runtime_error("cuValidate(sentence): Sentence embedding dimension mismatch.");
    }

    // Store and Reset State for this validation run
    int originalCurrentTokenCount = this->currentTokenCount;
    int originalPromptCount = this->promptCount;
    int originalBlockCount = this->blockCount;
    this->currentTokenCount = 0;
    this->blockCount = 1;
    // Note: validationError, validationMSE, validationCount should be reset before starting a validation run (e.g., in model::validate)
    this->tokenEmbed.assign(FULL_CONTEXT, std::vector<float>(d, 0.0f)); // Clear/Resize host tokenEmbed

    // Add the first token to host context (acts as initial prompt)
    if (!sentence.empty()) {
        this->tokenEmbed[0] = sentence[0]; // Only the first token
        this->currentTokenCount = 1;
        this->promptCount = 1;
    } else {
        return; // Cannot validate empty sentence
    }

    // --- Device Memory Allocation (Once for the whole sentence) ---
    float* d_tokenEmbed = nullptr; // This will hold the context on the device
    size_t totalTokenEmbedFloats = static_cast<size_t>(m) * CONTEXT_WIN * d; // Use m for full potential context
    size_t tokenEmbedBytes = totalTokenEmbedFloats * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_tokenEmbed, tokenEmbedBytes));
    CUDA_CHECK(cudaMemset(d_tokenEmbed, 0, tokenEmbedBytes)); // Zero out the buffer initially

    try {
        // --- Initial Context H->D ---
        // Copy only the first token to device context
        CUDA_CHECK(cudaMemcpy(d_tokenEmbed, sentence[0].data(), d * sizeof(float), cudaMemcpyHostToDevice));

        // --- Validate for each subsequent token ---
        for (size_t i = 1; i < sentence.size(); ++i) {
            if (this->currentTokenCount >= FULL_CONTEXT) {
                 std::cerr << "Warning: cuValidate(sentence) reached FULL_CONTEXT limit. Stopping validation early." << std::endl;
                 break;
            }

            std::vector<float>& expected_vec = sentence[i];
            std::string& expected_str = rString[i]; // Currently unused in logic

            // Call internal single-token validate (uses state *before* adding token i)
            cuValidate(expected_vec, expected_str);

            // --- Update Host State for the *next* prediction ---
            this->currentTokenCount += 1;
            this->blockCount = (this->currentTokenCount == 0) ? 1 : ((this->currentTokenCount - 1) / CONTEXT_WIN) + 1;
            this->promptCount = 1; // Subsequent steps are single token predictions

            // Add true embedding to host context
            if (this->currentTokenCount <= FULL_CONTEXT) { // Check if context is full
                if (static_cast<size_t>(currentTokenCount - 1) < this->tokenEmbed.size()) {
                    this->tokenEmbed[currentTokenCount - 1] = expected_vec;
                } 
                else {
                    std::cerr << "Warning: Host tokenEmbed size mismatch in cuValidate(sentence)." << std::endl;
                    // Resize host vector if needed (should ideally match FULL_CONTEXT)
                    this->tokenEmbed.resize(this->currentTokenCount);
                    this->tokenEmbed[currentTokenCount - 1] = expected_vec;
                }
                // Update Device Context (H->D) with the true token for the next step
                size_t offset_elements = static_cast<size_t>(this->currentTokenCount - 1) * d;
                if (offset_elements + d <= totalTokenEmbedFloats) {
                    CUDA_CHECK(cudaMemcpy(d_tokenEmbed + offset_elements, expected_vec.data(), d * sizeof(float), cudaMemcpyHostToDevice));
                } 
                else {
                    std::cerr << "Error: Attempting to write past allocated d_tokenEmbed buffer in cuValidate(sentence)." << std::endl;
                    break; // Stop processing if buffer bounds exceeded
                }
            }
        } // End loop over sentence
    }
    catch (const std::exception& e) {
        std::cerr << "Exception in cuValidate(sentence): " << e.what() << std::endl;
        if (d_tokenEmbed) cudaFree(d_tokenEmbed); // Cleanup on error
        // Restore original state on error?
        this->currentTokenCount = originalCurrentTokenCount;
        this->promptCount = originalPromptCount;
        this->blockCount = originalBlockCount;
        throw;
    }

    // --- Free Device Memory ---
    if (d_tokenEmbed)
        cudaFree(d_tokenEmbed);
}


/**
 * @brief (CUDA) Validate the transformer for prompt and response.
 * @param prompt Prompt token embeddings (on host).
 * @param response Response token embeddings (on host).
 * @param rString Tokens of the response (on host).
 */
void transformer::cuValidate(std::vector<std::vector<float>>& prompt, std::vector<std::vector<float>>& response, std::vector<std::string>& rString)
{
    // --- Basic validation ---
    if (prompt.empty()) throw std::runtime_error("cuValidate(prompt-response): Initial prompt cannot be empty.");
    if (response.empty() || response.size() != rString.size()) throw std::runtime_error("cuValidate(prompt-response): Response embeddings/strings mismatch or empty.");
    if ((!prompt.empty() && prompt[0].size() != static_cast<size_t>(d)) || (!response.empty() && response[0].size() != static_cast<size_t>(d))) throw std::runtime_error("cuValidate(prompt-response): Embedding dimension mismatch.");
    if (this->currentTokenCount + prompt.size() + response.size() > FULL_CONTEXT) std::cerr << "Warning: cuValidate(prompt-response) combined size exceeds FULL_CONTEXT." << std::endl;
    // Note: validationError, validationMSE, validationCount should be reset before starting a validation run (e.g., in model::validate)

    // Store and Reset State for this validation run
    int originalCurrentTokenCount = this->currentTokenCount;
    int originalPromptCount = this->promptCount;
    int originalBlockCount = this->blockCount;
    this->currentTokenCount = 0;
    this->blockCount = 1;
    this->tokenEmbed.assign(FULL_CONTEXT, std::vector<float>(d, 0.0f)); // Clear/Resize host tokenEmbed

    // --- Device Memory Allocation (Once for the whole sequence) ---
    float* d_tokenEmbed = nullptr; // Device context buffer
    size_t totalTokenEmbedFloats = static_cast<size_t>(m) * CONTEXT_WIN * d; // Use m for full potential context
    size_t tokenEmbedBytes = totalTokenEmbedFloats * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_tokenEmbed, tokenEmbedBytes));
    CUDA_CHECK(cudaMemset(d_tokenEmbed, 0, tokenEmbedBytes)); // Zero out the buffer initially

    // --- Process Prompt (Host context and Device context) ---
    for (size_t p = 0; p < prompt.size(); ++p) {
        if (this->currentTokenCount >= FULL_CONTEXT) {
            std::cerr << "Warning: Context full while processing prompt in cuValidate(prompt-response)." << std::endl;
            break;
        }
        // Update host context
        this->tokenEmbed[this->currentTokenCount] = prompt[p];
        // Update device context
        size_t offset_elements = static_cast<size_t>(this->currentTokenCount) * d;
        if (offset_elements + d <= totalTokenEmbedFloats) {
            CUDA_CHECK(cudaMemcpy(d_tokenEmbed + offset_elements, prompt[p].data(), d * sizeof(float), cudaMemcpyHostToDevice));
        } 
        else {
            std::cerr << "Error: Attempting to write past allocated d_tokenEmbed buffer during prompt processing in cuValidate(prompt-response)." << std::endl;
        }
        this->currentTokenCount++;
    }
    if (this->tokenEmbed.size() != FULL_CONTEXT) this->tokenEmbed.resize(FULL_CONTEXT, std::vector<float>(d, 0.0f));

    // Set state after processing prompt
    this->blockCount = (this->currentTokenCount == 0) ? 1 : ((this->currentTokenCount - 1) / CONTEXT_WIN) + 1;
    this->promptCount = this->currentTokenCount; // Size of the prompt just added

    try {
        // --- Validate Response ---
        for (size_t i = 0; i < response.size(); ++i) {
            if (this->currentTokenCount >= FULL_CONTEXT) {
                std::cerr << "Warning: cuValidate(prompt-response) reached FULL_CONTEXT limit during response." << std::endl;
                break;
            }

            std::vector<float>& expected_vec = response[i];
            std::string& expected_str = rString[i]; // Currently unused in logic

            // Call internal single-token validate (uses state *before* adding response token i)
            cuValidate(expected_vec, expected_str);

            // --- Update Host State for the *next* prediction ---
            this->currentTokenCount += 1;
            this->blockCount = (this->currentTokenCount == 0) ? 1 : ((this->currentTokenCount - 1) / CONTEXT_WIN) + 1;
            this->promptCount = 1; // Subsequent predictions are single token

            // Add true response token to host context
            if (this->currentTokenCount <= FULL_CONTEXT) { // Check context limit
                if (static_cast<size_t>(currentTokenCount - 1) < this->tokenEmbed.size()) {
                    this->tokenEmbed[currentTokenCount - 1] = expected_vec;
                }
                else {
                    std::cerr << "Warning: Host tokenEmbed size mismatch in cuValidate(prompt-response)." << std::endl;
                    this->tokenEmbed.resize(this->currentTokenCount); // Resize host vector if needed
                    this->tokenEmbed[currentTokenCount - 1] = expected_vec;
                }
                // Update Device Context (H->D) with the true token for the next step
                size_t offset_elements = static_cast<size_t>(this->currentTokenCount - 1) * d;
                if (offset_elements + d <= totalTokenEmbedFloats) {
                    CUDA_CHECK(cudaMemcpy(d_tokenEmbed + offset_elements, expected_vec.data(), d * sizeof(float), cudaMemcpyHostToDevice));
                }
                else {
                    std::cerr << "Error: Attempting to write past allocated d_tokenEmbed buffer during response processing in cuValidate(prompt-response)." << std::endl;
                    break; // Stop processing if buffer bounds exceeded
                }
            }

        } // End loop over response

    }
    catch (const std::exception& e) {
        std::cerr << "Exception in cuValidate(prompt-response): " << e.what() << std::endl;
        if (d_tokenEmbed) cudaFree(d_tokenEmbed); // Cleanup on error
        // Restore original state on error?
        this->currentTokenCount = originalCurrentTokenCount;
        this->promptCount = originalPromptCount;
        this->blockCount = originalBlockCount;
        throw;
    }
}


/**
 * @brief (CUDA) Validate transformers for continuous chats.
 * @param prompts All prompts (on host).
 * @param responses Token embeddings of all responses to the prompts (on host).
 * @param rString Tokens of the responses (on host).
 */
void transformer::cuValidate(std::vector<std::vector<std::vector<float>>>& prompts, std::vector<std::vector<std::vector<float>>>& responses,
                        std::vector<std::vector<std::string>>& rString)
{
    // Basic Validation
    if (prompts.size() != responses.size() || responses.size() != rString.size()) {
        throw std::runtime_error("cuValidate(chat): Mismatch in number of prompts, responses, and response strings.");
    }
    size_t total_tokens_to_add = 0;
    for (size_t i = 0; i < prompts.size(); ++i) {
        if (prompts[i].empty()) throw std::runtime_error("cuValidate(chat): Non-empty prompts required.");
        if (responses[i].empty() || responses[i].size() != rString[i].size()) throw std::runtime_error("cuValidate(chat): Response mismatch/empty.");
        if (!prompts[i].empty() && prompts[i][0].size() != static_cast<size_t>(d)) throw std::runtime_error("cuValidate(chat): Prompt dim mismatch.");
        if (!responses[i].empty() && responses[i][0].size() != static_cast<size_t>(d)) throw std::runtime_error("cuValidate(chat): Response dim mismatch.");
        total_tokens_to_add += prompts[i].size() + responses[i].size();
    }

    // Store and Reset State for the entire chat validation
    this->currentTokenCount = 0;
    this->blockCount = 1;
    this->tokenEmbed.assign(FULL_CONTEXT, std::vector<float>(d, 0.0f)); // Clear/Resize host tokenEmbed
    // Note: validationError, validationMSE, validationCount should be reset before starting a validation run (e.g., in model::validate)

    // Validate each chat turn by calling the prompt-response validator
    for (size_t turn = 0; turn < prompts.size(); ++turn) {
        std::vector<std::vector<float>>& currentPrompt = prompts[turn];
        std::vector<std::vector<float>>& currentResponse = responses[turn];
        std::vector<std::string>& currentRString = rString[turn];

        // Check context limit before processing turn
        if (this->currentTokenCount + currentPrompt.size() + currentResponse.size() > FULL_CONTEXT) {
             std::cerr << "Warning: cuValidate(chat) exceeds FULL_CONTEXT limit at turn " << turn << ". Stopping chat validation early." << std::endl;
             break;
        }

        // Call the prompt-response validator for this turn.
        // It will internally manage its own device memory and context updates based on the current state.
        cuValidate(currentPrompt, currentResponse, currentRString);
    }

    // Restore original state after chat validation? Or assume state persists?
    // Currently, the state reflects the end of the chat validation.
    // If restoration is needed, uncomment below:
    // this->currentTokenCount = originalCurrentTokenCount;
    // this->promptCount = originalPromptCount;
    // this->blockCount = originalBlockCount;
}
