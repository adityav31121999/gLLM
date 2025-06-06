
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
 * @brief (CUDA) Internal: Test one token prediction based on current context.
 * @param promptCount Number of tokens in the prompt. (Note: Might be redundant if cuForward uses contextSize)
 * @param currentTokenCount Number of tokens in the full context *before* this testing step.
 * @param blockCount Current block index (1-based) in the full context. (Note: Might not be needed if cuForward handles it based on currentTokenCount)
 * @param expected Expected token embedding (on host).
 * @param expString Expected token string (on host, currently unused in logic).
 */
void transformer::cuTest(std::vector<float>& expected, std::string& expString)
{
    // --- Basic Validation ---
    if (expected.size() != static_cast<size_t>(d)) {
        throw std::runtime_error("cuTest(single): Expected vector size mismatch.");
    }
    if (currentTokenCount >= FULL_CONTEXT) {
        std::cerr << "Warning: cuTest(single) called with currentTokenCount (" << currentTokenCount
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

        // Directly use mapped_data from the embeddings mat object
        CUDA_CHECK(cudaMemcpy(d_embeddings, this->embeddings.mapped_data, embeddingsBytes, cudaMemcpyHostToDevice));

        // Determine effective context size and block index
        int effective_context_size = currentTokenCount;
        int current_block_idx = (effective_context_size == 0) ? 1 : ((effective_context_size - 1) / CONTEXT_WIN) + 1;
        if (current_block_idx <= 0 || current_block_idx > m) {
            throw std::out_of_range("cuTest(single): Calculated current_block_idx (" + std::to_string(current_block_idx) + ") is out of range [1, " + std::to_string(m) + "].");
        }

        // --- Forward Pass ---
        cuForward(current_block_idx, effective_context_size, promptCount);

        // Copy the result from cuForward (this->otok on host) to device buffer for cuComputeOutput
        CUDA_CHECK(cudaMemcpy(d_otok_buffer, this->otok.data(), singleTokenBytes, cudaMemcpyHostToDevice));

        // --- Compute Prediction & Error ---
        cuComputeOutput(d_otok_buffer, d_embeddings, vocabsize, *d_indexForToken_ptr, d);
        CUDA_CHECK(cudaMemcpy(&host_indexForToken, d_indexForToken_ptr, indexBytes, cudaMemcpyDeviceToHost));

        current_error = errorofv(this->otok, expected); // Use this->otok directly
        current_mse = MSE(this->otok, expected);       // Use this->otok directly

        // NO BACKWARD PASS

        // Update Host State (Metrics Only)
        this->testCount++;
        this->testError += current_error;
        this->testMSE += current_mse;
        this->indexForToken = host_indexForToken;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception in cuTest(single): " << e.what() << std::endl;
        if (d_embeddings) cudaFree(d_embeddings);
        if (d_otok_buffer) cudaFree(d_otok_buffer);
        if (d_indexForToken_ptr) cudaFree(d_indexForToken_ptr);
        throw;
    }

    // --- Free Device Memory (Local buffers) ---
    CUDA_CHECK(cudaFree(d_embeddings));
    CUDA_CHECK(cudaFree(d_otok_buffer));
    CUDA_CHECK(cudaFree(d_indexForToken_ptr));
}


/**
 * @brief (CUDA) Test the transformer on sentences.
 * @param sentence Token embeddings of the sentence (on host).
 * @param rString Sentence tokens (on host).
 */
void transformer::cuTest(std::vector<std::vector<float>>& sentence, std::vector<std::string>& rString)
{
    // --- Basic validation ---
    if (sentence.size() > FULL_CONTEXT) {
        std::cerr << "Warning: cuTest(sentence) size exceeds FULL_CONTEXT. Testing might be truncated." << std::endl;
    }
    if (sentence.empty() || sentence.size() != rString.size()) {
        throw std::runtime_error("cuTest(sentence): Sentence embeddings/strings mismatch or empty.");
    }
    if (!sentence.empty() && sentence[0].size() != static_cast<size_t>(d)) {
        throw std::runtime_error("cuTest(sentence): Sentence embedding dimension mismatch.");
    }

    // Store and Reset State for this test run
    int originalCurrentTokenCount = this->currentTokenCount;
    int originalPromptCount = this->promptCount;
    int originalBlockCount = this->blockCount;
    this->currentTokenCount = 0;
    this->blockCount = 1;

    // Re-initialize tokenEmbed to the full context size for this transformer instance and zero it out.
    size_t full_context_rows = static_cast<size_t>(this->m) * CONTEXT_WIN;
    this->tokenEmbed = mat(full_context_rows, this->d); // Creates new mat, old one is destructed.
    if (this->tokenEmbed.mapped_data) { // Ensure mapping was successful
        std::fill_n(this->tokenEmbed.mapped_data, full_context_rows * this->d, 0.0f);
    }
    // Note: testError, testMSE, testCount should be reset before starting a test run (e.g., in model::test)

    // Add the first token to host context (acts as initial prompt)
    if (!sentence.empty()) {
        if (this->tokenEmbed.mapped_data && this->tokenEmbed.row > 0 && static_cast<size_t>(this->tokenEmbed.col) == sentence[0].size()) {
            std::copy(sentence[0].begin(), sentence[0].end(), this->tokenEmbed.mapped_data); // Copy to first row
        } // else: handle error or assume mat constructor/fill handles it
        this->currentTokenCount = 1;
        this->promptCount = 1;
    } 
    else {
        return;
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

        // --- Test subsequent tokens ---
        for (size_t i = 1; i < sentence.size(); ++i) {
            if (this->currentTokenCount >= FULL_CONTEXT) {
                std::cerr << "Warning: cuTest(sentence) reached FULL_CONTEXT limit. Stopping testing early." << std::endl;
                break;
            }

            std::vector<float>& expected_vec = sentence[i];
            std::string& expected_str = rString[i]; // Currently unused in logic

            // Call internal single-token test (uses state *before* adding token i)
            cuTest(expected_vec, expected_str);

            // --- Update Host State for the *next* prediction ---
            this->currentTokenCount += 1;
            this->blockCount = (this->currentTokenCount == 0) ? 1 : ((this->currentTokenCount - 1) / CONTEXT_WIN) + 1;
            this->promptCount = 1; // Subsequent steps are single token predictions

            // Add true embedding to host context
            if (this->currentTokenCount <= FULL_CONTEXT) { // Check if context is full
                size_t current_row_idx = static_cast<size_t>(currentTokenCount - 1);
                if (current_row_idx < static_cast<size_t>(this->tokenEmbed.row)) {
                    if (this->tokenEmbed.mapped_data && static_cast<size_t>(this->tokenEmbed.col) == expected_vec.size()) {
                        float* dest_ptr = this->tokenEmbed.mapped_data + current_row_idx * this->tokenEmbed.col;
                        std::copy(expected_vec.begin(), expected_vec.end(), dest_ptr);
                    } // else: handle error
                } else {
                    std::cerr << "Warning: Host tokenEmbed row index out of bounds in cuTest(sentence)." << std::endl;
                    break; // tokenEmbed is fixed size, cannot grow beyond full_context_rows
                }
                // Update Device Context (H->D) with the true token for the next step
                size_t offset_elements = static_cast<size_t>(this->currentTokenCount - 1) * d;
                if (offset_elements + d <= totalTokenEmbedFloats) {
                CUDA_CHECK(cudaMemcpy(d_tokenEmbed + offset_elements, expected_vec.data(), d * sizeof(float), cudaMemcpyHostToDevice));
                } else {
                    std::cerr << "Error: Attempting to write past allocated d_tokenEmbed buffer in cuTest(sentence)." << std::endl;
                    break; // Stop processing if buffer bounds exceeded
                }
            }
        } // End loop over sentence
    } 
    catch (const std::exception& e) {
        std::cerr << "Exception in cuTest(sentence): " << e.what() << std::endl;        
        if (d_tokenEmbed) cudaFree(d_tokenEmbed); // Cleanup on error
        // Restore original state on error?
        this->currentTokenCount = originalCurrentTokenCount;
        this->promptCount = originalPromptCount;
        this->blockCount = originalBlockCount;
        throw;
    }

    // --- Free Device Memory ---
    if (d_tokenEmbed) cudaFree(d_tokenEmbed);
}


/**
 * @brief (CUDA) Test the transformer for prompt and response.
 * @param prompt Prompt token embeddings (on host).
 * @param response Response token embeddings (on host).
 * @param rString Tokens of the response (on host).
 */
void transformer::cuTest(std::vector<std::vector<float>>& prompt, std::vector<std::vector<float>>& response, std::vector<std::string>& rString)
{
    // --- Basic validation ---
    if (prompt.empty()) throw std::runtime_error("cuTest(prompt-response): Initial prompt cannot be empty.");
    if (response.empty() || response.size() != rString.size()) throw std::runtime_error("cuTest(prompt-response): Response embeddings/strings mismatch or empty.");
    if ((!prompt.empty() && prompt[0].size() != static_cast<size_t>(d)) || (!response.empty() && response[0].size() != static_cast<size_t>(d))) throw std::runtime_error("cuTest(prompt-response): Embedding dimension mismatch.");
    if (this->currentTokenCount + prompt.size() + response.size() > FULL_CONTEXT) std::cerr << "Warning: cuTest(prompt-response) combined size exceeds FULL_CONTEXT." << std::endl;
    // Note: testError, testMSE, testCount should be reset before starting a test run (e.g., in model::test)

    // Store and Reset State for this test run
    int originalCurrentTokenCount = this->currentTokenCount;
    int originalPromptCount = this->promptCount;
    int originalBlockCount = this->blockCount;
    this->currentTokenCount = 0;
    this->blockCount = 1;

    size_t full_context_rows_pr = static_cast<size_t>(this->m) * CONTEXT_WIN;
    this->tokenEmbed = mat(full_context_rows_pr, this->d);
    if (this->tokenEmbed.mapped_data) {
        std::fill_n(this->tokenEmbed.mapped_data, full_context_rows_pr * this->d, 0.0f);
    }

    // --- Device Memory Allocation (Once for the whole sequence) ---
    float* d_tokenEmbed = nullptr; // Device context buffer
    size_t totalTokenEmbedFloats = static_cast<size_t>(m) * CONTEXT_WIN * d; // Use m for full potential context
    size_t tokenEmbedBytes = totalTokenEmbedFloats * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_tokenEmbed, tokenEmbedBytes));
    CUDA_CHECK(cudaMemset(d_tokenEmbed, 0, tokenEmbedBytes)); // Zero out the buffer initially

    // --- Process Prompt (Host context and Device context) ---
    for (size_t p = 0; p < prompt.size(); ++p) {
        if (this->currentTokenCount >= FULL_CONTEXT) {
            std::cerr << "Warning: Context full processing prompt in cuTest(prompt-response)." << std::endl;
            break;
        }
        // Update host context
        if (this->tokenEmbed.mapped_data && static_cast<size_t>(this->currentTokenCount) < static_cast<size_t>(this->tokenEmbed.row) && static_cast<size_t>(this->tokenEmbed.col) == prompt[p].size()) {
            float* dest_ptr = this->tokenEmbed.mapped_data + static_cast<size_t>(this->currentTokenCount) * this->tokenEmbed.col;
            std::copy(prompt[p].begin(), prompt[p].end(), dest_ptr);
        } else {
            std::cerr << "Error: Failed to copy prompt to tokenEmbed in cuTest(prompt-response)." << std::endl;
        }
        // Update device context
        size_t offset_elements = static_cast<size_t>(this->currentTokenCount) * d;
        if (offset_elements + d <= totalTokenEmbedFloats) {
            CUDA_CHECK(cudaMemcpy(d_tokenEmbed + offset_elements, prompt[p].data(), d * sizeof(float), cudaMemcpyHostToDevice));
        } 
        else {
            std::cerr << "Error: Attempting to write past allocated d_tokenEmbed buffer during prompt processing in cuTest(prompt-response)." << std::endl;
        }
        this->currentTokenCount++;
    }
    // tokenEmbed should already be full_context_rows_pr. This check is more of an assertion.
    if (static_cast<size_t>(this->tokenEmbed.row) != full_context_rows_pr) {
        std::cerr << "Warning: tokenEmbed rows (" << this->tokenEmbed.row << ") not matching expected (" << full_context_rows_pr << ") after prompt." << std::endl;
    }

    // Set state after processing prompt
    this->blockCount = (this->currentTokenCount == 0) ? 1 : ((this->currentTokenCount - 1) / CONTEXT_WIN) + 1;
    this->promptCount = this->currentTokenCount; // Size of the prompt just added

    try {
        // --- Test Response ---
        for (size_t i = 0; i < response.size(); ++i) {
            if (this->currentTokenCount >= FULL_CONTEXT) {
                std::cerr << "Warning: cuTest(prompt-response) reached FULL_CONTEXT limit during response." << std::endl;
                break;
            }

            std::vector<float>& expected_vec = response[i];
            std::string& expected_str = rString[i]; // Currently unused in logic

            // Call internal single-token test (uses state *before* adding response token i)
            cuTest(expected_vec, expected_str);

            // --- Update Host State for the *next* prediction ---
            this->currentTokenCount += 1;
            this->blockCount = (this->currentTokenCount == 0) ? 1 : ((this->currentTokenCount - 1) / CONTEXT_WIN) + 1;
            this->promptCount = 1; // Subsequent predictions are single token

            // Add true response token to host context
            if (this->currentTokenCount <= FULL_CONTEXT) { // Check context limit
                size_t current_row_idx_resp = static_cast<size_t>(currentTokenCount - 1);
                if (current_row_idx_resp < static_cast<size_t>(this->tokenEmbed.row)) {
                    if (this->tokenEmbed.mapped_data && static_cast<size_t>(this->tokenEmbed.col) == expected_vec.size()) {
                        float* dest_ptr = this->tokenEmbed.mapped_data + current_row_idx_resp * this->tokenEmbed.col;
                        std::copy(expected_vec.begin(), expected_vec.end(), dest_ptr);
                    } // else: handle error
                }
                else {
                    std::cerr << "Warning: Host tokenEmbed row index out of bounds in cuTest(prompt-response)." << std::endl;
                    break; // tokenEmbed is fixed size
                }
                // Update Device Context (H->D) with the true token for the next step
                size_t offset_elements = static_cast<size_t>(this->currentTokenCount - 1) * d;
                if (offset_elements + d <= totalTokenEmbedFloats) {
                    CUDA_CHECK(cudaMemcpy(d_tokenEmbed + offset_elements, expected_vec.data(), d * sizeof(float), cudaMemcpyHostToDevice));
                }
                else {
                    std::cerr << "Error: Attempting to write past allocated d_tokenEmbed buffer during response processing in cuTest(prompt-response)." << std::endl;
                    break; // Stop processing if buffer bounds exceeded
                }
            }
        } // End loop over response
    } 
    catch (const std::exception& e) {
        std::cerr << "Exception in cuTest(prompt-response): " << e.what() << std::endl;
        if (d_tokenEmbed) cudaFree(d_tokenEmbed); // Cleanup on error
        // Restore original state on error?
        this->currentTokenCount = originalCurrentTokenCount;
        this->promptCount = originalPromptCount;
        this->blockCount = originalBlockCount;
        throw;
    }
}


/**
 * @brief (CUDA) Test transformers for continuous chats.
 * @param prompts All prompts (on host).
 * @param responses Token embeddings of all responses to the prompts (on host).
 * @param rString Tokens of the responses (on host).
 */
void transformer::cuTest(std::vector<std::vector<std::vector<float>>>& prompts, std::vector<std::vector<std::vector<float>>>& responses,
                        std::vector<std::vector<std::string>>& rString)
{
    // Basic Validation
    if (prompts.size() != responses.size() || responses.size() != rString.size()) throw std::runtime_error("cuTest(chat): Mismatch in number of prompts, responses, and response strings.");
    size_t total_tokens_to_add = 0;
    for (size_t i = 0; i < prompts.size(); ++i) {
        if (prompts[i].empty()) throw std::runtime_error("cuTest(chat): Non-empty prompts required.");
        if (responses[i].empty() || responses[i].size() != rString[i].size()) throw std::runtime_error("cuTest(chat): Response mismatch/empty.");
        if (!prompts[i].empty() && prompts[i][0].size() != static_cast<size_t>(d)) throw std::runtime_error("cuTest(chat): Prompt dim mismatch.");
        if (!responses[i].empty() && responses[i][0].size() != static_cast<size_t>(d)) throw std::runtime_error("cuTest(chat): Response dim mismatch.");
        total_tokens_to_add += prompts[i].size() + responses[i].size();
    }

    // Store and Reset State for the entire chat test
    this->currentTokenCount = 0;
    this->blockCount = 1;

    size_t full_context_rows_chat = static_cast<size_t>(this->m) * CONTEXT_WIN;
    this->tokenEmbed = mat(full_context_rows_chat, this->d);
    if (this->tokenEmbed.mapped_data) {
        std::fill_n(this->tokenEmbed.mapped_data, full_context_rows_chat * this->d, 0.0f);
    }
    // Note: testError, testMSE, testCount should be reset before starting a test run (e.g., in model::test)

    // Test each chat turn by calling the prompt-response tester
    for (size_t turn = 0; turn < prompts.size(); ++turn) {
        std::vector<std::vector<float>>& currentPrompt = prompts[turn];
        std::vector<std::vector<float>>& currentResponse = responses[turn];
        std::vector<std::string>& currentRString = rString[turn];

        if (this->currentTokenCount + currentPrompt.size() + currentResponse.size() > FULL_CONTEXT) {
            std::cerr << "Warning: cuTest(chat) exceeds FULL_CONTEXT limit at turn " << turn << ". Stopping chat testing early." << std::endl;
            break;
        }
        // Call the prompt-response tester for this turn.
        // It will internally manage its own device memory and context updates based on the current state.
        cuTest(currentPrompt, currentResponse, currentRString);
    } 
    // End loop over chat turns
}
