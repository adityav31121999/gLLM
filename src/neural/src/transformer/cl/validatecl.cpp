
#ifdef USE_OPENCL

#include "include/transformer.hpp"
#include "include/block.hpp"
#include "include/attention.hpp"
#include "include/mlp.hpp" // For flatten/unflatten, errorofv, MSE
#include <maths.hpp>
#include <CL/cl.hpp>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <string>
#include <cmath> // For std::abs, std::max

/**
 * @brief (OpenCL) Validate the transformer for next token prediction (single token validation).
 * @param promptCount Number of tokens in the prompt.
 * @param currentTokenCount Number of tokens in the full context *before* this validation step.
 * @param blockCount Current block index (1-based) in the full context.
 * @param expected Expected token embedding (on host).
 * @param expString Expected token string (on host, currently unused in logic).
 */
void transformer::clValidate(int& promptCount, int& currentTokenCount, int& blockCount, std::vector<float>& expected, std::string& expString)
{
    // --- Basic Validation ---
    if (expected.size() != static_cast<size_t>(d)) {
        throw std::runtime_error("clValidate(single): Expected vector size mismatch. Expected " + std::to_string(d) + ", got " + std::to_string(expected.size()));
    }
    if (currentTokenCount >= FULL_CONTEXT) {
        // Allow validation up to the last token, but not beyond
        std::cerr << "Warning: clValidate(single) called with currentTokenCount (" << currentTokenCount << ") at or exceeding FULL_CONTEXT (" << FULL_CONTEXT << ")." << std::endl;
        // Depending on desired behavior, could return or throw here. Let's proceed but expect potential issues later.
    }
     if (blockCount <= 0 || blockCount > m) {
         throw std::out_of_range("clValidate(single): Initial blockCount (" + std::to_string(blockCount) + ") is out of range [1, " + std::to_string(m) + "].");
     }

    cl::Buffer d_tokenEmbed, d_embeddings; // No d_expected needed if error calculated on host
    int host_indexForToken = -1; // Host copy of the predicted index
    float current_error = 0.0f;
    float current_mse = 0.0f;

    try {
        // --- Device Buffer Allocation & H->D Transfer ---
        size_t totalTokenEmbedFloats = static_cast<size_t>(m) * CONTEXT_WIN * d;
        size_t tokenEmbedBytes = totalTokenEmbedFloats * sizeof(float);
        size_t embeddingsBytes = static_cast<size_t>(vocabsize) * d * sizeof(float);
        size_t singleTokenBytes = static_cast<size_t>(d) * sizeof(float);
        size_t outputBytes = singleTokenBytes; // Size of h_otok_buffer
        size_t indexBytes = sizeof(int);       // Size for the result index

        // Create buffers
        d_tokenEmbed = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY, tokenEmbedBytes); // Read-only for forward pass

        // Flatten and copy current tokenEmbed context to device
        std::vector<float> flat_host_tokenEmbed;
        flat_host_tokenEmbed.reserve(totalTokenEmbedFloats);
        for(int tk = 0; tk < this->currentTokenCount; ++tk) { // Copy existing context up to currentTokenCount
             if (tk < this->tokenEmbed.size() && this->tokenEmbed[tk].size() == static_cast<size_t>(d)) {
                 flat_host_tokenEmbed.insert(flat_host_tokenEmbed.end(), this->tokenEmbed[tk].begin(), this->tokenEmbed[tk].end());
             } else {
                 std::cerr << "Warning: Inconsistent tokenEmbed data at index " << tk << " during clValidate(single) setup." << std::endl;
                 std::vector<float> padding(d, 0.0f);
                 flat_host_tokenEmbed.insert(flat_host_tokenEmbed.end(), padding.begin(), padding.end());
             }
        }
        // Pad the rest of the buffer with zeros
        flat_host_tokenEmbed.resize(totalTokenEmbedFloats, 0.0f);
        this->clcontext.queue.enqueueWriteBuffer(d_tokenEmbed, CL_TRUE, 0, tokenEmbedBytes, flat_host_tokenEmbed.data());

        // Flatten and copy embeddings table
        std::vector<float> flat_embeddings = flatten(this->embeddings);
        d_embeddings = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, embeddingsBytes, flat_embeddings.data());

        // Determine effective context size and block index for the *current* state
        int effective_context_size = currentTokenCount;
        int current_block_idx = (effective_context_size == 0) ? 1 : ((effective_context_size - 1) / CONTEXT_WIN) + 1;
         if (current_block_idx <= 0 || current_block_idx > m) {
             throw std::out_of_range("clValidate(single): Calculated current_block_idx (" + std::to_string(current_block_idx) + ") is out of range [1, " + std::to_string(m) + "].");
         }

        // --- Forward Pass (Single Pass) ---
        // KdotQ calculation (if needed, depends on clForward implementation)
        // clParallelKdotQs(...); // Placeholder - Needs OpenCL adaptation based on data flow
        clForward(current_block_idx, effective_context_size, promptCount); // Operates on device data implicitly

        // --- Get EH output from the relevant block ---
        // Assume clForward updates the host t[block_idx-1].EH member
        std::vector<float> h_otok_buffer;
        if (current_block_idx > 0 && current_block_idx <= m) {
             h_otok_buffer = t[current_block_idx - 1].EH; // Get EH from the correct block's host copy
             if (h_otok_buffer.size() != static_cast<size_t>(d)) {
                 throw std::runtime_error("clValidate(single): EH buffer from block " + std::to_string(current_block_idx) + " has incorrect size.");
             }
        } else {
             throw std::runtime_error("clValidate(single): Invalid block index (" + std::to_string(current_block_idx) + ") before clComputeOutput.");
        }

        // --- Compute Prediction & Error ---
        // --- Start: Inline clComputeOutput Logic ---
        { // Scope for temporary compute output buffers
            cl::Buffer d_output, d_result_index;
            int result_index_val = -1; // Initialize

            try {
                // Create and copy output (EH) vector
                d_output = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, outputBytes, h_otok_buffer.data());
                // Create buffer for the kernel to write the result index
                d_result_index = cl::Buffer(this->clcontext.context, CL_MEM_WRITE_ONLY, indexBytes);

                // --- Get and Set Kernel Arguments ---
                cl::Kernel kernel = this->clcontext.kernels.at("compute_prediction");
                kernel.setArg(0, d_output);
                kernel.setArg(1, d_embeddings); // Use d_embeddings created earlier
                kernel.setArg(2, static_cast<cl_int>(d)); // dim
                kernel.setArg(3, static_cast<cl_int>(this->vocabsize)); // voc
                kernel.setArg(4, d_result_index); // Output buffer for the index

                // --- Enqueue Kernel ---
                cl::NDRange global(1);
                cl::NDRange local(1);
                cl_int err = this->clcontext.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
                if (err != CL_SUCCESS) {
                    throw cl::Error(err, "Failed to enqueue compute_prediction kernel");
                }

                // --- Read Result Back ---
                err = this->clcontext.queue.enqueueReadBuffer(d_result_index, CL_TRUE, 0, indexBytes, &result_index_val);
                if (err != CL_SUCCESS) {
                    throw cl::Error(err, "Failed to read result index buffer");
                }

                // --- Update Output Parameter ---
                host_indexForToken = result_index_val; // Update the loop's variable

            } catch (const cl::Error& err) {
                std::cerr << "OpenCL Error during inline clComputeOutput in clValidate(single): " << err.what() << " (" << err.err() << ")" << std::endl;
                host_indexForToken = -1; // Indicate error
                throw; // Re-throw
            }
            // Buffers d_output, d_result_index released by RAII
        }
        // --- End: Inline clComputeOutput Logic ---

        current_error = errorofv(h_otok_buffer, expected); // Host-side error calculation
        current_mse = MSE(h_otok_buffer, expected);        // Host-side MSE calculation

        // --- NO BACKWARD PASS ---

        // --- Update Host State ---
        this->validationCount++;
        this->validationError += current_error; // Accumulate validation error
        this->validationMSE += current_mse;     // Accumulate validation MSE
        this->indexForToken = host_indexForToken; // Store the predicted index if needed elsewhere

        // Update context counters for the *next* step (caller manages adding the actual next token)
        int previousTokenCount = this->currentTokenCount; // Store before incrementing
        this->currentTokenCount += 1;

        // Update blockCount for the *next* token
        this->blockCount = (this->currentTokenCount == 0) ? 1 : ((this->currentTokenCount - 1) / CONTEXT_WIN) + 1;
        // Update promptCount for the next step (relative to the current block)
        this->promptCount = this->currentTokenCount % CONTEXT_WIN;
        if (this->promptCount == 0 && this->currentTokenCount > 0) this->promptCount = CONTEXT_WIN;

        // NOTE: The host tokenEmbed is NOT updated here. The caller (e.g., clValidate(sentence))
        // is responsible for adding the *correct* next token (expected) to the host tokenEmbed
        // before the next call to clValidate(single).

    } catch (const cl::Error& err) {
        std::cerr << "OpenCL Error in clValidate(single): " << err.what() << " (" << err.err() << ")" << std::endl;
        throw; // Re-throw
    } catch (const std::exception& e) {
        std::cerr << "Standard Exception in clValidate(single): " << e.what() << std::endl;
        throw; // Re-throw
    }
    // Buffer release is handled by cl::Buffer destructors (RAII)
}


/**
 * @brief (OpenCL) Validate the transformer on sentences.
 * @param sentence Token embeddings of the sentence (on host).
 * @param rString Sentence tokens (on host).
 */
void transformer::clValidate(std::vector<std::vector<float>>& sentence, std::vector<std::string>& rString)
{
    // --- Basic validation ---
    if (sentence.size() > FULL_CONTEXT) {
        std::cerr << "Warning: clValidate(sentence) size (" << sentence.size() << ") exceeds FULL_CONTEXT (" << FULL_CONTEXT << "). Validation might be truncated." << std::endl;
        // Decide how to handle: truncate, throw, or proceed with caution.
    }
    if (sentence.empty() || sentence.size() != rString.size()) {
        throw std::runtime_error("clValidate(sentence): Sentence embeddings/strings mismatch or empty.");
    }
    if (!sentence.empty() && sentence[0].size() != static_cast<size_t>(d)) {
         throw std::runtime_error("clValidate(sentence): Sentence embedding dimension mismatch.");
    }

    // --- Store and Reset State for this validation run ---
    // Store original state if needed, then reset for this validation run
    int originalCurrentTokenCount = this->currentTokenCount;
    int originalBlockCount = this->blockCount;
    // Note: We don't reset validationError/MSE/Count here, assuming accumulation across calls.
    // If validation should be independent per call, reset them here.
    // float originalValidationError = validationError;
    // float originalValidationMSE = validationMSE;
    // int originalValidationCount = validationCount;
    // validationError = 0.0f; validationMSE = 0.0f; validationCount = 0;

    // Reset context state specifically for this sentence/passage validation
    this->currentTokenCount = 0; // Start from the beginning of the context for this sentence
    this->blockCount = 1;        // Start with the first block
    this->tokenEmbed.assign(FULL_CONTEXT, std::vector<float>(d, 0.0f)); // Clear/Resize host tokenEmbed

    // Add the first token as the initial context
    if (!sentence.empty()) {
        this->tokenEmbed[0] = sentence[0];
        this->currentTokenCount = 1;
        this->promptCount = 1; // The first token acts as the initial prompt
    } else {
        // Restore original state if needed before returning
        // this->currentTokenCount = originalCurrentTokenCount;
        // this->blockCount = originalBlockCount;
        return; // Cannot validate an empty sentence
    }

    // --- Validate for each subsequent token in the sentence (i=1 to N-1) ---
    for (size_t i = 1; i < sentence.size(); ++i) {
        if (this->currentTokenCount >= FULL_CONTEXT) {
             std::cerr << "Warning: clValidate(sentence) reached FULL_CONTEXT limit ("
                       << this->currentTokenCount << "). Stopping validation early at sentence index " << i << "." << std::endl;
             break;
        }

        // Target token for this iteration
        std::vector<float>& expected_vec = sentence[i];
        std::string& expected_str = rString[i]; // String is unused in clValidate(single) logic

        // Call the single-token validate function
        // It uses and increments currentTokenCount and blockCount internally
        clValidate(promptCount, currentTokenCount, blockCount, expected_vec, expected_str);

        // After validating token 'i', add its *true* embedding to the host context
        // for the *next* prediction step (i+1).
        // The clValidate(single) function increments currentTokenCount *before* this point.
        if (currentTokenCount <= FULL_CONTEXT) { // Check bounds (currentTokenCount was already incremented)
             // Ensure index is valid for host tokenEmbed
             if (static_cast<size_t>(currentTokenCount - 1) < this->tokenEmbed.size()) {
                 this->tokenEmbed[currentTokenCount - 1] = expected_vec; // Add the actual token embedding
             } else {
                  // This case might indicate an issue if tokenEmbed wasn't resized properly
                  std::cerr << "Warning: Host tokenEmbed size mismatch in clValidate(sentence) at index " << (currentTokenCount - 1) << std::endl;
                  // Optionally resize or handle error
                  this->tokenEmbed.resize(currentTokenCount);
                  this->tokenEmbed[currentTokenCount - 1] = expected_vec;
             }
        }

        // After the first prediction, subsequent steps predict one token based on the preceding ones.
        this->promptCount = 1;

    } // End loop over sentence tokens (i=1 to N-1)

    // Restore original state if validation is part of a larger process and state needs preserving
    // this->currentTokenCount = originalCurrentTokenCount;
    // this->blockCount = originalBlockCount;
}


/**
 * @brief (OpenCL) Validate the transformer for prompt and response.
 * @param prompt Prompt token embeddings (on host).
 * @param response Response token embeddings (on host).
 * @param rString Tokens of the response (on host).
 */
void transformer::clValidate(std::vector<std::vector<float>>& prompt, std::vector<std::vector<float>>& response, std::vector<std::string>& rString)
{
    // --- Basic validation ---
    if (prompt.empty()) {
        throw std::runtime_error("clValidate(prompt-response): Initial prompt cannot be empty.");
    }
    if (response.empty() || response.size() != rString.size()) {
        throw std::runtime_error("clValidate(prompt-response): Response embeddings/strings mismatch or empty.");
    }
    if ((!prompt.empty() && prompt[0].size() != static_cast<size_t>(d)) || (!response.empty() && response[0].size() != static_cast<size_t>(d))) {
        throw std::runtime_error("clValidate(prompt-response): Embedding dimension mismatch.");
    }
    if (this->currentTokenCount + prompt.size() + response.size() > FULL_CONTEXT) {
         std::cerr << "Warning: clValidate(prompt-response) combined size exceeds FULL_CONTEXT. Validation might be incomplete." << std::endl;
    }

    // --- Store and Reset State for this validation run ---
    int originalCurrentTokenCount = this->currentTokenCount;
    int originalBlockCount = this->blockCount;
    // Reset context state specifically for this prompt-response pair
    this->currentTokenCount = 0;
    this->blockCount = 1;
    this->tokenEmbed.assign(FULL_CONTEXT, std::vector<float>(d, 0.0f)); // Clear/Resize host tokenEmbed

    // --- Process Prompt (Add to host context only, device context handled in clValidate single) ---
    for (size_t p = 0; p < prompt.size(); ++p) {
        if (this->currentTokenCount >= FULL_CONTEXT) {
            std::cerr << "Warning: Context full while processing prompt in clValidate(prompt-response). Prompt truncated." << std::endl;
            break;
        }
        // Update host tracking
        if (static_cast<size_t>(this->currentTokenCount) < this->tokenEmbed.size()) {
             this->tokenEmbed[this->currentTokenCount] = prompt[p];
        } else {
             this->tokenEmbed.push_back(prompt[p]); // Should not happen if resized correctly
        }
        this->currentTokenCount++;
    }
    // Resize host vector if push_back was used (shouldn't be needed with assign)
    if (this->tokenEmbed.size() != FULL_CONTEXT) {
        this->tokenEmbed.resize(FULL_CONTEXT, std::vector<float>(d, 0.0f));
    }

    // Set initial promptCount and blockCount *after* processing the entire prompt
    this->blockCount = (this->currentTokenCount == 0) ? 1 : ((this->currentTokenCount - 1) / CONTEXT_WIN) + 1;
    // promptCount for the *first* prediction is the size of the prompt just added.
    this->promptCount = this->currentTokenCount; // Assuming prompt fits in first block or handled correctly

    // --- Validate for Response ---
    for (size_t i = 0; i < response.size(); ++i) {
         if (this->currentTokenCount >= FULL_CONTEXT) {
             std::cerr << "Warning: clValidate(prompt-response) reached FULL_CONTEXT limit ("
                       << this->currentTokenCount << ") during response. Stopping validation early at response index " << i << "." << std::endl;
             break;
         }

        std::vector<float>& expected_vec = response[i];
        std::string& expected_str = rString[i]; // Unused

        // Call single-token validate
        clValidate(promptCount, currentTokenCount, blockCount, expected_vec, expected_str);

        // After validating response token 'i', add its *true* embedding for the next prediction step (i+1)
        if (currentTokenCount <= FULL_CONTEXT) { // currentTokenCount was incremented by clValidate
             if (static_cast<size_t>(currentTokenCount - 1) < this->tokenEmbed.size()) {
                 this->tokenEmbed[currentTokenCount - 1] = expected_vec;
             } else {
                 std::cerr << "Warning: Host tokenEmbed size mismatch in clValidate(prompt-response) at index " << (currentTokenCount - 1) << std::endl;
                 this->tokenEmbed.resize(currentTokenCount);
                 this->tokenEmbed[currentTokenCount - 1] = expected_vec;
             }
        }

        // After the first response token prediction, subsequent predictions are based on a single preceding token context shift.
        this->promptCount = 1;

    } // End loop over response tokens

    // Restore original state if needed
    // this->currentTokenCount = originalCurrentTokenCount;
    // this->blockCount = originalBlockCount;
}


/**
 * @brief (OpenCL) Validate transformers for continuous chats.
 * @param prompts All prompts (on host).
 * @param responses Token embeddings of all responses to the prompts (on host).
 * @param rString Tokens of the responses (on host).
 */
void transformer::clValidate(std::vector<std::vector<std::vector<float>>>& prompts, std::vector<std::vector<std::vector<float>>>& responses,
                        std::vector<std::vector<std::string>>& rString)
{
    // --- Validation ---
    if (prompts.size() != responses.size() || responses.size() != rString.size()) {
        throw std::runtime_error("clValidate(chat): Mismatch in number of prompts, responses, and response strings.");
    }

    // Calculate total tokens required
    size_t total_tokens_to_add = 0;
    for (size_t i = 0; i < prompts.size(); ++i) {
        if (prompts[i].empty()) throw std::runtime_error("clValidate(chat): Non-empty prompts required at index " + std::to_string(i));
        if (responses[i].empty() || responses[i].size() != rString[i].size()) throw std::runtime_error("clValidate(chat): Response mismatch/empty at index " + std::to_string(i));
        if (!prompts[i].empty() && prompts[i][0].size() != static_cast<size_t>(d)) throw std::runtime_error("clValidate(chat): Prompt dim mismatch at index " + std::to_string(i));
        if (!responses[i].empty() && responses[i][0].size() != static_cast<size_t>(d)) throw std::runtime_error("clValidate(chat): Response dim mismatch at index " + std::to_string(i));
        total_tokens_to_add += prompts[i].size() + responses[i].size();
    }

    // --- Store and Reset State for the entire chat ---
    int originalCurrentTokenCount = this->currentTokenCount;
    int originalBlockCount = this->blockCount;
    // Reset context state for the chat
    this->currentTokenCount = 0;
    this->blockCount = 1;
    this->tokenEmbed.assign(FULL_CONTEXT, std::vector<float>(d, 0.0f)); // Clear/Resize host tokenEmbed

    // --- Validate each chat turn ---
    for (size_t turn = 0; turn < prompts.size(); ++turn) {
        std::vector<std::vector<float>>& currentPrompt = prompts[turn];
        std::vector<std::vector<float>>& currentResponse = responses[turn];
        std::vector<std::string>& currentRString = rString[turn];

        // Check if adding this turn exceeds context *before* processing
        if (this->currentTokenCount + currentPrompt.size() + currentResponse.size() > FULL_CONTEXT) {
             std::cerr << "Warning: clValidate(chat) exceeds FULL_CONTEXT limit at turn " << turn << ". Stopping chat validation early." << std::endl;
             break; // Stop processing further turns
        }

        // --- Process Prompt (Add to host context) ---
        int promptStartTokenCount = this->currentTokenCount; // Track where prompt starts
        for(int i = 0; i < currentPrompt.size(); ++i) {
            if (this->currentTokenCount >= FULL_CONTEXT) { /* Should have been caught above */ break; }
            this->tokenEmbed[this->currentTokenCount++] = currentPrompt[i];
        }

        // Set promptCount for the first prediction of this turn's response
        this->promptCount = this->currentTokenCount - promptStartTokenCount;
        // Update blockCount based on state *after* adding prompt
        this->blockCount = (this->currentTokenCount == 0) ? 1 : ((this->currentTokenCount - 1) / CONTEXT_WIN) + 1;

        // --- Validate the response tokens for this turn ---
        for(int i = 0; i < currentResponse.size(); ++i) {
            if (this->currentTokenCount >= FULL_CONTEXT) {
                 std::cerr << "Warning: clValidate(chat) context full during response in turn " << turn << "." << std::endl;
                 break; // Stop processing this response
            }
            std::vector<float>& expectedEmbedding = currentResponse[i];
            std::string& expectedString = currentRString[i]; // Unused

            // Call single-token validate
            clValidate(promptCount, currentTokenCount, blockCount, expectedEmbedding, expectedString);

            // Add true response token embedding to host context for next step
            if (currentTokenCount <= FULL_CONTEXT) { // currentTokenCount was incremented
                 this->tokenEmbed[currentTokenCount - 1] = expectedEmbedding;
            }

            // Subsequent predictions in the response sequence have promptCount=1
            this->promptCount = 1;
        }
        // Check if the inner loop broke due to full context
        if (this->currentTokenCount >= FULL_CONTEXT) {
            break; // Stop processing further turns if context became full during response
        }

    } // End loop over chat turns

    // Restore original state if needed
    // this->currentTokenCount = originalCurrentTokenCount;
    // this->blockCount = originalBlockCount;
}


#endif // USE_OPENCL