
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
 * @brief (OpenCL) Test the transformer for next token prediction (single token testing).
 * @param promptCount Number of tokens in the prompt.
 * @param currentTokenCount Number of tokens in the full context *before* this testing step.
 * @param blockCount Current block index (1-based) in the full context.
 * @param expected Expected token embedding (on host).
 * @param expString Expected token string (on host, currently unused in logic).
 */
void transformer::clTest(int& promptCount, int& currentTokenCount, int& blockCount, std::vector<float>& expected, std::string& expString)
{
    // --- Basic Validation ---
    if (expected.size() != static_cast<size_t>(d)) {
        throw std::runtime_error("clTest(single): Expected vector size mismatch. Expected " + std::to_string(d) + ", got " + std::to_string(expected.size()));
    }
    if (currentTokenCount >= FULL_CONTEXT) {
        std::cerr << "Warning: clTest(single) called with currentTokenCount (" << currentTokenCount << ") at or exceeding FULL_CONTEXT (" << FULL_CONTEXT << ")." << std::endl;
    }
     if (blockCount <= 0 || blockCount > m) {
         throw std::out_of_range("clTest(single): Initial blockCount (" + std::to_string(blockCount) + ") is out of range [1, " + std::to_string(m) + "].");
     }

    cl::Buffer d_tokenEmbed, d_embeddings;
    int host_indexForToken = -1;
    float current_error = 0.0f;
    float current_mse = 0.0f;

    try {
        // --- Device Buffer Allocation & H->D Transfer ---
        size_t totalTokenEmbedFloats = static_cast<size_t>(m) * CONTEXT_WIN * d;
        size_t tokenEmbedBytes = totalTokenEmbedFloats * sizeof(float);
        size_t embeddingsBytes = static_cast<size_t>(vocabsize) * d * sizeof(float);
        size_t singleTokenBytes = static_cast<size_t>(d) * sizeof(float);
        size_t outputBytes = singleTokenBytes;
        size_t indexBytes = sizeof(int);

        // Create buffers
        d_tokenEmbed = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY, tokenEmbedBytes);

        // Flatten and copy current tokenEmbed context to device
        std::vector<float> flat_host_tokenEmbed;
        flat_host_tokenEmbed.reserve(totalTokenEmbedFloats);
        for(int tk = 0; tk < this->currentTokenCount; ++tk) {
             if (tk < this->tokenEmbed.size() && this->tokenEmbed[tk].size() == static_cast<size_t>(d)) {
                 flat_host_tokenEmbed.insert(flat_host_tokenEmbed.end(), this->tokenEmbed[tk].begin(), this->tokenEmbed[tk].end());
             } else {
                 std::cerr << "Warning: Inconsistent tokenEmbed data at index " << tk << " during clTest(single) setup." << std::endl;
                 std::vector<float> padding(d, 0.0f);
                 flat_host_tokenEmbed.insert(flat_host_tokenEmbed.end(), padding.begin(), padding.end());
             }
        }
        flat_host_tokenEmbed.resize(totalTokenEmbedFloats, 0.0f);
        this->clcontext.queue.enqueueWriteBuffer(d_tokenEmbed, CL_TRUE, 0, tokenEmbedBytes, flat_host_tokenEmbed.data());

        // Flatten and copy embeddings table
        std::vector<float> flat_embeddings = flatten(this->embeddings);
        d_embeddings = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, embeddingsBytes, flat_embeddings.data());

        // Determine effective context size and block index
        int effective_context_size = currentTokenCount;
        int current_block_idx = (effective_context_size == 0) ? 1 : ((effective_context_size - 1) / CONTEXT_WIN) + 1;
         if (current_block_idx <= 0 || current_block_idx > m) {
             throw std::out_of_range("clTest(single): Calculated current_block_idx (" + std::to_string(current_block_idx) + ") is out of range [1, " + std::to_string(m) + "].");
         }

        // --- Forward Pass (Single Pass) ---
        clForward(current_block_idx, effective_context_size, promptCount);

        // --- Get EH output ---
        std::vector<float> h_otok_buffer;
        if (current_block_idx > 0 && current_block_idx <= m) {
             h_otok_buffer = t[current_block_idx - 1].EH;
             if (h_otok_buffer.size() != static_cast<size_t>(d)) {
                 throw std::runtime_error("clTest(single): EH buffer from block " + std::to_string(current_block_idx) + " has incorrect size.");
             }
        } else {
             throw std::runtime_error("clTest(single): Invalid block index (" + std::to_string(current_block_idx) + ") before clComputeOutput.");
        }

        // --- Compute Prediction & Error ---
        // --- Start: Inline clComputeOutput Logic ---
        {
            cl::Buffer d_output, d_result_index;
            int result_index_val = -1;
            try {
                d_output = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, outputBytes, h_otok_buffer.data());
                d_result_index = cl::Buffer(this->clcontext.context, CL_MEM_WRITE_ONLY, indexBytes);
                cl::Kernel kernel = this->clcontext.kernels.at("compute_prediction");
                kernel.setArg(0, d_output);
                kernel.setArg(1, d_embeddings);
                kernel.setArg(2, static_cast<cl_int>(d));
                kernel.setArg(3, static_cast<cl_int>(this->vocabsize));
                kernel.setArg(4, d_result_index);
                cl::NDRange global(1); cl::NDRange local(1);
                cl_int err = this->clcontext.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
                if (err != CL_SUCCESS) throw cl::Error(err, "Failed to enqueue compute_prediction kernel");
                err = this->clcontext.queue.enqueueReadBuffer(d_result_index, CL_TRUE, 0, indexBytes, &result_index_val);
                if (err != CL_SUCCESS) throw cl::Error(err, "Failed to read result index buffer");
                host_indexForToken = result_index_val;
            } catch (const cl::Error& err) {
                std::cerr << "OpenCL Error during inline clComputeOutput in clTest(single): " << err.what() << " (" << err.err() << ")" << std::endl;
                host_indexForToken = -1; throw;
            }
        }
        // --- End: Inline clComputeOutput Logic ---

        current_error = errorofv(h_otok_buffer, expected);
        current_mse = MSE(h_otok_buffer, expected);

        // --- NO BACKWARD PASS ---

        // --- Update Host State ---
        this->testCount++;
        this->testError += current_error; // Accumulate test error
        this->testMSE += current_mse;     // Accumulate test MSE
        this->indexForToken = host_indexForToken;

        // Update context counters for the *next* step
        this->currentTokenCount += 1;
        this->blockCount = (this->currentTokenCount == 0) ? 1 : ((this->currentTokenCount - 1) / CONTEXT_WIN) + 1;
        this->promptCount = this->currentTokenCount % CONTEXT_WIN;
        if (this->promptCount == 0 && this->currentTokenCount > 0) this->promptCount = CONTEXT_WIN;

        // Host tokenEmbed managed by caller

    } catch (const cl::Error& err) {
        std::cerr << "OpenCL Error in clTest(single): " << err.what() << " (" << err.err() << ")" << std::endl;
        throw;
    } catch (const std::exception& e) {
        std::cerr << "Standard Exception in clTest(single): " << e.what() << std::endl;
        throw;
    }
}


/**
 * @brief (OpenCL) Test the transformer on sentences.
 * @param sentence Token embeddings of the sentence (on host).
 * @param rString Sentence tokens (on host).
 */
void transformer::clTest(std::vector<std::vector<float>>& sentence, std::vector<std::string>& rString)
{
    // --- Basic validation ---
    if (sentence.size() > FULL_CONTEXT) {
        std::cerr << "Warning: clTest(sentence) size (" << sentence.size() << ") exceeds FULL_CONTEXT (" << FULL_CONTEXT << "). Testing might be truncated." << std::endl;
    }
    if (sentence.empty() || sentence.size() != rString.size()) {
        throw std::runtime_error("clTest(sentence): Sentence embeddings/strings mismatch or empty.");
    }
    if (!sentence.empty() && sentence[0].size() != static_cast<size_t>(d)) {
         throw std::runtime_error("clTest(sentence): Sentence embedding dimension mismatch.");
    }

    // --- Store and Reset State ---
    int originalCurrentTokenCount = this->currentTokenCount;
    int originalBlockCount = this->blockCount;
    // Reset context state
    this->currentTokenCount = 0;
    this->blockCount = 1;
    this->tokenEmbed.assign(FULL_CONTEXT, std::vector<float>(d, 0.0f));

    // Add first token to context
    if (!sentence.empty()) {
        this->tokenEmbed[0] = sentence[0];
        this->currentTokenCount = 1;
        this->promptCount = 1;
    } else {
        return; // Cannot test empty sentence
    }

    // --- Test subsequent tokens ---
    for (size_t i = 1; i < sentence.size(); ++i) {
        if (this->currentTokenCount >= FULL_CONTEXT) {
             std::cerr << "Warning: clTest(sentence) reached FULL_CONTEXT limit ("
                       << this->currentTokenCount << "). Stopping testing early at sentence index " << i << "." << std::endl;
             break;
        }

        std::vector<float>& expected_vec = sentence[i];
        std::string& expected_str = rString[i];

        // Call single-token test
        clTest(promptCount, currentTokenCount, blockCount, expected_vec, expected_str);

        // Add true embedding to host context for next step
        if (currentTokenCount <= FULL_CONTEXT) {
             if (static_cast<size_t>(currentTokenCount - 1) < this->tokenEmbed.size()) {
                 this->tokenEmbed[currentTokenCount - 1] = expected_vec;
             } else {
                  std::cerr << "Warning: Host tokenEmbed size mismatch in clTest(sentence) at index " << (currentTokenCount - 1) << std::endl;
                  this->tokenEmbed.resize(currentTokenCount);
                  this->tokenEmbed[currentTokenCount - 1] = expected_vec;
             }
        }
        this->promptCount = 1; // Subsequent steps

    } // End loop over sentence

    // Restore original state if needed
    // this->currentTokenCount = originalCurrentTokenCount;
    // this->blockCount = originalBlockCount;
}


/**
 * @brief (OpenCL) Test the transformer for prompt and response.
 * @param prompt Prompt token embeddings (on host).
 * @param response Response token embeddings (on host).
 * @param rString Tokens of the response (on host).
 */
void transformer::clTest(std::vector<std::vector<float>>& prompt, std::vector<std::vector<float>>& response, std::vector<std::string>& rString)
{
    // --- Basic validation ---
    if (prompt.empty()) throw std::runtime_error("clTest(prompt-response): Initial prompt cannot be empty.");
    if (response.empty() || response.size() != rString.size()) throw std::runtime_error("clTest(prompt-response): Response embeddings/strings mismatch or empty.");
    if ((!prompt.empty() && prompt[0].size() != static_cast<size_t>(d)) || (!response.empty() && response[0].size() != static_cast<size_t>(d))) throw std::runtime_error("clTest(prompt-response): Embedding dimension mismatch.");
    if (this->currentTokenCount + prompt.size() + response.size() > FULL_CONTEXT) std::cerr << "Warning: clTest(prompt-response) combined size exceeds FULL_CONTEXT." << std::endl;

    // --- Store and Reset State ---
    int originalCurrentTokenCount = this->currentTokenCount;
    int originalBlockCount = this->blockCount;
    this->currentTokenCount = 0;
    this->blockCount = 1;
    this->tokenEmbed.assign(FULL_CONTEXT, std::vector<float>(d, 0.0f));

    // --- Process Prompt (Host context) ---
    for (size_t p = 0; p < prompt.size(); ++p) {
        if (this->currentTokenCount >= FULL_CONTEXT) { std::cerr << "Warning: Context full processing prompt in clTest(prompt-response)." << std::endl; break; }
        this->tokenEmbed[this->currentTokenCount++] = prompt[p];
    }
    if (this->tokenEmbed.size() != FULL_CONTEXT) this->tokenEmbed.resize(FULL_CONTEXT, std::vector<float>(d, 0.0f));

    // Set initial state after prompt
    this->blockCount = (this->currentTokenCount == 0) ? 1 : ((this->currentTokenCount - 1) / CONTEXT_WIN) + 1;
    this->promptCount = this->currentTokenCount;

    // --- Test Response ---
    for (size_t i = 0; i < response.size(); ++i) {
         if (this->currentTokenCount >= FULL_CONTEXT) {
             std::cerr << "Warning: clTest(prompt-response) reached FULL_CONTEXT limit ("
                       << this->currentTokenCount << ") during response. Stopping testing early at response index " << i << "." << std::endl;
             break;
         }

        std::vector<float>& expected_vec = response[i];
        std::string& expected_str = rString[i];

        // Call single-token test
        clTest(promptCount, currentTokenCount, blockCount, expected_vec, expected_str);

        // Add true embedding to host context
        if (currentTokenCount <= FULL_CONTEXT) {
             if (static_cast<size_t>(currentTokenCount - 1) < this->tokenEmbed.size()) {
                 this->tokenEmbed[currentTokenCount - 1] = expected_vec;
             } else {
                 std::cerr << "Warning: Host tokenEmbed size mismatch in clTest(prompt-response) at index " << (currentTokenCount - 1) << std::endl;
                 this->tokenEmbed.resize(currentTokenCount);
                 this->tokenEmbed[currentTokenCount - 1] = expected_vec;
             }
        }
        this->promptCount = 1; // Subsequent predictions

    } // End loop over response

    // Restore original state if needed
    // this->currentTokenCount = originalCurrentTokenCount;
    // this->blockCount = originalBlockCount;
}


/**
 * @brief (OpenCL) Test transformers for continuous chats.
 * @param prompts All prompts (on host).
 * @param responses Token embeddings of all responses to the prompts (on host).
 * @param rString Tokens of the responses (on host).
 */
void transformer::clTest(std::vector<std::vector<std::vector<float>>>& prompts, std::vector<std::vector<std::vector<float>>>& responses,
                        std::vector<std::vector<std::string>>& rString)
{
    // --- Validation ---
    if (prompts.size() != responses.size() || responses.size() != rString.size()) throw std::runtime_error("clTest(chat): Mismatch in number of prompts, responses, and response strings.");
    size_t total_tokens_to_add = 0;
    for (size_t i = 0; i < prompts.size(); ++i) {
        if (prompts[i].empty()) throw std::runtime_error("clTest(chat): Non-empty prompts required at index " + std::to_string(i));
        if (responses[i].empty() || responses[i].size() != rString[i].size()) throw std::runtime_error("clTest(chat): Response mismatch/empty at index " + std::to_string(i));
        if (!prompts[i].empty() && prompts[i][0].size() != static_cast<size_t>(d)) throw std::runtime_error("clTest(chat): Prompt dim mismatch at index " + std::to_string(i));
        if (!responses[i].empty() && responses[i][0].size() != static_cast<size_t>(d)) throw std::runtime_error("clTest(chat): Response dim mismatch at index " + std::to_string(i));
        total_tokens_to_add += prompts[i].size() + responses[i].size();
    }

    // --- Store and Reset State ---
    int originalCurrentTokenCount = this->currentTokenCount;
    int originalBlockCount = this->blockCount;
    this->currentTokenCount = 0;
    this->blockCount = 1;
    this->tokenEmbed.assign(FULL_CONTEXT, std::vector<float>(d, 0.0f));

    // --- Test each chat turn ---
    for (size_t turn = 0; turn < prompts.size(); ++turn) {
        std::vector<std::vector<float>>& currentPrompt = prompts[turn];
        std::vector<std::vector<float>>& currentResponse = responses[turn];
        std::vector<std::string>& currentRString = rString[turn];

        if (this->currentTokenCount + currentPrompt.size() + currentResponse.size() > FULL_CONTEXT) {
             std::cerr << "Warning: clTest(chat) exceeds FULL_CONTEXT limit at turn " << turn << ". Stopping chat testing early." << std::endl;
             break;
        }

        // --- Process Prompt (Host context) ---
        int promptStartTokenCount = this->currentTokenCount;
        for(int i = 0; i < currentPrompt.size(); ++i) {
            if (this->currentTokenCount >= FULL_CONTEXT) break;
            this->tokenEmbed[this->currentTokenCount++] = currentPrompt[i];
        }
        this->promptCount = this->currentTokenCount - promptStartTokenCount;
        this->blockCount = (this->currentTokenCount == 0) ? 1 : ((this->currentTokenCount - 1) / CONTEXT_WIN) + 1;

        // --- Test response tokens ---
        for(int i = 0; i < currentResponse.size(); ++i) {
            if (this->currentTokenCount >= FULL_CONTEXT) {
                 std::cerr << "Warning: clTest(chat) context full during response in turn " << turn << "." << std::endl;
                 break;
            }
            std::vector<float>& expectedEmbedding = currentResponse[i];
            std::string& expectedString = currentRString[i];

            // Call single-token test
            clTest(promptCount, currentTokenCount, blockCount, expectedEmbedding, expectedString);

            // Add true response token to host context
            if (currentTokenCount <= FULL_CONTEXT) {
                 this->tokenEmbed[currentTokenCount - 1] = expectedEmbedding;
            }
            this->promptCount = 1; // Subsequent predictions
        }
        if (this->currentTokenCount >= FULL_CONTEXT) break; // Stop if context full during response

    } // End loop over chat turns

    // Restore original state if needed
    // this->currentTokenCount = originalCurrentTokenCount;
    // this->blockCount = originalBlockCount;
}


#endif // USE_OPENCL