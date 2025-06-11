
#ifdef USE_OPENCL

#include "include/transformer.hpp"
#include "include/block.hpp"
#include "include/attention.hpp"
#include "include/mlp.hpp" // For flatten/unflatten, errorofv, MSE
#include <maths.hpp>
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

    cl_int cl_err; // For OpenCL error codes

    try {
        // --- Device Buffer Allocation & H->D Transfer ---
        size_t totalTokenEmbedFloats = static_cast<size_t>(m) * CONTEXT_WIN * d;
        size_t tokenEmbedBytes = totalTokenEmbedFloats * sizeof(float);
        size_t embeddingsBytes = static_cast<size_t>(vocabsize) * d * sizeof(float);
        size_t singleTokenBytes = static_cast<size_t>(d) * sizeof(float);
        size_t outputBytes = singleTokenBytes; // Size of h_otok_buffer
        size_t indexBytes = sizeof(int);       // Size for the result index

        // Create buffers
        d_tokenEmbed = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY, tokenEmbedBytes, nullptr, &cl_err); CL_CHECK(cl_err); // Read-only for forward pass

        // Flatten and copy current tokenEmbed context to device
        // Flatten and copy current tokenEmbed context to device
        std::vector<float> flat_host_tokenEmbed;
        if (!this->tokenEmbed.mapped_data || this->tokenEmbed.row < 1 || this->tokenEmbed.col != static_cast<size_t>(d)) {
            throw std::runtime_error("clValidate(single): Invalid or uninitialized tokenEmbed (mat) dimensions.");
        }

        flat_host_tokenEmbed.reserve(totalTokenEmbedFloats);
        for(int tk = 0; tk < this->currentTokenCount; ++tk) { // Copy existing context up to currentTokenCount
            if (this->tokenEmbed.mapped_data &&
                static_cast<size_t>(tk) < this->tokenEmbed.row &&
                this->tokenEmbed.col == static_cast<size_t>(d))
            {
                // Calculate the pointer to the start of the tk-th row
                float* row_ptr = this->tokenEmbed.mapped_data + (static_cast<size_t>(tk) * this->tokenEmbed.col);
                // Insert the 'd' elements of this row into flat_host_tokenEmbed
                flat_host_tokenEmbed.insert(flat_host_tokenEmbed.end(), row_ptr, row_ptr + d);
             } else {
                std::cerr << "Warning: Inconsistent tokenEmbed (mat) data or bounds at index " << tk
                          << " (mat_rows: " << (this->tokenEmbed.mapped_data ? std::to_string(this->tokenEmbed.row) : "N/A")
                          << ", mat_cols: " << (this->tokenEmbed.mapped_data ? std::to_string(this->tokenEmbed.col) : "N/A")
                          << ", expected_cols: " << d << ") during clValidate(single) setup. Padding." << std::endl;
                 std::vector<float> padding(d, 0.0f);
                 flat_host_tokenEmbed.insert(flat_host_tokenEmbed.end(), padding.begin(), padding.end());
             }
        }
        // Pad the rest of the buffer with zeros
        flat_host_tokenEmbed.resize(totalTokenEmbedFloats, 0.0f);
        CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_tokenEmbed, CL_TRUE, 0, tokenEmbedBytes, flat_host_tokenEmbed.data()));

        // Flatten and copy embeddings table
        std::vector<float> flat_embeddings = ::flatten(this->embeddings); // Assuming global or from basic.hpp
        d_embeddings = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, embeddingsBytes, flat_embeddings.data(), &cl_err); CL_CHECK(cl_err);

        // Determine effective context size and block index for the *current* state
        int effective_context_size = currentTokenCount;
        int current_block_idx = (effective_context_size == 0) ? 1 : ((effective_context_size - 1) / CONTEXT_WIN) + 1;
         if (current_block_idx <= 0 || current_block_idx > m) {
             throw std::out_of_range("clValidate(single): Calculated current_block_idx (" + std::to_string(current_block_idx) + ") is out of range [1, " + std::to_string(m) + "].");
         }

        // --- Forward Pass (Single Pass) ---
        clForward(current_block_idx, effective_context_size, promptCount); // Operates on device data implicitly

        std::vector<float> h_otok_buffer;
        if (current_block_idx > 0 && current_block_idx <= m) {
             h_otok_buffer = this->otok; // Get EH from the correct block's host copy
             if (h_otok_buffer.size() != static_cast<size_t>(d)) {
                 throw std::runtime_error("clValidate(single): EH buffer from block " + std::to_string(current_block_idx) + " has incorrect size.");
             }
        } else {
             throw std::runtime_error("clValidate(single): Invalid block index (" + std::to_string(current_block_idx) + ") before clComputeOutput.");
        }

        { // Scope for temporary compute output buffers
            cl::Buffer d_output, d_result_index;
            int result_index_val = -1; // Initialize

            d_output = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, outputBytes, h_otok_buffer.data(), &cl_err); CL_CHECK(cl_err);
            d_result_index = cl::Buffer(this->clcontext.context, CL_MEM_WRITE_ONLY, indexBytes, nullptr, &cl_err); CL_CHECK(cl_err);

            try { // Keep try-catch for kernel name lookup
                // --- Get and Set Kernel Arguments ---
                cl::Kernel kernel = this->clcontext.kernels.at("compute_prediction");
                CL_CHECK(kernel.setArg(0, d_output));
                CL_CHECK(kernel.setArg(1, d_embeddings)); // Use d_embeddings created earlier
                CL_CHECK(kernel.setArg(2, static_cast<cl_int>(d))); // dim
                CL_CHECK(kernel.setArg(3, static_cast<cl_int>(this->vocabsize))); // voc
                CL_CHECK(kernel.setArg(4, d_result_index)); // Output buffer for the index

                // --- Enqueue Kernel ---
                cl::NDRange global(1);
                cl::NDRange local(1);
                CL_CHECK(this->clcontext.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local));

                // --- Read Result Back ---
                CL_CHECK(this->clcontext.queue.enqueueReadBuffer(d_result_index, CL_TRUE, 0, indexBytes, &result_index_val));

                // --- Update Output Parameter ---
                host_indexForToken = result_index_val; // Update the loop's variable
            } catch (const std::out_of_range& oor) { // Catch if kernel "compute_prediction" is not found
                std::cerr << "Error: Kernel 'compute_prediction' not found. " << oor.what() << std::endl;
                host_indexForToken = -1; // Indicate error
                throw; // Re-throw
            }
            // Buffers d_output, d_result_index released by RAII
        }
        // --- End: Inline clComputeOutput Logic ---

        current_error = errorofv(h_otok_buffer, expected); // Host-side error calculation
        current_mse = MSE(h_otok_buffer, expected);        // Host-side MSE calculation

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
    }
    catch (const std::runtime_error& e) {
        std::cerr << "OpenCL Runtime Error in clValidate(single): " << e.what() << std::endl;
        throw; // Re-throw
    }
    catch (const std::exception& e) {
        std::cerr << "Standard Exception in clValidate(single): " << e.what() << std::endl;
        throw; // Re-throw
    }
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
    this->currentTokenCount = 0; // Start from the beginning of the context for this sentence
    this->blockCount = 1;        // Start with the first block
    // Zero-fill the tokenEmbed mat
    if (this->tokenEmbed.mapped_data && this->tokenEmbed.row == FULL_CONTEXT && this->tokenEmbed.col == static_cast<size_t>(d)) {
        std::fill_n(this->tokenEmbed.mapped_data, static_cast<size_t>(FULL_CONTEXT) * d, 0.0f);
    }
    else {
        throw std::runtime_error("clValidate(sentence): tokenEmbed (mat) is not properly initialized to FULL_CONTEXT for zero-filling.");
    }

    // Add the first token as the initial context
    if (!sentence.empty() && sentence[0].size() == static_cast<size_t>(d)) {
        memcpy(this->tokenEmbed.mapped_data, sentence[0].data(), d * sizeof(float)); // Copy to the beginning of the mapped region
        this->currentTokenCount = 1;
        this->promptCount = 1; // The first token acts as the initial prompt
    }
    else {
        std::cerr << "Warning: Empty or invalid initial sentence embedding in clValidate(sentence). Skipping sentence." << std::endl;
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
        clValidate(promptCount, currentTokenCount, blockCount, expected_vec, expected_str);

        if (currentTokenCount <= FULL_CONTEXT) { // Check bounds (currentTokenCount was already incremented)
            if (this->tokenEmbed.mapped_data && static_cast<size_t>(currentTokenCount - 1) < this->tokenEmbed.row && this->tokenEmbed.col == static_cast<size_t>(d)) {
                float* dest_ptr = this->tokenEmbed.mapped_data + (static_cast<size_t>(currentTokenCount - 1) * this->d);
                memcpy(dest_ptr, expected_vec.data(), static_cast<size_t>(this->d) * sizeof(float));
            }
            else {
                std::cerr << "Warning: Host tokenEmbed (mat) out of bounds or not mapped in clValidate(sentence) at index " << (currentTokenCount - 1) << std::endl;
            }
        }

        // After the first prediction, subsequent steps predict one token based on the preceding ones.
        this->promptCount = 1;

    } // End loop over sentence tokens (i=1 to N-1)
}


/**
 * @brief (OpenCL) Validate the transformer for prompt and response.
 * @param prompt Prompt token embeddings (on host).
 * @param response Response token embeddings (on host).
 * @param rString Tokens of the response (on host).
 */
void transformer::clValidate(std::vector<std::vector<float>>& prompt, std::vector<std::vector<float>>& response, 
    std::vector<std::string>& rString)
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
    // Zero-fill the tokenEmbed mat
    if (this->tokenEmbed.mapped_data && this->tokenEmbed.row == FULL_CONTEXT && this->tokenEmbed.col == static_cast<size_t>(d)) {
        std::fill_n(this->tokenEmbed.mapped_data, static_cast<size_t>(FULL_CONTEXT) * d, 0.0f);
    }
    else {
        throw std::runtime_error("clValidate(prompt-response): tokenEmbed (mat) is not properly initialized to FULL_CONTEXT for zero-filling.");
    }

    // For K/Q computation
    cl::Buffer d_Q_cl, d_K_cl, d_mQ_cl, d_mK_cl, d_tok_cl;
    int embedding_dim_cl = d; // EMBEDDING is 'd'
    int mat_heights_cl = MATHEIGHTS;
    cl::Kernel kq_kernel;
    cl_int cl_err_kq;

    if (mat_heights_cl > 0 && x > 0 && y > 0) { // Only if attention is meaningful
        try {
            kq_kernel = cl::Kernel(this->clcontext.program, "kernelCompute_single_kq_vector", &cl_err_kq); CL_CHECK(cl_err_kq);
        } catch (const std::runtime_error& e) {
            std::cerr << "OpenCL Error creating kernel 'kernelCompute_single_kq_vector' in clValidate(prompt-response): " << e.what() << std::endl;
            throw;
        }
        size_t matheights_bytes_kq = static_cast<size_t>(mat_heights_cl) * sizeof(float);
        size_t embedding_bytes_loc_kq = static_cast<size_t>(embedding_dim_cl) * sizeof(float);
        size_t projection_matrix_bytes_kq = static_cast<size_t>(mat_heights_cl) * embedding_dim_cl * sizeof(float);

        d_Q_cl = cl::Buffer(this->clcontext.context, CL_MEM_WRITE_ONLY, matheights_bytes_kq, nullptr, &cl_err_kq); CL_CHECK(cl_err_kq);
        d_K_cl = cl::Buffer(this->clcontext.context, CL_MEM_WRITE_ONLY, matheights_bytes_kq, nullptr, &cl_err_kq); CL_CHECK(cl_err_kq);
        d_mQ_cl = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY, projection_matrix_bytes_kq, nullptr, &cl_err_kq); CL_CHECK(cl_err_kq);
        d_mK_cl = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY, projection_matrix_bytes_kq, nullptr, &cl_err_kq); CL_CHECK(cl_err_kq);
        d_tok_cl = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY, embedding_bytes_loc_kq, nullptr, &cl_err_kq); CL_CHECK(cl_err_kq);
    }

    // --- Process Prompt (Add to host context only, device context handled in clValidate single) ---
    for (size_t p = 0; p < prompt.size(); ++p) {
        if (this->currentTokenCount >= FULL_CONTEXT) {
            std::cerr << "Warning: Context full while processing prompt in clValidate(prompt-response). Prompt truncated." << std::endl;
            break;
        }
        // Update host tracking
        if (this->tokenEmbed.mapped_data && static_cast<size_t>(this->currentTokenCount) < this->tokenEmbed.row && this->tokenEmbed.col == static_cast<size_t>(d)) {
            float* dest_ptr = this->tokenEmbed.mapped_data + (static_cast<size_t>(this->currentTokenCount) * this->d);
            if (prompt[p].size() == static_cast<size_t>(d)) {
                memcpy(dest_ptr, prompt[p].data(), static_cast<size_t>(d) * sizeof(float));
            } else {
                std::cerr << "Warning: Prompt token dimension mismatch in clValidate(prompt-response) prompt processing at index " << p << std::endl;
            }
        }
        else {
            std::cerr << "Warning: currentTokenCount " << this->currentTokenCount << " is out of bounds for tokenEmbed (mat rows: " << this->tokenEmbed.row << ") or not mapped in clValidate(prompt-response) prompt processing." << std::endl;
        }
        this->currentTokenCount++;
    }
    if (this->tokenEmbed.row != FULL_CONTEXT) {
        throw std::runtime_error("clValidate(prompt-response): tokenEmbed (mat) does not have FULL_CONTEXT rows after prompt processing.");
    }

    // Set initial promptCount and blockCount *after* processing the entire prompt
    this->blockCount = (this->currentTokenCount == 0) ? 1 : ((this->currentTokenCount - 1) / CONTEXT_WIN) + 1;
    // promptCount for the *first* prediction is the size of the prompt just added.
    this->promptCount = this->currentTokenCount; // Assuming prompt fits in first block or handled correctly

    // --- K/Q Calculation for Prompt Tokens ---
    if (mat_heights_cl > 0 && x > 0 && y > 0) {
        size_t num_prompt_tokens = prompt.size();
        // blockCount is already updated based on currentTokenCount (which is num_prompt_tokens)
        int num_blocks_spanned_by_prompt = this->blockCount;
        
        size_t embedding_bytes_loc_kq = static_cast<size_t>(embedding_dim_cl) * sizeof(float);
        size_t projection_matrix_bytes_kq = static_cast<size_t>(mat_heights_cl) * embedding_dim_cl * sizeof(float);
        size_t matheights_bytes_kq = static_cast<size_t>(mat_heights_cl) * sizeof(float);

        for (int b_idx = 0; b_idx < num_blocks_spanned_by_prompt; ++b_idx) {
            for (int layer_idx = 0; layer_idx < x; ++layer_idx) {
                for (int parallel_idx = 0; parallel_idx < y; ++parallel_idx) {
                    auto& current_block_attention = t[b_idx].b[layer_idx][parallel_idx];
                    CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_mQ_cl, CL_TRUE, 0, projection_matrix_bytes_kq, current_block_attention.MQ.mapped_data));
                    CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_mK_cl, CL_TRUE, 0, projection_matrix_bytes_kq, current_block_attention.MK.mapped_data));

                    // Calculate K for prompt tokens in this block b_idx
                    size_t start_prompt_token_idx_for_block = static_cast<size_t>(b_idx) * CONTEXT_WIN;
                    size_t end_prompt_token_idx_for_block = std::min(num_prompt_tokens, (static_cast<size_t>(b_idx) + 1) * CONTEXT_WIN);

                    for (size_t p_glob = start_prompt_token_idx_for_block; p_glob < end_prompt_token_idx_for_block; ++p_glob) {
                        size_t p_local = p_glob % CONTEXT_WIN;
                        size_t host_offset = p_local * mat_heights_cl;
                        CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_tok_cl, CL_TRUE, 0, embedding_bytes_loc_kq, prompt[p_glob].data()));

                        kq_kernel.setArg(0, d_tok_cl); kq_kernel.setArg(1, d_mK_cl); kq_kernel.setArg(2, d_K_cl);
                        kq_kernel.setArg(3, embedding_dim_cl); kq_kernel.setArg(4, mat_heights_cl);
                        CL_CHECK(this->clcontext.queue.enqueueNDRangeKernel(kq_kernel, cl::NullRange, cl::NDRange(1), cl::NullRange));
                        if (current_block_attention.K.mapped_data && (host_offset + mat_heights_cl) <= (current_block_attention.K.row * current_block_attention.K.col)) {
                            CL_CHECK(this->clcontext.queue.enqueueReadBuffer(d_K_cl, CL_TRUE, 0, matheights_bytes_kq, current_block_attention.K.mapped_data + host_offset));
                        } 
                        else { 
                            std::cerr << "Error: Host K buffer invalid or out of bounds for prompt K, block " << b_idx << " in clValidate." << std::endl; 
                        }
                    }

                    // Calculate Q
                    if (b_idx == 0) { // For prompt tokens in block 0
                        for (size_t p_glob = start_prompt_token_idx_for_block; p_glob < end_prompt_token_idx_for_block; ++p_glob) {
                            size_t p_local = p_glob % CONTEXT_WIN;
                            size_t host_offset = p_local * mat_heights_cl;
                            CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_tok_cl, CL_TRUE, 0, embedding_bytes_loc_kq, prompt[p_glob].data()));
                            
                            kq_kernel.setArg(0, d_tok_cl); kq_kernel.setArg(1, d_mQ_cl); kq_kernel.setArg(2, d_Q_cl);
                            // Args 3,4 already set
                            CL_CHECK(this->clcontext.queue.enqueueNDRangeKernel(kq_kernel, cl::NullRange, cl::NDRange(1), cl::NullRange));
                            if (current_block_attention.Q.mapped_data && (host_offset + mat_heights_cl) <= (current_block_attention.Q.row * current_block_attention.Q.col)) {
                                CL_CHECK(this->clcontext.queue.enqueueReadBuffer(d_Q_cl, CL_TRUE, 0, matheights_bytes_kq, current_block_attention.Q.mapped_data + host_offset));
                            } 
                            else { 
                                std::cerr << "Error: Host Q buffer invalid or out of bounds for prompt Q, block 0 in clValidate." << std::endl; 
                            }
                        }
                    } 
                    else { // Q from previous block's EV for b_idx > 0
                        auto& prev_block_attention_for_ev = t[b_idx - 1].b[layer_idx][parallel_idx];
                        for (int k_ev = 0; k_ev < CONTEXT_WIN; ++k_ev) {
                            size_t host_q_offset = static_cast<size_t>(k_ev) * mat_heights_cl;
                            if (!prev_block_attention_for_ev.EV.mapped_data || (static_cast<size_t>(k_ev) * embedding_dim_cl + embedding_dim_cl) > (prev_block_attention_for_ev.EV.row * prev_block_attention_for_ev.EV.col)) {
                                std::cerr << "Warning: Prev block EV data invalid or out of bounds for index " << k_ev << " in clValidate K/Q for prompt." << std::endl; continue;
                            }
                            CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_tok_cl, CL_TRUE, 0, embedding_bytes_loc_kq, prev_block_attention_for_ev.EV.mapped_data + static_cast<size_t>(k_ev) * embedding_dim_cl));
                            kq_kernel.setArg(0, d_tok_cl); kq_kernel.setArg(1, d_mQ_cl); kq_kernel.setArg(2, d_Q_cl);
                            CL_CHECK(this->clcontext.queue.enqueueNDRangeKernel(kq_kernel, cl::NullRange, cl::NDRange(1), cl::NullRange));
                            if (current_block_attention.Q.mapped_data && (host_q_offset + mat_heights_cl) <= (current_block_attention.Q.row * current_block_attention.Q.col)) {
                                CL_CHECK(this->clcontext.queue.enqueueReadBuffer(d_Q_cl, CL_TRUE, 0, matheights_bytes_kq, current_block_attention.Q.mapped_data + host_q_offset));
                            } 
                            else { 
                                std::cerr << "Error: Host Q buffer invalid or out of bounds for EV Q, block " << b_idx << " in clValidate." << std::endl; 
                            }
                        }
                    }
                }
            }
        }
        this->clcontext.queue.finish();
    }

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
             if (this->tokenEmbed.mapped_data && static_cast<size_t>(currentTokenCount - 1) < this->tokenEmbed.row && this->tokenEmbed.col == static_cast<size_t>(d)) {
                float* dest_ptr = this->tokenEmbed.mapped_data + (static_cast<size_t>(currentTokenCount - 1) * this->d);
                memcpy(dest_ptr, expected_vec.data(), static_cast<size_t>(this->d) * sizeof(float));
             } else {
                 std::cerr << "Warning: Host tokenEmbed (mat) out of bounds or not mapped in clValidate(prompt-response) response processing at index " << (currentTokenCount - 1) << std::endl;
                 // For mat, resize is not an option. This is an error condition.
             }
        }

        // --- K/Q Calculation for the processed response token ---
        if (mat_heights_cl > 0 && x > 0 && y > 0) {
            // currentTokenCount was incremented by clValidate(single-token call)
            // The token just added to context is expected_vec (response[i])
            // Its global index in this validation run is this->currentTokenCount - 1
            int target_block_idx_resp = (this->currentTokenCount - 1) / CONTEXT_WIN;
            size_t qk_idx_in_block_resp = (static_cast<size_t>(this->currentTokenCount - 1) % CONTEXT_WIN);
            size_t host_offset_resp = qk_idx_in_block_resp * mat_heights_cl;
            
            size_t embedding_bytes_loc_kq = static_cast<size_t>(embedding_dim_cl) * sizeof(float);
            size_t projection_matrix_bytes_kq = static_cast<size_t>(mat_heights_cl) * embedding_dim_cl * sizeof(float);
            size_t matheights_bytes_kq = static_cast<size_t>(mat_heights_cl) * sizeof(float);

            if (target_block_idx_resp < 0 || static_cast<size_t>(target_block_idx_resp) >= t.size()) {
                 std::cerr << "Error: target_block_idx_resp out of range for K/Q recomputation of response token in clValidate." << std::endl;
            } 
            else {
                for (int layer_idx = 0; layer_idx < x; ++layer_idx) {
                    for (int parallel_idx = 0; parallel_idx < y; ++parallel_idx) {
                        auto& attention_head = t[target_block_idx_resp].b[layer_idx][parallel_idx];
                        CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_mQ_cl, CL_TRUE, 0, projection_matrix_bytes_kq, attention_head.MQ.mapped_data));
                        CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_mK_cl, CL_TRUE, 0, projection_matrix_bytes_kq, attention_head.MK.mapped_data));
                        CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_tok_cl, CL_TRUE, 0, embedding_bytes_loc_kq, expected_vec.data()));

                        kq_kernel.setArg(0, d_tok_cl); kq_kernel.setArg(1, d_mQ_cl); kq_kernel.setArg(2, d_Q_cl);
                        kq_kernel.setArg(3, embedding_dim_cl); kq_kernel.setArg(4, mat_heights_cl);
                        CL_CHECK(this->clcontext.queue.enqueueNDRangeKernel(kq_kernel, cl::NullRange, cl::NDRange(1), cl::NullRange));
                        if (attention_head.Q.mapped_data && (host_offset_resp + mat_heights_cl) <= (attention_head.Q.row * attention_head.Q.col)) CL_CHECK(this->clcontext.queue.enqueueReadBuffer(d_Q_cl, CL_TRUE, 0, matheights_bytes_kq, attention_head.Q.mapped_data + host_offset_resp)); else {std::cerr << "Error: Host Q invalid for response K/Q, block " << target_block_idx_resp << " in clValidate." << std::endl;}

                        kq_kernel.setArg(0, d_tok_cl); kq_kernel.setArg(1, d_mK_cl); kq_kernel.setArg(2, d_K_cl);
                        CL_CHECK(this->clcontext.queue.enqueueNDRangeKernel(kq_kernel, cl::NullRange, cl::NDRange(1), cl::NullRange));
                        if (attention_head.K.mapped_data && (host_offset_resp + mat_heights_cl) <= (attention_head.K.row * attention_head.K.col)) CL_CHECK(this->clcontext.queue.enqueueReadBuffer(d_K_cl, CL_TRUE, 0, matheights_bytes_kq, attention_head.K.mapped_data + host_offset_resp)); else {std::cerr << "Error: Host K invalid for response K/Q, block " << target_block_idx_resp << " in clValidate." << std::endl;}
                    }
                }
                this->clcontext.queue.finish();
            }
        }
        // After the first response token prediction, subsequent predictions are based on a single preceding token context shift.
        this->promptCount = 1;

    } // End loop over response tokens
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
    // Zero-fill the tokenEmbed mat
    if (this->tokenEmbed.mapped_data && this->tokenEmbed.row == FULL_CONTEXT && this->tokenEmbed.col == static_cast<size_t>(d)) {
        std::fill_n(this->tokenEmbed.mapped_data, static_cast<size_t>(FULL_CONTEXT) * d, 0.0f);
    } else {
        throw std::runtime_error("clValidate(chat): tokenEmbed (mat) is not properly initialized to FULL_CONTEXT for zero-filling.");
    }

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
            if (this->tokenEmbed.mapped_data && static_cast<size_t>(this->currentTokenCount) < this->tokenEmbed.row && this->tokenEmbed.col == static_cast<size_t>(d)) {
                float* dest_ptr = this->tokenEmbed.mapped_data + (static_cast<size_t>(this->currentTokenCount) * this->d);
                memcpy(dest_ptr, currentPrompt[i].data(), static_cast<size_t>(this->d) * sizeof(float));
            } else {
                 std::cerr << "Warning: Host tokenEmbed (mat) out of bounds or not mapped in clValidate(chat) prompt processing at turn " << turn << ", token " << i << std::endl;
                 break; 
            }
            this->currentTokenCount++;
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
                if (this->tokenEmbed.mapped_data && static_cast<size_t>(currentTokenCount - 1) < this->tokenEmbed.row && this->tokenEmbed.col == static_cast<size_t>(d)) {
                    float* dest_ptr = this->tokenEmbed.mapped_data + (static_cast<size_t>(currentTokenCount - 1) * this->d);
                    memcpy(dest_ptr, expectedEmbedding.data(), static_cast<size_t>(this->d) * sizeof(float));
                } else {
                    std::cerr << "Warning: Host tokenEmbed (mat) out of bounds or not mapped in clValidate(chat) response processing at turn " << turn << ", token " << i << std::endl;
                }
            }

            // Subsequent predictions in the response sequence have promptCount=1
            this->promptCount = 1;
        }
        // Check if the inner loop broke due to full context
        if (this->currentTokenCount >= FULL_CONTEXT) {
            break; // Stop processing further turns if context became full during response
        }

    } // End loop over chat turns
}


#endif // USE_OPENCL