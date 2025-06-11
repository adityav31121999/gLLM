
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
    cl_int cl_err;
    int host_indexForToken = -1;
    float current_error = 0.0f;
    float current_mse = 0.0f;

    // For K/Q computation (not strictly needed for single token test if context is pre-set,
    // but good for consistency if this function were to be part of a larger sequence
    // where K/Q for the *added* token needs to be computed for subsequent steps.
    // However, clTest(single) typically *doesn't* add to a persistent K/Q state for future calls itself.)

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
            if (this->tokenEmbed.mapped_data &&
                static_cast<size_t>(tk) < this->tokenEmbed.row &&
                this->tokenEmbed.col == static_cast<size_t>(d))
            {
                // Calculate the pointer to the start of the tk-th row
                float* row_ptr = this->tokenEmbed.mapped_data + (static_cast<size_t>(tk) * this->tokenEmbed.col);
                // Insert the 'd' elements of this row into flat_host_tokenEmbed
                flat_host_tokenEmbed.insert(flat_host_tokenEmbed.end(), row_ptr, row_ptr + d);
            }
            else {
                std::cerr << "Warning: Inconsistent tokenEmbed (mat) data or bounds at index " << tk
                          << " (mat_rows: " << (this->tokenEmbed.mapped_data ? std::to_string(this->tokenEmbed.row) : "N/A")
                          << ", mat_cols: " << (this->tokenEmbed.mapped_data ? std::to_string(this->tokenEmbed.col) : "N/A")
                          << ", expected_cols: " << d << ") during clTest(single) setup. Padding." << std::endl;
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
             h_otok_buffer = otok;
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
            // cl_err is already declared at the top of the clTest function
            d_output = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, outputBytes, h_otok_buffer.data(), &cl_err); CL_CHECK(cl_err);
            d_result_index = cl::Buffer(this->clcontext.context, CL_MEM_WRITE_ONLY, indexBytes, nullptr, &cl_err); CL_CHECK(cl_err);
            try { // Keep try-catch for kernel name lookup, or ensure kernel name is always valid
                cl::Kernel kernel = this->clcontext.kernels.at("compute_prediction");
                CL_CHECK(kernel.setArg(0, d_output));
                CL_CHECK(kernel.setArg(1, d_embeddings));
                CL_CHECK(kernel.setArg(2, static_cast<cl_int>(d)));
                CL_CHECK(kernel.setArg(3, static_cast<cl_int>(this->vocabsize)));
                CL_CHECK(kernel.setArg(4, d_result_index));
                cl::NDRange global(1); cl::NDRange local(1);
                CL_CHECK(this->clcontext.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local));
                CL_CHECK(this->clcontext.queue.enqueueReadBuffer(d_result_index, CL_TRUE, 0, indexBytes, &result_index_val));
                host_indexForToken = result_index_val;
            } catch (const std::out_of_range& oor) { // Catch if kernel "compute_prediction" is not found
                std::cerr << "Error: Kernel 'compute_prediction' not found. " << oor.what() << std::endl;
                host_indexForToken = -1; 
                throw; // Re-throw as this is a critical setup issue
            } // CL_CHECK will throw std::runtime_error for OpenCL API errors
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

    } catch (const std::exception& e) { // Catches std::runtime_error from CL_CHECK and other std exceptions
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
    this->currentTokenCount = 0; // Start from the beginning of the context for this sentence
    this->blockCount = 1;        // Start with the first block
    // Zero-fill the tokenEmbed mat
    if (this->tokenEmbed.mapped_data && this->tokenEmbed.row == FULL_CONTEXT && this->tokenEmbed.col == static_cast<size_t>(d)) {
        std::fill_n(this->tokenEmbed.mapped_data, static_cast<size_t>(FULL_CONTEXT) * d, 0.0f);
    } else {
        throw std::runtime_error("clTest(sentence): tokenEmbed (mat) is not properly initialized to FULL_CONTEXT for zero-filling.");
    }
    
    // Add first token to context
    if (!sentence.empty()) {
        setRow(this->tokenEmbed, 0, sentence[0]);
        this->currentTokenCount = 1;
        this->promptCount = 1;
    }
    else {
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
             if (this->tokenEmbed.mapped_data && static_cast<size_t>(currentTokenCount - 1) < this->tokenEmbed.row && this->tokenEmbed.col == static_cast<size_t>(d)) {
                float* dest_ptr = this->tokenEmbed.mapped_data + (static_cast<size_t>(currentTokenCount - 1) * this->d);
                memcpy(dest_ptr, expected_vec.data(), static_cast<size_t>(this->d) * sizeof(float));
             } else {
                  std::cerr << "Warning: Host tokenEmbed size mismatch in clTest(sentence) at index " << (currentTokenCount - 1) << std::endl;
                  // For mat, resize is not an option. This is an error condition.
             }
        }
        this->promptCount = 1; // Subsequent steps

    } // End loop over sentence
}


/**
 * @brief (OpenCL) Test the transformer for prompt and response.
 * @param prompt Prompt token embeddings (on host).
 * @param response Response token embeddings (on host).
 * @param rString Tokens of the response (on host).
 */
void transformer::clTest(std::vector<std::vector<float>>& prompt, std::vector<std::vector<float>>& response, 
    std::vector<std::string>& rString)
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
    // Zero-fill the tokenEmbed mat
    if (this->tokenEmbed.mapped_data && this->tokenEmbed.row == FULL_CONTEXT && this->tokenEmbed.col == static_cast<size_t>(d)) {
        std::fill_n(this->tokenEmbed.mapped_data, static_cast<size_t>(FULL_CONTEXT) * d, 0.0f);
    }
    else {
        throw std::runtime_error("clTest(prompt-response): tokenEmbed (mat) is not properly initialized to FULL_CONTEXT for zero-filling.");
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
            std::cerr << "OpenCL Error creating kernel 'kernelCompute_single_kq_vector' in clTest(prompt-response): " << e.what() << std::endl;
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

    // --- Process Prompt (Host context) ---
    for (size_t p = 0; p < prompt.size(); ++p) {
        if (this->currentTokenCount >= FULL_CONTEXT) { std::cerr << "Warning: Context full processing prompt in clTest(prompt-response)." << std::endl; break; }
        if (this->tokenEmbed.mapped_data && static_cast<size_t>(this->currentTokenCount) < this->tokenEmbed.row && this->tokenEmbed.col == static_cast<size_t>(d)) {
            float* dest_ptr = this->tokenEmbed.mapped_data + (static_cast<size_t>(this->currentTokenCount) * this->d);
            memcpy(dest_ptr, prompt[p].data(), static_cast<size_t>(this->d) * sizeof(float));
        }
        else {
            std::cerr << "Warning: Host tokenEmbed out of bounds or not mapped in clTest(prompt-response) prompt processing at index " << p << std::endl;
            break;
        }
        this->currentTokenCount++;
    }
    if (this->tokenEmbed.row != FULL_CONTEXT) { // Check if mat is correctly sized
        throw std::runtime_error("clTest(prompt-response): tokenEmbed (mat) does not have FULL_CONTEXT rows after prompt processing.");
    }

    // Set initial state after prompt
    this->blockCount = (this->currentTokenCount == 0) ? 1 : ((this->currentTokenCount - 1) / CONTEXT_WIN) + 1;
    this->promptCount = this->currentTokenCount;

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
                        } else { std::cerr << "Error: Host K buffer invalid or out of bounds for prompt K, block " << b_idx << std::endl; }
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
                                std::cerr << "Error: Host Q buffer invalid or out of bounds for prompt Q, block 0" << std::endl; 
                            }
                        }
                    }
                    else { // Q from previous block's EV for b_idx > 0
                        auto& prev_block_attention_for_ev = t[b_idx - 1].b[layer_idx][parallel_idx];
                        for (int k_ev = 0; k_ev < CONTEXT_WIN; ++k_ev) {
                            size_t host_q_offset = static_cast<size_t>(k_ev) * mat_heights_cl;
                            if (!prev_block_attention_for_ev.EV.mapped_data || (static_cast<size_t>(k_ev) * embedding_dim_cl + embedding_dim_cl) > (prev_block_attention_for_ev.EV.row * prev_block_attention_for_ev.EV.col)) {
                                std::cerr << "Warning: Prev block EV data invalid or out of bounds for index " << k_ev << " in clTest K/Q for prompt." << std::endl; continue;
                            }
                            CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_tok_cl, CL_TRUE, 0, embedding_bytes_loc_kq, prev_block_attention_for_ev.EV.mapped_data + static_cast<size_t>(k_ev) * embedding_dim_cl));
                            kq_kernel.setArg(0, d_tok_cl); kq_kernel.setArg(1, d_mQ_cl); kq_kernel.setArg(2, d_Q_cl);
                            CL_CHECK(this->clcontext.queue.enqueueNDRangeKernel(kq_kernel, cl::NullRange, cl::NDRange(1), cl::NullRange));
                            if (current_block_attention.Q.mapped_data && (host_q_offset + mat_heights_cl) <= (current_block_attention.Q.row * current_block_attention.Q.col)) {
                                CL_CHECK(this->clcontext.queue.enqueueReadBuffer(d_Q_cl, CL_TRUE, 0, matheights_bytes_kq, current_block_attention.Q.mapped_data + host_q_offset));
                            }
                            else { 
                                std::cerr << "Error: Host Q buffer invalid or out of bounds for EV Q, block " << b_idx << std::endl; 
                            }
                        }
                    }
                }
            }
        }
        this->clcontext.queue.finish();
    }

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
             if (this->tokenEmbed.mapped_data && static_cast<size_t>(currentTokenCount - 1) < this->tokenEmbed.row && this->tokenEmbed.col == static_cast<size_t>(d)) {
                float* dest_ptr = this->tokenEmbed.mapped_data + (static_cast<size_t>(currentTokenCount - 1) * this->d);
                memcpy(dest_ptr, expected_vec.data(), static_cast<size_t>(this->d) * sizeof(float));
             } else {
                 std::cerr << "Warning: Host tokenEmbed size mismatch in clTest(prompt-response) at index " << (currentTokenCount - 1) << std::endl;
             }
        }

        // --- K/Q Calculation for the processed response token ---
        if (mat_heights_cl > 0 && x > 0 && y > 0) {
            // currentTokenCount was incremented by clTest(single-token call)
            // The token just added to context is expected_vec (response[i])
            // Its global index in this test run is this->currentTokenCount - 1
            int target_block_idx_resp = (this->currentTokenCount - 1) / CONTEXT_WIN;
            size_t qk_idx_in_block_resp = (static_cast<size_t>(this->currentTokenCount - 1) % CONTEXT_WIN);
            size_t host_offset_resp = qk_idx_in_block_resp * mat_heights_cl;
            
            size_t embedding_bytes_loc_kq = static_cast<size_t>(embedding_dim_cl) * sizeof(float);
            size_t projection_matrix_bytes_kq = static_cast<size_t>(mat_heights_cl) * embedding_dim_cl * sizeof(float);
            size_t matheights_bytes_kq = static_cast<size_t>(mat_heights_cl) * sizeof(float);

            if (target_block_idx_resp < 0 || static_cast<size_t>(target_block_idx_resp) >= t.size()) {
                std::cerr << "Error: target_block_idx_resp out of range for K/Q recomputation of response token in clTest." << std::endl;
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
                        if (attention_head.Q.mapped_data && (host_offset_resp + mat_heights_cl) <= (attention_head.Q.row * attention_head.Q.col)) 
                            CL_CHECK(this->clcontext.queue.enqueueReadBuffer(d_Q_cl, CL_TRUE, 0, matheights_bytes_kq, attention_head.Q.mapped_data + host_offset_resp)); 
                        else {
                            std::cerr << "Error: Host Q invalid for response K/Q, block " << target_block_idx_resp << std::endl;
                        }

                        kq_kernel.setArg(0, d_tok_cl); kq_kernel.setArg(1, d_mK_cl); kq_kernel.setArg(2, d_K_cl);
                        CL_CHECK(this->clcontext.queue.enqueueNDRangeKernel(kq_kernel, cl::NullRange, cl::NDRange(1), cl::NullRange));
                        if (attention_head.K.mapped_data && (host_offset_resp + mat_heights_cl) <= (attention_head.K.row * attention_head.K.col)) 
                            CL_CHECK(this->clcontext.queue.enqueueReadBuffer(d_K_cl, CL_TRUE, 0, matheights_bytes_kq, attention_head.K.mapped_data + host_offset_resp)); 
                        else {
                            std::cerr << "Error: Host K invalid for response K/Q, block " << target_block_idx_resp << std::endl;
                        }
                    }
                }
                this->clcontext.queue.finish();
            }
        }
        this->promptCount = 1; // Subsequent predictions

    } // End loop over response
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
    // Zero-fill the tokenEmbed mat
    if (this->tokenEmbed.mapped_data && this->tokenEmbed.row == FULL_CONTEXT && this->tokenEmbed.col == static_cast<size_t>(d)) {
        std::fill_n(this->tokenEmbed.mapped_data, static_cast<size_t>(FULL_CONTEXT) * d, 0.0f);
    }
    else {
        throw std::runtime_error("clTest(chat): tokenEmbed (mat) is not properly initialized to FULL_CONTEXT for zero-filling.");
    }

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
            if (this->tokenEmbed.mapped_data && static_cast<size_t>(this->currentTokenCount) < this->tokenEmbed.row && this->tokenEmbed.col == static_cast<size_t>(d)) {
                float* dest_ptr = this->tokenEmbed.mapped_data + (static_cast<size_t>(this->currentTokenCount) * this->d);
                memcpy(dest_ptr, currentPrompt[i].data(), static_cast<size_t>(this->d) * sizeof(float));
            }
            else {
                std::cerr << "Warning: Host tokenEmbed out of bounds or not mapped in clTest(chat) prompt processing at turn " << turn << ", token " << i << std::endl;
                break;
            }
            this->currentTokenCount++;
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
                float* dest_ptr = this->tokenEmbed.mapped_data + (static_cast<size_t>(currentTokenCount - 1) * this->d);
                memcpy(dest_ptr, expectedEmbedding.data(), static_cast<size_t>(this->d) * sizeof(float));
            }
            this->promptCount = 1; // Subsequent predictions
        }
        if (this->currentTokenCount >= FULL_CONTEXT) break; // Stop if context full during response

    } // End loop over chat turns
}

#endif
