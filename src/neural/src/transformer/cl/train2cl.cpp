#ifdef USE_OPENCL
#include "include/transformer.hpp"
#include <maths.hpp>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <string>
#include <cmath>


/**
 * @brief (OpenCL) Train the transformer for prompt and response.
 * @param prompt Prompt token embeddings (on host).
 * @param response Response token embeddings (on host).
 * @param rString Tokens of the response (on host).
 */
void transformer::clTrain(std::vector<std::vector<float>>& prompt, std::vector<std::vector<float>>& response, 
        std::vector<std::string>& rString)
{
    // --- Basic validation ---
    if (prompt.empty()) {
        throw std::runtime_error("clTrain(prompt-response): Initial prompt cannot be empty.");
    }
    // Warning for large prompts, but allow up to CONTEXT_WIN
    if (prompt.size() > CONTEXT_WIN) {
        std::cerr << "Warning: Prompt size (" << prompt.size() << ") exceeds context window (" << CONTEXT_WIN << "). Ensure this is intended." << std::endl;
    }
    if (response.empty() || response.size() != rString.size()) {
        throw std::runtime_error("clTrain(prompt-response): Response embeddings/strings mismatch or empty.");
    }
    if ((!prompt.empty() && prompt[0].size() != static_cast<size_t>(d)) || (!response.empty() && response[0].size() != static_cast<size_t>(d))) {
        throw std::runtime_error("clTrain(prompt-response): Embedding dimension mismatch.");
    }
    if (this->currentTokenCount + prompt.size() + response.size() > FULL_CONTEXT) {
        throw std::runtime_error("clTrain(prompt-response): Adding prompt and response exceeds FULL_CONTEXT limit.");
    }

    cl::Buffer d_tokenEmbed, d_embeddings, d_expected_response_token;
    float current_error = 0.0f;
    float prev_Error = 0.0f;
    int initial_token_count = this->currentTokenCount; // Store initial count
    int initial_epochs = this->epochs;
    int resCount = 0;

    cl_int cl_err; // For OpenCL error codes
    std::cout << "Current Token Count: " << this->currentTokenCount << std::endl;
    std::cout << "Prompt Size: " << prompt.size() << std::endl;
    std::cout << "Response Size: " << response.size() << std::endl; 

    // For K/Q computation
    cl::Buffer d_Q_cl, d_K_cl, d_mQ_cl, d_mK_cl, d_tok_cl;
    int embedding_dim_cl = EMBEDDING;
    int mat_heights_cl = MATHEIGHTS;
    cl::Kernel kq_kernel;

    if (mat_heights_cl > 0) {
        try {
            kq_kernel = cl::Kernel(this->clcontext.program, "kernelCompute_single_kq_vector", &cl_err); CL_CHECK(cl_err);
        } 
        catch (const std::runtime_error& e) { // Catch runtime errors from CL_CHECK
            std::cerr << "OpenCL Error creating kernel 'kernelCompute_single_kq_vector': " << e.what() << std::endl;
            this->epochs = initial_epochs;
            throw;
        }
    }

    try {
        size_t totalTokenEmbedFloats = static_cast<size_t>(m) * CONTEXT_WIN * d;
        size_t tokenEmbedBytes = totalTokenEmbedFloats * sizeof(float);
        size_t embeddingsBytes = static_cast<size_t>(vocabsize) * d * sizeof(float);
        size_t singleTokenBytes = static_cast<size_t>(d) * sizeof(float);
        size_t outputBytes = singleTokenBytes; // Size of h_otok_buffer
        size_t indexBytes = sizeof(int);       // Size for the result index

        // Create buffers
        std::cout << "Here for tokenEMbed and expected token buffer" << std::endl;
        d_tokenEmbed = cl::Buffer(this->clcontext.context, CL_MEM_READ_WRITE, tokenEmbedBytes, nullptr, &cl_err); CL_CHECK(cl_err);
        d_expected_response_token = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY, singleTokenBytes, nullptr, &cl_err); CL_CHECK(cl_err);

        size_t matheights_bytes = static_cast<size_t>(mat_heights_cl) * sizeof(float);
        size_t embedding_bytes_loc = static_cast<size_t>(embedding_dim_cl) * sizeof(float);
        size_t projection_matrix_bytes = static_cast<size_t>(mat_heights_cl) * embedding_dim_cl * sizeof(float);

        std::cout << "Here for k and q buffers" << std::endl;
        d_Q_cl = cl::Buffer(this->clcontext.context, CL_MEM_WRITE_ONLY, matheights_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        d_K_cl = cl::Buffer(this->clcontext.context, CL_MEM_WRITE_ONLY, matheights_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        d_mQ_cl = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY, projection_matrix_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        d_mK_cl = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY, projection_matrix_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        d_tok_cl = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY, embedding_bytes_loc, nullptr, &cl_err); CL_CHECK(cl_err);
        std::cout << "bufferes done" << std::endl;
        // Prepare initial context buffer content (existing context)
        std::vector<float> flat_host_tokenEmbed;
        flat_host_tokenEmbed.reserve(totalTokenEmbedFloats);
        if (this->tokenEmbed.mapped_data && this->tokenEmbed.row >= static_cast<size_t>(this->currentTokenCount) && this->tokenEmbed.col == static_cast<size_t>(d)) {
            for (int tk = 0; tk < this->currentTokenCount; ++tk) { // Copy existing context
                float* row_ptr = this->tokenEmbed.mapped_data + (static_cast<size_t>(tk) * this->d);
                flat_host_tokenEmbed.insert(flat_host_tokenEmbed.end(), row_ptr, row_ptr + this->d);
            }
        }
        else if (this->currentTokenCount > 0) {
            std::cerr << "Warning: Host tokenEmbed (mat) not properly initialized or too small for currentTokenCount ("
                      << this->currentTokenCount << ", mat_rows: " << this->tokenEmbed.row
                      << ") in clTrain(prompt-response) setup." << std::endl;
            for (int tk = 0; tk < this->currentTokenCount; ++tk) {
                std::vector<float> padding(d, 0.0f);
                flat_host_tokenEmbed.insert(flat_host_tokenEmbed.end(), padding.begin(), padding.end());
            }
        }
        // Pad the rest (up to where the prompt will start)
        flat_host_tokenEmbed.resize(static_cast<size_t>(this->currentTokenCount) * d, 0.0f);

        // Write existing context to device
        if (!flat_host_tokenEmbed.empty()) {
            CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_tokenEmbed, CL_TRUE, 0, flat_host_tokenEmbed.size() * sizeof(float), flat_host_tokenEmbed.data()));
        }

        // Flatten and copy embeddings table
        std::vector<float> flat_embeddings = ::flatten(this->embeddings); // Assuming global or from basic.hpp
        d_embeddings = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, embeddingsBytes, flat_embeddings.data(), &cl_err); CL_CHECK(cl_err);

        // --- Process Prompt (Add to context on Host and Device) ---
        if ((initial_token_count % CONTEXT_WIN) + prompt.size() > CONTEXT_WIN) {
            throw std::runtime_error("clTrain(prompt, response): Prompt exceeds current block capacity when starting.");
        }
        std::cout << "push prompt to token embed buffer" << std::endl;
        for (size_t p = 0; p < prompt.size(); ++p) {
            // Update device buffer
            size_t offset_bytes = static_cast<size_t>(initial_token_count + p) * d * sizeof(float);
            if (offset_bytes + singleTokenBytes > tokenEmbedBytes) {
                throw std::out_of_range("clTrain(prompt-response): Offset exceeds buffer bounds when writing prompt token.");
            }
            CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_tokenEmbed, CL_TRUE, offset_bytes, singleTokenBytes, prompt[p].data()));

            // Update host tracking
            if (this->tokenEmbed.mapped_data && static_cast<size_t>(initial_token_count + p) < this->tokenEmbed.row && this->tokenEmbed.col == static_cast<size_t>(d)) {
                float* dest_ptr = this->tokenEmbed.mapped_data + (static_cast<size_t>(initial_token_count + p) * this->d);
                if (prompt[p].size() == static_cast<size_t>(d)) {
                    memcpy(dest_ptr, prompt[p].data(), singleTokenBytes);
                }
                else {
                    std::cerr << "Error: prompt[p] size mismatch for host tokenEmbed (mat) update in clTrain(prompt-response)." << std::endl;
                }
            }
            else {
                std::cerr << "Error: Host tokenEmbed (mat) not properly initialized or out of bounds for prompt[p] update in clTrain(prompt-response)." << std::endl;
            }
            this->currentTokenCount++;
        }
        std::cout << "current token count after prompt alloted to tokenEmbed: " << currentTokenCount << std::endl;

        // Copy prompt D->D from d_tokenEmbed into d_EV of each head in block 0
        size_t prompt_bytes = prompt.size() * d * sizeof(float);
        size_t prompt_start_offset_bytes = initial_token_count * d * sizeof(float);
        for (int i = 0; i < x; ++i) { // Layers
            for (int j = 0; j < y; ++j) { // Parallels
                cl::Buffer& d_head_ev = t[0].b[i][j].getDeviceEVBuffer(); // Assuming getter exists
                size_t dest_offset_bytes = (initial_token_count % CONTEXT_WIN) * d * sizeof(float); // Correct offset within the block's context window
                CL_CHECK(this->clcontext.queue.enqueueCopyBuffer(d_tokenEmbed, d_head_ev, prompt_start_offset_bytes, dest_offset_bytes, prompt_bytes));
            }
        }
        this->blockCount = (this->currentTokenCount == 0) ? 1 : ((this->currentTokenCount - 1) / CONTEXT_WIN) + 1;
        this->promptCount = prompt.size();

        std::cout << "Here for response training" << std::endl;
        // --- Train for Response ---
        for (size_t i = 0; i < response.size(); ++i) {
            std::vector<float> h_otok_buffer(d, 0.0f);
            if (this->currentTokenCount >= FULL_CONTEXT) {
                std::cerr << "Warning: clTrain(prompt-response) reached FULL_CONTEXT limit ("
                        << this->currentTokenCount << ") during response. Stopping training early at response index " << i << "." << std::endl;
                break;
            }

            std::vector<float> expected_vec(d, 0.0f);
            for(size_t j = 0; j < response[i].size(); ++j) {
                expected_vec[j] = response[i][j];
            }
            std::string& expected_str = rString[i];

            // Copy target token H->D
            CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_expected_response_token, CL_TRUE, 0, singleTokenBytes, expected_vec.data()));

            int effective_context_size = this->currentTokenCount; // Context size *before* adding response[i]
            int current_block_idx = this->blockCount; // Block index based on current context size

            if (current_block_idx <= 0 || current_block_idx > m) {
                throw std::out_of_range("clTrain(prompt-response): Calculated current_block_idx (" 
                                        + std::to_string(current_block_idx) + ") is out of range [1, " 
                                        + std::to_string(m) + "].");
            }

            // Compute K/Q vectors from prompt and/or previous block's EV
            if (mat_heights_cl > 0 && x > 0 && y > 0) {
                int current_processing_block_idx = this->blockCount;
                size_t embedding_bytes_loc = static_cast<size_t>(embedding_dim_cl) * sizeof(float);
                size_t projection_matrix_bytes = static_cast<size_t>(mat_heights_cl) * embedding_dim_cl * sizeof(float);
                size_t matheights_bytes = static_cast<size_t>(mat_heights_cl) * sizeof(float);

                if (current_processing_block_idx == 1) {
                    for (int layer_idx = 0; layer_idx < x; ++layer_idx) {
                        for (int parallel_idx = 0; parallel_idx < y; ++parallel_idx) {
                            auto& attention_head = t[0].b[layer_idx][parallel_idx];
                            CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_mQ_cl, CL_TRUE, 0, projection_matrix_bytes, attention_head.MQ.mapped_data));
                            CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_mK_cl, CL_TRUE, 0, projection_matrix_bytes, attention_head.MK.mapped_data));

                            for (size_t k = 0; k < prompt.size(); ++k) {
                                size_t qk_vector_idx_in_block = (static_cast<size_t>(initial_token_count) % CONTEXT_WIN) + k;
                                if (qk_vector_idx_in_block >= CONTEXT_WIN) {
                                    std::cerr << "Warning: qk_vector_idx_in_block ("<< qk_vector_idx_in_block << ") exceeds CONTEXT_WIN in clTrain K/Q for prompt (block 1)." << std::endl;
                                    continue;
                                }
                                size_t host_qk_offset = qk_vector_idx_in_block * mat_heights_cl;

                                CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_tok_cl, CL_TRUE, 0, embedding_bytes_loc, prompt[k].data()));
                                
                                kq_kernel.setArg(0, d_tok_cl); kq_kernel.setArg(1, d_mQ_cl); kq_kernel.setArg(2, d_Q_cl);
                                kq_kernel.setArg(3, embedding_dim_cl); kq_kernel.setArg(4, mat_heights_cl);
                                CL_CHECK(this->clcontext.queue.enqueueNDRangeKernel(kq_kernel, cl::NullRange, cl::NDRange(1), cl::NullRange));
                                if (attention_head.Q.mapped_data && (host_qk_offset + mat_heights_cl) <= (attention_head.Q.row * attention_head.Q.col)) {
                                    CL_CHECK(this->clcontext.queue.enqueueReadBuffer(d_Q_cl, CL_TRUE, 0, matheights_bytes, attention_head.Q.mapped_data + host_qk_offset));
                                } 
                                else { 
                                    std::cerr << "Error: Host Q buffer invalid or out of bounds for block 0." << std::endl;
                                    break;
                                }

                                kq_kernel.setArg(0, d_tok_cl); kq_kernel.setArg(1, d_mK_cl); kq_kernel.setArg(2, d_K_cl);
                                CL_CHECK(this->clcontext.queue.enqueueNDRangeKernel(kq_kernel, cl::NullRange, cl::NDRange(1), cl::NullRange));
                                if (attention_head.K.mapped_data && (host_qk_offset + mat_heights_cl) <= (attention_head.K.row * attention_head.K.col)) {
                                    CL_CHECK(this->clcontext.queue.enqueueReadBuffer(d_K_cl, CL_TRUE, 0, matheights_bytes, attention_head.K.mapped_data + host_qk_offset));
                                } 
                                else { 
                                    std::cerr << "Error: Host K buffer invalid or out of bounds for block 0." << std::endl;
                                    break;
                                }
                            }
                        }
                    }
                    this->clcontext.queue.finish();
                }
                else if (current_processing_block_idx > 1) {
                    for (int layer_idx = 0; layer_idx < x; ++layer_idx) {
                        for (int parallel_idx = 0; parallel_idx < y; ++parallel_idx) {
                            auto& current_block_attention = t[current_processing_block_idx - 1].b[layer_idx][parallel_idx];
                            auto& prev_block_attention = t[current_processing_block_idx - 2].b[layer_idx][parallel_idx];

                            CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_mQ_cl, CL_TRUE, 0, projection_matrix_bytes, current_block_attention.MQ.mapped_data));
                            CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_mK_cl, CL_TRUE, 0, projection_matrix_bytes, current_block_attention.MK.mapped_data));

                            // Keys from current prompt tokens for the current block
                            for (size_t k = 0; k < prompt.size(); ++k) {
                                size_t qk_vector_idx_in_block = (static_cast<size_t>(initial_token_count) % CONTEXT_WIN) + k;
                                if (qk_vector_idx_in_block >= CONTEXT_WIN) { std::cerr << "Warning: qk_vector_idx_in_block ("<< qk_vector_idx_in_block << ") exceeds CONTEXT_WIN in clTrain K/Q for prompt (block N)." << std::endl; continue; }
                                size_t host_k_offset = qk_vector_idx_in_block * mat_heights_cl;
                                CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_tok_cl, CL_TRUE, 0, embedding_bytes_loc, prompt[k].data()));
                                kq_kernel.setArg(0, d_tok_cl); kq_kernel.setArg(1, d_mK_cl); kq_kernel.setArg(2, d_K_cl);
                                kq_kernel.setArg(3, embedding_dim_cl); kq_kernel.setArg(4, mat_heights_cl);
                                CL_CHECK(this->clcontext.queue.enqueueNDRangeKernel(kq_kernel, cl::NullRange, cl::NDRange(1), cl::NullRange));
                                if (current_block_attention.K.mapped_data && (host_k_offset + mat_heights_cl) <= (current_block_attention.K.row * current_block_attention.K.col)) {
                                    CL_CHECK(this->clcontext.queue.enqueueReadBuffer(d_K_cl, CL_TRUE, 0, matheights_bytes, current_block_attention.K.mapped_data + host_k_offset));
                                } 
                                else { 
                                    std::cerr << "Error: Host K buffer invalid or out of bounds for block " << current_processing_block_idx << std::endl;
                                }
                            }

                            // Queries from previous block's EV for the current block
                            for (int k_ev = 0; k_ev < CONTEXT_WIN; ++k_ev) {
                                size_t q_vector_idx_in_block = k_ev;
                                size_t host_q_offset = q_vector_idx_in_block * mat_heights_cl;
                                if (!prev_block_attention.EV.mapped_data || (static_cast<size_t>(k_ev) * embedding_dim_cl + embedding_dim_cl) > (prev_block_attention.EV.row * prev_block_attention.EV.col)) {
                                    std::cerr << "Warning: Prev block EV data invalid or out of bounds for index " << k_ev << " in clTrain K/Q." << std::endl; continue;
                                }
                                CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_tok_cl, CL_TRUE, 0, embedding_bytes_loc, prev_block_attention.EV.mapped_data + static_cast<size_t>(k_ev) * embedding_dim_cl));
                                kq_kernel.setArg(0, d_tok_cl); kq_kernel.setArg(1, d_mQ_cl); kq_kernel.setArg(2, d_Q_cl);
                                CL_CHECK(this->clcontext.queue.enqueueNDRangeKernel(kq_kernel, cl::NullRange, cl::NDRange(1), cl::NullRange));
                                if (current_block_attention.Q.mapped_data && (host_q_offset + mat_heights_cl) <= (current_block_attention.Q.row * current_block_attention.Q.col)) {
                                    CL_CHECK(this->clcontext.queue.enqueueReadBuffer(d_Q_cl, CL_TRUE, 0, matheights_bytes, current_block_attention.Q.mapped_data + host_q_offset));
                                } 
                                else { 
                                    std::cerr << "Error: Host Q buffer invalid or out of bounds for block " << current_processing_block_idx << std::endl; 
                                }
                            }
                        }
                    }
                    this->clcontext.queue.finish();
                }
            }

            if(resCount > 0) {
                size_t embedding_bytes_loc_recompute = static_cast<size_t>(embedding_dim_cl) * sizeof(float);
                size_t projection_matrix_bytes_recompute = static_cast<size_t>(mat_heights_cl) * embedding_dim_cl * sizeof(float);
                size_t matheights_bytes_recompute = static_cast<size_t>(mat_heights_cl) * sizeof(float);
                if (this->blockCount == 1) {
                    for (int layer_idx = 0; layer_idx < x; ++layer_idx) {
                        for (int parallel_idx = 0; parallel_idx < y; ++parallel_idx) {
                            auto& attention_head = t[0].b[layer_idx][parallel_idx];
                            CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_mQ_cl, CL_TRUE, 0, projection_matrix_bytes_recompute, attention_head.MQ.mapped_data));
                            CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_mK_cl, CL_TRUE, 0, projection_matrix_bytes_recompute, attention_head.MK.mapped_data));

                            // Recompute K/Q for response tokens processed so far in *this* training iteration (up to response[i-1])
                            for (size_t k_resp = 0; k_resp < i; ++k_resp) { // `i` is the current response token index
                                size_t qk_vec_idx = (static_cast<size_t>(initial_token_count + prompt.size() + k_resp) % CONTEXT_WIN);
                                if (qk_vec_idx >= CONTEXT_WIN) continue;
                                size_t host_offset = qk_vec_idx * mat_heights_cl;

                                CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_tok_cl, CL_TRUE, 0, embedding_bytes_loc_recompute, response[k_resp].data()));

                                kq_kernel.setArg(0, d_tok_cl); kq_kernel.setArg(1, d_mQ_cl); kq_kernel.setArg(2, d_Q_cl);
                                CL_CHECK(this->clcontext.queue.enqueueNDRangeKernel(kq_kernel, cl::NullRange, cl::NDRange(1), cl::NullRange));
                                if (attention_head.Q.mapped_data) 
                                    CL_CHECK(this->clcontext.queue.enqueueReadBuffer(d_Q_cl, CL_TRUE, 0, matheights_bytes_recompute, attention_head.Q.mapped_data + host_offset));
                                
                                kq_kernel.setArg(0, d_tok_cl); kq_kernel.setArg(1, d_mK_cl); kq_kernel.setArg(2, d_K_cl);
                                CL_CHECK(this->clcontext.queue.enqueueNDRangeKernel(kq_kernel, cl::NullRange, cl::NDRange(1), cl::NullRange));
                                if (attention_head.K.mapped_data) 
                                    CL_CHECK(this->clcontext.queue.enqueueReadBuffer(d_K_cl, CL_TRUE, 0, matheights_bytes_recompute, attention_head.K.mapped_data + host_offset));
                            }
                        }
                    }
                    this->clcontext.queue.finish();
                }
                else if (this->blockCount > 1) {
                    for (int layer_idx = 0; layer_idx < x; ++layer_idx) {
                        for (int parallel_idx = 0; parallel_idx < y; ++parallel_idx) {
                            auto& current_block_attention = t[this->blockCount - 1].b[layer_idx][parallel_idx];
                            auto& prev_block_attention = t[this->blockCount - 2].b[layer_idx][parallel_idx];
                            CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_mQ_cl, CL_TRUE, 0, projection_matrix_bytes_recompute, current_block_attention.MQ.mapped_data));
                            CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_mK_cl, CL_TRUE, 0, projection_matrix_bytes_recompute, current_block_attention.MK.mapped_data));

                            // K/Q for response tokens processed so far in *this* training iteration (up to response[i-1]) that fall into this block
                            for (size_t k_resp = 0; k_resp < i; ++k_resp) {
                                size_t global_resp_token_idx = static_cast<size_t>(initial_token_count) + prompt.size() + k_resp;
                                if (global_resp_token_idx < static_cast<size_t>((this->blockCount - 1) * CONTEXT_WIN)) continue;
                                if (global_resp_token_idx >= static_cast<size_t>(this->blockCount * CONTEXT_WIN)) break;

                                size_t qk_vec_idx_in_block = global_resp_token_idx % CONTEXT_WIN;
                                size_t host_offset = qk_vec_idx_in_block * mat_heights_cl;

                                CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_tok_cl, CL_TRUE, 0, embedding_bytes_loc_recompute, response[k_resp].data()));
                                kq_kernel.setArg(0, d_tok_cl); kq_kernel.setArg(1, d_mQ_cl); kq_kernel.setArg(2, d_Q_cl); CL_CHECK(this->clcontext.queue.enqueueNDRangeKernel(kq_kernel, cl::NullRange, cl::NDRange(1), cl::NullRange));
                                if (current_block_attention.Q.mapped_data) 
                                    CL_CHECK(this->clcontext.queue.enqueueReadBuffer(d_Q_cl, CL_TRUE, 0, matheights_bytes_recompute, current_block_attention.Q.mapped_data + host_offset));
                                
                                kq_kernel.setArg(0, d_tok_cl); kq_kernel.setArg(1, d_mK_cl); kq_kernel.setArg(2, d_K_cl); CL_CHECK(this->clcontext.queue.enqueueNDRangeKernel(kq_kernel, cl::NullRange, cl::NDRange(1), cl::NullRange));
                                if (current_block_attention.K.mapped_data) 
                                    CL_CHECK(this->clcontext.queue.enqueueReadBuffer(d_K_cl, CL_TRUE, 0, matheights_bytes_recompute, current_block_attention.K.mapped_data + host_offset));
                            }
                        }
                    }
                    this->clcontext.queue.finish();
                }
            }

            // --- Training Loop for response token i ---
            int j = 0; // Epoch counter
            // current_error = 1.0f; // Error is calculated fresh each iteration
            while (j < this->epochs) { // Changed loop condition to match CUDA (j < epochs)
                // --- Forward Pass ---
                // Pass the promptCount relevant for the *current* block/context state
                int current_prompt_count_in_block = effective_context_size % CONTEXT_WIN;
                if (current_prompt_count_in_block == 0 && effective_context_size > 0) current_prompt_count_in_block = CONTEXT_WIN;
                clForward(current_block_idx, effective_context_size, current_prompt_count_in_block);
                std::cout << "current block: " << current_block_idx << " & current token count: " << currentTokenCount << std::endl;
                
                unsigned long long host_indexForToken = -1;
                h_otok_buffer = this->otok;
                if (h_otok_buffer.size() != static_cast<size_t>(d)) {
                    throw std::runtime_error("clTrain(prompt-response): this->otok from clForward has incorrect size. Expected " + std::to_string(d) + ", got " + std::to_string(this->otok.size()));
                }
                // use kernelComputePrediction for output prediction
                {
                    cl::Buffer d_otok_buffer, d_result_index_buffer;
                    try {
                        size_t otok_bytes = h_otok_buffer.size() * sizeof(float);
                        d_otok_buffer = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, otok_bytes, h_otok_buffer.data(), &cl_err); CL_CHECK(cl_err);
                        d_result_index_buffer = cl::Buffer(this->clcontext.context, CL_MEM_WRITE_ONLY, sizeof(cl_int), nullptr, &cl_err); CL_CHECK(cl_err);

                        cl::Kernel kernel = this->clcontext.kernels.at("kernelComputePrediction");

                        CL_CHECK(kernel.setArg(0, d_otok_buffer));
                        CL_CHECK(kernel.setArg(1, d_embeddings));
                        CL_CHECK(kernel.setArg(2, d_result_index_buffer));
                        CL_CHECK(kernel.setArg(3, static_cast<cl_int>(this->d)));
                        CL_CHECK(kernel.setArg(4, static_cast<cl_int>(this->vocabsize)));

                        cl::NDRange global(1);
                        cl::NDRange local(1);
                        CL_CHECK(this->clcontext.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local));

                        int result_idx = -1;
                        CL_CHECK(this->clcontext.queue.enqueueReadBuffer(d_result_index_buffer, CL_TRUE, 0, sizeof(cl_int), &result_idx));
                        host_indexForToken = result_idx;
                        this->indexForToken = result_idx; // Also update the class member
                    } 
                    catch (const std::exception& e) {
                        std::cerr << "Error during kernelComputePrediction in clTrain: " << e.what() << std::endl;
                        throw;
                    }
                }

                // calculate error
                current_error = crossEntropy(h_otok_buffer, expected_vec);
                // Unified logging and convergence check
                std::string predicted_token_str = (host_indexForToken >= 0 && host_indexForToken < static_cast<unsigned long long>(tokens.size()))
                                                  ? tokens[host_indexForToken]
                                                  : "INVALID_INDEX";

                std::cout << "Computed token is -> " << predicted_token_str << " (index: " << host_indexForToken << ") | with BCE error " << current_error << " | MAE Error " << MAE(h_otok_buffer, response[i]) << std::endl;

                if (predicted_token_str == expected_str && predicted_token_str != "INVALID_INDEX") {
                    size_t offset_bytes = static_cast<size_t>(effective_context_size) * d * sizeof(float);
                    if (offset_bytes + outputBytes > tokenEmbedBytes) {
                        throw std::out_of_range("clTrain(prompt-response): Offset exceeds buffer bounds when writing converged response token.");
                    }
                    std::cout << "indexForToken: " << this->indexForToken << " | host_indexForToken: " << host_indexForToken << " | Epoch Count: " << epochCount << " | Current Token Count " << currentTokenCount << std::endl;
                    CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_tokenEmbed, CL_TRUE, offset_bytes, outputBytes, expected_vec.data())); // Write expected_vec (target EH)
                    if(predicted_token_str == "</s>"){
                        // learning = learnby;
                        totalLearning += learning;
                        prev_Error = current_error;
                        j++;
                        std::cout << "--------------->>>>>>>>>>>>> To next LINE >>>>>>>>>>>>>>>>-------------" << std::endl;
                        break;
                    }
                    else {
                        // learning = learnby;
                        totalLearning += learning;
                        prev_Error = current_error;
                        j++;
                        std::cout << "---------------------------- To next token ------------->>>>>>>>>>>>>>>" << std::endl;
                        break;
                    }
                }
                else if (j == this->epochs - 1) {
                    if (predicted_token_str != expected_str) {
                        std::cout << "Increasing Epoch Count by 10 '-'" << std::endl;
                        this->epochs += 10;
                    }
                }

                if(j > 0) {
                    if(current_error <= prev_Error) {
                        if(j <= 6)   
                            learning *= 1.05;
                        else if (j % 6 == 0)
                            learning *= (1 + (j/6)*0.05);
                    }
                    else {
                        if(j <= 6)   
                            learning *= 0.95;
                        else if (j % 6 == 0)
                            learning *= (1 - (j/6)*0.05);
                    }
                }
                prev_Error = current_error;
                // --- Backward Pass ---
                clBackward(expected_vec, current_block_idx, clip_norm);

                int current_processing_block_idx_recompute = this->blockCount; // Or current_block_idx if more appropriate
                size_t embedding_bytes_loc_recompute = static_cast<size_t>(embedding_dim_cl) * sizeof(float);
                size_t projection_matrix_bytes_recompute = static_cast<size_t>(mat_heights_cl) * embedding_dim_cl * sizeof(float);
                size_t matheights_bytes_recompute = static_cast<size_t>(mat_heights_cl) * sizeof(float);

                // std::cout << "current block: " << current_block_idx << " & current token count: " << currentTokenCount << std::endl; // Original print
                j++;
            }

            resCount += 1;
            // --- Update Host State ---
            this->trainCount++;
            this->epochCount += j;
            this->error += current_error;

            // Add the *converged/expected* token to the host context tracking
            if (this->tokenEmbed.mapped_data && static_cast<size_t>(effective_context_size) < this->tokenEmbed.row && this->tokenEmbed.col == static_cast<size_t>(d)) {
                float* dest_ptr = this->tokenEmbed.mapped_data + (static_cast<size_t>(effective_context_size) * this->d);
                if (expected_vec.size() == static_cast<size_t>(d)) { // Ensure expected_vec is correctly sized
                    memcpy(dest_ptr, expected_vec.data(), singleTokenBytes); // Use expected_vec (target EH)
                } 
                else {
                    std::cerr << "Error: expected_vec size mismatch for host tokenEmbed (mat) update in clTrain(prompt-response)." << std::endl;
                }
            } 
            else {
                std::cerr << "Error: Host tokenEmbed (mat) not properly initialized or out of bounds for h_otok_buffer update in clTrain(prompt-response)." << std::endl;
            }
            this->currentTokenCount++;

            // Update blockCount and promptCount for the *next* iteration
            this->blockCount = (this->currentTokenCount == 0) ? 1 : ((this->currentTokenCount -1) / CONTEXT_WIN) + 1;
        }
    }
    catch (const std::exception& e) { // Catch other standard exceptions like std::out_of_range
        std::cerr << "Standard Exception in clTrain(prompt-response): " << e.what() << std::endl; // This will catch std::out_of_range from kernels.at()
        this->epochs = initial_epochs;
        throw;
    }
}


/**
 * @brief (OpenCL) Train transformers for continuous chats.
 *        Mirrors the logic of transformer::cuTrain(..., prompts, ..., responses, ...).
 * @param prompts All prompts (on host).
 * @param responses Token embeddings of all responses to the prompts (on host).
 * @param rString Tokens of the responses (on host).
 */
void transformer::clTrain(std::vector<std::vector<std::vector<float>>>& prompts, std::vector<std::vector<std::vector<float>>>& responses,
                        std::vector<std::vector<std::string>>& rString)
{
    // --- Validation ---
    if (prompts.size() != responses.size() || responses.size() != rString.size()) {
        throw std::runtime_error("clTrain(chat): Mismatch in number of prompts (" + std::to_string(prompts.size())
                                 + "), responses (" + std::to_string(responses.size())
                                 + "), and response strings (" + std::to_string(rString.size()) + ").");
    }

    // Calculate total tokens required to check against FULL_CONTEXT
    size_t total_tokens_to_add = 0;
    for (size_t i = 0; i < prompts.size(); ++i) {
        if (prompts[i].empty()) {
            throw std::runtime_error("clTrain(chat): Chat training requires non-empty prompts at index " + std::to_string(i));
        }
        if (responses[i].empty() || responses[i].size() != rString[i].size()) {
            throw std::runtime_error("clTrain(chat): Response embeddings/strings mismatch or empty at index " + std::to_string(i));
        }
        // Add dimension checks for embeddings
        if (!prompts[i].empty() && prompts[i][0].size() != static_cast<size_t>(d)) {
            throw std::runtime_error("clTrain(chat): Prompt embedding dimension mismatch at index " + std::to_string(i));
        }
         if (!responses[i].empty() && responses[i][0].size() != static_cast<size_t>(d)) {
            throw std::runtime_error("clTrain(chat): Response embedding dimension mismatch at index " + std::to_string(i));
        }
        total_tokens_to_add += prompts[i].size() + responses[i].size();
    }

    if (this->currentTokenCount + total_tokens_to_add > FULL_CONTEXT) {
        throw std::runtime_error("clTrain(chat): Total tokens required (" 
                                + std::to_string(this->currentTokenCount + total_tokens_to_add)
                                + ") exceeds FULL_CONTEXT (" + std::to_string(FULL_CONTEXT) + ").");
    }

    // --- Train for each chat turn ---
    for (size_t i = 0; i < prompts.size(); ++i) {
        try {
            clTrain(prompts[i], responses[i], rString[i]);
        } 
        catch (const std::exception& e) {
            std::cerr << "Error during chat training turn " << i << ": " << e.what() << std::endl;
            throw;
        }
    }
}

#endif // USE_OPENCL
