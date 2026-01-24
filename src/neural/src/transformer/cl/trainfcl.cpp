#ifdef USE_CL
#include "include/transformer.hpp"
#include <maths.hpp>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <string>
#include <cmath> // For std::abs, std::max


/**
 * @brief (OpenCL) Train the transformer for next token prediction (single token training).
 *        Mirrors the logic of transformer::cuTrain(..., std::vector<float>& expected, ...).
 * @param promptCount Number of tokens in the prompt.
 * @param currentTokenCount Number of tokens in the full context *before* this training step.
 * @param blockCount Current block index (1-based) in the full context.
 * @param expected Expected token embedding (on host).
 * @param expString Expected token string (on host, used for convergence check).
 */
void transformer::clTrain(std::vector<float>& expected, std::string& expString)
{
    // --- Basic Validation ---
    if (expected.size() != static_cast<size_t>(d)) {
        throw std::runtime_error("clTrain(single): Expected vector size mismatch. Expected " + std::to_string(d) + ", got " + std::to_string(expected.size()));
    }
    if (currentTokenCount >= FULL_CONTEXT) {
        throw std::runtime_error("clTrain(single): Cannot train, FULL_CONTEXT limit reached (" + std::to_string(currentTokenCount) + ").");
    }
    if (blockCount <= 0 || blockCount > m) {
         throw std::out_of_range("clTrain(single): Initial blockCount (" + std::to_string(blockCount) + ") is out of range [1, " + std::to_string(m) + "].");
    }

    cl::Buffer d_tokenEmbed, d_embeddings, d_expected;
    this->indexForToken = -1; // Host copy of the predicted index
    float current_error = 1.0f; // Initialize error high
    int initial_epochs = this->epochs; // Store initial epochs setting

    cl_int cl_err; // For OpenCL error codes
    std::cout << "Current Token Count Before Training: " << currentTokenCount << std::endl;

    try {
        float prev_Error = 0.0f;        // previous iterations error
        // --- Device Buffer Allocation & H->D Transfer ---
        // Calculate the total size needed for all blocks' context windows
        size_t totalTokenEmbedFloats = static_cast<size_t>(m) * CONTEXT_WIN * d;
        size_t tokenEmbedBytes = totalTokenEmbedFloats * sizeof(float);
        size_t embeddingsBytes = static_cast<size_t>(vocabsize) * d * sizeof(float);
        size_t expectedBytes = expected.size() * sizeof(float);
        size_t singleTokenBytes = static_cast<size_t>(d) * sizeof(float);
        size_t outputBytes = singleTokenBytes; // Size of h_otok_buffer
        size_t indexBytes = sizeof(int);       // Size for the result index

        // Create buffers
        d_tokenEmbed = cl::Buffer(this->clcontext.context, CL_MEM_READ_WRITE, tokenEmbedBytes, nullptr, &cl_err); CL_CHECK(cl_err);
        d_expected = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY, expectedBytes, nullptr, &cl_err); CL_CHECK(cl_err);

        // Flatten and copy current tokenEmbed context to device
        std::vector<float> flat_host_tokenEmbed;
        flat_host_tokenEmbed.reserve(totalTokenEmbedFloats); // Reserve space for all blocks
        if (this->tokenEmbed.mapped_data && this->tokenEmbed.row >= static_cast<size_t>(this->currentTokenCount) && this->tokenEmbed.col == static_cast<size_t>(d)) {
            for (int tk = 0; tk < this->currentTokenCount; ++tk) {
                float* row_ptr = this->tokenEmbed.mapped_data + (static_cast<size_t>(tk) * this->d);
                flat_host_tokenEmbed.insert(flat_host_tokenEmbed.end(), row_ptr, row_ptr + this->d);
            }
        }
        else if (this->currentTokenCount > 0) {
            // Handle error or inconsistent state: tokenEmbed not ready or too small
            std::cerr << "Warning: Host tokenEmbed (mat) not properly initialized or too small for currentTokenCount ("
                      << this->currentTokenCount << ", mat_rows: " << this->tokenEmbed.row
                      << ") in clTrain(single) setup." << std::endl;
            // Fill with padding if proceeding
            for (int tk = 0; tk < this->currentTokenCount; ++tk) {
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

        // Copy expected vector
        CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_expected, CL_TRUE, 0, expectedBytes, expected.data()));

        // Determine effective context size and block index for the *current* state
        int effective_context_size = currentTokenCount;
        // Recalculate current_block_idx based on effective_context_size just to be sure
        int current_block_idx = (effective_context_size == 0) ? 1 : ((effective_context_size) / CONTEXT_WIN) + 1;
        if (current_block_idx <= 0 || current_block_idx > m) {
            throw std::out_of_range("clTrain(single): Calculated current_block_idx (" + std::to_string(current_block_idx) + ") is out of range [1, " + std::to_string(m) + "].");
        }
    
        // --- Training Loop ---
        prev_Error = 0.0f;
        int j = 0; // Epoch counter for this token
        this->sequence1Count = 1; // Initialize sequence1Count for single token training

        while (j <= this->epochs) // Use member variable epochs
        {
            // --- Forward Pass ---
            // KdotQ calculation (if needed, depends on clForward implementation)
            if (this->inTraining) {
                // clParallelKdotQs(...); // Placeholder - Needs OpenCL adaptation based on data flow
            }
            clForward(current_block_idx, effective_context_size, this->sequence1Count); // Operates on device data implicitly

            // --- Get EH output from the relevant block ---
            std::cout << "clTrain(single): Getting output for block " << current_block_idx << " with effective context size " << effective_context_size << std::endl;
            std::vector<float> h_otok_buffer(d, 0.0f); // Initialize with zeros
            h_otok_buffer = this->otok;
            if (h_otok_buffer.size() != static_cast<size_t>(d)) {
                throw std::runtime_error("clTrain(single): this->otok from clForward has incorrect size. Expected " + std::to_string(d) + ", got " + std::to_string(h_otok_buffer.size()));
            }
            current_error = binaryCrossEntropy(h_otok_buffer, expected); // Host-side error calculation
            std::cout << "Training Error: " << current_error << " for token: " << expString << std::endl;
            bool converged = (current_error < 0.01);
            if (!converged && this->indexForToken >= 0 && this->indexForToken < static_cast<int>(tokens.size())) {
                converged = (tokens[this->indexForToken] == expString);
            }

            if (converged) {
                size_t offset_bytes = static_cast<size_t>(effective_context_size) * d * sizeof(float);
                if (offset_bytes + singleTokenBytes > tokenEmbedBytes) {
                    throw std::out_of_range("clTrain(single): Calculated offset for writing converged token exceeds buffer bounds.");
                }
                CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_tokenEmbed, CL_TRUE, offset_bytes, singleTokenBytes, expected.data())); // Write expected token
                break;
            }

            if (current_error >= 0.01 && j == this->epochs) {
                bool predicted_matches = (this->indexForToken >= 0 && this->indexForToken < static_cast<int>(tokens.size()) && tokens[this->indexForToken] == expString);
                if (!predicted_matches) {
                    this->epochs += 10;
                }
                else {
                    size_t offset_bytes = static_cast<size_t>(effective_context_size) * d * sizeof(float);
                    if (offset_bytes + singleTokenBytes > tokenEmbedBytes) {
                        throw std::out_of_range("clTrain(single): Calculated offset for writing converged token exceeds buffer bounds.");
                    }
                    CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_tokenEmbed, CL_TRUE, offset_bytes, singleTokenBytes, expected.data()));
                    break;
                }
            }

            // update learning rate starting from second epoch and specific conditions
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

            clBackward(expected, current_block_idx);
            totalLearning += learning;
            prev_Error = current_error;
            j++;
        }

        this->trainCount++;
        this->epochCount += j; // Add epochs spent on this token
        this->error += current_error; // Accumulate final error
        int previousTokenCount = this->currentTokenCount;
        this->currentTokenCount += 1; // Increment context size *after* training for the token

        // Update host tokenEmbed with the *expected* value
        if (this->tokenEmbed.mapped_data && static_cast<size_t>(previousTokenCount) < this->tokenEmbed.row && this->tokenEmbed.col == static_cast<size_t>(d)) {
            float* dest_ptr = this->tokenEmbed.mapped_data + (static_cast<size_t>(previousTokenCount) * this->d);
            if (expected.size() == static_cast<size_t>(d)) {
                memcpy(dest_ptr, expected.data(), singleTokenBytes); // singleTokenBytes = d * sizeof(float)
            }
            else {
                std::cerr << "Error: Expected vector size mismatch for host tokenEmbed (mat) update in clTrain(single)." << std::endl;
            }
        }
        else {
            std::cerr << "Error: Host tokenEmbed (mat) not properly initialized or out of bounds (row: " << previousTokenCount << ", mat_rows: " << this->tokenEmbed.row << ") for update in clTrain(single)." << std::endl;
        }

        this->blockCount = (this->currentTokenCount == 0) ? 1 : ((this->currentTokenCount) / CONTEXT_WIN) + 1;
        // Update sequence1Count for the next step (relative to the current block)
        this->sequence1Count = this->currentTokenCount % CONTEXT_WIN;
        if (this->sequence1Count == 0 && this->currentTokenCount > 0)
            this->sequence1Count = CONTEXT_WIN;
    }
    catch (const std::runtime_error& e) {
        std::cerr << "Standard Exception in clTrain(single): " << e.what() << std::endl;
        this->epochs = initial_epochs;
        throw; // Re-throw
    }
}

/**
 * @brief (OpenCL) Train the transformer for next token prediction (single token training) using
 *          embedding-deEmbedding. Contextualisation for embeddings and deEmbeddings.
 * @param promptCount Number of tokens in the prompt.
 * @param currentTokenCount Number of tokens in the full context *before* this training step.
 * @param blockCount Current block index (1-based) in the full context.
 * @param expected Expected token embedding (on host).
 * @param expString Expected token string (on host, used for convergence check).
 */
void transformer::clTrainContext(std::vector<float>& expected, std::string& expString)
{
    // --- Basic Validation ---
    if (expected.size() != static_cast<size_t>(d)) {
        throw std::runtime_error("clTrainCONTEXT(single): Expected vector size mismatch. Expected " + std::to_string(d) + ", got " + std::to_string(expected.size()));
    }
    if (currentTokenCount >= FULL_CONTEXT) {
        throw std::runtime_error("clTrainCONTEXT(single): Cannot train, FULL_CONTEXT limit reached (" + std::to_string(currentTokenCount) + ").");
    }
    if (blockCount <= 0 || blockCount > m) {
         throw std::out_of_range("clTrainCONTEXT(single): Initial blockCount (" + std::to_string(blockCount) + ") is out of range [1, " + std::to_string(m) + "].");
    }

    cl::Buffer d_tokenEmbed, d_embeddings, d_expected;
    cl::Buffer d_Q_cl, d_K_cl, d_mQ_cl, d_mK_cl, d_tok_cl;
    this->indexForToken = -1;
    float current_error = 1.0f;
    int initial_epochs = this->epochs;
    cl_int cl_err; // For OpenCL error codes
    cl::Kernel kq_kernel; // Key and query calculation
    std::cout << "Current Token Count Before Training: " << currentTokenCount << std::endl;

    if (CONTEXT_WIN > 0) {
        try {
            kq_kernel = cl::Kernel(this->clcontext.program, "kernelCompute_single_kq_vector", &cl_err);
            CL_CHECK(cl_err);
        } 
        catch (const std::runtime_error& e) { // Catch runtime errors from CL_CHECK
            std::cerr << "OpenCL Error creating kernel 'kernelCompute_single_kq_vector': " << e.what() << std::endl;
            this->epochs = initial_epochs;
            throw;
        }
    }

    try {
        float prev_Error = 0.0f;
        float learnby = learning;
        int initial_token_count = this->currentTokenCount;
        // --- Device Buffer Allocation & H->D Transfer ---
        // Calculate the total size needed for all blocks' context windows
        size_t totalTokenEmbedFloats = static_cast<size_t>(m) * CONTEXT_WIN * d;
        size_t tokenEmbedBytes = totalTokenEmbedFloats * sizeof(float);
        size_t embeddingsBytes = static_cast<size_t>(vocabsize) * d * sizeof(float);
        size_t singleTokenBytes = static_cast<size_t>(d) * sizeof(float);
        size_t outputBytes = singleTokenBytes; // Size of h_otok_buffer
        size_t indexBytes = sizeof(int);       // Size for the result index
        size_t matheights_bytes = static_cast<size_t>(CONTEXT_WIN) * sizeof(float);
        size_t projection_matrix_bytes = static_cast<size_t>(CONTEXT_WIN) * EMBEDDING * sizeof(float);

        // Create buffers
        d_tokenEmbed = cl::Buffer(this->clcontext.context, CL_MEM_READ_WRITE, tokenEmbedBytes, nullptr, &cl_err); CL_CHECK(cl_err);
        d_expected = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY, singleTokenBytes, nullptr, &cl_err); CL_CHECK(cl_err);
        cl::Buffer d_deEmbeddings;

        // Flatten and copy current tokenEmbed context to device
        std::vector<float> flat_host_tokenEmbed;
        flat_host_tokenEmbed.reserve(totalTokenEmbedFloats); // Reserve space for all blocks
        if (this->tokenEmbed.mapped_data && this->tokenEmbed.row >= static_cast<size_t>(this->currentTokenCount) && this->tokenEmbed.col == static_cast<size_t>(d)) {
            for (int tk = 0; tk < this->currentTokenCount; ++tk) {
                float* row_ptr = this->tokenEmbed.mapped_data + (static_cast<size_t>(tk) * this->d);
                flat_host_tokenEmbed.insert(flat_host_tokenEmbed.end(), row_ptr, row_ptr + this->d);
            }
        }
        else if (this->currentTokenCount > 0) {
            // Handle error or inconsistent state: tokenEmbed not ready or too small
            std::cerr << "Warning: Host tokenEmbed (mat) not properly initialized or too small for currentTokenCount ("
                      << this->currentTokenCount << ", mat_rows: " << this->tokenEmbed.row
                      << ") in clTrainCONTEXT(single) setup." << std::endl;
            // Fill with padding if proceeding
            for (int tk = 0; tk < this->currentTokenCount; ++tk) {
                std::vector<float> padding(d, 0.0f);
                flat_host_tokenEmbed.insert(flat_host_tokenEmbed.end(), padding.begin(), padding.end());
            }
        }
        // Pad the rest of the buffer with zeros
        flat_host_tokenEmbed.resize(totalTokenEmbedFloats, 0.0f);
        CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_tokenEmbed, CL_TRUE, 0, tokenEmbedBytes, flat_host_tokenEmbed.data()));

        // Flatten and copy embeddings table
        std::vector<float> flat_deEmbeddings = ::flatten(this->deEmbeddings);

        // Copy expected vector
        CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_expected, CL_TRUE, 0, singleTokenBytes, expected.data()));

        // Determine effective context size and block index for the *current* state
        int effective_context_size = currentTokenCount;
        // Recalculate current_block_idx based on effective_context_size just to be sure
        int current_block_idx = (effective_context_size == 0) ? 1 : ((effective_context_size) / CONTEXT_WIN) + 1;
        if (current_block_idx <= 0 || current_block_idx > m) {
            throw std::out_of_range("clTrainCONTEXT(single): Calculated current_block_idx (" + std::to_string(current_block_idx) + ") is out of range [1, " + std::to_string(m) + "].");
        }
    
        // --- Training Loop ---
        prev_Error = 0.0f;
        int j = 0; // Epoch counter for this token
        this->sequence1Count = 1; // Initialize sequence1Count

        while (j <= this->epochs) // Use member variable epochs
        {
            int host_indexForToken = -1;
            // keys and queries for heads of respective blocks
            if(current_block_idx == 1) {
                for (int layer_idx = 0; layer_idx < x; ++layer_idx) {
                    for (int parallel_idx = 0; parallel_idx < y; ++parallel_idx) {
                        auto& attention_head = blocks[0].b[layer_idx][parallel_idx];
                        CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_mQ_cl, CL_TRUE, 0, projection_matrix_bytes, attention_head.MQ.mapped_data));
                        CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_mK_cl, CL_TRUE, 0, projection_matrix_bytes, attention_head.MK.mapped_data));

                        for (size_t k = 0; k < initial_token_count; ++k) {
                            size_t qk_vector_idx_in_block = (static_cast<size_t>(initial_token_count) % CONTEXT_WIN) + k;
                            if (qk_vector_idx_in_block >= CONTEXT_WIN) {
                                std::cerr << "Warning: qk_vector_idx_in_block ("<< qk_vector_idx_in_block << ") exceeds CONTEXT_WIN in clTrainCONTEXT K/Q for prompt (block 1)." << std::endl;
                                continue;
                            }
                            size_t host_qk_offset = qk_vector_idx_in_block * CONTEXT_WIN;
                            CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_tok_cl, CL_TRUE, 0, embeddingsBytes, tokenEmbed(k).data()));

                            kq_kernel.setArg(0, d_tok_cl); kq_kernel.setArg(1, d_mQ_cl); kq_kernel.setArg(2, d_Q_cl);
                            kq_kernel.setArg(3, EMBEDDING); kq_kernel.setArg(4, CONTEXT_WIN);
                            CL_CHECK(this->clcontext.queue.enqueueNDRangeKernel(kq_kernel, cl::NullRange, cl::NDRange(1), cl::NullRange));
                            if (attention_head.Q.mapped_data && (host_qk_offset + CONTEXT_WIN) <= (attention_head.Q.row * attention_head.Q.col)) {
                                CL_CHECK(this->clcontext.queue.enqueueReadBuffer(d_Q_cl, CL_TRUE, 0, matheights_bytes, attention_head.Q.mapped_data + host_qk_offset));
                            } 
                            else { 
                                std::cerr << "Error: Host Q buffer invalid or out of bounds for block 0." << std::endl;
                                break;
                            }

                            kq_kernel.setArg(0, d_tok_cl); kq_kernel.setArg(1, d_mK_cl); kq_kernel.setArg(2, d_K_cl);
                            CL_CHECK(this->clcontext.queue.enqueueNDRangeKernel(kq_kernel, cl::NullRange, cl::NDRange(1), cl::NullRange));
                            if (attention_head.K.mapped_data && (host_qk_offset + CONTEXT_WIN) <= (attention_head.K.row * attention_head.K.col)) {
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
            else {
                int current_processing_block_idx = this->blockCount;
                for (int layer_idx = 0; layer_idx < x; ++layer_idx) {
                    for (int parallel_idx = 0; parallel_idx < y; ++parallel_idx) {
                        auto& current_block_attention = blocks[current_processing_block_idx - 1].b[layer_idx][parallel_idx];
                        auto& prev_block_attention = blocks[current_processing_block_idx - 2].b[layer_idx][parallel_idx];

                        CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_mQ_cl, CL_TRUE, 0, projection_matrix_bytes, current_block_attention.MQ.mapped_data));
                        CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_mK_cl, CL_TRUE, 0, projection_matrix_bytes, current_block_attention.MK.mapped_data));

                        // Keys from current prompt tokens for the current block
                        for (size_t k = 0; k < initial_token_count; ++k) {
                            size_t qk_vector_idx_in_block = (static_cast<size_t>(initial_token_count) % CONTEXT_WIN) + k;
                            if (qk_vector_idx_in_block >= CONTEXT_WIN) { std::cerr << "Warning: qk_vector_idx_in_block ("<< qk_vector_idx_in_block << ") exceeds CONTEXT_WIN in clTrainCONTEXT K/Q for prompt (block N)." << std::endl; continue; }
                            size_t host_k_offset = qk_vector_idx_in_block * CONTEXT_WIN;
                            CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_tok_cl, CL_TRUE, 0, embeddingsBytes, tokenEmbed(k).data()));
                            kq_kernel.setArg(0, d_tok_cl); kq_kernel.setArg(1, d_mK_cl); kq_kernel.setArg(2, d_K_cl);
                            kq_kernel.setArg(3, EMBEDDING); kq_kernel.setArg(4, CONTEXT_WIN);
                            CL_CHECK(this->clcontext.queue.enqueueNDRangeKernel(kq_kernel, cl::NullRange, cl::NDRange(1), cl::NullRange));
                            if (current_block_attention.K.mapped_data && (host_k_offset + CONTEXT_WIN) <= (current_block_attention.K.row * current_block_attention.K.col)) {
                                CL_CHECK(this->clcontext.queue.enqueueReadBuffer(d_K_cl, CL_TRUE, 0, matheights_bytes, current_block_attention.K.mapped_data + host_k_offset));
                            }
                            else { 
                                std::cerr << "Error: Host K buffer invalid or out of bounds for block " << current_processing_block_idx << std::endl;
                            }
                        }

                        // Queries from previous block's EV for the current block
                        for (int k_ev = 0; k_ev < CONTEXT_WIN; ++k_ev) {
                            size_t q_vector_idx_in_block = k_ev;
                            size_t host_q_offset = q_vector_idx_in_block * CONTEXT_WIN;
                            if (!prev_block_attention.EV.mapped_data || (static_cast<size_t>(k_ev) * EMBEDDING + EMBEDDING) > (prev_block_attention.EV.row * prev_block_attention.EV.col)) {
                                std::cerr << "Warning: Prev block EV data invalid or out of bounds for index " << k_ev << " in clTrainCONTEXT K/Q." << std::endl; continue;
                            }
                            CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_tok_cl, CL_TRUE, 0, embeddingsBytes, prev_block_attention.EV.mapped_data + static_cast<size_t>(k_ev) * EMBEDDING));
                            kq_kernel.setArg(0, d_tok_cl); kq_kernel.setArg(1, d_mQ_cl); kq_kernel.setArg(2, d_Q_cl);
                            CL_CHECK(this->clcontext.queue.enqueueNDRangeKernel(kq_kernel, cl::NullRange, cl::NDRange(1), cl::NullRange));
                            if (current_block_attention.Q.mapped_data && (host_q_offset + CONTEXT_WIN) <= (current_block_attention.Q.row * current_block_attention.Q.col)) {
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

            // --- Forward Pass ---
            clForward(current_block_idx, effective_context_size, this->sequence1Count);
            std::vector<float> h_otok_buffer(d*x, 0.0f);
            if (h_otok_buffer.size() != static_cast<size_t>(d*x)) {
                throw std::runtime_error("clTrainCONTEXT(sentence): this->otok from clForward has incorrect size.");
            }

            // use kernelComputePredictionWithScores for output prediction
            {
                // store EH of all PAs to single vector
                if (y > 0) {
                    for (int j = 0; j < x; ++j) {
                        const std::vector<float>& eh_vector = blocks[blockCount-1].b[j][y - 1].EH;
                        if (eh_vector.size() != static_cast<size_t>(d)) {
                            throw std::runtime_error("clForward: EH vector size mismatch during host accumulation for head ["
                                                    + std::to_string(j) + "][" + std::to_string(y - 1) + "]. Expected "
                                                    + std::to_string(d) + ", got " + std::to_string(eh_vector.size()));
                        }
                        for (int k = 0; k < EMBEDDING; ++k) {
                            h_otok_buffer[j*EMBEDDING + k] += eh_vector[k];
                        }
                    }
                    // set max for infinity and zero for nan in otok and h_otok_buffer
                    // h_otok_buffer in function
                    for(size_t k_dim = 0; k_dim < static_cast<size_t>(this->d * this->x); k_dim++) {
                        if (std::isnan(h_otok_buffer[k_dim])) { h_otok_buffer[k_dim] = 0.0001f; }
                        else if (std::isinf(h_otok_buffer[k_dim])) { h_otok_buffer[k_dim] = std::copysign((std::numeric_limits<float>::max)(), h_otok_buffer[k_dim]); }
                    }
                    // otok in transformer
                    if (this->otok.size() != static_cast<size_t>(this->d)) {
                        throw std::runtime_error("clTrain(sentence): this->otok from clForward has incorrect size.");
                    }
                    for(size_t k_dim = 0; k_dim < static_cast<size_t>(this->d); k_dim++) { // Loop `d` times
                        if (std::isnan(otok[k_dim])) { otok[k_dim] = 0.0001f; }
                        else if (std::isinf(this->otok[k_dim])) { otok[k_dim] = std::copysign((std::numeric_limits<float>::max)(), otok[k_dim]); }
                    }
                }
                else {
                    std::cerr << "Warning: clTrainr called with y=0 columns. Cannot accumulate EH." << std::endl;
                }

                cl::Buffer d_otok_buffer, d_predictions, d_result_index_buffer;
                try {
                    d_deEmbeddings = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, embeddingsBytes, flat_deEmbeddings.data(), &cl_err); CL_CHECK(cl_err);
                    size_t h_otok_bytes = this->d * this->x * sizeof(float);
                    d_otok_buffer = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, h_otok_bytes, h_otok_buffer.data(), &cl_err); CL_CHECK(cl_err);
                    d_predictions = cl::Buffer(this->clcontext.context, CL_MEM_WRITE_ONLY, vocabsize*sizeof(float), nullptr, &cl_err); CL_CHECK(cl_err);
                    d_result_index_buffer = cl::Buffer(this->clcontext.context, CL_MEM_WRITE_ONLY, sizeof(cl_int), nullptr, &cl_err); CL_CHECK(cl_err);
                    cl::Kernel kernel = this->clcontext.kernels.at("kernelComputePredictionWithScores");
                    CL_CHECK(kernel.setArg(0, d_otok_buffer));
                    CL_CHECK(kernel.setArg(1, d_deEmbeddings));
                    CL_CHECK(kernel.setArg(2, d_predictions));
                    CL_CHECK(kernel.setArg(3, d_result_index_buffer));
                    CL_CHECK(kernel.setArg(4, static_cast<cl_int>(this->d * this->x)));
                    CL_CHECK(kernel.setArg(5, static_cast<cl_int>(this->vocabsize)));
                    cl::NDRange global(1);
                    cl::NDRange local(1);
                    CL_CHECK(this->clcontext.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local));
                    CL_CHECK(this->clcontext.queue.enqueueReadBuffer(d_result_index_buffer, CL_TRUE, 0, sizeof(cl_int), &host_indexForToken));
                    CL_CHECK(this->clcontext.queue.enqueueReadBuffer(d_predictions, CL_TRUE, 0, vocabsize * sizeof(float), pred.data()));
                    this->indexForToken = host_indexForToken;
                }
                catch (const std::exception& e) {
                    std::cerr << "Error during kernelComputePredictionWithScores in clTrainCONTEXT: " << e.what() << std::endl;
                    throw;
                }
            }

            // --- Get EH output from the relevant block ---
            std::cout << "clTrainCONTEXT(single): Getting output for block " << current_block_idx << " with effective context size " << effective_context_size << std::endl;
            if (h_otok_buffer.size() != static_cast<size_t>(d * x)) {
                throw std::runtime_error("clTrainCONTEXT(single): this->otok from clForward has incorrect size. Expected " + std::to_string(d) + ", got " + std::to_string(h_otok_buffer.size()));
            }
            current_error = crossEntropy(oneHotEncode, pred);
            std::string predicted_token_str = (host_indexForToken >= 0 && host_indexForToken < static_cast<int>(tokens.size()))
                                                ? tokens[host_indexForToken] : "INVALID_INDEX";
            std::cout << "Computed token is -> " << predicted_token_str
                        << " (index: " << host_indexForToken
                        << ") | BCE LOSS " << current_error
                        << " | Epoch Count: " << j 
                        << " | Learning Rate: " << this->learning << std::endl;

            if (this->tokens[host_indexForToken] == expString && this->tokens[host_indexForToken] != "INVALID_INDEX") {
                std::cout << "--------------------- To next token ------------->>>>>>>>>>>>>>>>>" << std::endl;
                totalLearning += learning;
                break;
            }
            else if (j == this->epochs - 1) {
                if (this->tokens[host_indexForToken] != expString) {
                    std::cout << "Increasing Epoch Count by 10 '-'" << std::endl;
                    this->epochs += 10;
                }
            }

            // update learning rate starting from second epoch and specific conditions
            if(j > 0) {
                if(current_error <= prev_Error) {
                    if(j <= 6)   
                        learning *= 1.1;
                    else if (j % 6 == 0)
                        learning *= (1.05 + (j/6)*0.05);
                }
                else {
                    if(j <= 6)   
                        learning *= 0.95;
                    else if (j % 6 == 0)
                        learning *= (0.95 - (j/6)*0.01);
                }
            }

            // modify the de-embeddings and get gradients for backprop
            std::vector<float> gradEH(this->d * this->x, 0.0f);
            clUpdateDeEmbeddings(deEmbeddings, pred, oneHotEncode, this->learning, gradEH);
            // get expected target for backprop
            std::vector<float> exp(this->d * this->x, 0.0f);
            exp = deEmbeddings(host_indexForToken);
            std::vector<std::vector<float>> targets_for_heads(x, std::vector<float>(EMBEDDING, 0.0f));
            for(int head_idx = 0; head_idx < x; ++head_idx) {
                for(int eidx = 0; eidx < EMBEDDING; ++eidx) {
                    targets_for_heads[head_idx][eidx] = exp[(head_idx * EMBEDDING) + eidx];
                }
            }
            // backpropagate through the attention units
            clBackwardContext(targets_for_heads, current_block_idx);
            // update embeddings which are in use
            clUpdateEmbeddings(embeddings, blocks[blockCount-1].gradToken, this->learning, vocabsize);
            clUpdateEmbeddings(tokenEmbed, blocks[blockCount-1].gradToken, this->learning, effective_context_size);

            totalLearning += learning;
            prev_Error = current_error;
            j++;
        }
        learning = learnby;

        this->trainCount++;
        this->epochCount += j;
        this->error += current_error;
        int previousTokenCount = this->currentTokenCount;
        this->currentTokenCount += 1;

        this->blockCount = (this->currentTokenCount == 0) ? 1 : ((this->currentTokenCount) / CONTEXT_WIN) + 1;
        // Update sequence1Count for the next step (relative to the current block)
        this->sequence1Count = this->currentTokenCount % CONTEXT_WIN;
        if (this->sequence1Count == 0 && this->currentTokenCount > 0)
            this->sequence1Count = CONTEXT_WIN;
    }
    catch (const std::runtime_error& e) {
        std::cerr << "Standard Exception in rclTrainCONTEXT(single): " << e.what() << std::endl;
        this->epochs = initial_epochs;
        throw;
    }
}

#endif