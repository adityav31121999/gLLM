#ifdef USE_OPENCL
#include "include/transformer.hpp"
#include <maths.hpp>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <string>
#include <cmath>
#include <limits>

/**
 * @brief (OpenCL) Train the transformer for next token prediction (single token training) using
 *          embedding-deEmbedding. Contextualisation for embeddings and deEmbeddings.
 * @param promptCount Number of tokens in the prompt.
 * @param currentTokenCount Number of tokens in the full context *before* this training step.
 * @param blockCount Current block index (1-based) in the full context.
 * @param expected Expected token embedding (on host).
 * @param expString Expected token string (on host, used for convergence check).
 */
void transformer::clTrainContext(int& promptCount, int& currentTokenCount, int& blockCount, std::vector<float>& expected, std::string& expString)
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

    if (MATHEIGHTS > 0) {
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
        size_t matheights_bytes = static_cast<size_t>(MATHEIGHTS) * sizeof(float);
        size_t projection_matrix_bytes = static_cast<size_t>(MATHEIGHTS) * EMBEDDING * sizeof(float);

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
        while (j <= this->epochs) // Use member variable epochs
        {
            int host_indexForToken = -1;
            // keys and queries for heads of respective blocks
            if(current_block_idx == 1) {
                for (int layer_idx = 0; layer_idx < x; ++layer_idx) {
                    for (int parallel_idx = 0; parallel_idx < y; ++parallel_idx) {
                        auto& attention_head = t[0].b[layer_idx][parallel_idx];
                        CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_mQ_cl, CL_TRUE, 0, projection_matrix_bytes, attention_head.MQ.mapped_data));
                        CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_mK_cl, CL_TRUE, 0, projection_matrix_bytes, attention_head.MK.mapped_data));

                        for (size_t k = 0; k < initial_token_count; ++k) {
                            size_t qk_vector_idx_in_block = (static_cast<size_t>(initial_token_count) % CONTEXT_WIN) + k;
                            if (qk_vector_idx_in_block >= CONTEXT_WIN) {
                                std::cerr << "Warning: qk_vector_idx_in_block ("<< qk_vector_idx_in_block << ") exceeds CONTEXT_WIN in clTrainCONTEXT K/Q for prompt (block 1)." << std::endl;
                                continue;
                            }
                            size_t host_qk_offset = qk_vector_idx_in_block * MATHEIGHTS;
                            CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_tok_cl, CL_TRUE, 0, embeddingsBytes, tokenEmbed(k).data()));

                            kq_kernel.setArg(0, d_tok_cl); kq_kernel.setArg(1, d_mQ_cl); kq_kernel.setArg(2, d_Q_cl);
                            kq_kernel.setArg(3, EMBEDDING); kq_kernel.setArg(4, MATHEIGHTS);
                            CL_CHECK(this->clcontext.queue.enqueueNDRangeKernel(kq_kernel, cl::NullRange, cl::NDRange(1), cl::NullRange));
                            if (attention_head.Q.mapped_data && (host_qk_offset + MATHEIGHTS) <= (attention_head.Q.row * attention_head.Q.col)) {
                                CL_CHECK(this->clcontext.queue.enqueueReadBuffer(d_Q_cl, CL_TRUE, 0, matheights_bytes, attention_head.Q.mapped_data + host_qk_offset));
                            } 
                            else { 
                                std::cerr << "Error: Host Q buffer invalid or out of bounds for block 0." << std::endl;
                                break;
                            }

                            kq_kernel.setArg(0, d_tok_cl); kq_kernel.setArg(1, d_mK_cl); kq_kernel.setArg(2, d_K_cl);
                            CL_CHECK(this->clcontext.queue.enqueueNDRangeKernel(kq_kernel, cl::NullRange, cl::NDRange(1), cl::NullRange));
                            if (attention_head.K.mapped_data && (host_qk_offset + MATHEIGHTS) <= (attention_head.K.row * attention_head.K.col)) {
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
                        auto& current_block_attention = t[current_processing_block_idx - 1].b[layer_idx][parallel_idx];
                        auto& prev_block_attention = t[current_processing_block_idx - 2].b[layer_idx][parallel_idx];

                        CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_mQ_cl, CL_TRUE, 0, projection_matrix_bytes, current_block_attention.MQ.mapped_data));
                        CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_mK_cl, CL_TRUE, 0, projection_matrix_bytes, current_block_attention.MK.mapped_data));

                        // Keys from current prompt tokens for the current block
                        for (size_t k = 0; k < initial_token_count; ++k) {
                            size_t qk_vector_idx_in_block = (static_cast<size_t>(initial_token_count) % CONTEXT_WIN) + k;
                            if (qk_vector_idx_in_block >= CONTEXT_WIN) { std::cerr << "Warning: qk_vector_idx_in_block ("<< qk_vector_idx_in_block << ") exceeds CONTEXT_WIN in clTrainCONTEXT K/Q for prompt (block N)." << std::endl; continue; }
                            size_t host_k_offset = qk_vector_idx_in_block * MATHEIGHTS;
                            CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_tok_cl, CL_TRUE, 0, embeddingsBytes, tokenEmbed(k).data()));
                            kq_kernel.setArg(0, d_tok_cl); kq_kernel.setArg(1, d_mK_cl); kq_kernel.setArg(2, d_K_cl);
                            kq_kernel.setArg(3, EMBEDDING); kq_kernel.setArg(4, MATHEIGHTS);
                            CL_CHECK(this->clcontext.queue.enqueueNDRangeKernel(kq_kernel, cl::NullRange, cl::NDRange(1), cl::NullRange));
                            if (current_block_attention.K.mapped_data && (host_k_offset + MATHEIGHTS) <= (current_block_attention.K.row * current_block_attention.K.col)) {
                                CL_CHECK(this->clcontext.queue.enqueueReadBuffer(d_K_cl, CL_TRUE, 0, matheights_bytes, current_block_attention.K.mapped_data + host_k_offset));
                            }
                            else { 
                                std::cerr << "Error: Host K buffer invalid or out of bounds for block " << current_processing_block_idx << std::endl;
                            }
                        }

                        // Queries from previous block's EV for the current block
                        for (int k_ev = 0; k_ev < CONTEXT_WIN; ++k_ev) {
                            size_t q_vector_idx_in_block = k_ev;
                            size_t host_q_offset = q_vector_idx_in_block * MATHEIGHTS;
                            if (!prev_block_attention.EV.mapped_data || (static_cast<size_t>(k_ev) * EMBEDDING + EMBEDDING) > (prev_block_attention.EV.row * prev_block_attention.EV.col)) {
                                std::cerr << "Warning: Prev block EV data invalid or out of bounds for index " << k_ev << " in clTrainCONTEXT K/Q." << std::endl; continue;
                            }
                            CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_tok_cl, CL_TRUE, 0, embeddingsBytes, prev_block_attention.EV.mapped_data + static_cast<size_t>(k_ev) * EMBEDDING));
                            kq_kernel.setArg(0, d_tok_cl); kq_kernel.setArg(1, d_mQ_cl); kq_kernel.setArg(2, d_Q_cl);
                            CL_CHECK(this->clcontext.queue.enqueueNDRangeKernel(kq_kernel, cl::NullRange, cl::NDRange(1), cl::NullRange));
                            if (current_block_attention.Q.mapped_data && (host_q_offset + MATHEIGHTS) <= (current_block_attention.Q.row * current_block_attention.Q.col)) {
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
            clForward(current_block_idx, effective_context_size, promptCount);
            std::vector<float> h_otok_buffer(d*x, 0.0f);
            if (h_otok_buffer.size() != static_cast<size_t>(d*x)) {
                throw std::runtime_error("clTrainCONTEXT(sentence): this->otok from clForward has incorrect size.");
            }

            // use kernelComputePredictionWithScores for output prediction
            {
                // store EH of all PAs to single vector
                if (y > 0) {
                    for (int j = 0; j < x; ++j) {
                        const std::vector<float>& eh_vector = t[blockCount-1].b[j][y - 1].EH;
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
            clUpdateDeEmbeddings(deEmbeddings, otok, pred, oneHotEncode, host_indexForToken, this->learning, lambda_L1, lambda_L2, gradEH);
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
            std::vector<float> tok = tokenEmbed(currentTokenCount-1);
            clUpdateEmbeddings(tok, this->learning, lambda_L1, lambda_L2, t[blockCount-1].gradToken);
            setRow(this->tokenEmbed, currentTokenCount-1, tok);

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
        // Update promptCount for the next step (relative to the current block)
        this->promptCount = this->currentTokenCount % CONTEXT_WIN;
        if (this->promptCount == 0 && this->currentTokenCount > 0)
            this->promptCount = CONTEXT_WIN;
    }
    catch (const std::runtime_error& e) {
        std::cerr << "Standard Exception in rclTrainCONTEXT(single): " << e.what() << std::endl;
        this->epochs = initial_epochs;
        throw;
    }
}


/**
 * @brief (OpenCL) Train the transformer on sentences and paragraphsusing embedding-deEmbedding.
 *          Contextualisation for embeddings and deEmbeddings.
 * @param sentence Token embeddings of the sentence (on host).
 * @param rString Sentence tokens (on host).
 */
void transformer::clTrainContext(std::vector<std::vector<float>>& sentence, std::vector<std::string>& rString)
{
    // --- Basic validation ---
    if (sentence.size() > FULL_CONTEXT) {
        throw std::runtime_error("clTrainCONTEXT(sentence): Sentence size (" + std::to_string(sentence.size()) + ") exceeds FULL_CONTEXT (" + std::to_string(FULL_CONTEXT) + ").");
    }
    if (sentence.empty() || sentence.size() != rString.size()) {
        std::cout << "sentence.size(): " << sentence.size() << ", rString.size(): " << rString.size() << std::endl;
        throw std::runtime_error("clTrainCONTEXT(sentence): Sentence embeddings/strings mismatch or empty.");
    }
    if (!sentence.empty() && sentence[0].size() != static_cast<size_t>(d)) {
        throw std::runtime_error("clTrainCONTEXT(sentence): Sentence embedding dimension mismatch. Expected " + std::to_string(d) + ", got " + std::to_string(sentence[0].size()));
    }

    cl_int cl_err;          // error
    cl::Kernel kq_kernel;   // key and query calculation
    cl::Buffer d_tokenEmbed, d_embeddings, d_deEmbeddings, d_expected_token;    // embeddings
    cl::Buffer d_Q_cl, d_K_cl, d_mQ_cl, d_mK_cl, d_tok_cl;  // matrices and output
    int initial_epochs_per_token_limit = this->epochs;      // Store initial limit, if you don't modify 'epochs' within loop
    int initial_token_count = this->currentTokenCount;      // Store initial count
    float learn = learning;

    try {

        // --- Device Buffer Allocation & H->D Transfer ---
        size_t totalTokenEmbedFloats = static_cast<size_t>(m) * CONTEXT_WIN * d;
        size_t tokenEmbedBytes = totalTokenEmbedFloats * sizeof(float);
        size_t embeddingsBytes = static_cast<size_t>(vocabsize) * d * sizeof(float);
        size_t singleTokenBytes = static_cast<size_t>(d) * sizeof(float);
        size_t outputBytes = singleTokenBytes; // Size of h_otok_buffer
        size_t matheights_bytes = static_cast<size_t>(MATHEIGHTS) * sizeof(float);
        size_t embedding_bytes_loc = static_cast<size_t>(EMBEDDING) * sizeof(float); // Corrected to size_t
        size_t projection_matrix_bytes = static_cast<size_t>(MATHEIGHTS) * EMBEDDING * sizeof(float);

        // Create buffers
        d_tokenEmbed = cl::Buffer(this->clcontext.context, CL_MEM_READ_WRITE, tokenEmbedBytes, nullptr, &cl_err); CL_CHECK(cl_err);
        d_expected_token = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY, singleTokenBytes, nullptr, &cl_err); CL_CHECK(cl_err);
        d_Q_cl = cl::Buffer(this->clcontext.context, CL_MEM_WRITE_ONLY, matheights_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        d_K_cl = cl::Buffer(this->clcontext.context, CL_MEM_WRITE_ONLY, matheights_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        d_mQ_cl = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY, projection_matrix_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        d_mK_cl = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY, projection_matrix_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        d_tok_cl = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY, embedding_bytes_loc, nullptr, &cl_err); CL_CHECK(cl_err);

        // Prepare initial context buffer content
        std::vector<float> flat_host_tokenEmbed(totalTokenEmbedFloats, 0.0f);
        if (this->tokenEmbed.mapped_data && this->tokenEmbed.row >= static_cast<size_t>(this->currentTokenCount) && this->tokenEmbed.col == static_cast<size_t>(d)) {
            for (int tk = 0; tk < this->currentTokenCount; ++tk) {
                float* row_ptr = this->tokenEmbed.mapped_data + (static_cast<size_t>(tk) * this->d);
                flat_host_tokenEmbed.insert(flat_host_tokenEmbed.end(), row_ptr, row_ptr + this->d);
            }
        }
        else if (this->currentTokenCount > 0) {
            std::cerr << "Warning: Host tokenEmbed (mat) not properly initialized or too small for currentTokenCount ("
                      << this->currentTokenCount << ", mat_rows: " << this->tokenEmbed.row
                      << ") in clTrainCONTEXT(sentence) setup." << std::endl;
            for (int tk = 0; tk < this->currentTokenCount; ++tk) {
                std::vector<float> padding(d, 0.0f);
                flat_host_tokenEmbed.insert(flat_host_tokenEmbed.end(), padding.begin(), padding.end());
            }
        }
        // Add the first token of the sentence to the initial buffer content
        if (!sentence.empty()) {
            flat_host_tokenEmbed.insert(flat_host_tokenEmbed.end(), sentence[0].begin(), sentence[0].end());
        }
        flat_host_tokenEmbed.resize(totalTokenEmbedFloats, 0.0f);
        CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_tokenEmbed, CL_TRUE, 0, tokenEmbedBytes, flat_host_tokenEmbed.data()));

        // Flatten and copy embeddings table
        std::vector<float> flat_deEmbeddings = ::flatten(this->deEmbeddings);       // de-embeddings

        // --- Initialize Host State ---
        // Add the first token to the host context tracking *before* the loop
        if (!sentence.empty()) {
            if (this->tokenEmbed.mapped_data && static_cast<size_t>(this->currentTokenCount) < this->tokenEmbed.row && this->tokenEmbed.col == static_cast<size_t>(d)) {
                float* dest_ptr = this->tokenEmbed.mapped_data + (static_cast<size_t>(this->currentTokenCount) * this->d);
                if (sentence[0].size() == static_cast<size_t>(d)) {
                    memcpy(dest_ptr, sentence[0].data(), singleTokenBytes);
                }
                else {
                    std::cerr << "Error: sentence[0] size mismatch for host tokenEmbed (mat) update in clTrainCONTEXT(sentence)." << std::endl;
                }
            }
            else {
                std::cerr << "Error: Host tokenEmbed (mat) not properly initialized or out of bounds for sentence[0] update in clTrainCONTEXT(sentence)." << std::endl;
            }
            this->currentTokenCount++; // Increment after successful or attempted copy
        }

        // start taking attention score for first token and then perform training
        // otherwise starting from zero means trying to perform training without attention score
        this->blockCount = 1; // Start with first block
        this->promptCount = currentTokenCount; // Number of tokens in the prompt (initial context)
        kq_kernel = cl::Kernel(this->clcontext.program, "kernelCompute_single_kq_vector", &cl_err);
        CL_CHECK(cl_err);

        // --- Train for each subsequent token in the sentence (i=1 to N-1) ---
        for (size_t i = 1; i < sentence.size(); ++i) { // i is the index of the TARGET token in 'sentence'
            if (this->currentTokenCount >= FULL_CONTEXT) {
                std::cerr << "Warning: clTrainCONTEXT(sentence) reached FULL_CONTEXT limit ("
                          << this->currentTokenCount << "). Stopping training early at sentence index " << i << "." << std::endl;
                break;
            }

            // Target token for this iteration
            std::vector<float>& expected_vec = sentence[i]; // This is the GROUND TRUTH target
            std::string& expected_str = rString[i];
            // Copy target token H->D into the dedicated buffer
            CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_expected_token, CL_TRUE, 0, singleTokenBytes, expected_vec.data()));
            
            // Context size *before* adding current token `i`.
            // This represents the length of the input sequence for the current forward pass.
            int effective_context_size = this->currentTokenCount; 
            int current_block_idx = this->blockCount; // Block index based on current context size

            if (current_block_idx <= 0 || current_block_idx > m) {
                throw std::out_of_range("clTrainCONTEXT(sentence): Calculated current_block_idx (" + std::to_string(current_block_idx) + ") is out of range [1, " + std::to_string(m) + "].");
            }

            // --- Training Loop for token i ---
            int j = 0; // Epoch counter for this token
            t[0].tokenCount = i; // Ensure this is correctly used by block (seems to be token_count in attention heads)
            float current_error = 0.0f; // Reset for each token's training loop
            std::cout << "Training token " << i << "/" << sentence.size() << ": '" << expected_str << "' @ " << expIndex[i] << std::endl;
            oneHotEncode.resize(this->vocabsize, 0.0f);     // set to 0s
            if (i >= expIndex.size() || expIndex[i] < 0 || expIndex[i] >= this->vocabsize) {
                std::cerr << "Warning: expIndex[" << i << "] is out of bounds or invalid for vocabulary size " << this->vocabsize << "." << std::endl;
            } else {
                oneHotEncode[expIndex[i]] = 1.0f;           // set 1 for expected token
            }
            // The epoch loop for a single target token
            // The `this->epochs` member now serves as a soft cap, but the LR scheduler can cause earlier exit
            float prev_error = 1.0f;

            while (j < initial_epochs_per_token_limit) { // Use initial limit for fixed epochs per token

                // recalculate keys and queries for heads of respective blocks
                if(current_block_idx == 1) {
                    for (int layer_idx = 0; layer_idx < x; ++layer_idx) {
                        for (int parallel_idx = 0; parallel_idx < y; ++parallel_idx) {
                            auto& attention_head = t[0].b[layer_idx][parallel_idx];
                            CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_mQ_cl, CL_TRUE, 0, projection_matrix_bytes, attention_head.MQ.mapped_data));
                            CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_mK_cl, CL_TRUE, 0, projection_matrix_bytes, attention_head.MK.mapped_data));

                            for (size_t k = 0; k < i; ++k) {
                                size_t qk_vector_idx_in_block = (static_cast<size_t>(initial_token_count) % CONTEXT_WIN) + k;
                                if (qk_vector_idx_in_block >= CONTEXT_WIN) {
                                    std::cerr << "Warning: qk_vector_idx_in_block ("<< qk_vector_idx_in_block << ") exceeds CONTEXT_WIN in clTrainCONTEXT K/Q for prompt (block 1)." << std::endl;
                                    continue;
                                }
                                size_t host_qk_offset = qk_vector_idx_in_block * MATHEIGHTS;
                                CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_tok_cl, CL_TRUE, 0, embedding_bytes_loc, sentence[k].data()));
                                // query calculation
                                kq_kernel.setArg(0, d_tok_cl); kq_kernel.setArg(1, d_mQ_cl); kq_kernel.setArg(2, d_Q_cl);
                                kq_kernel.setArg(3, static_cast<cl_int>(EMBEDDING)); kq_kernel.setArg(4, static_cast<cl_int>(MATHEIGHTS));
                                CL_CHECK(this->clcontext.queue.enqueueNDRangeKernel(kq_kernel, cl::NullRange, cl::NDRange(1), cl::NullRange));
                                if (attention_head.Q.mapped_data && (host_qk_offset + MATHEIGHTS) <= (attention_head.Q.row * attention_head.Q.col)) {
                                    CL_CHECK(this->clcontext.queue.enqueueReadBuffer(d_Q_cl, CL_TRUE, 0, matheights_bytes, attention_head.Q.mapped_data + host_qk_offset));
                                }
                                else {
                                    std::cerr << "Error: Host Q buffer invalid or out of bounds for block 0." << std::endl;
                                    break;
                                }

                                // key calculation
                                kq_kernel.setArg(0, d_tok_cl); kq_kernel.setArg(1, d_mK_cl); kq_kernel.setArg(2, d_K_cl);
                                kq_kernel.setArg(3, static_cast<cl_int>(EMBEDDING)); kq_kernel.setArg(4, static_cast<cl_int>(MATHEIGHTS));
                                CL_CHECK(this->clcontext.queue.enqueueNDRangeKernel(kq_kernel, cl::NullRange, cl::NDRange(1), cl::NullRange));
                                if (attention_head.K.mapped_data && (host_qk_offset + MATHEIGHTS) <= (attention_head.K.row * attention_head.K.col)) {
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
                            auto& current_block_attention = t[current_processing_block_idx - 1].b[layer_idx][parallel_idx];
                            auto& prev_block_attention = t[current_processing_block_idx - 2].b[layer_idx][parallel_idx];
                            CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_mQ_cl, CL_TRUE, 0, projection_matrix_bytes, current_block_attention.MQ.mapped_data));
                            CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_mK_cl, CL_TRUE, 0, projection_matrix_bytes, current_block_attention.MK.mapped_data));

                            // Keys from current prompt tokens for the current block
                            for (size_t k = 0; k < i; ++k) {
                                size_t qk_vector_idx_in_block = (static_cast<size_t>(initial_token_count) % CONTEXT_WIN) + k;
                                if (qk_vector_idx_in_block >= CONTEXT_WIN) { std::cerr << "Warning: qk_vector_idx_in_block ("<< qk_vector_idx_in_block << ") exceeds CONTEXT_WIN in clTrainCONTEXT K/Q for prompt (block N)." << std::endl; continue; }
                                size_t host_k_offset = qk_vector_idx_in_block * MATHEIGHTS;
                                CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_tok_cl, CL_TRUE, 0, embedding_bytes_loc, sentence[k].data()));
                                kq_kernel.setArg(0, d_tok_cl); kq_kernel.setArg(1, d_mK_cl); kq_kernel.setArg(2, d_K_cl);
                                kq_kernel.setArg(3, EMBEDDING); kq_kernel.setArg(4, MATHEIGHTS);
                                CL_CHECK(this->clcontext.queue.enqueueNDRangeKernel(kq_kernel, cl::NullRange, cl::NDRange(1), cl::NullRange));
                                if (current_block_attention.K.mapped_data && (host_k_offset + MATHEIGHTS) <= (current_block_attention.K.row * current_block_attention.K.col)) {
                                    CL_CHECK(this->clcontext.queue.enqueueReadBuffer(d_K_cl, CL_TRUE, 0, matheights_bytes, current_block_attention.K.mapped_data + host_k_offset));
                                }
                                else { 
                                    std::cerr << "Error: Host K buffer invalid or out of bounds for block " << current_processing_block_idx << std::endl;
                                }
                            }

                            // Queries from previous block's EV for the current block
                            for (int k_ev = 0; k_ev < CONTEXT_WIN; ++k_ev) {
                                size_t q_vector_idx_in_block = k_ev;
                                size_t host_q_offset = q_vector_idx_in_block * MATHEIGHTS;
                                if (!prev_block_attention.EV.mapped_data || (static_cast<size_t>(k_ev) * EMBEDDING + EMBEDDING) > (prev_block_attention.EV.row * prev_block_attention.EV.col)) {
                                    std::cerr << "Warning: Prev block EV data invalid or out of bounds for index " << k_ev << " in clTrainCONTEXT K/Q." << std::endl; continue;
                                }
                                CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_tok_cl, CL_TRUE, 0, embedding_bytes_loc, prev_block_attention.EV.mapped_data + static_cast<size_t>(k_ev) * EMBEDDING));
                                kq_kernel.setArg(0, d_tok_cl); kq_kernel.setArg(1, d_mQ_cl); kq_kernel.setArg(2, d_Q_cl);
                                kq_kernel.setArg(3, EMBEDDING); kq_kernel.setArg(4, MATHEIGHTS);
                                CL_CHECK(this->clcontext.queue.enqueueNDRangeKernel(kq_kernel, cl::NullRange, cl::NDRange(1), cl::NullRange));
                                if (current_block_attention.Q.mapped_data && (host_q_offset + MATHEIGHTS) <= (current_block_attention.Q.row * current_block_attention.Q.col)) {
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
                int host_indexForToken = -1;
                clForward(current_block_idx, effective_context_size, promptCount);

                for(size_t k_dim = 0; k_dim < static_cast<size_t>(this->d * this->x); k_dim++) {
                    if (std::isnan(this->otok[k_dim])) { this->otok[k_dim] = 0.0001f; }
                    else if (std::isinf(this->otok[k_dim])) { this->otok[k_dim] = std::copysign((std::numeric_limits<float>::max)(), this->otok[k_dim]); }
                }
                if (this->otok.size() != static_cast<size_t>(this->d * this->x)) {
                    throw std::runtime_error("clTrainCONTEXT(sentence): this->otok from clForward has incorrect size.");
                }
                std::cout << "current block: " << current_block_idx << " & current token count: " << currentTokenCount << std::endl;

                // use kernelComputePredictionWithScores for output prediction
                {
                    cl::Buffer d_otok_buffer, d_predictions, d_result_index_buffer;
                    try {
                        d_deEmbeddings = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, embeddingsBytes, flat_deEmbeddings.data(), &cl_err); CL_CHECK(cl_err);
                        size_t otok_bytes = this->otok.size() * sizeof(float);
                        d_otok_buffer = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, otok_bytes, this->otok.data(), &cl_err); CL_CHECK(cl_err);
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

                // get error from predicted tokens vs one hot encode for token 
                current_error = crossEntropy(oneHotEncode, pred);
                std::string predicted_token_str = (host_indexForToken >= 0 && host_indexForToken < static_cast<unsigned long long>(tokens.size()))
                                                  ? tokens[host_indexForToken] : "INVALID_INDEX";
                std::cout << "Computed token is -> " << predicted_token_str
                          << " (index: " << host_indexForToken
                          << ") | BCE LOSS: " << current_error
                          << " | Epoch Count: " << j 
                          << " | Learning Rate: " << this->learning << std::endl;
                
                // --- Early exit if token is predicted correctly ---
                if (predicted_token_str == expected_str && predicted_token_str != "INVALID_INDEX") {
                    std::cout << "Token '" << expected_str << "' predicted correctly after " << j+1 << " epochs. Moving to next token." << std::endl;
                    totalLearning += this->learning;
                    learning = learn;           // reset to original
                    break;
                }

                // modify the learning rate after first epoch
                if(j > 0) {
                    if(current_error <= prev_error) {
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
                clUpdateDeEmbeddings(deEmbeddings, otok, pred, oneHotEncode, host_indexForToken, this->learning, lambda_L1, lambda_L2, gradEH);
                // get expected target for backprop
                std::vector<float> exp(this->d * this->x, 0.0f);
                std::vector<std::vector<float>> targets_for_heads(x, std::vector<float>(EMBEDDING, 0.0f));
                for(int head_idx = 0; head_idx < x; ++head_idx) {
                    for(int eidx = 0; eidx < EMBEDDING; ++eidx) {
                        exp[(head_idx * EMBEDDING) + eidx] = otok[(head_idx * EMBEDDING) + eidx]
                                        - (learning*gradEH[(head_idx * EMBEDDING) + eidx]
                                        + lambda_L1*deEmbeddings(host_indexForToken, eidx)
                                        + lambda_L2*deEmbeddings(host_indexForToken, eidx));
                        targets_for_heads[head_idx][eidx] = exp[(head_idx * EMBEDDING) + eidx];
                    }
                }
                // backpropagate
                clBackwardContext(targets_for_heads, current_block_idx);
                // update embeddings which are in use
                std::vector<float> tok = tokenEmbed(currentTokenCount-1);
                clUpdateEmbeddings(tok, this->learning, lambda_L1, lambda_L2, t[blockCount-1].gradToken);
                setRow(this->tokenEmbed, currentTokenCount-1, tok);

                totalLearning += learning;
                j++;
                prev_error = current_error;
            }
            learning = learn;           // reset to original
            // --- Update Host State after current token's training loop completes ---
            this->trainCount++;
            this->epochCount += j;
            this->error += current_error;
            this->currentTokenCount++;
            // Update blockCount after currentTokenCount has been incremented (only when local context if full)
            this->blockCount = (this->currentTokenCount == 0) ? 1 : ((this->currentTokenCount - 1) / CONTEXT_WIN) + 1;
        }
        std::cout << "--------------->>>>>>>>>>>>> Training for this sentence complete. >>>>>>>>>>>>>>>>-------------" << std::endl;
    }
    catch (const std::runtime_error& e) { // Catches std::runtime_error from CL_CHECK
        std::cerr << "Standard Exception in clTrainCONTEXT(sentence): " << e.what() << std::endl;
        this->epochs = initial_epochs_per_token_limit; // Restore original epoch limit if an exception aborts
        throw;
    }
}

#endif
