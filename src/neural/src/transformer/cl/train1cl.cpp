#ifdef USE_OPENCL
#include "include/transformer.hpp"
#include <maths.hpp>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <string>
#include <cmath> // For std::abs, std::max


/**
 * @brief (OpenCL) Train the transformer for next token prediction (single token training).
 * @param promptCount Number of tokens in the prompt.
 * @param currentTokenCount Number of tokens in the full context *before* this training step.
 * @param blockCount Current block index (1-based) in the full context.
 * @param expected Expected token embedding (on host).
 * @param expString Expected token string (on host, used for convergence check).
 */
void transformer::clTrain(int& promptCount, int& currentTokenCount, int& blockCount, std::vector<float>& expected, std::string& expString)
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
        float learnby = learning;
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
        while (j <= this->epochs) // Use member variable epochs
        {
            long long int host_indexForToken = -1;
            // --- Forward Pass ---
            // KdotQ calculation (if needed, depends on clForward implementation)
            if (this->inTraining) {
                // clParallelKdotQs(...); <- already in clForward
            }
            clForward(current_block_idx, effective_context_size, promptCount); // Operates on device data implicitly
            std::vector<float> h_otok_buffer(d, 0.0f); // Initialize with zeros
            h_otok_buffer = this->otok; // clForward populates this
            if (h_otok_buffer.size() != static_cast<size_t>(d)) {
                throw std::runtime_error("clTrain(sentence): this->otok from clForward has incorrect size.");
            }
            // use kernelComputePrediction for output prediction
            {
                cl::Buffer d_otok_buffer, d_result_index_buffer;
                try {
                    size_t otok_bytes = h_otok_buffer.size() * sizeof(float);
                    d_otok_buffer = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, otok_bytes, h_otok_buffer.data(), &cl_err); CL_CHECK(cl_err);
                    d_result_index_buffer = cl::Buffer(this->clcontext.context, CL_MEM_WRITE_ONLY, sizeof(cl_int), nullptr, &cl_err); CL_CHECK(cl_err);

                    cl::Kernel kernel = this->clcontext.kernels.at("kernelComputePrediction");
                    CL_CHECK(kernel.setArg(0, d_otok_buffer));  CL_CHECK(kernel.setArg(1, d_embeddings));
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
            // --- Get EH output from the relevant block ---
            std::cout << "clTrain(single): Getting output for block " << current_block_idx << " with effective context size " << effective_context_size << std::endl;
            h_otok_buffer = this->otok;
            if (h_otok_buffer.size() != static_cast<size_t>(d)) {
                throw std::runtime_error("clTrain(single): this->otok from clForward has incorrect size. Expected " + std::to_string(d) + ", got " + std::to_string(h_otok_buffer.size()));
            }
            current_error = crossEntropy(h_otok_buffer, expected); // Host-side error calculation
            std::cout << "Training Error: " << current_error << " for token: " << expString << std::endl;
            if (this->tokens[host_indexForToken] == expString && this->tokens[host_indexForToken] != "INVALID_INDEX") {
                size_t offset_bytes = static_cast<size_t>(effective_context_size) * d * sizeof(float);
                if (offset_bytes + outputBytes > tokenEmbedBytes) {
                    throw std::out_of_range("clTrain(prompt-response): Offset exceeds buffer bounds when writing converged response token.");
                }
                std::cout << "indexForToken: " << this->indexForToken << " | host_indexForToken: " << host_indexForToken << " | Epoch Count for this token: " << j << " | Current Token Count " << currentTokenCount << std::endl;
                CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_tokenEmbed, CL_TRUE, offset_bytes, outputBytes, expected.data()));
                if(this->tokens[host_indexForToken]  == "@#0"){
                    std::cout << "--------------->>>>>>>>>>>>> To next LINE >>>>>>>>>>>>>>>>-------------" << std::endl;
                }
                else {
                    std::cout << "--------------------- To next token ------------->>>>>>>>>>>>>>>>>" << std::endl;
                    totalLearning += learning;
                    break;
                }
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
                        learning *= (1.05 + (j/6)*0.15);
                }
                else {
                    if(j <= 6)   
                        learning *= 0.95;
                    else if (j % 6 == 0)
                        learning *= (0.95 - (j/6)*0.01);
                }
            }

            if(current_block_idx == 1) {
                clBackward(expected);
                // t[0].serialise(t[0].blockFilePath); // Save the first block after training
            }
            else {
                clBackward(expected, current_block_idx);
                // t[current_block_idx-1].serialise(t[current_block_idx-1].blockFilePath);
            }
            totalLearning += learning;
            prev_Error = current_error;
            j++;
        }
        learning = learnby;

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
        // Update promptCount for the next step (relative to the current block)
        this->promptCount = this->currentTokenCount % CONTEXT_WIN;
        if (this->promptCount == 0 && this->currentTokenCount > 0)
            this->promptCount = CONTEXT_WIN;
    }
    catch (const std::runtime_error& e) {
        std::cerr << "Standard Exception in clTrain(single): " << e.what() << std::endl;
        this->epochs = initial_epochs;
        throw; // Re-throw
    }
}


/**
 * @brief (OpenCL) Train the transformer on sentences and paragraphs.
 * @param sentence Token embeddings of the sentence (on host).
 * @param rString Sentence tokens (on host).
 */
void transformer::clTrain(std::vector<std::vector<float>>& sentence, std::vector<std::string>& rString)
{
    // --- Basic validation ---
    if (sentence.size() > FULL_CONTEXT) {
        throw std::runtime_error("clTrain(sentence): Sentence size (" + std::to_string(sentence.size()) + ") exceeds FULL_CONTEXT (" + std::to_string(FULL_CONTEXT) + ").");
    }
    if (sentence.empty() || sentence.size() != rString.size()) {
        std::cout << "sentence.size(): " << sentence.size() << ", rString.size(): " << rString.size() << std::endl;
        throw std::runtime_error("clTrain(sentence): Sentence embeddings/strings mismatch or empty.");
    }
    if (!sentence.empty() && sentence[0].size() != static_cast<size_t>(d)) {
        throw std::runtime_error("clTrain(sentence): Sentence embedding dimension mismatch. Expected " + std::to_string(d) + ", got " + std::to_string(sentence[0].size()));
    }

    cl_int cl_err;          // error
    cl::Kernel kq_kernel;   // key and query calculation
    cl::Buffer d_tokenEmbed, d_embeddings, d_expected_token;    // embeddings
    cl::Buffer d_Q_cl, d_K_cl, d_mQ_cl, d_mK_cl, d_tok_cl;      // matrices and output
    float current_error = 1.0f;
    int initial_epochs = this->epochs;
    int initial_token_count = this->currentTokenCount; // Store initial count

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
        int learnby = learning;
        float prev_Error = 0.0f;    // error from previous epoch for same token
        // --- Device Buffer Allocation & H->D Transfer ---
        size_t totalTokenEmbedFloats = static_cast<size_t>(m) * CONTEXT_WIN * d;
        size_t tokenEmbedBytes = totalTokenEmbedFloats * sizeof(float);
        size_t embeddingsBytes = static_cast<size_t>(vocabsize) * d * sizeof(float);
        size_t singleTokenBytes = static_cast<size_t>(d) * sizeof(float);
        size_t outputBytes = singleTokenBytes; // Size of h_otok_buffer
        size_t indexBytes = sizeof(int);       // Size for the result index
        size_t matheights_bytes = static_cast<size_t>(MATHEIGHTS) * sizeof(float);
        size_t embedding_bytes_loc = static_cast<size_t>(EMBEDDING) * sizeof(float);
        size_t projection_matrix_bytes = static_cast<size_t>(MATHEIGHTS) * EMBEDDING * sizeof(float);

        // Create buffers
        d_tokenEmbed = cl::Buffer(this->clcontext.context, CL_MEM_READ_WRITE, tokenEmbedBytes, nullptr, &cl_err); CL_CHECK(cl_err);
        d_expected_token = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY, singleTokenBytes, nullptr, &cl_err); CL_CHECK(cl_err); // Buffer for the target token
        d_Q_cl = cl::Buffer(this->clcontext.context, CL_MEM_WRITE_ONLY, matheights_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        d_K_cl = cl::Buffer(this->clcontext.context, CL_MEM_WRITE_ONLY, matheights_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        d_mQ_cl = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY, projection_matrix_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        d_mK_cl = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY, projection_matrix_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        d_tok_cl = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY, embedding_bytes_loc, nullptr, &cl_err); CL_CHECK(cl_err);

        // Prepare initial context buffer content
        std::vector<float> flat_host_tokenEmbed(totalTokenEmbedFloats, 0.0f);
        if (this->tokenEmbed.mapped_data && this->tokenEmbed.row >= static_cast<size_t>(this->currentTokenCount) && this->tokenEmbed.col == static_cast<size_t>(d)) {
            for (int tk = 0; tk < this->currentTokenCount; ++tk) { // Copy existing context
                float* row_ptr = this->tokenEmbed.mapped_data + (static_cast<size_t>(tk) * this->d);
                flat_host_tokenEmbed.insert(flat_host_tokenEmbed.end(), row_ptr, row_ptr + this->d);
            }
        } 
        else if (this->currentTokenCount > 0) {
            std::cerr << "Warning: Host tokenEmbed (mat) not properly initialized or too small for currentTokenCount ("
                      << this->currentTokenCount << ", mat_rows: " << this->tokenEmbed.row
                      << ") in clTrain(sentence) setup." << std::endl;
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
        std::vector<float> flat_embeddings = ::flatten(this->embeddings); // Assuming global or from basic.hpp
        d_embeddings = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, embeddingsBytes, flat_embeddings.data(), &cl_err); CL_CHECK(cl_err);

        // --- Initialize Host State ---
        // Add the first token to the host context tracking *before* the loop
        if (!sentence.empty()) {
            if (this->tokenEmbed.mapped_data && static_cast<size_t>(this->currentTokenCount) < this->tokenEmbed.row && this->tokenEmbed.col == static_cast<size_t>(d)) {
                float* dest_ptr = this->tokenEmbed.mapped_data + (static_cast<size_t>(this->currentTokenCount) * this->d);
                if (sentence[0].size() == static_cast<size_t>(d)) {
                    memcpy(dest_ptr, sentence[0].data(), singleTokenBytes);
                }
                else {
                    std::cerr << "Error: sentence[0] size mismatch for host tokenEmbed (mat) update in clTrain(sentence)." << std::endl;
                }
            }
            else {
                std::cerr << "Error: Host tokenEmbed (mat) not properly initialized or out of bounds for sentence[0] update in clTrain(sentence)." << std::endl;
            }
            this->currentTokenCount++; // Increment after successful or attempted copy
        }

        // start taking attention score for first token and then perform trainin
        // otherwise starting from zero means trying to perform training without attention score
        this->blockCount = 1, this->promptCount = 1;

        // --- Train for each subsequent token in the sentence (i=1 to N-1) ---
        for (size_t i = 1; i < sentence.size(); ++i) {
            if (this->currentTokenCount >= FULL_CONTEXT) {
                std::cerr << "Warning: clTrain(sentence) reached FULL_CONTEXT limit ("
                          << this->currentTokenCount << "). Stopping training early at sentence index " << i << "." << std::endl;
                break;
            }

            // Target token for this iteration
            std::vector<float>& expected_vec = sentence[i];
            std::string& expected_str = rString[i];

            // Copy target token H->D into the dedicated buffer
            CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_expected_token, CL_TRUE, 0, singleTokenBytes, expected_vec.data()));

            int effective_context_size = this->currentTokenCount; // Context size *before* adding token i
            int current_block_idx = this->blockCount; // Block index based on current context size

            if (current_block_idx <= 0 || current_block_idx > m) {
                throw std::out_of_range("clTrain(sentence): Calculated current_block_idx (" + std::to_string(current_block_idx) + ") is out of range [1, " + std::to_string(m) + "].");
            }

            // --- Training Loop for token i ---
            int j = 0; // Epoch counter for this token
            current_error = 1.0f;
            prev_Error = 0.0f;

            std::cout << "Training token " << i << "/" << sentence.size() << ": '" << expected_str << "'" << std::endl;
            while (j <= this->epochs) {
                // keys and queries for heads of respective blocks
                if(current_block_idx == 1) {
                    for (int layer_idx = 0; layer_idx < x; ++layer_idx) {
                        for (int parallel_idx = 0; parallel_idx < y; ++parallel_idx) {
                            auto& attention_head = t[0].b[layer_idx][parallel_idx];
                            CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_mQ_cl, CL_TRUE, 0, projection_matrix_bytes, attention_head.MQ.mapped_data));
                            CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_mK_cl, CL_TRUE, 0, projection_matrix_bytes, attention_head.MK.mapped_data));

                            for (size_t k = 0; k < i; ++k) {
                                size_t qk_vector_idx_in_block = (static_cast<size_t>(initial_token_count) % CONTEXT_WIN) + k;
                                if (qk_vector_idx_in_block >= CONTEXT_WIN) {
                                    std::cerr << "Warning: qk_vector_idx_in_block ("<< qk_vector_idx_in_block << ") exceeds CONTEXT_WIN in clTrain K/Q for prompt (block 1)." << std::endl;
                                    continue;
                                }
                                size_t host_qk_offset = qk_vector_idx_in_block * MATHEIGHTS;
                                CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_tok_cl, CL_TRUE, 0, embedding_bytes_loc, sentence[k].data()));

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
                            for (size_t k = 0; k < i; ++k) {
                                size_t qk_vector_idx_in_block = (static_cast<size_t>(initial_token_count) % CONTEXT_WIN) + k;
                                if (qk_vector_idx_in_block >= CONTEXT_WIN) { std::cerr << "Warning: qk_vector_idx_in_block ("<< qk_vector_idx_in_block << ") exceeds CONTEXT_WIN in clTrain K/Q for prompt (block N)." << std::endl; continue; }
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
                                    std::cerr << "Warning: Prev block EV data invalid or out of bounds for index " << k_ev << " in clTrain K/Q." << std::endl; continue;
                                }
                                CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_tok_cl, CL_TRUE, 0, embedding_bytes_loc, prev_block_attention.EV.mapped_data + static_cast<size_t>(k_ev) * EMBEDDING));
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
                long long int host_indexForToken = -1;
                // --- Get EH output ---
                std::vector<float> h_otok_buffer(d, 0.0f); // Initialize with zeros
                h_otok_buffer = this->otok; // clForward populates this
                if (h_otok_buffer.size() != static_cast<size_t>(d)) {
                    throw std::runtime_error("clTrain(sentence): this->otok from clForward has incorrect size.");
                }
                std::cout << "current block: " << current_block_idx << " & current token count: " << currentTokenCount << std::endl;
                
                // use kernelComputePrediction for output prediction
                {
                    cl::Buffer d_otok_buffer, d_result_index_buffer;
                    try {
                        size_t otok_bytes = h_otok_buffer.size() * sizeof(float);
                        d_otok_buffer = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, otok_bytes, h_otok_buffer.data(), &cl_err); CL_CHECK(cl_err);
                        d_result_index_buffer = cl::Buffer(this->clcontext.context, CL_MEM_WRITE_ONLY, sizeof(cl_int), nullptr, &cl_err); CL_CHECK(cl_err);

                        cl::Kernel kernel = this->clcontext.kernels.at("kernelComputePrediction");
                        CL_CHECK(kernel.setArg(0, d_otok_buffer));  CL_CHECK(kernel.setArg(1, d_embeddings));
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

                current_error = crossEntropy(h_otok_buffer, expected_vec);
                std::string predicted_token_str = (host_indexForToken >= 0 && host_indexForToken < static_cast<long long int>(tokens.size()))
                                                  ? tokens[host_indexForToken] : "INVALID_INDEX";
                std::cout << "Computed token is -> " << predicted_token_str << " (index: " << host_indexForToken << ") | with BCE error " << current_error << " | MAE Error " << MAE(h_otok_buffer, sentence[i]) << std::endl;

                if (predicted_token_str == expected_str && predicted_token_str != "INVALID_INDEX") {
                    size_t offset_bytes = static_cast<size_t>(effective_context_size) * d * sizeof(float);
                    if (offset_bytes + outputBytes > tokenEmbedBytes) {
                        throw std::out_of_range("clTrain(prompt-response): Offset exceeds buffer bounds when writing converged response token.");
                    }
                    std::cout << "indexForToken: " << this->indexForToken << " | host_indexForToken: " << host_indexForToken << " | Epoch Count for this token: " << j << " | Current Token Count " << currentTokenCount << std::endl;
                    CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_tokenEmbed, CL_TRUE, offset_bytes, outputBytes, expected_vec.data())); // Write expected_vec (target EH)
                    if(predicted_token_str == "@#0"){
                        std::cout << "--------------->>>>>>>>>>>>> To next LINE >>>>>>>>>>>>>>>>-------------" << std::endl;
                    }
                    else {
                        std::cout << "--------------------- To next token ------------->>>>>>>>>>>>>>>>>" << std::endl;
                        totalLearning += learning;
                        break;
                    }
                }
                else if (j == this->epochs - 1) {
                    if (predicted_token_str != expected_str) {
                        std::cout << "Increasing Epoch Count by 10 '-'" << std::endl;
                        this->epochs += 10;
                    }
                }
                // update learning
                if(current_error < prev_Error) {
                    learning = 0.95 * learning;
                }
                else {
                    learning = 1.05 * learning;
                }
                // --- Backward Pass ---
                clBackward(expected_vec, current_block_idx);
                learning = learnby;
                totalLearning += learning;
                prev_Error = current_error;
                j++;
            } // End training loop for token i

            // --- Update Host State ---
            this->trainCount++;
            this->epochCount += j;
            this->error += current_error;

            if (this->tokenEmbed.mapped_data && static_cast<size_t>(this->currentTokenCount) < this->tokenEmbed.row && this->tokenEmbed.col == static_cast<size_t>(d)) {
                float* dest_ptr = this->tokenEmbed.mapped_data + (static_cast<size_t>(this->currentTokenCount) * this->d);
                if (expected_vec.size() == static_cast<size_t>(d)) {
                    memcpy(dest_ptr, expected_vec.data(), singleTokenBytes);
                }
                else {
                    std::cerr << "Error: expected_vec size mismatch for host tokenEmbed (mat) update in clTrain(sentence)." << std::endl;
                }
            }
            else {
                std::cerr << "Error: Host tokenEmbed (mat) not properly initialized or out of bounds for expected_vec update in clTrain(sentence)." << std::endl;
            }
            this->currentTokenCount++; // Increment after successful or attempted copy

            // Update blockCount for the *next* iteration
            // t[blockCount-1].serialise(t[blockCount-1].blockFilePath);
            this->blockCount = (this->currentTokenCount == 0) ? 1 : ((this->currentTokenCount) / CONTEXT_WIN) + 1;
        }
    }
    catch (const std::runtime_error& e) { // Catches std::runtime_error from CL_CHECK
        std::cerr << "Standard Exception in clTrain(sentence): " << e.what() << std::endl;
        this->epochs = initial_epochs;
        throw;
    }
    // Buffers released by RAII
}

#endif // USE_OPENCL
