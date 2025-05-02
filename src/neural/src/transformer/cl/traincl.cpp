#ifdef USE_OPENCL

#include "include/transformer.hpp"
#include "include/block.hpp"
#include "include/attention.hpp"
#include "include/mlp.hpp" // For flatten/unflatten, errorofv
#include <maths.hpp>
#include <CL/cl.hpp>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <string>
#include <cmath> // For std::abs, std::max

// Forward declaration (already present in the provided code)
// void transformer::clComputeOutput(std::vector<float>& output, std::vector<std::vector<float>>& embeddings, int& voc, int& index);

/**
 * @brief (OpenCL) Train the transformer for next token prediction (single token training).
 *        Mirrors the logic of transformer::cuTrain(..., std::vector<float>& expected, ...).
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
    int host_indexForToken = -1; // Host copy of the predicted index
    float current_error = 1.0f; // Initialize error high
    int initial_epochs = this->epochs; // Store initial epochs setting

    try {
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
        d_tokenEmbed = cl::Buffer(this->clcontext.context, CL_MEM_READ_WRITE, tokenEmbedBytes);
        d_expected = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY, expectedBytes);

        // Flatten and copy current tokenEmbed context to device
        std::vector<float> flat_host_tokenEmbed;
        flat_host_tokenEmbed.reserve(totalTokenEmbedFloats); // Reserve space for all blocks
        for(int tk = 0; tk < this->currentTokenCount; ++tk) { // Copy existing context
             if (tk < this->tokenEmbed.size() && this->tokenEmbed[tk].size() == static_cast<size_t>(d)) {
                 flat_host_tokenEmbed.insert(flat_host_tokenEmbed.end(), this->tokenEmbed[tk].begin(), this->tokenEmbed[tk].end());
             } else {
                 // Handle potential inconsistencies or pad if necessary
                 std::cerr << "Warning: Inconsistent tokenEmbed data at index " << tk << " during clTrain(single) setup." << std::endl;
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

        // Copy expected vector
        this->clcontext.queue.enqueueWriteBuffer(d_expected, CL_TRUE, 0, expectedBytes, expected.data());

        // Determine effective context size and block index for the *current* state
        int effective_context_size = currentTokenCount;
        // Recalculate current_block_idx based on effective_context_size just to be sure
        int current_block_idx = (effective_context_size == 0) ? 1 : ((effective_context_size - 1) / CONTEXT_WIN) + 1;
        if (current_block_idx <= 0 || current_block_idx > m) {
             throw std::out_of_range("clTrain(single): Calculated current_block_idx (" + std::to_string(current_block_idx) + ") is out of range [1, " + std::to_string(m) + "].");
        }


        // --- Training Loop ---
        int j = 0; // Epoch counter for this token
        while (j <= this->epochs) // Use member variable epochs
        {
            // --- Forward Pass ---
            // KdotQ calculation (if needed, depends on clForward implementation)
            if (this->inTraining) {
                 // clParallelKdotQs(...); // Placeholder - Needs OpenCL adaptation based on data flow
            }
            clForward(current_block_idx, effective_context_size, promptCount); // Operates on device data implicitly

            // --- Get EH output from the relevant block ---
            // Assume clForward updates the host t[block_idx-1].EH member
            std::vector<float> h_otok_buffer;
            if (current_block_idx > 0 && current_block_idx <= m) {
                 h_otok_buffer = t[current_block_idx - 1].EH; // Get EH from the correct block's host copy
                 if (h_otok_buffer.size() != static_cast<size_t>(d)) {
                     throw std::runtime_error("clTrain(single): EH buffer from block " + std::to_string(current_block_idx) + " has incorrect size.");
                 }
            } else {
                 throw std::runtime_error("clTrain(single): Invalid block index (" + std::to_string(current_block_idx) + ") before clComputeOutput.");
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
                    std::cerr << "OpenCL Error during inline clComputeOutput in clTrain(single): " << err.what() << " (" << err.err() << ")" << std::endl;
                    host_indexForToken = -1; // Indicate error
                    throw; // Re-throw
                }
                // Buffers d_output, d_result_index released by RAII
            }
            // --- End: Inline clComputeOutput Logic ---

            current_error = errorofv(h_otok_buffer, expected); // Host-side error calculation

            // --- Convergence Check ---
            bool converged = (current_error < 0.01);
            if (!converged && host_indexForToken >= 0 && host_indexForToken < static_cast<int>(tokens.size())) {
                converged = (tokens[host_indexForToken] == expString);
            }

            if (converged) {
                // Converged: Update the main token embedding buffer on the device with the *expected* value
                size_t offset_bytes = static_cast<size_t>(effective_context_size) * d * sizeof(float);
                // Ensure offset is within bounds
                 if (offset_bytes + singleTokenBytes > tokenEmbedBytes) {
                    throw std::out_of_range("clTrain(single): Calculated offset for writing converged token exceeds buffer bounds.");
                 }
                 this->clcontext.queue.enqueueWriteBuffer(d_tokenEmbed, CL_TRUE, offset_bytes, singleTokenBytes, expected.data()); // Write expected token
                break; // Exit training loop
            }

            // --- Adjust Epochs if Needed ---
            // Check if it's the last planned epoch and still not converged
            if (current_error >= 0.01 && j == this->epochs) {
                 // Check if the predicted token matches the expected string even if error is high
                 bool predicted_matches = (host_indexForToken >= 0 && host_indexForToken < static_cast<int>(tokens.size()) && tokens[host_indexForToken] == expString);
                 if (!predicted_matches) {
                    this->epochs += 10; // Increase total epochs if not converging by error or string match
                    // Optional: Add a maximum epoch limit to prevent infinite loops
                    // if (this->epochs > MAX_EPOCHS_ALLOWED) { throw std::runtime_error("Max epochs reached"); }
                 } else {
                    // If string matches but error is high, maybe stop anyway or log it.
                    // For now, let's break as the string match is considered convergence.
                     size_t offset_bytes = static_cast<size_t>(effective_context_size) * d * sizeof(float);
                     if (offset_bytes + singleTokenBytes > tokenEmbedBytes) {
                         throw std::out_of_range("clTrain(single): Calculated offset for writing converged token exceeds buffer bounds.");
                     }
                     this->clcontext.queue.enqueueWriteBuffer(d_tokenEmbed, CL_TRUE, offset_bytes, singleTokenBytes, expected.data());
                     break;
                 }
            }

            // --- Backward Pass ---
            clBackward(expected, current_block_idx);

            j++; // Increment epoch counter for this token
        } // End training loop

        // --- Update Host State ---
        this->trainCount++;
        this->epochCount += j; // Add epochs spent on this token
        this->error += current_error; // Accumulate final error
        int previousTokenCount = this->currentTokenCount;
        this->currentTokenCount += 1; // Increment context size *after* training for the token

        // Update host tokenEmbed with the *expected* value
        if (previousTokenCount < this->tokenEmbed.size()) {
            this->tokenEmbed[previousTokenCount] = expected;
        } else {
            this->tokenEmbed.push_back(expected);
        }
        // Resize if needed, though push_back handles growth
        if (this->tokenEmbed.size() != static_cast<size_t>(this->currentTokenCount)) {
             this->tokenEmbed.resize(this->currentTokenCount); // Ensure host vector size matches
             // If resizing larger, might need default initialization, but push_back is preferred
        }


        // Update blockCount for the *next* token
        this->blockCount = (this->currentTokenCount == 0) ? 1 : ((this->currentTokenCount - 1) / CONTEXT_WIN) + 1;
        // Update promptCount for the next step (relative to the current block)
        this->promptCount = this->currentTokenCount % CONTEXT_WIN;
        if (this->promptCount == 0 && this->currentTokenCount > 0) this->promptCount = CONTEXT_WIN;


        // --- Optional: Read back the entire updated tokenEmbed buffer to the host member ---
        // This might be redundant if the host state is updated correctly above,
        // but uncomment if strict synchronization is needed for subsequent operations.
        /*
        flat_host_tokenEmbed.resize(totalTokenEmbedFloats); // Ensure size
        queue.enqueueReadBuffer(d_tokenEmbed, CL_TRUE, 0, tokenEmbedBytes, flat_host_tokenEmbed.data());
        // Unflatten flat_host_tokenEmbed back into this->tokenEmbed
        this->tokenEmbed.resize(this->currentTokenCount); // Resize host vector first
        for(int tk = 0; tk < this->currentTokenCount; ++tk) {
            if (this->tokenEmbed[tk].size() != static_cast<size_t>(d)) {
                this->tokenEmbed[tk].resize(d);
            }
            size_t start_idx = static_cast<size_t>(tk) * d;
            if (start_idx + d <= flat_host_tokenEmbed.size()) {
                 std::copy(flat_host_tokenEmbed.begin() + start_idx,
                           flat_host_tokenEmbed.begin() + start_idx + d,
                           this->tokenEmbed[tk].begin());
            } else {
                 std::cerr << "Warning: Read buffer index out of bounds during unflattening in clTrain(single)." << std::endl;
                 // Handle error or fill with zeros
                 std::fill(this->tokenEmbed[tk].begin(), this->tokenEmbed[tk].end(), 0.0f);
            }
        }
        */


    } catch (const cl::Error& err) {
        std::cerr << "OpenCL Error in clTrain(single): " << err.what() << " (" << err.err() << ")" << std::endl;
        this->epochs = initial_epochs; // Reset epochs on error
        throw; // Re-throw
    } catch (const std::exception& e) {
        std::cerr << "Standard Exception in clTrain(single): " << e.what() << std::endl;
        this->epochs = initial_epochs; // Reset epochs on error
        throw; // Re-throw
    }
    // Buffer release is handled by cl::Buffer destructors (RAII)
}


/**
 * @brief (OpenCL) Train the transformer on sentences.
 *        Mirrors the logic of transformer::cuTrain(std::vector<std::vector<float>>& sentence, ...).
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
        throw std::runtime_error("clTrain(sentence): Sentence embeddings/strings mismatch or empty.");
    }
    if (!sentence.empty() && sentence[0].size() != static_cast<size_t>(d)) {
         throw std::runtime_error("clTrain(sentence): Sentence embedding dimension mismatch. Expected " + std::to_string(d) + ", got " + std::to_string(sentence[0].size()));
    }
     if (this->currentTokenCount + sentence.size() > FULL_CONTEXT) {
         throw std::runtime_error("clTrain(sentence): Adding sentence exceeds FULL_CONTEXT limit.");
     }

    cl::Buffer d_tokenEmbed, d_embeddings, d_expected_token;
    int host_indexForToken = -1;
    float current_error = 1.0f;
    int initial_epochs = this->epochs;

    try {
        // --- Device Buffer Allocation & H->D Transfer ---
        size_t totalTokenEmbedFloats = static_cast<size_t>(m) * CONTEXT_WIN * d;
        size_t tokenEmbedBytes = totalTokenEmbedFloats * sizeof(float);
        size_t embeddingsBytes = static_cast<size_t>(vocabsize) * d * sizeof(float);
        size_t singleTokenBytes = static_cast<size_t>(d) * sizeof(float);
        size_t outputBytes = singleTokenBytes; // Size of h_otok_buffer
        size_t indexBytes = sizeof(int);       // Size for the result index

        // Create buffers
        d_tokenEmbed = cl::Buffer(this->clcontext.context, CL_MEM_READ_WRITE, tokenEmbedBytes);
        d_expected_token = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY, singleTokenBytes); // Buffer for the target token

        // Prepare initial context buffer content
        std::vector<float> flat_host_tokenEmbed;
        flat_host_tokenEmbed.reserve(totalTokenEmbedFloats);
        for(int tk = 0; tk < this->currentTokenCount; ++tk) { // Copy existing context
             if (tk < this->tokenEmbed.size() && this->tokenEmbed[tk].size() == static_cast<size_t>(d)) {
                flat_host_tokenEmbed.insert(flat_host_tokenEmbed.end(), this->tokenEmbed[tk].begin(), this->tokenEmbed[tk].end());
             } else {
                 std::cerr << "Warning: Inconsistent tokenEmbed data at index " << tk << " during clTrain(sentence) setup." << std::endl;
                 std::vector<float> padding(d, 0.0f);
                 flat_host_tokenEmbed.insert(flat_host_tokenEmbed.end(), padding.begin(), padding.end());
             }
        }
        // Add the first token of the sentence to the initial buffer content
        if (!sentence.empty()) {
            flat_host_tokenEmbed.insert(flat_host_tokenEmbed.end(), sentence[0].begin(), sentence[0].end());
        }
        // Pad the rest of the buffer
        flat_host_tokenEmbed.resize(totalTokenEmbedFloats, 0.0f);
        // Write initial buffer content to device
        this->clcontext.queue.enqueueWriteBuffer(d_tokenEmbed, CL_TRUE, 0, tokenEmbedBytes, flat_host_tokenEmbed.data());

        // Flatten and copy embeddings table
        std::vector<float> flat_embeddings = flatten(this->embeddings);
        d_embeddings = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, embeddingsBytes, flat_embeddings.data());


        // --- Initialize Host State ---
        // Add the first token to the host context tracking *before* the loop
        if (!sentence.empty()) {
            int previousTokenCount = this->currentTokenCount;
            if (previousTokenCount < this->tokenEmbed.size()) {
                 this->tokenEmbed[previousTokenCount] = sentence[0];
            } else {
                 this->tokenEmbed.push_back(sentence[0]);
            }
            this->currentTokenCount++;
            // Resize if needed
            if (this->tokenEmbed.size() != static_cast<size_t>(this->currentTokenCount)) {
                 this->tokenEmbed.resize(this->currentTokenCount);
            }
        }
        // Set initial promptCount and blockCount based on the state *after* adding the first token
        this->blockCount = (this->currentTokenCount == 0) ? 1 : ((this->currentTokenCount - 1) / CONTEXT_WIN) + 1;
        this->promptCount = this->currentTokenCount % CONTEXT_WIN;
        if (this->promptCount == 0 && this->currentTokenCount > 0) this->promptCount = CONTEXT_WIN;


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
            this->clcontext.queue.enqueueWriteBuffer(d_expected_token, CL_TRUE, 0, singleTokenBytes, expected_vec.data());

            int effective_context_size = this->currentTokenCount; // Context size *before* adding token i
            int current_block_idx = this->blockCount; // Block index based on current context size

            if (current_block_idx <= 0 || current_block_idx > m) {
                 throw std::out_of_range("clTrain(sentence): Calculated current_block_idx (" + std::to_string(current_block_idx) + ") is out of range [1, " + std::to_string(m) + "].");
            }

            // --- Training Loop for token i ---
            int j = 0; // Epoch counter for this token
            current_error = 1.0f;
            host_indexForToken = -1;

            while (j <= this->epochs) {
                // --- Forward Pass ---
                if (this->inTraining) {
                    // clParallelKdotQs(...); // Placeholder
                }
                clForward(current_block_idx, effective_context_size, promptCount);

                // --- Get EH output ---
                std::vector<float> h_otok_buffer;
                 if (current_block_idx > 0 && current_block_idx <= m) {
                    h_otok_buffer = t[current_block_idx - 1].EH; // Get host copy
                     if (h_otok_buffer.size() != static_cast<size_t>(d)) {
                         throw std::runtime_error("clTrain(sentence): EH buffer from block " + std::to_string(current_block_idx) + " has incorrect size.");
                     }
                 } else {
                     // This case should have been caught earlier, but double-check
                     throw std::runtime_error("clTrain(sentence): Invalid block index (" + std::to_string(current_block_idx) + ") before clComputeOutput.");
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
                        std::cerr << "OpenCL Error during inline clComputeOutput in clTrain(sentence): " << err.what() << " (" << err.err() << ")" << std::endl;
                        host_indexForToken = -1; // Indicate error
                        throw; // Re-throw
                    }
                    // Buffers d_output, d_result_index released by RAII
                }
                // --- End: Inline clComputeOutput Logic ---

                current_error = errorofv(h_otok_buffer, expected_vec);

                // --- Convergence Check ---
                bool converged = (current_error < 0.01);
                if (!converged && host_indexForToken >= 0 && host_indexForToken < static_cast<int>(tokens.size())) {
                    converged = (tokens[host_indexForToken] == expected_str);
                }

                if (converged) {
                    // Converged: Update device token buffer with the *expected* value at the current position
                    size_t offset_bytes = static_cast<size_t>(effective_context_size) * d * sizeof(float);
                     if (offset_bytes + singleTokenBytes > tokenEmbedBytes) {
                        throw std::out_of_range("clTrain(sentence): Calculated offset for writing converged token exceeds buffer bounds.");
                     }
                     this->clcontext.queue.enqueueWriteBuffer(d_tokenEmbed, CL_TRUE, offset_bytes, singleTokenBytes, expected_vec.data());
                    break; // Exit training loop for this token
                }

                // --- Adjust Epochs ---
                if (current_error >= 0.01 && j == this->epochs) {
                    bool predicted_matches = (host_indexForToken >= 0 && host_indexForToken < static_cast<int>(tokens.size()) && tokens[host_indexForToken] == expected_str);
                    if (!predicted_matches) {
                        this->epochs += 10;
                    } else {
                        // String matches, break even if error is high
                         size_t offset_bytes = static_cast<size_t>(effective_context_size) * d * sizeof(float);
                         if (offset_bytes + singleTokenBytes > tokenEmbedBytes) {
                             throw std::out_of_range("clTrain(sentence): Calculated offset for writing converged token exceeds buffer bounds.");
                         }
                         this->clcontext.queue.enqueueWriteBuffer(d_tokenEmbed, CL_TRUE, offset_bytes, singleTokenBytes, expected_vec.data());
                         break;
                    }
                }

                // --- Backward Pass ---
                clBackward(expected_vec, current_block_idx);

                j++;
            } // End training loop for token i

            // --- Update Host State ---
            this->trainCount++;
            this->epochCount += j;
            this->error += current_error;

            // Add the *converged/expected* token to the host context tracking
            int previousTokenCount = this->currentTokenCount;
            if (previousTokenCount < this->tokenEmbed.size()) {
                 this->tokenEmbed[previousTokenCount] = expected_vec;
            } else {
                 this->tokenEmbed.push_back(expected_vec);
            }
            this->currentTokenCount++;
            // Resize if needed
            if (this->tokenEmbed.size() != static_cast<size_t>(this->currentTokenCount)) {
                 this->tokenEmbed.resize(this->currentTokenCount);
            }

            // Update blockCount and promptCount for the *next* iteration
            this->blockCount = (this->currentTokenCount == 0) ? 1 : ((this->currentTokenCount - 1) / CONTEXT_WIN) + 1;
            this->promptCount = this->currentTokenCount % CONTEXT_WIN;
            if (this->promptCount == 0 && this->currentTokenCount > 0) this->promptCount = CONTEXT_WIN;

        } // End loop over sentence tokens (i=1 to N-1)

        // --- Optional: Final readback of tokenEmbed buffer ---
        /*
        flat_host_tokenEmbed.resize(totalTokenEmbedFloats);
        queue.enqueueReadBuffer(d_tokenEmbed, CL_TRUE, 0, tokenEmbedBytes, flat_host_tokenEmbed.data());
        // Unflatten logic... (similar to clTrain(single))
        */

    } catch (const cl::Error& err) {
        std::cerr << "OpenCL Error in clTrain(sentence): " << err.what() << " (" << err.err() << ")" << std::endl;
        this->epochs = initial_epochs;
        throw;
    } catch (const std::exception& e) {
        std::cerr << "Standard Exception in clTrain(sentence): " << e.what() << std::endl;
        this->epochs = initial_epochs;
        throw;
    }
    // Buffers released by RAII
}


/**
 * @brief (OpenCL) Train the transformer for prompt and response.
 *        Mirrors the logic of transformer::cuTrain(..., prompt, ..., response, ...).
 * @param prompt Prompt token embeddings (on host).
 * @param response Response token embeddings (on host).
 * @param rString Tokens of the response (on host).
 */
void transformer::clTrain(std::vector<std::vector<float>>& prompt, std::vector<std::vector<float>>& response, std::vector<std::string>& rString)
{
    // --- Basic validation ---
    if (prompt.empty()) {
        throw std::runtime_error("clTrain(prompt-response): Initial prompt cannot be empty.");
    }
    // Warning for large prompts, but allow up to CONTEXT_WIN
    if (prompt.size() > CONTEXT_WIN) {
        std::cerr << "Warning: Prompt size (" << prompt.size() << ") exceeds context window (" << CONTEXT_WIN << "). Ensure this is intended." << std::endl;
        // Consider truncating prompt if it exceeds CONTEXT_WIN and multi-block prompts aren't handled.
        // For now, assume it fits or multi-block processing handles it.
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
    int host_indexForToken = -1;
    float current_error = 1.0f;
    int initial_token_count = this->currentTokenCount; // Store initial count
    int initial_epochs = this->epochs;

    try {
        // --- Device Buffer Allocation & H->D Transfer ---
        size_t totalTokenEmbedFloats = static_cast<size_t>(m) * CONTEXT_WIN * d;
        size_t tokenEmbedBytes = totalTokenEmbedFloats * sizeof(float);
        size_t embeddingsBytes = static_cast<size_t>(vocabsize) * d * sizeof(float);
        size_t singleTokenBytes = static_cast<size_t>(d) * sizeof(float);
        size_t outputBytes = singleTokenBytes; // Size of h_otok_buffer
        size_t indexBytes = sizeof(int);       // Size for the result index

        // Create buffers
        d_tokenEmbed = cl::Buffer(this->clcontext.context, CL_MEM_READ_WRITE, tokenEmbedBytes);
        d_expected_response_token = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY, singleTokenBytes);

            // If currentTokenCount > 0, copy existing host tokenEmbed H->D
            // std::vector<float> initial_flat_context = flatten_range(this->tokenEmbed, 0, initial_token_count);
            // this->clcontext.queue.enqueueWriteBuffer(d_tokenEmbed, CL_TRUE, 0, initial_flat_context.size() * sizeof(float), initial_flat_context.data());

        // Prepare initial context buffer content (existing context)
        std::vector<float> flat_host_tokenEmbed;
        flat_host_tokenEmbed.reserve(totalTokenEmbedFloats);
        for(int tk = 0; tk < this->currentTokenCount; ++tk) { // Copy existing context
             if (tk < this->tokenEmbed.size() && this->tokenEmbed[tk].size() == static_cast<size_t>(d)) {
                 flat_host_tokenEmbed.insert(flat_host_tokenEmbed.end(), this->tokenEmbed[tk].begin(), this->tokenEmbed[tk].end());
             } else {
                  std::cerr << "Warning: Inconsistent tokenEmbed data at index " << tk << " during clTrain(prompt-response) setup." << std::endl;
                 std::vector<float> padding(d, 0.0f);
                 flat_host_tokenEmbed.insert(flat_host_tokenEmbed.end(), padding.begin(), padding.end());
             }
        }
        // Pad the rest (up to where the prompt will start)
        flat_host_tokenEmbed.resize(static_cast<size_t>(this->currentTokenCount) * d, 0.0f);

        // Write existing context to device
        if (!flat_host_tokenEmbed.empty()) {
            this->clcontext.queue.enqueueWriteBuffer(d_tokenEmbed, CL_TRUE, 0, flat_host_tokenEmbed.size() * sizeof(float), flat_host_tokenEmbed.data());
        }

        // Flatten and copy embeddings table
        std::vector<float> flat_embeddings = flatten(this->embeddings);
        d_embeddings = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, embeddingsBytes, flat_embeddings.data());


        // --- Process Prompt (Add to context on Host and Device) ---
            if (initial_token_count + prompt.size() > CONTEXT_WIN) {
                 throw std::runtime_error("clTrain(prompt, response): Prompt exceeds first block capacity when starting.");
            }

        for (size_t p = 0; p < prompt.size(); ++p) {
            // Update device buffer
                size_t offset_bytes = static_cast<size_t>(initial_token_count + p) * d * sizeof(float);
             if (offset_bytes + singleTokenBytes > tokenEmbedBytes) {
                 throw std::out_of_range("clTrain(prompt-response): Offset exceeds buffer bounds when writing prompt token.");
             }
             this->clcontext.queue.enqueueWriteBuffer(d_tokenEmbed, CL_TRUE, offset_bytes, singleTokenBytes, prompt[p].data());

            // Update host tracking
                if (static_cast<size_t>(initial_token_count + p) < this->tokenEmbed.size()) {
                     this->tokenEmbed[initial_token_count + p] = prompt[p];
            } else {
                 this->tokenEmbed.push_back(prompt[p]);
            }
            this->currentTokenCount++;
            }
            // Resize host vector if push_back was used
            if (this->tokenEmbed.size() != static_cast<size_t>(this->currentTokenCount)) {
                this->tokenEmbed.resize(this->currentTokenCount);
            }

            // Copy prompt D->D from d_tokenEmbed into d_EV of each head in block 0
            size_t prompt_bytes = prompt.size() * d * sizeof(float);
            size_t prompt_start_offset_bytes = initial_token_count * d * sizeof(float);
            for (int i = 0; i < x; ++i) { // Layers
                for (int j = 0; j < y; ++j) { // Parallels
                    cl::Buffer& d_head_ev = t[0].b[i][j].getDeviceEVBuffer(); // Assuming getter exists
                    size_t dest_offset_bytes = initial_token_count * d * sizeof(float);
                    // Add size checks if necessary
                    this->clcontext.queue.enqueueCopyBuffer(d_tokenEmbed, d_head_ev, prompt_start_offset_bytes, dest_offset_bytes, prompt_bytes);
                }
        }
        // Update blockCount and promptCount *after* processing the entire prompt
        this->blockCount = (this->currentTokenCount == 0) ? 1 : ((this->currentTokenCount - 1) / CONTEXT_WIN) + 1;
        // promptCount here refers to the number of tokens *within the current block* that are part of the prompt being processed *now*.
        // This definition seems ambiguous compared to the single-token training.
        // Let's follow the logic from cuTrain: promptCount is the size of the prompt just added.
        this->promptCount = prompt.size();
        // Adjust promptCount if it spans block boundaries? The CUDA code doesn't show this adjustment.
        // Let's assume promptCount is just the length of the prompt added in this call.


        // --- Train for Response ---
        for (size_t i = 0; i < response.size(); ++i) {
             if (this->currentTokenCount >= FULL_CONTEXT) {
                 std::cerr << "Warning: clTrain(prompt-response) reached FULL_CONTEXT limit ("
                           << this->currentTokenCount << ") during response. Stopping training early at response index " << i << "." << std::endl;
                 break;
             }

            std::vector<float>& expected_vec = response[i];
            std::string& expected_str = rString[i];

            // Copy target token H->D
            this->clcontext.queue.enqueueWriteBuffer(d_expected_response_token, CL_TRUE, 0, singleTokenBytes, expected_vec.data());

            int effective_context_size = this->currentTokenCount; // Context size *before* adding response[i]
            int current_block_idx = this->blockCount; // Block index based on current context size

             if (current_block_idx <= 0 || current_block_idx > m) {
                 // If strictly training block 1, this should ideally be checked earlier or always be 1
                 throw std::out_of_range("clTrain(prompt-response): Calculated current_block_idx (" + std::to_string(current_block_idx) + ") is out of range [1, " + std::to_string(m) + "].");
             }

            // --- Training Loop for response token i ---
            int j = 0; // Epoch counter
            current_error = 1.0f;
            host_indexForToken = -1;
            // --- Get EH output (Host copy for prediction setup) ---
            std::vector<float> h_otok_buffer;
            while (j <= this->epochs) {
                // --- Forward Pass ---
                 if (this->inTraining) {
                    // clParallelKdotQs(...); // Placeholder
                }
                // Pass the promptCount relevant for the *current* block/context state
                int current_prompt_count_in_block = effective_context_size % CONTEXT_WIN;
                 if (current_prompt_count_in_block == 0 && effective_context_size > 0) current_prompt_count_in_block = CONTEXT_WIN;

                clForward(current_block_idx, effective_context_size, current_prompt_count_in_block);

                if (current_block_idx > 0 && current_block_idx <= m) {
                    h_otok_buffer = t[current_block_idx - 1].EH; // Get host copy
                    // TODO: Ideally clForward updates a device buffer, and we use that directly.
                    // This H->D copy below is inefficient but matches the structure.
                    if (h_otok_buffer.size() != static_cast<size_t>(d)) {
                        throw std::runtime_error("clTrain(prompt-response): EH buffer from block " + std::to_string(current_block_idx) + " has incorrect size.");
                    }
                } 
                else {
                    throw std::runtime_error("clTrain(prompt-response): Invalid block index (" + std::to_string(current_block_idx) + ") before clComputeOutput.");
                }
                cl::Buffer d_output, d_result_index;
                // --- Compute Prediction & Error ---
                // --- Start: Inline clComputeOutput Logic ---
                { // Scope for temporary compute output buffers
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
                        std::cerr << "OpenCL Error during inline clComputeOutput in clTrain(prompt-response): " << err.what() << " (" << err.err() << ")" << std::endl;
                        host_indexForToken = -1; // Indicate error
                        throw; // Re-throw
                    }
                    // Buffers d_output, d_result_index released by RAII
                }
                // --- End: Inline clComputeOutput Logic ---

                current_error = errorofv(h_otok_buffer, expected_vec);

                // --- Convergence Check ---
                bool converged = (current_error < 0.01);
                if (!converged && host_indexForToken >= 0 && host_indexForToken < static_cast<int>(tokens.size())) {
                    converged = (tokens[host_indexForToken] == expected_str);
                }

                if (converged) {
                    // CPU Logic: Update context with predicted EH
                    size_t offset_bytes = static_cast<size_t>(effective_context_size) * d * sizeof(float);
                     if (offset_bytes + singleTokenBytes > tokenEmbedBytes) {
                         throw std::out_of_range("clTrain(prompt-response): Offset exceeds buffer bounds when writing converged response token.");
                     }
                     this->clcontext.queue.enqueueCopyBuffer(d_output, d_tokenEmbed, 0, offset_bytes, outputBytes); // Copy from d_output (predicted EH) D->D
                    break; // Exit training loop
                }

                // --- Adjust Epochs ---
                if (current_error >= 0.01 && j == this->epochs) {
                     bool predicted_matches = (host_indexForToken >= 0 && host_indexForToken < static_cast<int>(tokens.size()) && tokens[host_indexForToken] == expected_str);
                     if (!predicted_matches) {
                        this->epochs += 10;
                     } else {
                        // String matches, break even if error is high
                         size_t offset_bytes = static_cast<size_t>(effective_context_size) * d * sizeof(float);
                         if (offset_bytes + outputBytes > tokenEmbedBytes) { // Use outputBytes
                             throw std::out_of_range("clTrain(prompt-response): Offset exceeds buffer bounds when writing converged response token.");
                         }
                         this->clcontext.queue.enqueueCopyBuffer(d_output, d_tokenEmbed, 0, offset_bytes, outputBytes); // Copy from d_output (predicted EH) D->D
                        break;
                     }
                }

                // --- Backward Pass ---
                clBackward(expected_vec, current_block_idx);

                j++;
            } // End training loop for response token i

            // --- Update Host State ---
            this->trainCount++;
            this->epochCount += j;
            this->error += current_error;

            // Add the *converged/expected* token to the host context tracking
            // CPU Logic: Update host context with predicted EH (copied D->H)
            if (static_cast<size_t>(effective_context_size) < this->tokenEmbed.size()) {
                 this->tokenEmbed[effective_context_size] = h_otok_buffer; // Use the h_otok_buffer (predicted EH)
            } else { // Should not happen if size is managed correctly
                 this->tokenEmbed.push_back(expected_vec);
            }
            this->currentTokenCount++;
             if (this->tokenEmbed.size() != static_cast<size_t>(this->currentTokenCount)) {
                 this->tokenEmbed.resize(this->currentTokenCount);
             }

            // Update blockCount and promptCount for the *next* iteration
            this->blockCount = (this->currentTokenCount == 0) ? 1 : ((this->currentTokenCount - 1) / CONTEXT_WIN) + 1;
            this->promptCount = this->currentTokenCount % CONTEXT_WIN; // Count within the current block
            if (this->promptCount == 0 && this->currentTokenCount > 0) this->promptCount = CONTEXT_WIN;

        } // End loop over response tokens

        // --- Optional: Final readback of tokenEmbed buffer ---
        /*
        flat_host_tokenEmbed.resize(totalTokenEmbedFloats);
        queue.enqueueReadBuffer(d_tokenEmbed, CL_TRUE, 0, tokenEmbedBytes, flat_host_tokenEmbed.data());
        // Unflatten logic...
        */

    } catch (const cl::Error& err) {
        std::cerr << "OpenCL Error in clTrain(prompt-response): " << err.what() << " (" << err.err() << ")" << std::endl;
        this->epochs = initial_epochs;
        throw;
    } catch (const std::exception& e) {
        std::cerr << "Standard Exception in clTrain(prompt-response): " << e.what() << std::endl;
        this->epochs = initial_epochs;
        throw;
    }
    // Buffers released by RAII
} // End clTrain(prompt, response, rString)


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
        throw std::runtime_error("clTrain(chat): Total tokens required (" + std::to_string(this->currentTokenCount + total_tokens_to_add)
                                 + ") exceeds FULL_CONTEXT (" + std::to_string(FULL_CONTEXT) + ").");
    }

    // --- Train for each chat turn ---
    for (size_t i = 0; i < prompts.size(); ++i) {
        try {
            // Call the prompt-response trainer for each turn
            // Ensure the arguments are correctly passed (non-const references)
            clTrain(prompts[i], responses[i], rString[i]);
        } catch (const std::exception& e) {
            // Log error and re-throw to halt chat training on error
            std::cerr << "Error during chat training turn " << i << ": " << e.what() << std::endl;
            throw; // Re-throw by default
        }
    }
}


/**
 * @brief (OpenCL) Computes the predicted token index based on the output embedding.
 *        This is a host-side wrapper that uses the 'compute_prediction' OpenCL kernel.
 * @param output The final output embedding vector (EH) from the transformer (host).
 * @param embeddings The full token embedding table (host, shape [vocabsize][d]).
 * @param voc Vocabulary size.
 * @param index Output parameter to store the predicted token index.
 */
void transformer::clComputeOutput(std::vector<float>& output, std::vector<std::vector<float>>& embeddings, int& voc, int& index) {
    // --- Validation ---
    if (output.size() != static_cast<size_t>(d)) {
        throw std::runtime_error("clComputeOutput: Output vector dimension mismatch. Expected " + std::to_string(d) + ", got " + std::to_string(output.size()));
    }
    if (embeddings.empty() || embeddings.size() != static_cast<size_t>(voc)) {
         throw std::runtime_error("clComputeOutput: Embeddings table row count mismatch. Expected " + std::to_string(voc) + ", got " + std::to_string(embeddings.size()));
    }
    if (voc > 0 && embeddings[0].size() != static_cast<size_t>(d)) {
        throw std::runtime_error("clComputeOutput: Embeddings table column count (dimension) mismatch. Expected " + std::to_string(d) + ", got " + std::to_string(embeddings[0].size()));
    }
    if (voc <= 0) {
        std::cerr << "Warning: clComputeOutput called with vocabulary size <= 0." << std::endl;
        index = -1; // Indicate error or undefined result
        return;
    }


    cl::Buffer d_output, d_embeddings_compute, d_result_index; // Use a different name for embeddings buffer if needed
    int result_index_val = -1; // Initialize with an invalid index

    try {
        // --- Allocate Buffers ---
        size_t outputBytes = output.size() * sizeof(float);
        size_t embeddingsBytes = static_cast<size_t>(voc) * d * sizeof(float);
        size_t indexBytes = sizeof(int);

        // Create and copy output (EH) vector
        d_output = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, outputBytes, output.data());

        // Flatten and copy embeddings table
        std::vector<float> flat_embeddings = flatten(embeddings); // Flatten [voc][d] -> [voc*d]
        d_embeddings_compute = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, embeddingsBytes, flat_embeddings.data());

        // Create buffer for the kernel to write the result index
        d_result_index = cl::Buffer(this->clcontext.context, CL_MEM_WRITE_ONLY, indexBytes);

        // --- Get and Set Kernel Arguments ---
        cl::Kernel kernel = this->clcontext.kernels.at("compute_prediction");
        kernel.setArg(0, d_output);
        kernel.setArg(1, d_embeddings_compute); // Pass the correct buffer
        kernel.setArg(2, static_cast<cl_int>(d)); // dim
        kernel.setArg(3, static_cast<cl_int>(voc)); // voc
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
        index = result_index_val;

    } catch (const cl::Error& err) {
        // Catch OpenCL specific errors
        std::cerr << "OpenCL Error in clComputeOutput: " << err.what() << " (" << err.err() << ")" << std::endl;
        index = -1; // Indicate error
        throw; // Re-throw
    } catch (const std::exception& e) {
        // Catch standard C++ exceptions (e.g., from flatten, getKernel)
        std::cerr << "Standard Exception in clComputeOutput: " << e.what() << std::endl;
        index = -1; // Indicate error
        throw; // Re-throw
    }
    // Buffers d_output, d_embeddings_compute, d_result_index are automatically released by RAII
}


#endif // USE_OPENCL
