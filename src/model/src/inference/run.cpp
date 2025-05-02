
// model.cpp: implementation of Model class
#include "include/model.hpp"
#include <iostream>
#include <stdexcept>
#include <filesystem>
#include <neural.hpp>

#ifdef USE_CUDA
    #define CUDA_CHECK(call) do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            throw std::runtime_error(cudaGetErrorString(err)); \
        } \
    } while (0)
#endif

/**
 * @brief run model for conversation
 */
void model::runModel()
{
    std::cout << "You are now running the model " << info.modelName << std::endl;
    bool savechat;      // 1 to save chat
    bool newchat;       // 1 for new chat, 0 for endchat
    // take input
    T.currentTokenCount = 0;
    while(T.currentTokenCount < FULL_CONTEXT) {
        std::cout << "Enter prompt: "; // Use 'total' member variable
        takeInput();
        if (this->chat != nullptr) {
            // Check if the file pointer is valid
            fprintf(this->chat, "Response:\n");
            fflush(this->chat); // Ensure it's written immediately (optional but good for logging)
        }
        #ifdef USE_CUDA
            // use cuda
            int rCount = 0;
            auto start_time = std::chrono::high_resolution_clock::now();
            std::cout << "\nResponse: ";
            // --- Device Memory Allocation ---
            float* d_tokenEmbed = nullptr;      // Holds all token embeddings (prompt + generated) up to FULL_CONTEXT
            float* d_embeddings = nullptr;      // Holds the vocabulary embeddings
            float* d_EVuse = nullptr;           // Holds the EV state from the previous block (flattened: x * y * CONTEXT_WIN * d)
            float* d_tokForBlock = nullptr;     // Holds the token embeddings for the current block's context window
            float* d_otok = nullptr;            // Holds the output token embedding from cuForward (size d)

            // Determine sizes
            size_t full_context_bytes = static_cast<size_t>(FULL_CONTEXT) * d * sizeof(float);
            size_t vocab_bytes = static_cast<size_t>(this->T.vocabsize) * d * sizeof(float);
            size_t context_win_bytes = static_cast<size_t>(CONTEXT_WIN) * d * sizeof(float);
            // EVuse needs space for all attention heads' EV states for a full context window
            size_t ev_use_bytes = static_cast<size_t>(x) * y * CONTEXT_WIN * d * sizeof(float);
            size_t otok_bytes = static_cast<size_t>(d) * sizeof(float);

            CUDA_CHECK(cudaMalloc(&d_tokenEmbed, full_context_bytes));
            CUDA_CHECK(cudaMalloc(&d_embeddings, vocab_bytes));
            CUDA_CHECK(cudaMalloc(&d_EVuse, ev_use_bytes)); // Allocate space for EV context passing
            CUDA_CHECK(cudaMalloc(&d_tokForBlock, context_win_bytes));
            CUDA_CHECK(cudaMalloc(&d_otok, otok_bytes)); // d_otok is managed internally by cuForward now

            // --- Initial Data Transfer ---
            // Flatten and copy vocabulary embeddings H->D (once)
            std::vector<float> flat_embeddings;
            flat_embeddings.reserve(static_cast<size_t>(this->T.vocabsize) * d);
            for (const auto& embed : this->T.embeddings) {
                if (embed.size() != static_cast<size_t>(d)) throw std::runtime_error("Inconsistent embedding dimension in vocabulary.");
                flat_embeddings.insert(flat_embeddings.end(), embed.begin(), embed.end());
            }
            CUDA_CHECK(cudaMemcpy(d_embeddings, flat_embeddings.data(), vocab_bytes, cudaMemcpyHostToDevice));
            flat_embeddings.clear(); // Free host memory
            int previousTokenCount = this->T.currentTokenCount; // Store count before adding prompt

            // Get embeddings for the prompt tokens and copy H->D incrementally
            std::vector<float> prompt_embeddings_flat;
            prompt_embeddings_flat.reserve(static_cast<size_t>(this->T.promptCount) * d);
            for (int i = 0; i < this->T.promptCount; ++i) {
                std::vector<float> current_embed(d);
                // Index in mTokens is previousTokenCount + i
                this->T.getEmbedding(this->T.mTokens[previousTokenCount + i], current_embed);
                prompt_embeddings_flat.insert(prompt_embeddings_flat.end(), current_embed.begin(), current_embed.end());
            }
            // Copy the new prompt embeddings to the correct offset in d_tokenEmbed
            CUDA_CHECK(cudaMemcpy(d_tokenEmbed + static_cast<size_t>(previousTokenCount) * d,
                                prompt_embeddings_flat.data(),
                                prompt_embeddings_flat.size() * sizeof(float),
                                cudaMemcpyHostToDevice));
            prompt_embeddings_flat.clear(); // Free host memory


            if (this->T.currentTokenCount >= FULL_CONTEXT) {
                throw std::runtime_error("TOKEN LIMIT REACHED AT FULL CONTEXT during prompt processing!");
                break; // Should be unreachable due to throw
            }

            // --- Prompt Placement & Initial KdotQ (Mirroring CPU logic) ---
            // Calculate offset within the current block's window
            int c = (this->T.blockCount == 0) ? previousTokenCount : (previousTokenCount % CONTEXT_WIN);

            int tokens_in_current_block_before_prompt = c;
            int space_in_current_block = CONTEXT_WIN - tokens_in_current_block_before_prompt;

            if (this->T.promptCount <= space_in_current_block) {
                // Case 1: Prompt fits entirely within the current block
                if (this->T.blockCount > 0) {
                    // Copy relevant part of d_tokenEmbed to d_tokForBlock if needed by cuParallelKdotQs
                    // This depends on how cuParallelKdotQs expects input for non-first blocks.
                    // Assuming it uses d_tokenEmbed directly with offsets.
                }
                // Pre-calculate KdotQ for the prompt tokens added to the current block
                // We need to tell cuParallelKdotQs how many *new* tokens (promptCount) were added
                // starting at which index (previousTokenCount) within the full context.
                for (int col = 0; col < y; ++col) {
                    // Pass the count of *new* prompt tokens and the total count *before* the prompt
                    int effectivePromptCount = this->T.promptCount;
                    this->T.cuParallelKdotQs(effectivePromptCount, previousTokenCount, this->T.blockCount, col, isSelf, this->T.inTraining);
                }
                CUDA_CHECK(cudaDeviceSynchronize()); // Ensure KdotQ calculation is done
            } 
            else {
                // Case 2: Prompt spans across the current and next block
                int m2 = space_in_current_block; // Tokens fitting in the current block
                int m1 = this->T.promptCount - m2;       // Tokens going to the next block

                // Process the first part (m2 tokens) in the current block
                if (m2 > 0) {
                    for (int col = 0; col < y; ++col) {
                        // Calculate KdotQ for the m2 tokens added
                        int effectivePromptCount = m2;
                        this->T.cuParallelKdotQs(effectivePromptCount, previousTokenCount, this->T.blockCount, col, isSelf, this->T.inTraining);
                    }
                    CUDA_CHECK(cudaDeviceSynchronize()); // Ensure KdotQ for m2 is done
                }

                // --- Transition to the next block ---
                // Copy the EV state from the completed block (t[0]) to d_EVuse (D->D)
                size_t head_ev_bytes = static_cast<size_t>(CONTEXT_WIN) * d * sizeof(float);

                // Check if the total size matches the allocated buffer size
                if (static_cast<size_t>(x) * y * head_ev_bytes != ev_use_bytes) {
                    throw std::runtime_error("Mismatch between calculated EVuse size and allocated buffer size during block transition.");
                }

                // Loop through each attention head in the completed block (t[0])
                for (int i = 0; i < x; ++i) { // Iterate through layers (rows)
                    for (int j = 0; j < y; ++j) { // Iterate through parallels (columns)
                        try {
                            // Get the device pointer for the EV state of the current head (i, j)
                            float* d_src_ev_ptr = this->T.t[0].b[i][j].d_EV; // Directly access public member

                            // Calculate the destination offset in the flattened d_EVuse buffer
                            size_t dest_offset_elements = (static_cast<size_t>(i) * y + j) * CONTEXT_WIN * d;

                            // Boundary check (optional but good practice)
                            if (dest_offset_elements * sizeof(float) + head_ev_bytes > ev_use_bytes) {
                                throw std::out_of_range("EVuse destination offset out of bounds for head [" +
                                                        std::to_string(i) + "][" + std::to_string(j) + "].");
                            }

                            // Enqueue the device-to-device memory copy
                            CUDA_CHECK(cudaMemcpy(d_EVuse + dest_offset_elements, // Destination pointer
                                                d_src_ev_ptr,                  // Source pointer
                                                head_ev_bytes,                 // Size in bytes
                                                cudaMemcpyDeviceToDevice));    // Type of copy

                        }
                        catch (const std::exception& e) {
                            // Catch potential errors from getDeviceEVPointer() or other issues
                            std::cerr << "Error getting/copying EV pointer for head [" << i << "][" << j << "]: " << e.what() << std::endl;
                            throw; // Re-throw standard exceptions
                        }
                    } // End loop columns (j)
                } // End loop rows (i)

                // Ensure all copy operations are completed before proceeding
                CUDA_CHECK(cudaDeviceSynchronize());

                this->T.blockCount += 1; // Increment block count *once*
                int start_idx_for_new_block = this->T.currentTokenCount - CONTEXT_WIN; // Start index in d_tokenEmbed
                if (start_idx_for_new_block < 0) start_idx_for_new_block = 0; // Should not happen if logic is correct

                CUDA_CHECK(cudaMemcpy(d_tokForBlock,
                                    d_tokenEmbed + static_cast<size_t>(start_idx_for_new_block) * d,
                                    context_win_bytes,
                                    cudaMemcpyDeviceToDevice));


                // Pre-calculate KdotQ for the second part (m1 tokens) in the *new* block
                if (m1 > 0) {
                    // The "previous token count" for this calculation is effectively the start of the new block
                    int start_of_new_block_count = this->T.currentTokenCount - m1;
                    for (int col = 0; col < y; ++col) {
                        int effectivePromptCount = m1;
                        // Pass the new blockCount
                        this->T.cuParallelKdotQs(effectivePromptCount, start_of_new_block_count, this->T.blockCount, col, isSelf, 
                            this->T.inTraining);
                    }
                    CUDA_CHECK(cudaDeviceSynchronize()); // Ensure KdotQ for m1 is done
                }
            }

            std::cout << "Response: ";
            while (1) {
                // --- Core Forward Pass ---
                int generationPromptCount = 0; // Indicate we are in generation phase
                this->T.cuForward(this->T.blockCount, this->T.currentTokenCount, generationPromptCount); // This updates t[0]'s internal state (EH, EV) on device
                CUDA_CHECK(cudaDeviceSynchronize()); // Ensure forward pass is complete

                // --- Get Output Token ---
                cuComputeOutput(d_otok, d_embeddings, this->T.vocabsize, this->T.indexForToken, d);
                CUDA_CHECK(cudaDeviceSynchronize()); // Ensure output computation is done

                // --- Update State (Host & Device) ---
                if (this->T.indexForToken < 0 || this->T.indexForToken >= this->T.vocabsize) {
                    std::cerr << "\nError: Invalid token index computed: " << this->T.indexForToken << std::endl;
                    break; // Exit generation loop on error
                }

                // Get the embedding for the new token on the host
                const std::vector<float>& next_token_embed_h = this->T.embeddings[this->T.indexForToken];
                // Copy the new embedding H->D into d_tokenEmbed at the current position
                CUDA_CHECK(cudaMemcpy(d_tokenEmbed + static_cast<size_t>(this->T.currentTokenCount) * d,
                                    next_token_embed_h.data(),
                                    static_cast<size_t>(d) * sizeof(float),
                                    cudaMemcpyHostToDevice));

                // Store the token string (host)
                this->T.mTokens[this->T.currentTokenCount] = this->T.tokens[this->T.indexForToken];

                // Print the token (host)
                std::cout << this->T.mTokens[this->T.currentTokenCount] << " " << std::flush;

                // Increment token count (host)
                this->T.currentTokenCount += 1;
                rCount += 1;

                // --- Context Window / Block Transition Check ---
                if (this->T.currentTokenCount % CONTEXT_WIN == 0 && this->T.currentTokenCount > 0) {
                    // Copy the EV state from the completed block (t[0]) to d_EVuse (D->D)
                    size_t head_ev_bytes = static_cast<size_t>(CONTEXT_WIN) * d * sizeof(float);
                    if (static_cast<size_t>(x) * y * head_ev_bytes != ev_use_bytes) {
                        throw std::runtime_error("Mismatch between calculated EVuse size and allocated buffer size during block transition in generation.");
                    }

                    for (int i = 0; i < x; ++i) { // Iterate through layers (rows)
                        for (int j = 0; j < y; ++j) { // Iterate through parallels (columns)
                            try {
                                float* d_src_ev_ptr = this->T.t[0].b[i][j].d_EV; // Directly access public member
                                size_t dest_offset_elements = (static_cast<size_t>(i) * y + j) * CONTEXT_WIN * d;

                                if (dest_offset_elements * sizeof(float) + head_ev_bytes > ev_use_bytes) {
                                    throw std::out_of_range("EVuse destination offset out of bounds during generation for head [" +
                                                            std::to_string(i) + "][" + std::to_string(j) + "].");
                                }

                                CUDA_CHECK(cudaMemcpy(d_EVuse + dest_offset_elements, d_src_ev_ptr, head_ev_bytes, cudaMemcpyDeviceToDevice));
                            }
                            catch (const std::exception& e) {
                                std::cerr << "Error getting/copying EV pointer during generation for head [" << i << "][" << j << "]: " << e.what() << std::endl;
                                throw;
                            }
                        }
                    }
                    CUDA_CHECK(cudaDeviceSynchronize()); // Ensure copies are done before next block uses EVuse (if applicable)

                    this->T.blockCount += 1; // Increment block count

                    // Update d_tokForBlock for the new block (copy last CONTEXT_WIN embeddings D->D)
                    int start_idx_for_new_block = this->T.currentTokenCount - CONTEXT_WIN;
                    CUDA_CHECK(cudaMemcpy(d_tokForBlock,
                                        d_tokenEmbed + static_cast<size_t>(start_idx_for_new_block) * d,
                                        context_win_bytes,
                                        cudaMemcpyDeviceToDevice));
                }

                // --- Termination Checks ---
                if (this->T.currentTokenCount == FULL_CONTEXT) {
                    std::cout << "\nTOKEN LIMIT REACHED AT FULL CONTEXT!";
                    break;
                }
                if (this->T.tokens[this->T.indexForToken] == TERMINATE) {
                    break; // End generation for this prompt
                }

            } // End of response generation loop
            if (this->chat != nullptr) {
                // Check if the file pointer is valid
                fprintf(this->chat, "%s ", this->T.mTokens[this->T.currentTokenCount-1].c_str());
                fflush(this->chat); // Ensure it's written immediately (optional but good for logging)
            }
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            double seconds = duration.count() / 1000000.0;
            std::cout << "\nTime taken to predict tokens of response: " << seconds << " seconds" << std::endl;
            if (seconds > 0) {
                std::cout << "Token Rate: " << static_cast<float>(rCount / seconds) << " tokens/second" << std::endl;
            } 
            else {
                std::cout << "Token Rate: N/A (duration too short)" << std::endl;
            }
            std::cout << std::endl;
            CUDA_CHECK(cudaFree(d_tokenEmbed));
            CUDA_CHECK(cudaFree(d_embeddings));
            CUDA_CHECK(cudaFree(d_EVuse));
            CUDA_CHECK(cudaFree(d_tokForBlock));
            CUDA_CHECK(cudaFree(d_otok)); // Free if allocated externally
            // Ready for next prompt
        #elif USE_OPENCL
            // use opencl
            cl::Buffer d_tokenEmbed;      // Holds all token embeddings (prompt + generated) up to FULL_CONTEXT
            cl::Buffer d_embeddings;      // Holds the vocabulary embeddings
            cl::Buffer d_EVuse;           // Holds the EV state from the previous block (flattened: x * y * CONTEXT_WIN * d)
            cl::Buffer d_tokForBlock;     // Holds the token embeddings for the current block's context window
            // d_otok is managed internally by clForward/clComputeOutput

            // Determine sizes
            size_t full_context_bytes = static_cast<size_t>(FULL_CONTEXT) * d * sizeof(float);
            size_t vocab_bytes = static_cast<size_t>(this->T.vocabsize) * d * sizeof(float);
            size_t context_win_bytes = static_cast<size_t>(CONTEXT_WIN) * d * sizeof(float);
            // EVuse needs space for all attention heads' EV states for a full context window
            // Structure: [x][y][CONTEXT_WIN][d] flattened
            size_t ev_use_bytes = static_cast<size_t>(x) * y * CONTEXT_WIN * d * sizeof(float);
            size_t otok_bytes = static_cast<size_t>(d) * sizeof(float); // Size of EH output
            size_t indexBytes = sizeof(int); // Size for the result index
            int previousTokenCount = this->T.currentTokenCount;
            // Create Buffers
            d_tokenEmbed = cl::Buffer(this->clcontext.context, CL_MEM_READ_WRITE, full_context_bytes);
            d_embeddings = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY, vocab_bytes); // Read-only for inference
            d_EVuse = cl::Buffer(this->clcontext.context, CL_MEM_READ_WRITE, ev_use_bytes); // Holds state between blocks
            d_tokForBlock = cl::Buffer(this->clcontext.context, CL_MEM_READ_WRITE, context_win_bytes); // Holds current block's context

            // Get embeddings for the prompt tokens and copy H->D incrementally
            std::vector<float> prompt_embeddings_flat;
            prompt_embeddings_flat.reserve(static_cast<size_t>(this->T.promptCount) * d);
            for (int i = 0; i < this->T.promptCount; ++i) {
                std::vector<float> current_embed(d);
                // Index in mTokens is previousTokenCount + i
                this->T.getEmbedding(this->T.mTokens[previousTokenCount + i], current_embed);
                prompt_embeddings_flat.insert(prompt_embeddings_flat.end(), current_embed.begin(), current_embed.end());
            }
            // Copy the new prompt embeddings to the correct offset in d_tokenEmbed
            size_t prompt_offset_bytes = static_cast<size_t>(previousTokenCount) * d * sizeof(float);
            size_t prompt_bytes = prompt_embeddings_flat.size() * sizeof(float);
            if (prompt_offset_bytes + prompt_bytes > full_context_bytes) {
                throw std::runtime_error("Prompt copy exceeds d_tokenEmbed bounds.");
            }
            this->clcontext.queue.enqueueWriteBuffer(d_tokenEmbed, CL_TRUE, prompt_offset_bytes, prompt_bytes, prompt_embeddings_flat.data());
            prompt_embeddings_flat.clear(); // Free host memory


            // --- Prompt Placement & Initial KdotQ ---
            // Calculate offset within the current block's window
            int current_block_idx_0based = (this->T.blockCount == 0) ? 0 : this->T.blockCount - 1; // Assuming blockCount is 1-based from training
            int tokens_in_block_before_prompt = (current_block_idx_0based == 0) ? previousTokenCount : (previousTokenCount % CONTEXT_WIN);
            int space_in_current_block = CONTEXT_WIN - tokens_in_block_before_prompt;

            if (this->T.promptCount <= space_in_current_block) {
                // Case 1: Prompt fits entirely within the current block
                // Pre-calculate KdotQ for the prompt tokens added
                for (int col = 0; col < y; ++col) {
                    int effectivePromptCount = this->T.promptCount; // Number of new tokens
                    // The 'currentTokenCount' argument to clParallelKdotQs should be the count *before* adding the prompt
                    this->T.clParallelKdotQs(effectivePromptCount, previousTokenCount, this->T.blockCount, col, isSelf, this->T.inTraining);
                }
                this->clcontext.queue.finish(); // Ensure KdotQ calculation is done
            } 
            else {
                // Case 2: Prompt spans across the current and next block
                int m2 = space_in_current_block; // Tokens fitting in the current block
                int m1 = this->T.promptCount - m2;       // Tokens going to the next block

                // Process the first part (m2 tokens) in the current block
                if (m2 > 0) {
                    for (int col = 0; col < y; ++col) {
                        int effectivePromptCount = m2;
                        this->T.clParallelKdotQs(effectivePromptCount, previousTokenCount, this->T.blockCount, col, isSelf, this->T.inTraining);
                    }
                    this->clcontext.queue.finish(); // Ensure KdotQ for m2 is done
                }

                // --- Transition to the next block ---
                // Copy the EV state from the completed block (t[0]) to d_EVuse (D->D)
                size_t head_ev_bytes = static_cast<size_t>(CONTEXT_WIN) * d * sizeof(float);

                // Check if the total size matches the allocated buffer size
                if (static_cast<size_t>(x) * y * head_ev_bytes != ev_use_bytes) {
                    throw std::runtime_error("Mismatch between calculated EVuse size and allocated buffer size during block transition.");
                }

                // Loop through each attention head in the completed block (t[0])
                for (int i = 0; i < x; ++i) { // Iterate through layers (rows)
                    for (int j = 0; j < y; ++j) { // Iterate through parallels (columns)
                        try {
                            // Get the device buffer for the EV state of the current head (i, j)
                            cl::Buffer& src_ev_buffer = this->T.t[0].b[i][j].getDeviceEVBuffer(); // Use the newly added getter

                            // Calculate the destination offset in the flattened d_EVuse buffer
                            size_t dest_offset_bytes = (static_cast<size_t>(i) * y + j) * head_ev_bytes;

                            // Boundary check for safety
                            if (dest_offset_bytes + head_ev_bytes > ev_use_bytes) {
                                throw std::out_of_range("EVuse destination offset out of bounds for head [" +
                                                        std::to_string(i) + "][" + std::to_string(j) + "].");
                            }

                            // Enqueue the device-to-device buffer copy
                            cl_int err = this->clcontext.queue.enqueueCopyBuffer(src_ev_buffer, d_EVuse, 0, dest_offset_bytes, head_ev_bytes);
                            if (err != CL_SUCCESS) {
                                std::string error_msg = "Failed to enqueue EV copy for head [" + std::to_string(i) + "][" + std::to_string(j) + "]";
                                throw cl::Error(err, error_msg.c_str()); // Use .c_str() here
                            }
                        }
                        catch (const cl::Error& clErr) {
                            std::cerr << "OpenCL Error getting/copying EV buffer for head [" << i << "][" << j << "]: "
                                    << clErr.what() << " (" << clErr.err() << ")" << std::endl;
                            throw; // Re-throw OpenCL errors
                        }
                        catch (const std::exception& e) {
                            // Catch potential errors from getDeviceEVBuffer() or other issues
                            std::cerr << "Error getting/copying EV buffer for head [" << i << "][" << j << "]: " << e.what() << std::endl;
                            throw; // Re-throw standard exceptions
                        }
                    } // End loop columns (j)
                } // End loop rows (i)

                // Ensure all copy operations are completed before proceeding
                this->clcontext.queue.finish();

                this->T.blockCount += 1; // Increment block count *once*

                // Prepare d_tokForBlock for the *new* block (contains the m1 tokens + previous context)
                // Copy the last CONTEXT_WIN tokens from d_tokenEmbed to d_tokForBlock
                int start_idx_for_new_block = this->T.currentTokenCount - CONTEXT_WIN; // Start index in d_tokenEmbed
                size_t copy_offset_bytes = static_cast<size_t>(start_idx_for_new_block) * d * sizeof(float);
                if (copy_offset_bytes + context_win_bytes > full_context_bytes) {
                    throw std::runtime_error("d_tokForBlock source offset out of bounds.");
                }
                this->clcontext.queue.enqueueCopyBuffer(d_tokenEmbed, d_tokForBlock, copy_offset_bytes, 0, context_win_bytes);
                this->clcontext.queue.finish(); // Ensure copy is done


                // Pre-calculate KdotQ for the second part (m1 tokens) in the *new* block
                if (m1 > 0) {
                    // The "previous token count" for this calculation is effectively the start of the new block's relevant context
                    int start_of_new_block_count = this->T.currentTokenCount - m1;
                    for (int col = 0; col < y; ++col) {
                        int effectivePromptCount = m1;
                        // Pass the new blockCount
                        this->T.clParallelKdotQs(effectivePromptCount, start_of_new_block_count, this->T.blockCount, col, isSelf, this->T.inTraining);
                    }
                    this->clcontext.queue.finish(); // Ensure KdotQ for m1 is done
                }
            }

            // --- Response Generation Loop (OpenCL) ---
            int rCount = 0;
            auto start_time = std::chrono::high_resolution_clock::now();
            std::cout << "Response: ";

            while (1) {
                // --- Core Forward Pass ---
                // clForward handles attention, MLPs, and state updates for the current step.
                // It needs the current block index and total token count.
                // Pass 0 for promptCount during generation.
                int generationPromptCount = 0; // Indicate generation phase
                this->T.clForward(this->T.blockCount, this->T.currentTokenCount, generationPromptCount); // Updates t[0]'s internal state (EH, EV)
                this->clcontext.queue.finish(); // Ensure forward pass is complete

                // --- Get Output Token ---
                // Assume clForward updated the host t[0].EH. If not, read back from device.
                if (this->T.t[0].EH.size() != static_cast<size_t>(d)) {
                    throw std::runtime_error("EH size in t[0] is incorrect after clForward.");
                }

                // --- Start: Inline Prediction Kernel Launch ---
                { // Scope for temporary prediction buffers
                    cl::Buffer d_output;       // Buffer for EH input to kernel
                    cl::Buffer d_result_index; // Buffer for kernel output index
                    int result_index_val = -1; // Host variable to hold the result

                    try {
                        // Create and copy EH to device buffer
                        d_output = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, otok_bytes, this->T.t[0].EH.data());
                        // Create buffer for the kernel to write the result index
                        d_result_index = cl::Buffer(this->clcontext.context, CL_MEM_WRITE_ONLY, indexBytes);

                        // Get the kernel
                        cl::Kernel kernel = this->clcontext.kernels.at("compute_prediction"); // Use member function

                        // Set arguments based on the kernel signature
                        kernel.setArg(0, d_output);
                        kernel.setArg(1, d_embeddings); // Use existing buffer
                        kernel.setArg(2, static_cast<cl_int>(this->d)); // dim
                        kernel.setArg(3, static_cast<cl_int>(this->T.vocabsize)); // voc
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
                        this->T.indexForToken = result_index_val; // Update class member

                    } 
                    catch (const cl::Error& err) {
                        std::cerr << "OpenCL Error during inline prediction kernel launch: " << err.what() << " (" << err.err() << ")" << std::endl;
                        this->T.indexForToken = -1; // Indicate error
                        throw; // Re-throw
                    }
                    // Buffers d_output, d_result_index released by RAII
                }
                // --- Update State (Host & Device) ---
                if (this->T.indexForToken < 0 || this->T.indexForToken >= this->T.vocabsize) {
                    std::cerr << "\nError: Invalid token index computed: " << this->T.indexForToken << std::endl;
                    break; // Exit generation loop on error
                }

                // Get the embedding for the new token on the host
                const std::vector<float>& next_token_embed_h = this->T.embeddings[this->T.indexForToken];
                // Copy the new embedding H->D into d_tokenEmbed at the current position
                size_t next_token_offset_bytes = static_cast<size_t>(this->T.currentTokenCount) * d * sizeof(float);
                size_t next_token_bytes = static_cast<size_t>(d) * sizeof(float);
                if (next_token_offset_bytes + next_token_bytes > full_context_bytes) {
                    throw std::runtime_error("Next token copy exceeds d_tokenEmbed bounds.");
                }
                this->clcontext.queue.enqueueWriteBuffer(d_tokenEmbed, CL_TRUE, next_token_offset_bytes, next_token_bytes, next_token_embed_h.data());

                // Store the token string (host)
                if (this->T.currentTokenCount >= this->T.mTokens.size()) {
                    this->T.mTokens.resize(this->T.currentTokenCount + 1); // Ensure space
                }
                this->T.mTokens[this->T.currentTokenCount] = this->T.tokens[this->T.indexForToken];

                // Print the token (host)
                std::cout << this->T.mTokens[this->T.currentTokenCount] << " " << std::flush;

                // Increment token count (host)
                this->T.currentTokenCount += 1;
                rCount += 1;

                // --- Context Window / Block Transition Check ---
                if (this->T.currentTokenCount % CONTEXT_WIN == 0 && this->T.currentTokenCount > 0 && this->T.currentTokenCount < FULL_CONTEXT) {
                    // Copy the EV state from the completed block (t[0]) to d_EVuse (D->D)
                    std::cerr << "\nWarning: Device-to-device copy for EV->EVuse not implemented yet in clRun generation loop.\n" << std::endl;
                    // Placeholder logic (see prompt handling section)
                    // ... D->D copy loop using enqueueCopyBuffer ...
                    // queue.finish();

                    this->T.blockCount += 1; // Increment block count

                    // Update d_tokForBlock for the new block (copy last CONTEXT_WIN embeddings D->D)
                    int start_idx_for_new_block = this->T.currentTokenCount - CONTEXT_WIN;
                    size_t copy_offset_bytes = static_cast<size_t>(start_idx_for_new_block) * d * sizeof(float);
                    if (copy_offset_bytes + context_win_bytes > full_context_bytes) {
                        throw std::runtime_error("d_tokForBlock source offset out of bounds during generation.");
                    }
                    this->clcontext.queue.enqueueCopyBuffer(d_tokenEmbed, d_tokForBlock, copy_offset_bytes, 0, context_win_bytes);
                    this->clcontext.queue.finish(); // Ensure copy is done

                    // Optional: Pre-calculate KdotQ for the new block if needed.
                }

                // --- Termination Checks ---
                if (this->T.currentTokenCount == FULL_CONTEXT) {
                    std::cout << "\nTOKEN LIMIT REACHED AT FULL CONTEXT (" << FULL_CONTEXT << ")!" << std::endl;
                    break;
                }
                if (this->T.tokens[this->T.indexForToken] == TERMINATE) {
                    break; // End generation for this prompt
                }
                if (this->chat != nullptr) {
                    // Check if the file pointer is valid
                    fprintf(this->chat, "%s ", this->T.mTokens[this->T.currentTokenCount-1].c_str());
                    fflush(this->chat); // Ensure it's written immediately (optional but good for logging)
                }
            } // End of response generation loop

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            double seconds = duration.count() / 1000000.0;
            std::cout << "\nTime taken to predict " << rCount << " tokens of response: " << seconds << " seconds" << std::endl;
            if (seconds > 0 && rCount > 0) {
                std::cout << "Token Rate: " << static_cast<double>(rCount) / seconds << " tokens/second" << std::endl;
            } else {
                std::cout << "Token Rate: N/A" << std::endl;
            }
            std::cout << std::endl;
        #elif USE_CPU
            // use cpp
            auto start_time = std::chrono::high_resolution_clock::now();
            for(int i = 0; i < this->T.promptCount; i++) {
                // get token embeddings from 'embeddings' vector in transformer class
                // tokenEmbed[currentTokenCount+i] = embeddings[];
                this->T.getEmbedding(this->T.mTokens[this->T.currentTokenCount+i], this->T.tokenEmbed[this->T.currentTokenCount+i]);
            }
            if(this->T.currentTokenCount + this->T.promptCount >= FULL_CONTEXT) {
                throw std::runtime_error("TOKEN LIMIT REACHED AT FULL CONTEXT!");
                break;
            }
            int c = std::abs(this->T.currentTokenCount - (this->T.blockCount-1)*CONTEXT_WIN);
            // under local context
            if(c + this->T.promptCount <= CONTEXT_WIN) {
                // when first block, tokenEmbed is directly utilised
                if(this->T.blockCount > 1) {
                    for(int k = 0; k < this->T.promptCount; k++) {
                        this->T.tokForBlock[c + k] = this->T.tokenEmbed[this->T.currentTokenCount + k];
                    }
                }
                for(int i = 0; i < x; i++) {
                    for(int j = 0; j < y; j++) {
                        for(int k = 0; k < this->T.promptCount; k++) {
                            this->T.t[0].b[i][j].EV[c-1+k] = this->T.tokenEmbed[this->T.currentTokenCount + k];
                        }
                    }
                }
            }
            // if it goes over context window, increment to next block
            if(c + this->T.promptCount > CONTEXT_WIN) {
                int m1 = c + this->T.promptCount - CONTEXT_WIN;     // part of prompt in next block
                int m2 = CONTEXT_WIN - c;   // available space in this block
                // add prompt to EVs and tokforblock
                for(int i = 0; i < x; i++) {
                    for(int j = 0; j < y; j++) {
                        for(int k = 0; k < m2; k++) {
                            this->T.t[0].b[i][j].EV[c-1+k] = this->T.tokenEmbed[this->T.currentTokenCount + k];
                        }
                    }
                }
                for(int i = 0; i < m2; i++) {
                    this->T.tokForBlock[i] = this->T.tokenEmbed[this->T.currentTokenCount + i];
                }
                this->T.computeKdotQs(this->T.promptCount, this->T.currentTokenCount, this->T.blockCount, isSelf, this->T.inTraining);
                // token limit reached for first block
                this->T.currentTokenCount += m1;
                // set vertical retention vectors
                for(int i = 0; i < x; i++) {
                    for(int j = 0; j < y; j++) {
                        for(int k = 0; k < CONTEXT_WIN; k++) {
                            this->T.EVuse[i][j][k] = this->T.t[0].b[i][j].EV[k];
                        }
                    }
                }
                this->T.blockCount += 1;
                for(int i = 0; i < CONTEXT_WIN; i++) {
                    this->T.tokForBlock[i] = this->T.tokenEmbed[this->T.currentTokenCount - CONTEXT_WIN + i];
                }
                this->T.currentTokenCount += m2;
                this->T.blockCount += 1;
            }
            // caculate response
            int rCount = 0;
            while (1) {
                int k, l;   // for row and column sum
                // forprop for EH and EV
                for(int i = 0; i < x; i++) {
                    for(int j = 0; j < y; j++) {
                        // block specific KdotQ
                        if(this->T.blockCount == 1) {
                            computeKdotQ(this->T.t[0].b[i][j].KdotQ, this->T.tokenEmbed, this->T.t[0].b[i][j].qkCache, this->T.currentTokenCount, 
                                this->T.promptCount, isSelf);
                        }
                        else {
                            computeKdotQ(this->T.t[0].b[i][j].KdotQ, this->T.tokForBlock, this->T.EVuse[i][j], this->T.t[0].b[i][j].qkCache, 
                                this->T.currentTokenCount, this->T.promptCount, this->T.blockCount, isSelf);
                        }
                        // number of tokens in context window of this block
                        int count = std::abs(this->T.currentTokenCount - n * (this->T.blockCount-1));
                        // calculate KdotQ and head
                        std::vector<std::vector<float>> head(count, std::vector<float>(count, 0.0f));
                        head = LOTA(this->T.t[0].b[i][j].KdotQ, count, isSelf);
                        // get weighted sums
                        for(int w = 0; w < count; w++) {
                            k = 0;
                            l = 0;
                            for(int z = 0; z < (isSelf ? w : count); w++) {    
                                k += head[w][z];    // row sum
                                l += head[z][w];    // column sum
                            }
                            // ti*k, dh = weighted sums horizontal
                            this->T.t[0].b[i][j].dh = this->T.t[0].b[i][j].dh + (k * ((this->T.blockCount == 1) ? this->T.tokenEmbed[i] : this->T.tokForBlock[i]));
                            // ti*l, dv = weighted sums vertical
                            this->T.t[0].b[i][j].dv = this->T.t[0].b[i][j].dv + (l * ((this->T.blockCount == 1) ? this->T.tokenEmbed[i] : this->T.EVuse[i][j][count]));
                        }
                        this->T.t[0].b[i][j].dh = dot(this->T.t[0].b[i][j].dh, this->T.t[0].b[i][j].khCache);
                        this->T.t[0].b[i][j].dv = dot(this->T.t[0].b[i][j].dv, this->T.t[0].b[i][j].qvCache);
                        // get the required change from MLPs
                        this->T.t[0].b[i][j].hor.input = this->T.t[0].b[i][j].EH + this->T.t[0].b[i][j].dh;
                        this->T.t[0].b[i][j].ver.input = this->T.EVuse[i][j][this->T.currentTokenCount] + this->T.t[0].b[i][j].dv;
                        this->T.t[0].b[i][j].hor.forward(d, l);
                        this->T.t[0].b[i][j].ver.forward(d, l);
                        // AND gate for the final output
                        this->T.t[0].b[i][j].EH += ReLU(this->T.t[0].b[i][j].hor.output);
                        this->T.t[0].b[i][j].EV[i] += ReLU(this->T.t[0].b[i][j].ver.output);
                    }
                    this->T.t[0].EH += this->T.t[0].b[i][y-1].EH;
                }
                computeOutput(this->T.t[0].EH, this->T.embeddings, this->T.vocabsize, this->T.indexForToken);
                this->T.tokenEmbed[this->T.currentTokenCount] = this->T.embeddings[this->T.indexForToken];
                this->T.mTokens[this->T.currentTokenCount] = this->T.tokens[this->T.indexForToken];
                std::cout << this->T.mTokens[this->T.currentTokenCount] << " ";
                this->T.currentTokenCount += 1;
                // check for local context
                if(this->T.currentTokenCount%CONTEXT_WIN == 0) {
                    // set vertical context retention for next blocks
                    for(int i = 0; i < x; i++) {
                        for(int j = 0; j < y; j++) {
                            for(int k = 0; k < CONTEXT_WIN; k++) {
                                this->T.EVuse[i][j][k] = this->T.t[0].b[i][j].EV[k];
                            }
                        }
                    }
                    this->T.blockCount += 1;
                }
                // check for maximum token limit
                if(this->T.currentTokenCount == FULL_CONTEXT) {
                    std::cout << "\nTOKEN LIMIT REACHED AT FULL CONTEXT!";
                    break;
                }
                // check for terminating word '@#0'
                if(this->T.tokens[this->T.indexForToken] == TERMINATE) {
                    break;
                }
                rCount += 1;
                if (this->chat != nullptr) {
                    // Check if the file pointer is valid
                    fprintf(this->chat, "%s ", this->T.mTokens[this->T.currentTokenCount-1].c_str());
                    fflush(this->chat); // Ensure it's written immediately (optional but good for logging)
                }
            }
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            std::cout << "Time taken to predict tokens of response: "<< duration.count()/1000000.0 << " seconds" << std::endl;
            std::cout << "Token Rate: " << static_cast<float>(rCount/(duration.count()/1000000.0)) << " tokens/second" << std::endl;
            std::cout << std::endl;
        #endif
        if (this->chat != nullptr) {
            // Check if the file pointer is valid
            fprintf(this->chat, "\n");
            fflush(this->chat); // Ensure it's written immediately (optional but good for logging)
        }
    }
    // save model
    std::cout << "SAVE CHAT IN FILE (1 for save): ";
    std::cin >> savechat;
    if(savechat == 1) { saveChat(); }
    std::cout << "NEW CHAT (1) or END (0): ";
    std::cin >> newchat;
    // continue chatting or end chat
    (newchat == 1) ? newChat() : endChat();
}
