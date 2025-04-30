#ifdef USE_OPENCL

#include "include/transformer.hpp"
#include "include/mlp.hpp"
#include "include/attention.hpp"
#include "include/block.hpp"
#include <maths.hpp>
#include <CL/cl.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <chrono>
#include <stdexcept>
#include <cmath> // For std::abs
#include <limits> // For numeric_limits

/**
 * @brief Run transformer with OpenCL using model parameters for inference.
 * Uses clParallelKdotQs for initial KdotQ calculation based on prompt.
 * Uses clForward for subsequent token generation steps within a block.
 * Uses compute_prediction kernel to determine the next token index.
 * Handles context window transitions and block management on the device.
 */
void transformer::clRun() {
    // Set for inference
    inTraining = 0;
    if (t.empty()) {
         throw std::runtime_error("Transformer block 't' is not initialized in clRun.");
    }
    if (t[0].b.empty() || t[0].b[0].empty()) {
        throw std::runtime_error("Attention heads within the transformer block 't[0]' are not initialized in clRun.");
    }

    // --- Device Memory Allocation ---
    cl::Buffer d_tokenEmbed;      // Holds all token embeddings (prompt + generated) up to FULL_CONTEXT
    cl::Buffer d_embeddings;      // Holds the vocabulary embeddings
    cl::Buffer d_EVuse;           // Holds the EV state from the previous block (flattened: x * y * CONTEXT_WIN * d)
    cl::Buffer d_tokForBlock;     // Holds the token embeddings for the current block's context window
    // d_otok is managed internally by clForward/clComputeOutput

    // Determine sizes
    size_t full_context_bytes = static_cast<size_t>(FULL_CONTEXT) * d * sizeof(float);
    size_t vocab_bytes = static_cast<size_t>(vocabsize) * d * sizeof(float);
    size_t context_win_bytes = static_cast<size_t>(CONTEXT_WIN) * d * sizeof(float);
    // EVuse needs space for all attention heads' EV states for a full context window
    // Structure: [x][y][CONTEXT_WIN][d] flattened
    size_t ev_use_bytes = static_cast<size_t>(x) * y * CONTEXT_WIN * d * sizeof(float);
    size_t otok_bytes = static_cast<size_t>(d) * sizeof(float); // Size of EH output
    size_t indexBytes = sizeof(int); // Size for the result index

    try {
        // Create Buffers
        d_tokenEmbed = cl::Buffer(this->clcontext.context, CL_MEM_READ_WRITE, full_context_bytes);
        d_embeddings = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY, vocab_bytes); // Read-only for inference
        d_EVuse = cl::Buffer(this->clcontext.context, CL_MEM_READ_WRITE, ev_use_bytes); // Holds state between blocks
        d_tokForBlock = cl::Buffer(this->clcontext.context, CL_MEM_READ_WRITE, context_win_bytes); // Holds current block's context

        // --- Initial Data Transfer ---
        // Flatten and copy vocabulary embeddings H->D (once)
        std::vector<float> flat_embeddings;
        flat_embeddings.reserve(static_cast<size_t>(vocabsize) * d);
        for (const auto& embed : embeddings) {
            if (embed.size() != static_cast<size_t>(d)) throw std::runtime_error("Inconsistent embedding dimension in vocabulary.");
            flat_embeddings.insert(flat_embeddings.end(), embed.begin(), embed.end());
        }
        this->clcontext.queue.enqueueWriteBuffer(d_embeddings, CL_TRUE, 0, vocab_bytes, flat_embeddings.data());
        flat_embeddings.clear(); // Free host memory

        // --- Main Inference Loop ---
        while (1) {
            // --- Prompt Handling (Host) ---
            std::string userPrompt;
            std::cout << "ENTER PROMPT (LIMIT " << PROMPT_THRESHOLD << " TOKENS, type 'quit' to exit): ";
            // Using std::getline to read the whole line, including spaces
            // Clear potential leftover newline from previous input if any
            if (std::cin.peek() == '\n') {
                std::cin.ignore();
            }
            std::getline(std::cin, userPrompt);

            if (userPrompt == "quit") {
                break; // Exit the main loop
            }
            if (userPrompt.empty()) {
                continue; // Handle empty input
            }


            int previousTokenCount = currentTokenCount; // Store count before adding prompt
            promptCount = tokenise(userPrompt, mTokens, currentTokenCount); // currentTokenCount is updated

            if (promptCount == 0) {
                std::cout << "Could not tokenize prompt." << std::endl;
                currentTokenCount = previousTokenCount; // Revert count
                continue;
            }
            if (currentTokenCount >= FULL_CONTEXT) {
                std::cerr << "\nError: Adding prompt exceeds FULL_CONTEXT limit (" << FULL_CONTEXT << "). Please restart or shorten prompt." << std::endl;
                currentTokenCount = previousTokenCount; // Revert count
                mTokens.resize(currentTokenCount); // Revert token list
                continue; // Ask for prompt again
            }


            // Get embeddings for the prompt tokens and copy H->D incrementally
            std::vector<float> prompt_embeddings_flat;
            prompt_embeddings_flat.reserve(static_cast<size_t>(promptCount) * d);
            for (int i = 0; i < promptCount; ++i) {
                std::vector<float> current_embed(d);
                // Index in mTokens is previousTokenCount + i
                getEmbedding(mTokens[previousTokenCount + i], current_embed);
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
            int current_block_idx_0based = (blockCount == 0) ? 0 : blockCount - 1; // Assuming blockCount is 1-based from training
            int tokens_in_block_before_prompt = (current_block_idx_0based == 0) ? previousTokenCount : (previousTokenCount % CONTEXT_WIN);
            int space_in_current_block = CONTEXT_WIN - tokens_in_block_before_prompt;

            if (promptCount <= space_in_current_block) {
                // Case 1: Prompt fits entirely within the current block
                // Pre-calculate KdotQ for the prompt tokens added
                for (int col = 0; col < y; ++col) {
                    int effectivePromptCount = promptCount; // Number of new tokens
                    // The 'currentTokenCount' argument to clParallelKdotQs should be the count *before* adding the prompt
                    clParallelKdotQs(effectivePromptCount, previousTokenCount, blockCount, col, isSelf, inTraining);
                }
                this->clcontext.queue.finish(); // Ensure KdotQ calculation is done
            } else {
                // Case 2: Prompt spans across the current and next block
                int m2 = space_in_current_block; // Tokens fitting in the current block
                int m1 = promptCount - m2;       // Tokens going to the next block

                // Process the first part (m2 tokens) in the current block
                if (m2 > 0) {
                    for (int col = 0; col < y; ++col) {
                        int effectivePromptCount = m2;
                        clParallelKdotQs(effectivePromptCount, previousTokenCount, blockCount, col, isSelf, inTraining);
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
                            cl::Buffer& src_ev_buffer = t[0].b[i][j].getDeviceEVBuffer(); // Use the newly added getter

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

                blockCount += 1; // Increment block count *once*

                // Prepare d_tokForBlock for the *new* block (contains the m1 tokens + previous context)
                // Copy the last CONTEXT_WIN tokens from d_tokenEmbed to d_tokForBlock
                int start_idx_for_new_block = currentTokenCount - CONTEXT_WIN; // Start index in d_tokenEmbed
                size_t copy_offset_bytes = static_cast<size_t>(start_idx_for_new_block) * d * sizeof(float);
                if (copy_offset_bytes + context_win_bytes > full_context_bytes) {
                    throw std::runtime_error("d_tokForBlock source offset out of bounds.");
                }
                this->clcontext.queue.enqueueCopyBuffer(d_tokenEmbed, d_tokForBlock, copy_offset_bytes, 0, context_win_bytes);
                this->clcontext.queue.finish(); // Ensure copy is done


                // Pre-calculate KdotQ for the second part (m1 tokens) in the *new* block
                if (m1 > 0) {
                    // The "previous token count" for this calculation is effectively the start of the new block's relevant context
                    int start_of_new_block_count = currentTokenCount - m1;
                    for (int col = 0; col < y; ++col) {
                        int effectivePromptCount = m1;
                        // Pass the new blockCount
                        clParallelKdotQs(effectivePromptCount, start_of_new_block_count, blockCount, col, isSelf, inTraining);
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
                clForward(blockCount, currentTokenCount, generationPromptCount); // Updates t[0]'s internal state (EH, EV)
                this->clcontext.queue.finish(); // Ensure forward pass is complete

                // --- Get Output Token ---
                // Assume clForward updated the host t[0].EH. If not, read back from device.
                if (t[0].EH.size() != static_cast<size_t>(d)) {
                    throw std::runtime_error("EH size in t[0] is incorrect after clForward.");
                }

                // --- Start: Inline Prediction Kernel Launch ---
                { // Scope for temporary prediction buffers
                    cl::Buffer d_output;       // Buffer for EH input to kernel
                    cl::Buffer d_result_index; // Buffer for kernel output index
                    int result_index_val = -1; // Host variable to hold the result

                    try {
                        // Create and copy EH to device buffer
                        d_output = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, otok_bytes, t[0].EH.data());
                        // Create buffer for the kernel to write the result index
                        d_result_index = cl::Buffer(this->clcontext.context, CL_MEM_WRITE_ONLY, indexBytes);

                        // Get the kernel
                        cl::Kernel kernel = this->clcontext.kernels.at("compute_prediction"); // Use member function

                        // Set arguments based on the kernel signature
                        kernel.setArg(0, d_output);
                        kernel.setArg(1, d_embeddings); // Use existing buffer
                        kernel.setArg(2, static_cast<cl_int>(this->d)); // dim
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
                        this->indexForToken = result_index_val; // Update class member

                    } catch (const cl::Error& err) {
                        std::cerr << "OpenCL Error during inline prediction kernel launch: " << err.what() << " (" << err.err() << ")" << std::endl;
                        this->indexForToken = -1; // Indicate error
                        throw; // Re-throw
                    }
                    // Buffers d_output, d_result_index released by RAII
                }
                // --- End: Inline Prediction Kernel Launch ---


                // --- Update State (Host & Device) ---
                if (indexForToken < 0 || indexForToken >= vocabsize) {
                    std::cerr << "\nError: Invalid token index computed: " << indexForToken << std::endl;
                    break; // Exit generation loop on error
                }

                // Get the embedding for the new token on the host
                const std::vector<float>& next_token_embed_h = embeddings[indexForToken];
                // Copy the new embedding H->D into d_tokenEmbed at the current position
                size_t next_token_offset_bytes = static_cast<size_t>(currentTokenCount) * d * sizeof(float);
                size_t next_token_bytes = static_cast<size_t>(d) * sizeof(float);
                if (next_token_offset_bytes + next_token_bytes > full_context_bytes) {
                    throw std::runtime_error("Next token copy exceeds d_tokenEmbed bounds.");
                }
                this->clcontext.queue.enqueueWriteBuffer(d_tokenEmbed, CL_TRUE, next_token_offset_bytes, next_token_bytes, next_token_embed_h.data());

                // Store the token string (host)
                if (currentTokenCount >= mTokens.size()) {
                    mTokens.resize(currentTokenCount + 1); // Ensure space
                }
                mTokens[currentTokenCount] = tokens[indexForToken];

                // Print the token (host)
                std::cout << mTokens[currentTokenCount] << " " << std::flush;

                // Increment token count (host)
                currentTokenCount += 1;
                rCount += 1;

                // --- Context Window / Block Transition Check ---
                if (currentTokenCount % CONTEXT_WIN == 0 && currentTokenCount > 0 && currentTokenCount < FULL_CONTEXT) {
                    // Copy the EV state from the completed block (t[0]) to d_EVuse (D->D)
                    std::cerr << "\nWarning: Device-to-device copy for EV->EVuse not implemented yet in clRun generation loop.\n" << std::endl;
                    // Placeholder logic (see prompt handling section)
                    // ... D->D copy loop using enqueueCopyBuffer ...
                    // queue.finish();

                    blockCount += 1; // Increment block count

                    // Update d_tokForBlock for the new block (copy last CONTEXT_WIN embeddings D->D)
                    int start_idx_for_new_block = currentTokenCount - CONTEXT_WIN;
                    size_t copy_offset_bytes = static_cast<size_t>(start_idx_for_new_block) * d * sizeof(float);
                    if (copy_offset_bytes + context_win_bytes > full_context_bytes) {
                        throw std::runtime_error("d_tokForBlock source offset out of bounds during generation.");
                    }
                    this->clcontext.queue.enqueueCopyBuffer(d_tokenEmbed, d_tokForBlock, copy_offset_bytes, 0, context_win_bytes);
                    this->clcontext.queue.finish(); // Ensure copy is done

                    // Optional: Pre-calculate KdotQ for the new block if needed.
                }

                // --- Termination Checks ---
                if (currentTokenCount == FULL_CONTEXT) {
                    std::cout << "\nTOKEN LIMIT REACHED AT FULL CONTEXT (" << FULL_CONTEXT << ")!" << std::endl;
                    break;
                }
                if (tokens[indexForToken] == TERMINATE) {
                    break; // End generation for this prompt
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

            // Ready for next prompt
        } // End of main inference loop (while(1))

    } catch (const cl::Error& err) {
        std::cerr << "OpenCL Error in transformer::clRun: " << err.what() << " (" << err.err() << ")" << std::endl;
        // Cleanup is handled by RAII for cl::Buffer
        throw; // Re-throw
    } catch (const std::exception& e) {
        std::cerr << "Error in transformer::clRun: " << e.what() << std::endl;
        // Cleanup is handled by RAII for cl::Buffer
        throw; // Re-throw
    }

    // --- Final Cleanup ---
    // cl::Buffer objects (d_tokenEmbed, d_embeddings, etc.) are automatically released by their destructors.
    std::cout << "Exiting clRun." << std::endl;
}

#endif // USE_OPENCL
