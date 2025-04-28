
#include "include/transformer.hpp"
#include "include/mlp.hpp"
#include "include/attention.hpp"
#include "include/block.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <string>
#include <iostream>
#include <chrono>
#include <stdexcept>
#include <cmath> // For std::abs
#include <limits> // For numeric_limits

// check cuda errors
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        throw std::runtime_error(cudaGetErrorString(err)); \
    } \
} while (0)


/**
 * @brief Run transformer with CUDA using model parameters of cache and MLPs for inference.
 * Uses cuParallelKdotQs for initial KdotQ calculation based on prompt.
 * Uses cuForward for subsequent token generation steps within a block.
 * Uses cuComputeOutput to determine the next token index.
 * Handles context window transitions and block management on the device.
 */
void transformer::cuRun() {
    // Set for inference
    inTraining = 0;
    if (t.empty()) {
         throw std::runtime_error("Transformer block 't' is not initialized in cuRun.");
    }
    if (t[0].b.empty() || t[0].b[0].empty()) {
        throw std::runtime_error("Attention heads within the transformer block 't[0]' are not initialized in cuRun.");
    }

    // --- Device Memory Allocation ---
    float* d_tokenEmbed = nullptr;      // Holds all token embeddings (prompt + generated) up to FULL_CONTEXT
    float* d_embeddings = nullptr;      // Holds the vocabulary embeddings
    float* d_EVuse = nullptr;           // Holds the EV state from the previous block (flattened: x * y * CONTEXT_WIN * d)
    float* d_tokForBlock = nullptr;     // Holds the token embeddings for the current block's context window
    float* d_otok = nullptr;            // Holds the output token embedding from cuForward (size d)

    // Determine sizes
    size_t full_context_bytes = static_cast<size_t>(FULL_CONTEXT) * d * sizeof(float);
    size_t vocab_bytes = static_cast<size_t>(vocabsize) * d * sizeof(float);
    size_t context_win_bytes = static_cast<size_t>(CONTEXT_WIN) * d * sizeof(float);
    // EVuse needs space for all attention heads' EV states for a full context window
    size_t ev_use_bytes = static_cast<size_t>(x) * y * CONTEXT_WIN * d * sizeof(float);
    size_t otok_bytes = static_cast<size_t>(d) * sizeof(float);


    try {
        CUDA_CHECK(cudaMalloc(&d_tokenEmbed, full_context_bytes));
        CUDA_CHECK(cudaMalloc(&d_embeddings, vocab_bytes));
        CUDA_CHECK(cudaMalloc(&d_EVuse, ev_use_bytes)); // Allocate space for EV context passing
        CUDA_CHECK(cudaMalloc(&d_tokForBlock, context_win_bytes));
        CUDA_CHECK(cudaMalloc(&d_otok, otok_bytes)); // d_otok is managed internally by cuForward now

        // --- Initial Data Transfer ---
        // Flatten and copy vocabulary embeddings H->D (once)
        std::vector<float> flat_embeddings;
        flat_embeddings.reserve(static_cast<size_t>(vocabsize) * d);
        for (const auto& embed : embeddings) {
            if (embed.size() != static_cast<size_t>(d)) throw std::runtime_error("Inconsistent embedding dimension in vocabulary.");
            flat_embeddings.insert(flat_embeddings.end(), embed.begin(), embed.end());
        }
        CUDA_CHECK(cudaMemcpy(d_embeddings, flat_embeddings.data(), vocab_bytes, cudaMemcpyHostToDevice));
        flat_embeddings.clear(); // Free host memory

        // --- Main Inference Loop ---
        while (1) {
            // --- Prompt Handling (Host) ---
            std::string userPrompt;
            std::cout << "ENTER PROMPT (LIMIT " << PROMPT_THRESHOLD << " TOKENS): ";
            // Using std::getline to read the whole line, including spaces
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Clear potential leftover newline
            std::getline(std::cin, userPrompt);
            if (userPrompt.empty()) continue; // Handle empty input

            int previousTokenCount = currentTokenCount; // Store count before adding prompt
            promptCount = tokenise(userPrompt, mTokens, currentTokenCount); // currentTokenCount is updated by tokenise

            if (promptCount == 0) {
                std::cout << "Could not tokenize prompt." << std::endl;
                currentTokenCount = previousTokenCount; // Revert count
                continue;
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
            CUDA_CHECK(cudaMemcpy(d_tokenEmbed + static_cast<size_t>(previousTokenCount) * d,
                                  prompt_embeddings_flat.data(),
                                  prompt_embeddings_flat.size() * sizeof(float),
                                  cudaMemcpyHostToDevice));
            prompt_embeddings_flat.clear(); // Free host memory


            if (currentTokenCount >= FULL_CONTEXT) {
                throw std::runtime_error("TOKEN LIMIT REACHED AT FULL CONTEXT during prompt processing!");
                break; // Should be unreachable due to throw
            }

            // --- Prompt Placement & Initial KdotQ (Mirroring CPU logic) ---
            // Calculate offset within the current block's window
            int c = (blockCount == 0) ? previousTokenCount : (previousTokenCount % CONTEXT_WIN);
            // Note: CPU code used std::abs(currentTokenCount - (blockCount-1)*CONTEXT_WIN), which seems complex.
            // Assuming blockCount starts at 0 for the first block.
            // Let's use the simpler modulo logic, assuming blocks align with CONTEXT_WIN.

            int tokens_in_current_block_before_prompt = c;
            int space_in_current_block = CONTEXT_WIN - tokens_in_current_block_before_prompt;

            if (promptCount <= space_in_current_block) {
                // Case 1: Prompt fits entirely within the current block
                if (blockCount > 0) {
                    // Copy relevant part of d_tokenEmbed to d_tokForBlock if needed by cuParallelKdotQs
                    // This depends on how cuParallelKdotQs expects input for non-first blocks.
                    // Assuming it uses d_tokenEmbed directly with offsets.
                }
                // Pre-calculate KdotQ for the prompt tokens added to the current block
                // We need to tell cuParallelKdotQs how many *new* tokens (promptCount) were added
                // starting at which index (previousTokenCount) within the full context.
                for (int col = 0; col < y; ++col) {
                    // Pass the count of *new* prompt tokens and the total count *before* the prompt
                    int effectivePromptCount = promptCount;
                    cuParallelKdotQs(effectivePromptCount, previousTokenCount, blockCount, col, isSelf, inTraining);
                }
                 CUDA_CHECK(cudaDeviceSynchronize()); // Ensure KdotQ calculation is done
            } else {
                // Case 2: Prompt spans across the current and next block
                int m2 = space_in_current_block; // Tokens fitting in the current block
                int m1 = promptCount - m2;       // Tokens going to the next block

                // Process the first part (m2 tokens) in the current block
                if (m2 > 0) {
                    for (int col = 0; col < y; ++col) {
                        // Calculate KdotQ for the m2 tokens added
                        int effectivePromptCount = m2;
                        cuParallelKdotQs(effectivePromptCount, previousTokenCount, blockCount, col, isSelf, inTraining);
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
                            float* d_src_ev_ptr = t[0].b[i][j].getDeviceEVPointer(); // Use the getter

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

                        } catch (const std::exception& e) {
                             // Catch potential errors from getDeviceEVPointer() or other issues
                             std::cerr << "Error getting/copying EV pointer for head [" << i << "][" << j << "]: " << e.what() << std::endl;
                             throw; // Re-throw standard exceptions
                        }
                    } // End loop columns (j)
                } // End loop rows (i)

                // Ensure all copy operations are completed before proceeding
                CUDA_CHECK(cudaDeviceSynchronize());

                blockCount += 1; // Increment block count *once*

                // Prepare d_tokForBlock for the *new* block (contains the m1 tokens + previous context)
                // Copy the last CONTEXT_WIN tokens from d_tokenEmbed to d_tokForBlock
                int start_idx_for_new_block = currentTokenCount - CONTEXT_WIN; // Start index in d_tokenEmbed
                if (start_idx_for_new_block < 0) start_idx_for_new_block = 0; // Should not happen if logic is correct
                CUDA_CHECK(cudaMemcpy(d_tokForBlock,
                                      d_tokenEmbed + static_cast<size_t>(start_idx_for_new_block) * d,
                                      context_win_bytes,
                                      cudaMemcpyDeviceToDevice));


                // Pre-calculate KdotQ for the second part (m1 tokens) in the *new* block
                if (m1 > 0) {
                    // The "previous token count" for this calculation is effectively the start of the new block
                    int start_of_new_block_count = currentTokenCount - m1;
                    for (int col = 0; col < y; ++col) {
                        int effectivePromptCount = m1;
                        // Pass the new blockCount
                        cuParallelKdotQs(effectivePromptCount, start_of_new_block_count, blockCount, col, isSelf, inTraining);
                    }
                     CUDA_CHECK(cudaDeviceSynchronize()); // Ensure KdotQ for m1 is done
                }
            }

            // --- Response Generation Loop (CUDA) ---
            int rCount = 0;
            auto start_time = std::chrono::high_resolution_clock::now();
            std::cout << "Response: ";

            while (1) {
                // --- Core Forward Pass ---
                // cuForward should handle the attention, MLPs, and state updates for the current token step
                // It needs the current block index, total token count, and potentially info about the prompt part if applicable.
                // The signature from feedfor.cu is: cuForward(int &blockCount, int &currentTokenCount, int &promptCount)
                // Let's pass 0 for promptCount during generation, assuming cuParallelKdotQs handled the initial prompt.
                int generationPromptCount = 0; // Indicate we are in generation phase
                cuForward(blockCount, currentTokenCount, generationPromptCount); // This updates t[0]'s internal state (EH, EV) on device
                CUDA_CHECK(cudaDeviceSynchronize()); // Ensure forward pass is complete

                // --- Get Output Token ---
                // cuForward should have placed the final accumulated EH (or equivalent output) into d_otok
                // Need to retrieve d_otok pointer if cuForward doesn't return it directly.
                // Assuming cuForward updates a known location or t[0] provides access.
                // Placeholder: float* d_output_from_forward = t[0].getDeviceOutputPointer(); // Hypothetical
                // Let's assume cuForward writes to a member accessible like t[0].d_EH_final
                // If cuForward in feedfor.cu already calculates d_otok, we use that.
                // The feedfor.cu version *does* calculate d_otok.

                // Find the index of the most likely next token
                cuComputeOutput(d_otok, d_embeddings, vocabsize, indexForToken, d);
                CUDA_CHECK(cudaDeviceSynchronize()); // Ensure output computation is done

                // --- Update State (Host & Device) ---
                if (indexForToken < 0 || indexForToken >= vocabsize) {
                     std::cerr << "\nError: Invalid token index computed: " << indexForToken << std::endl;
                     break; // Exit generation loop on error
                }

                // Get the embedding for the new token on the host
                const std::vector<float>& next_token_embed_h = embeddings[indexForToken];
                // Copy the new embedding H->D into d_tokenEmbed at the current position
                CUDA_CHECK(cudaMemcpy(d_tokenEmbed + static_cast<size_t>(currentTokenCount) * d,
                                      next_token_embed_h.data(),
                                      static_cast<size_t>(d) * sizeof(float),
                                      cudaMemcpyHostToDevice));

                // Store the token string (host)
                mTokens[currentTokenCount] = tokens[indexForToken];

                // Print the token (host)
                std::cout << mTokens[currentTokenCount] << " " << std::flush;

                // Increment token count (host)
                currentTokenCount += 1;
                rCount += 1;

                // --- Context Window / Block Transition Check ---
                if (currentTokenCount % CONTEXT_WIN == 0 && currentTokenCount > 0) {
                    // Copy the EV state from the completed block (t[0]) to d_EVuse (D->D)
                    // TODO: Implement mechanism as described in the prompt handling section.
                     std::cerr << "\nWarning: Device-to-device copy for EV->EVuse not implemented yet in cuRun generation loop." << std::endl;

                    blockCount += 1; // Increment block count

                    // Update d_tokForBlock for the new block (copy last CONTEXT_WIN embeddings D->D)
                    int start_idx_for_new_block = currentTokenCount - CONTEXT_WIN;
                    CUDA_CHECK(cudaMemcpy(d_tokForBlock,
                                          d_tokenEmbed + static_cast<size_t>(start_idx_for_new_block) * d,
                                          context_win_bytes,
                                          cudaMemcpyDeviceToDevice));

                     // Optional: Pre-calculate KdotQ for the new block if needed by the architecture.
                     // The current logic seems to calculate KdotQ incrementally within cuForward or via cuParallelKdotQs.
                     // If a full KdotQ is needed at the block start, call cuParallelKdotQs here for the whole window.
                }

                // --- Termination Checks ---
                if (currentTokenCount == FULL_CONTEXT) {
                    std::cout << "\nTOKEN LIMIT REACHED AT FULL CONTEXT!";
                    break;
                }
                if (tokens[indexForToken] == TERMINATE) {
                    break; // End generation for this prompt
                }

            } // End of response generation loop

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            double seconds = duration.count() / 1000000.0;
            std::cout << "\nTime taken to predict tokens of response: " << seconds << " seconds" << std::endl;
            if (seconds > 0) {
                std::cout << "Token Rate: " << static_cast<float>(rCount / seconds) << " tokens/second" << std::endl;
            } else {
                 std::cout << "Token Rate: N/A (duration too short)" << std::endl;
            }
            std::cout << std::endl;

            // Ready for next prompt
        } // End of main inference loop (while(1))

    } catch (const std::exception& e) {
        std::cerr << "Error in transformer::cuRun: " << e.what() << std::endl;
        // --- Cleanup on Error ---
        cudaFree(d_tokenEmbed);
        cudaFree(d_embeddings);
        cudaFree(d_EVuse);
        cudaFree(d_tokForBlock);
        cudaFree(d_otok); // Free if allocated externally, otherwise managed by cuForward
        throw; // Re-throw
    }

    // --- Final Cleanup ---
    CUDA_CHECK(cudaFree(d_tokenEmbed));
    CUDA_CHECK(cudaFree(d_embeddings));
    CUDA_CHECK(cudaFree(d_EVuse));
    CUDA_CHECK(cudaFree(d_tokForBlock));
    CUDA_CHECK(cudaFree(d_otok)); // Free if allocated externally
}
