#ifdef USE_OPENCL
#include "include/transformer.hpp"
#include <maths.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <chrono>
#include <stdexcept>
#include <cmath>        // For std::abs
#include <limits>       // For numeric_limits


// Helper struct to hold per-head sub-buffers for inference
struct allBuffers {
    // imp: context_win = matheights
    // imp: full_context = context_win x number_of_blocks
    cl::Buffer d_qk, d_kh, d_qv;                // d x d
    cl::Buffer d_K, d_Q, d_KdotQ, d_head;       // context_win x context_win
    cl::Buffer d_row_sums, d_col_sums;          // context_win (sum(Kdot[i,j]) for i row or col)
    cl::Buffer d_dh_accum, d_dv_accum;          // d (sum(accum[i]*d_K(i)))
    cl::Buffer d_h, d_v, d_EH;                  // d
    cl::Buffer d_EV;                            // context_win x d
    cl::Buffer d_hin, d_vin;                    // d
    cl::Buffer d_hout, d_vout;                  // d
    cl::Buffer d_relu_hout, d_relu_vout;        // d
    cl::Buffer d_hor_a, d_hor_b;                // d
    cl::Buffer d_ver_a, d_ver_b;                // d
    cl::Buffer d_mlp_pre_activation;            // d

    std::vector<cl::Buffer> d_hor_activations;  // layers x d
    std::vector<cl::Buffer> d_ver_activations;  // layers x d
    std::vector<cl::Buffer> d_hor_weights;      // (layers-1) x (d x d)
    std::vector<cl::Buffer> d_ver_weights;      // (layers-1) x (d x d)
};


/**
 * @brief run transformer to get sequence2 from sequence1
 */
void transformer::clRunBuffer()
{
    // Set for inference
    inTraining = 0;
    if (blocks.empty()) {
            throw std::runtime_error("Transformer block 't' is not initialized in clRun.");
    }
    if (blocks[0].b.empty() || blocks[0].b[0].empty()) {
        throw std::runtime_error("Attention heads within the transformer block 't[0]' are not initialized in clRun.");
    }

    // --- Device Memory Allocation ---
    cl::Buffer d_tokenEmbed;      // Holds all token embeddings (sequence1 + generated) up to FULL_CONTEXT
    cl::Buffer d_embeddings;      // Holds the vocabulary embeddings
    cl::Buffer d_EVuse;           // Holds the EV state from the previous block (flattened: x * y * CONTEXT_WIN * d)
    cl::Buffer d_tokForBlock;     // Holds the token embeddings for the current block's context window

    // Determine sizes
    size_t full_context_bytes = static_cast<size_t>(FULL_CONTEXT) * d * sizeof(float);
    size_t vocab_bytes = static_cast<size_t>(vocabsize) * d * sizeof(float);
    size_t context_win_bytes = static_cast<size_t>(CONTEXT_WIN) * d * sizeof(float);
    // EVuse needs space for all attention heads' EV states for a full context window
    // Structure: [x][y][CONTEXT_WIN][d] flattened
    size_t ev_use_bytes = static_cast<size_t>(x) * y * CONTEXT_WIN * d * sizeof(float);
    size_t otok_bytes = static_cast<size_t>(d) * sizeof(float); // Size of EH output
    size_t indexBytes = sizeof(int); // Size for the result index
    int previousTokenCount = currentTokenCount;
    if (currentTokenCount >= FULL_CONTEXT) {
        throw std::runtime_error("\nError: Adding sequence1 exceeds FULL_CONTEXT limit (" + std::to_string(FULL_CONTEXT) + "). Please restart or shorten sequence1.");
    }

    cl_int cl_err; // For OpenCL error codes

    try {
        // Create Buffers
        d_tokenEmbed = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, full_context_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        d_embeddings = cl::Buffer(clcontext.context, CL_MEM_READ_ONLY, vocab_bytes, nullptr, &cl_err); CL_CHECK(cl_err); // Read-only for inference
        d_EVuse = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, ev_use_bytes, nullptr, &cl_err); CL_CHECK(cl_err); // Holds state between blocks
        d_tokForBlock = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, context_win_bytes, nullptr, &cl_err); CL_CHECK(cl_err); // Holds current block's context

        // --- Initial Data Transfer ---
        // Flatten and copy vocabulary embeddings H->D (once)
        // Copy vocabulary embeddings (from mat.mapped_data) H->D (once)
        if (!embeddings.mapped_data) {
            throw std::runtime_error("Vocabulary embeddings (mat) are not mapped.");
        }
        if (embeddings.row != vocabsize || embeddings.col != d) {
            throw std::runtime_error("Vocabulary embeddings (mat) dimensions mismatch vocabsize or embedding dimension.");
        }
        if (embeddings.mapped_size < vocab_bytes) {
            throw std::runtime_error("Vocabulary embeddings (mat) mapped size is insufficient.");
        }
        CL_CHECK(clcontext.queue.enqueueWriteBuffer(d_embeddings, CL_TRUE, 0, vocab_bytes, embeddings.mapped_data));

        // --- Main Inference Loop ---
        // Get embeddings for the sequence1 tokens and copy H->D incrementally
        std::vector<float> sequence1_embeddings_flat;
        sequence1_embeddings_flat.reserve(static_cast<size_t>(sequence1Count) * d);
        for (int i = 0; i < sequence1Count; ++i) {
            std::vector<float> current_embed(d);
            // Index in mTokens is previousTokenCount + i
            getEmbedding(mTokens[previousTokenCount + i], current_embed);
            sequence1_embeddings_flat.insert(sequence1_embeddings_flat.end(), current_embed.begin(), current_embed.end());
        }
        // Copy the new sequence1 embeddings to the correct offset in d_tokenEmbed
        size_t sequence1_offset_bytes = static_cast<size_t>(previousTokenCount) * d * sizeof(float);
        size_t sequence1_bytes = sequence1_embeddings_flat.size() * sizeof(float);
        if (sequence1_offset_bytes + sequence1_bytes > full_context_bytes) {
                throw std::runtime_error("Sequence1 copy exceeds d_tokenEmbed bounds.");
        }
        CL_CHECK(clcontext.queue.enqueueWriteBuffer(d_tokenEmbed, CL_TRUE, sequence1_offset_bytes, sequence1_bytes, sequence1_embeddings_flat.data()));
        sequence1_embeddings_flat.clear(); // Free host memory

        // --- Sequence1 Placement & Initial KdotQ ---
        // Calculate offset within the current block's window
        int current_block_idx_0based = (blockCount == 0) ? 0 : blockCount - 1; // Assuming blockCount is 1-based from training
        int tokens_in_block_before_sequence1 = (current_block_idx_0based == 0) ? previousTokenCount : (previousTokenCount % CONTEXT_WIN);
        int space_in_current_block = CONTEXT_WIN - tokens_in_block_before_sequence1;

        if (sequence1Count <= space_in_current_block) {
            // Case 1: Sequence1 fits entirely within the current block
            // Pre-calculate KdotQ for the sequence1 tokens added
            int effectiveSequence1Count = sequence1Count; // Number of new tokens
            clKdotQ4Infer(effectiveSequence1Count, previousTokenCount, blockCount, isSelf, inTraining);
            CL_CHECK(clcontext.queue.finish()); // Ensure KdotQ calculation is done
        }
        else {
            // Case 2: Sequence1 spans across the current and next block
            int m2 = space_in_current_block; // Tokens fitting in the current block
            int m1 = sequence1Count - m2;       // Tokens going to the next block

            // Process the first part (m2 tokens) in the current block
            if (m2 > 0) {
                int effectiveSequence1Count = m2;
                clKdotQ4Infer(effectiveSequence1Count, previousTokenCount, blockCount, isSelf, inTraining);
                CL_CHECK(clcontext.queue.finish()); // Ensure KdotQ for m2 is done
            }

            // --- Transition to the next block ---
            // Copy the EV state from the completed block (blocks[0]) to d_EVuse (D->D)
            size_t head_ev_bytes = static_cast<size_t>(CONTEXT_WIN) * d * sizeof(float);

            // Check if the total size matches the allocated buffer size
            if (static_cast<size_t>(x) * y * head_ev_bytes != ev_use_bytes) {
                throw std::runtime_error("Mismatch between calculated EVuse size and allocated buffer size during block transition.");
            }

            // Loop through each attention head in the completed block (blocks[0])
            for (int i = 0; i < x; ++i) { // Iterate through layers (rows)
                for (int j = 0; j < y; ++j) { // Iterate through parallels (columns)
                    try {
                        // Get the device buffer for the EV state of the current head (i, j)
                        cl::Buffer& src_ev_buffer = blocks[0].b[i][j].getDeviceEVBuffer(); // Use the newly added getter

                        // Calculate the destination offset in the flattened d_EVuse buffer
                        size_t dest_offset_bytes = (static_cast<size_t>(i) * y + j) * head_ev_bytes;

                        // Boundary check for safety
                        if (dest_offset_bytes + head_ev_bytes > ev_use_bytes) {
                            throw std::out_of_range("EVuse destination offset out of bounds for head [" +
                                                        std::to_string(i) + "][" + std::to_string(j) + "].");
                        }

                        // Enqueue the device-to-device buffer copy
                        CL_CHECK(clcontext.queue.enqueueCopyBuffer(src_ev_buffer, d_EVuse, 0, dest_offset_bytes, head_ev_bytes));
                    }
                    catch (const std::exception& e) {
                        // Catch potential errors from getDeviceEVBuffer() or other issues
                        std::cerr << "Standard Exception getting/copying EV buffer for head [" << i << "][" << j << "]: " << e.what() << std::endl;
                        throw; // Re-throw standard exceptions
                    }
                }
            }

            // Ensure all copy operations are completed before proceeding
            CL_CHECK(clcontext.queue.finish());

            blockCount += 1; // Increment block count *once*

            // Prepare d_tokForBlock for the *new* block (contains the m1 tokens + previous context)
            // Copy the last CONTEXT_WIN tokens from d_tokenEmbed to d_tokForBlock
            int start_idx_for_new_block = currentTokenCount - CONTEXT_WIN; // Start index in d_tokenEmbed
            size_t copy_offset_bytes = static_cast<size_t>(start_idx_for_new_block) * d * sizeof(float);
            if (copy_offset_bytes + context_win_bytes > full_context_bytes) {
                throw std::runtime_error("d_tokForBlock source offset out of bounds.");
            }
            CL_CHECK(clcontext.queue.enqueueCopyBuffer(d_tokenEmbed, d_tokForBlock, copy_offset_bytes, 0, context_win_bytes));
            CL_CHECK(clcontext.queue.finish()); // Ensure copy is done

            // Pre-calculate KdotQ for the second part (m1 tokens) in the *new* block
            if (m1 > 0) {
                // The "previous token count" for this calculation is effectively the start of the new block's relevant context
                int start_of_new_block_count = currentTokenCount - m1;
                int effectiveSequence1Count = m1;
                clKdotQ4Infer(effectiveSequence1Count, start_of_new_block_count, blockCount, isSelf, inTraining);
                CL_CHECK(clcontext.queue.finish()); // Ensure KdotQ for m1 is done
            }
        }

        // --- Sequence2 Generation Loop (OpenCL) ---
        int rCount = 0;
        auto start_time = std::chrono::high_resolution_clock::now();
        std::cout << "Sequence2: ";

        while (1) {
            // --- Core Forward Pass ---
            // clForward handles attention, MLPs, and state updates for the current step.
            // It needs the current block index and total token count.
            // Pass 0 for sequence1Count during generation.
            int generationSequence1Count = 0; // Indicate generation phase
            clForward(blockCount, currentTokenCount, generationSequence1Count); // Updates blocks[0]'s internal state (EH, EV)
            CL_CHECK(clcontext.queue.finish()); // Ensure forward pass is complete

            // --- Get Output Token ---
            // Assume clForward updated the host blocks[0].EH. If not, read back from device.
            if (otok.size() != static_cast<size_t>(d)) {
                throw std::runtime_error("EH size in blocks[0] is incorrect after clForward.");
            }

            {
                cl::Buffer d_output;        // Buffer for EH input to kernel
                cl::Buffer d_result_index;  // Buffer for kernel output index
                int result_index_val = -1;  // Host variable to hold the result
                d_output = cl::Buffer(clcontext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, otok_bytes, otok.data(), &cl_err); CL_CHECK(cl_err);
                d_result_index = cl::Buffer(clcontext.context, CL_MEM_WRITE_ONLY, indexBytes, nullptr, &cl_err); CL_CHECK(cl_err);

                try {
                    cl::Kernel kernel = clcontext.kernels.at((contextTrain == 0) ? "kernelComputePrediction" : "kernelComputePredictionWithScores"); // Use member function
                    CL_CHECK(kernel.setArg(0, d_output));
                    CL_CHECK(kernel.setArg(1, d_embeddings));
                    CL_CHECK(kernel.setArg(2, d_result_index));
                    CL_CHECK(kernel.setArg(3, static_cast<cl_int>(d)));
                    CL_CHECK(kernel.setArg(4, static_cast<cl_int>(vocabsize)));
                    cl::NDRange global(1);
                    cl::NDRange local(1);
                    CL_CHECK(clcontext.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local));
                    CL_CHECK(clcontext.queue.enqueueReadBuffer(d_result_index, CL_TRUE, 0, indexBytes, &result_index_val));
                    indexForToken = result_index_val; // Update class member
                } 
                catch (const std::out_of_range& oor) { // Catch if kernel "compute_prediction" is not found
                    std::cerr << "Error: Kernel 'kernelComputePrediction' not found. " << oor.what() << std::endl;
                    indexForToken = -1; // Indicate error
                    throw; // Re-throw
                }
            }

            // --- Update State (Host & Device) ---
            if (indexForToken < 0 || indexForToken >= vocabsize) {
                std::cerr << "\nError: Invalid token index computed: " << indexForToken << std::endl;
                break; // Exit generation loop on error
            }

            // Get the embedding for the new token on the host
            // Access the row for indexForToken directly from embeddings.mapped_data
            if (!embeddings.mapped_data) {
                throw std::runtime_error("Vocabulary embeddings (mat) are not mapped for next token fetch.");
            }
            if (indexForToken < 0 || indexForToken >= embeddings.row) { // embeddings.row should be vocabsize
                throw std::out_of_range("indexForToken is out of bounds for vocabulary embeddings.");
            }
            // Calculate pointer to the start of the specific token's embedding data
            float* next_token_embed_ptr = embeddings.mapped_data + (static_cast<size_t>(indexForToken) * d);

            // Copy the new embedding H->D into d_tokenEmbed at the current position
            size_t next_token_offset_bytes = static_cast<size_t>(currentTokenCount) * d * sizeof(float);
            size_t next_token_bytes = static_cast<size_t>(d) * sizeof(float);
            if (next_token_offset_bytes + next_token_bytes > full_context_bytes) {
                throw std::runtime_error("Next token copy exceeds d_tokenEmbed bounds.");
            }
            CL_CHECK(clcontext.queue.enqueueWriteBuffer(d_tokenEmbed, CL_TRUE, next_token_offset_bytes, next_token_bytes, next_token_embed_ptr));

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
                // Copy the EV state from the completed block (blocks[0]) to d_EVuse (D->D)
                std::cerr << "\nWarning: Device-to-device copy for EV->EVuse not implemented yet in clRun generation loop.\n" << std::endl;
                // queue.finish();

                blockCount += 1; // Increment block count

                // Update d_tokForBlock for the new block (copy last CONTEXT_WIN embeddings D->D)
                int start_idx_for_new_block = currentTokenCount - CONTEXT_WIN;
                size_t copy_offset_bytes = static_cast<size_t>(start_idx_for_new_block) * d * sizeof(float);
                if (copy_offset_bytes + context_win_bytes > full_context_bytes) {
                    throw std::runtime_error("d_tokForBlock source offset out of bounds during generation.");
                }
                CL_CHECK(clcontext.queue.enqueueCopyBuffer(d_tokenEmbed, d_tokForBlock, copy_offset_bytes, 0, context_win_bytes));
                CL_CHECK(clcontext.queue.finish()); // Ensure copy is done

                // Optional: Pre-calculate KdotQ for the new block if needed.
            }

            // --- Termination Checks ---
            if (currentTokenCount == FULL_CONTEXT) {
                std::cout << "\nTOKEN LIMIT REACHED AT FULL CONTEXT (" << FULL_CONTEXT << ")!" << std::endl;
                break;
            }
            if (tokens[indexForToken] == "</s>") {
                break; // End generation for this sequence1
            }
        } // End of sequence2 generation loop
        resCount = rCount;
    }
    catch (const std::exception& e) { // Catches std::runtime_error from CL_CHECK and other std exceptions
        std::cerr << "Error in transformer::clRun: " << e.what() << std::endl;
        // Cleanup is handled by RAII for cl::Buffer
        throw; // Re-throw
    }
}

#endif // USE_OPENCL
