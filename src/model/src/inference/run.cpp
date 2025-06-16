
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
void model::runModel(const std::string& binDirectory)
{
    std::cout << "You are now running the model " << info.modelName << std::endl;
    bool savechat;      // 1 to save chat
    bool newchat;       // 1 for new chat, 0 for endchat
    // take input
    T.currentTokenCount = 0;
    // Construct file paths once, as 'dir' is constant here.
    std::filesystem::path baseDirFs(binDirectory);
    std::string path_qk_bin = (baseDirFs / "QK.bin").string();
    std::string path_qv_bin = (baseDirFs / "QV.bin").string();
    std::string path_kh_bin = (baseDirFs / "KH.bin").string();
    std::string path_hor_bin = (baseDirFs / "HOR.bin").string();
    std::string path_ver_bin = (baseDirFs / "VER.bin").string();

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
        if (!this->T.embeddings.mapped_data || 
            this->T.embeddings.row != this->T.vocabsize || 
            this->T.embeddings.col != d) {
            throw std::runtime_error("CUDA: Vocabulary embeddings mat is not properly initialized or dimensions mismatch.");
        }
        CUDA_CHECK(cudaMemcpy(d_embeddings, this->T.embeddings.mapped_data, vocab_bytes, cudaMemcpyHostToDevice));

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
            if (this->T.blockCount > 1) {
                int start_idx_in_full_context = T.currentTokenCount - CONTEXT_WIN;
                // Safety check: Ensure the start index isn't negative (can happen if total tokens < CONTEXT_WIN)
                if (start_idx_in_full_context < 0) start_idx_in_full_context = 0;

                // Calculate the source pointer offset within d_tokenEmbed
                float* src_ptr = d_tokenEmbed + static_cast<size_t>(start_idx_in_full_context) * d;

                // Perform the device-to-device copy
                CUDA_CHECK(cudaMemcpy(d_tokForBlock,          // Destination buffer
                                        src_ptr,                // Source pointer within d_tokenEmbed
                                        context_win_bytes,      // Number of bytes to copy (size of d_tokForBlock)
                                        cudaMemcpyDeviceToDevice)); // Type of copy
            }
            for (int col = 0; col < y; ++col) {
                // Pass the count of *new* prompt tokens and the total count *before* the prompt
                int effectivePromptCount = this->T.promptCount;
                this->T.cuParallelKdotQs(effectivePromptCount, previousTokenCount, this->T.blockCount, col, isSelf, this->T.inTraining);
            }
            CUDA_CHECK(cudaDeviceSynchronize()); // Ensure KdotQ calculation is done
        }
        else {
            // Case 2: Prompt spans across the current and next block
            int m2 = space_in_current_block;
            int m1 = this->T.promptCount - m2;

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
            for (int i = 0; i < x; ++i) {
                for (int j = 0; j < y; ++j) {
                    try {
                        // Get the device pointer for the EV state of the current head (i, j)
                        float* d_src_ev_ptr = this->T.t[0].b[i][j].d_EV;

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
                        throw;
                    }
                }
            }

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
            this->T.cuForward(this->T.blockCount, this->T.currentTokenCount, generationPromptCount);
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
            std::vector<float> next_token_embed_h = this->T.embeddings(this->T.indexForToken);
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

                for (int i = 0; i < x; ++i) {
                    for (int j = 0; j < y; ++j) {
                        try {
                            float* d_src_ev_ptr = this->T.t[0].b[i][j].d_EV;
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
        cl_int cl_err; // For OpenCL error codes used with CL_CHECK
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
        d_tokenEmbed = cl::Buffer(this->clcontext.context, CL_MEM_READ_WRITE, full_context_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
        d_embeddings = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY, vocab_bytes, nullptr, &cl_err); CL_CHECK(cl_err); // Read-only for inference
        d_EVuse = cl::Buffer(this->clcontext.context, CL_MEM_READ_WRITE, ev_use_bytes, nullptr, &cl_err); CL_CHECK(cl_err); // Holds state between blocks
        d_tokForBlock = cl::Buffer(this->clcontext.context, CL_MEM_READ_WRITE, context_win_bytes, nullptr, &cl_err); CL_CHECK(cl_err); // Holds current block's context

        // Initial Data Transfer: Vocabulary embeddings H->D (once)
        if (!this->T.embeddings.mapped_data ||
            this->T.embeddings.row != this->T.vocabsize ||
            this->T.embeddings.col != d) {
            throw std::runtime_error("OpenCL: Vocabulary embeddings mat is not properly initialized or dimensions mismatch.");
        }
        CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_embeddings, CL_TRUE, 0, vocab_bytes, this->T.embeddings.mapped_data));

        std::string modelFileFullPath = binDirectory; // Define modelFileFullPath, assuming clForward might need it
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
        CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_tokenEmbed, CL_TRUE, prompt_offset_bytes, prompt_bytes, prompt_embeddings_flat.data()));
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
            CL_CHECK(this->clcontext.queue.finish()); // Ensure KdotQ calculation is done
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
                CL_CHECK(this->clcontext.queue.finish()); // Ensure KdotQ for m2 is done
            }

            // --- Transition to the next block ---
            // Copy the EV state from the completed block (t[0]) to d_EVuse (D->D)
            size_t head_ev_bytes = static_cast<size_t>(CONTEXT_WIN) * d * sizeof(float);

            // Check if the total size matches the allocated buffer size
            if (static_cast<size_t>(x) * y * head_ev_bytes != ev_use_bytes) {
                throw std::runtime_error("Mismatch between calculated EVuse size and allocated buffer size during block transition.");
            }

            // Loop through each attention head in the completed block (t[0])
            for (int i = 0; i < x; ++i) {
                for (int j = 0; j < y; ++j) {
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
                        CL_CHECK(this->clcontext.queue.enqueueCopyBuffer(src_ev_buffer, d_EVuse, 0, dest_offset_bytes, head_ev_bytes));
                    }
                    catch (const std::runtime_error& e) { // Catches CL_CHECK errors
                        std::cerr << "OpenCL Runtime Error getting/copying EV buffer for head [" << i << "][" << j << "]: "
                                    << e.what() << std::endl;
                        throw;
                    }
                    catch (const std::exception& e) {
                        // Catch potential errors from getDeviceEVBuffer() or other issues
                        std::cerr << "Standard Exception getting/copying EV buffer for head [" << i << "][" << j << "]: " << e.what() << std::endl;
                        throw; // Re-throw standard exceptions
                    }
                }
            }
            // Ensure all copy operations are completed before proceeding
            CL_CHECK(this->clcontext.queue.finish());

            this->T.blockCount += 1; // Increment block count *once*

            // Prepare d_tokForBlock for the *new* block (contains the m1 tokens + previous context)
            // Copy the last CONTEXT_WIN tokens from d_tokenEmbed to d_tokForBlock
            int start_idx_for_new_block = this->T.currentTokenCount - CONTEXT_WIN; // Start index in d_tokenEmbed
            size_t copy_offset_bytes = static_cast<size_t>(start_idx_for_new_block) * d * sizeof(float);
            if (copy_offset_bytes + context_win_bytes > full_context_bytes) {
                throw std::runtime_error("d_tokForBlock source offset out of bounds.");
            }
            CL_CHECK(this->clcontext.queue.enqueueCopyBuffer(d_tokenEmbed, d_tokForBlock, copy_offset_bytes, 0, context_win_bytes));
            CL_CHECK(this->clcontext.queue.finish()); // Ensure copy is done


            // Pre-calculate KdotQ for the second part (m1 tokens) in the *new* block
            if (m1 > 0) {
                // The "previous token count" for this calculation is effectively the start of the new block's relevant context
                int start_of_new_block_count = this->T.currentTokenCount - m1;
                for (int col = 0; col < y; ++col) {
                    int effectivePromptCount = m1;
                    // Pass the new blockCount
                    this->T.clParallelKdotQs(effectivePromptCount, start_of_new_block_count, this->T.blockCount, col, isSelf, this->T.inTraining);
                }
                CL_CHECK(this->clcontext.queue.finish()); // Ensure KdotQ for m1 is done
            }
        }

        // --- Response Generation Loop (OpenCL) ---
        int rCount = 0;
        auto start_time = std::chrono::high_resolution_clock::now();
        std::cout << "Response: ";

        while (1) {
            // --- Core Forward Pass ---
            int generationPromptCount = 0; // Indicate generation phase
            this->T.clForward(this->T.blockCount, this->T.currentTokenCount, generationPromptCount);
            CL_CHECK(this->clcontext.queue.finish()); // Ensure forward pass is complete

            // --- Get Output Token ---
            if (T.otok.size() != static_cast<size_t>(d)) {
                throw std::runtime_error("EH size in t[0] is incorrect after clForward.");
            }

            // --- Start: Inline Prediction Kernel Launch ---
            { // Scope for temporary prediction buffers
                cl::Buffer d_output;       // Buffer for EH input to kernel
                cl::Buffer d_result_index; // Buffer for kernel output index
                int result_index_val = -1; // Host variable to hold the result

                try {
                    // Create and copy EH to device buffer
                    d_output = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, otok_bytes, T.otok.data(), &cl_err); CL_CHECK(cl_err);
                    // Create buffer for the kernel to write the result index
                    d_result_index = cl::Buffer(this->clcontext.context, CL_MEM_WRITE_ONLY, indexBytes, nullptr, &cl_err); CL_CHECK(cl_err);

                    // Get the kernel
                    cl::Kernel kernel = this->clcontext.kernels.at("compute_prediction"); 

                    // Set arguments based on the kernel signature
                    CL_CHECK(kernel.setArg(0, d_output));
                    CL_CHECK(kernel.setArg(1, d_embeddings)); // Use existing buffer
                    CL_CHECK(kernel.setArg(2, static_cast<cl_int>(this->d))); // dim
                    CL_CHECK(kernel.setArg(3, static_cast<cl_int>(this->T.vocabsize))); // voc
                    CL_CHECK(kernel.setArg(4, d_result_index)); // Output buffer for the index

                    // --- Enqueue Kernel ---
                    cl::NDRange global(1);
                    cl::NDRange local(1);
                    CL_CHECK(this->clcontext.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local));

                    // --- Read Result Back ---
                    CL_CHECK(this->clcontext.queue.enqueueReadBuffer(d_result_index, CL_TRUE, 0, indexBytes, &result_index_val));

                    // --- Update Output Parameter ---
                    this->T.indexForToken = result_index_val; // Update class member

                }
                catch (const std::out_of_range& oor) { // For kernels.at()
                    std::cerr << "OpenCL Error: Kernel 'compute_prediction' not found: " << oor.what() << std::endl;
                    this->T.indexForToken = -1;
                    throw;
                }
                catch (const std::runtime_error& e) { // Catches CL_CHECK errors
                    std::cerr << "OpenCL Runtime Error during inline prediction kernel launch: " << e.what() << std::endl;
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
            if (this->T.indexForToken < 0 || this->T.indexForToken >= this->T.embeddings.row) {
                throw std::out_of_range("OpenCL: indexForToken out of bounds for embeddings mat.");
            }
            float* next_token_embed_ptr = this->T.embeddings.mapped_data + (static_cast<size_t>(this->T.indexForToken) * d);

            // Copy the new embedding H->D into d_tokenEmbed at the current position
            size_t next_token_offset_bytes = static_cast<size_t>(this->T.currentTokenCount) * d * sizeof(float);
            size_t next_token_bytes = static_cast<size_t>(d) * sizeof(float);
            if (next_token_offset_bytes + next_token_bytes > full_context_bytes) {
                throw std::runtime_error("Next token copy exceeds d_tokenEmbed bounds.");
            }
            CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_tokenEmbed, CL_TRUE, next_token_offset_bytes, next_token_bytes, next_token_embed_ptr));

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

                this->T.blockCount += 1; // Increment block count

                // Update d_tokForBlock for the new block (copy last CONTEXT_WIN embeddings D->D)
                int start_idx_for_new_block = this->T.currentTokenCount - CONTEXT_WIN;
                size_t copy_offset_bytes = static_cast<size_t>(start_idx_for_new_block) * d * sizeof(float);
                if (copy_offset_bytes + context_win_bytes > full_context_bytes) {
                    throw std::runtime_error("d_tokForBlock source offset out of bounds during generation.");
                }
                CL_CHECK(this->clcontext.queue.enqueueCopyBuffer(d_tokenEmbed, d_tokForBlock, copy_offset_bytes, 0, context_win_bytes));
                CL_CHECK(this->clcontext.queue.finish()); // Ensure copy is done
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
            // this->T.getEmbedding(this->T.mTokens[this->T.currentTokenCount+i], this->T.tokenEmbed[this->T.currentTokenCount+i]); // Incorrect
            std::vector<float> temp_embed(d);
            this->T.getEmbedding(this->T.mTokens[this->T.currentTokenCount+i], temp_embed);
            setRow(this->T.tokenEmbed, this->T.currentTokenCount+i, temp_embed);
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
                    // this->T.tokForBlock[c + k] = this->T.tokenEmbed[this->T.currentTokenCount + k]; // Incorrect
                    setRow(this->T.tokForBlock, c + k, this->T.tokenEmbed(this->T.currentTokenCount + k));
                }
            }
            for(int i = 0; i < x; i++) {
                for(int j = 0; j < y; j++) {
                    for(int k = 0; k < this->T.promptCount; k++) {
                        // this->T.t[0].b[i][j].EV[c-1+k] = this->T.tokenEmbed[this->T.currentTokenCount + k]; // Incorrect
                        setRow(this->T.t[0].b[i][j].EV, c-1+k, this->T.tokenEmbed(this->T.currentTokenCount + k));
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
                        // this->T.t[0].b[i][j].EV[c-1+k] = this->T.tokenEmbed[this->T.currentTokenCount + k]; // Incorrect
                        setRow(this->T.t[0].b[i][j].EV, c-1+k, this->T.tokenEmbed(this->T.currentTokenCount + k));
                    }
                }
            }
            for(int i = 0; i < m2; i++) {
                // this->T.tokForBlock[i] = this->T.tokenEmbed[this->T.currentTokenCount + i]; // Incorrect
                setRow(this->T.tokForBlock, i, this->T.tokenEmbed(this->T.currentTokenCount + i));
            }
            this->T.computeKdotQs(this->T.promptCount, this->T.currentTokenCount, this->T.blockCount, isSelf, this->T.inTraining);
            // token limit reached for first block
            this->T.currentTokenCount += m1;
            // set vertical retention vectors
            for(int i = 0; i < x; i++) {
                for(int j = 0; j < y; j++) {
                    for(int k = 0; k < CONTEXT_WIN; k++) {
                        this->T.EVuse[i][j][k] = this->T.t[0].b[i][j].EV(k);
                    }
                }
            }
            this->T.blockCount += 1;
            for(int i = 0; i < CONTEXT_WIN; i++) {
                // this->T.tokForBlock[i] = this->T.tokenEmbed[this->T.currentTokenCount - CONTEXT_WIN + i]; // Incorrect
                setRow(this->T.tokForBlock, i, this->T.tokenEmbed(this->T.currentTokenCount - CONTEXT_WIN + i));
            }
            this->T.currentTokenCount += m2;
            this->T.blockCount += 1;
        }
        // caculate response
        int rCount = 0;
        while (1) {
            float k, l;   // for row and column sum
            // forprop for EH and EV
            for(int i = 0; i < x; i++) {
                for(int j = 0; j < y; j++) {
                    // block specific KdotQ
                    fetchmat(T.t[0].b[i][j].qkCache, T.blockCount, i, j, path_qk_bin);
                    fetchmat(T.t[0].b[i][j].qvCache, T.blockCount, i, j, path_qv_bin);
                    fetchmat(T.t[0].b[i][j].khCache, T.blockCount, i, j, path_kh_bin);
                    fetchmlp(T.t[0].b[i][j].hor, T.blockCount, i, j, path_hor_bin);
                    fetchmlp(T.t[0].b[i][j].ver, T.blockCount, i, j, path_ver_bin);
                    if(this->T.blockCount == 1) {
                        // Adapting to use make2dVector pattern, assuming computeKdotQ expects std::vector<std::vector<float>>
                        // Determine relevant rows/cols for make2dVector based on currentTokenCount, promptCount etc.
                        int kdotq_rows = std::min(this->T.currentTokenCount, CONTEXT_WIN); // Example, adjust as needed
                        int kdotq_cols = kdotq_rows; // For self-attention
                        std::vector<std::vector<float>> kdotq_vec = this->T.t[0].b[i][j].KdotQ.make2dVector(this->T.t[0].b[i][j].KdotQ, kdotq_rows, kdotq_cols);
                        std::vector<std::vector<float>> tokenembed_vec = this->T.tokenEmbed.make2dVector(this->T.tokenEmbed, kdotq_rows, d);
                        computeKdotQ(kdotq_vec, tokenembed_vec, this->T.t[0].b[i][j].qkCache, this->T.currentTokenCount, this->T.promptCount, isSelf);
                        // Note: If computeKdotQ modifies kdotq_vec, changes are not written back to mat KdotQ unless handled.
                    }
                    else {
                        int kdotq_rows = std::min(this->T.currentTokenCount % CONTEXT_WIN, CONTEXT_WIN); // Example
                        if (kdotq_rows == 0 && this->T.currentTokenCount > 0) kdotq_rows = CONTEXT_WIN; // If at boundary
                        int kdotq_cols = kdotq_rows;
                        std::vector<std::vector<float>> kdotq_vec = this->T.t[0].b[i][j].KdotQ.make2dVector(this->T.t[0].b[i][j].KdotQ, kdotq_rows, kdotq_cols);
                        std::vector<std::vector<float>> tokforblock_vec = this->T.tokForBlock.make2dVector(this->T.tokForBlock, kdotq_rows, d);
                        computeKdotQ(kdotq_vec, tokforblock_vec, this->T.EVuse[i][j], this->T.t[0].b[i][j].qkCache, this->T.currentTokenCount, this->T.promptCount, this->T.blockCount, isSelf);
                    }
                    // number of tokens in context window of this block
                    int count = std::abs(this->T.currentTokenCount - n * (this->T.blockCount-1));
                    // calculate KdotQ and head
                    mat head = LOTA(this->T.t[0].b[i][j].KdotQ, count, isSelf);

                    // Initialize dh and dv to zeros, similar to forward.cpp
                    this->T.t[0].b[i][j].dh.assign(d, 0.0f);
                    this->T.t[0].b[i][j].dv.assign(d, 0.0f);

                    // get weighted sums
                    for(int w = 0; w < count; w++) {
                        k = 0;          // row sum
                        l = 0;          // column sum
                        // Corrected loop limit for self-attention and loop variable, added boundary checks
                        int limit_z = isSelf ? (w + 1) : count;
                        limit_z = std::min(limit_z, static_cast<int>(head.col)); // Ensure z is within head's column bounds
                        for(int z = 0; z < limit_z; z++) { // Corrected loop increment to z++
                            if (w < head.row) { // Ensure w is within head's row bounds
                                k += head(w, z);
                            }
                            if (z < head.row && w < head.col) { // Ensure z is within head's row and w within col bounds for l_sum
                                l += head(z, w);
                            }
                        }

                        std::vector<float> term_h_source = (this->T.blockCount == 1) ? this->T.tokenEmbed(w) : this->T.tokForBlock(w); // w is the token index
                        std::vector<float> term_v_source = (this->T.blockCount == 1) ? this->T.tokenEmbed(w) : this->T.EVuse[i][j][w]; // w is the token index

                        for(int m_idx = 0; m_idx < d; ++m_idx) {
                            this->T.t[0].b[i][j].dh[m_idx] += (k * term_h_source[m_idx]);
                            this->T.t[0].b[i][j].dv[m_idx] += (l * term_v_source[m_idx]);
                        }
                    }
                    this->T.t[0].b[i][j].dh = dot(this->T.t[0].b[i][j].dh, this->T.t[0].b[i][j].khCache);
                    this->T.t[0].b[i][j].dv = dot(this->T.t[0].b[i][j].dv, this->T.t[0].b[i][j].qvCache);

                    for(size_t m_idx = 0; m_idx < this->T.t[0].b[i][j].EH.size(); ++m_idx) {
                        this->T.t[0].b[i][j].hor.input[m_idx] = this->T.t[0].b[i][j].EH[m_idx] + this->T.t[0].b[i][j].dh[m_idx];
                    }

                    // Align ver.input calculation with forward.cpp
                    this->T.t[0].b[i][j].ver.input.assign(d, 0.0f); // Reset ver.input
                    for (int token_iter = 0; token_iter < count; ++token_iter) {
                        std::vector<float> ev_row_for_sum = this->T.t[0].b[i][j].EV(token_iter);
                        for (int m_idx = 0; m_idx < d; ++m_idx) {
                            this->T.t[0].b[i][j].ver.input[m_idx] += ev_row_for_sum[m_idx];
                        }
                    }
                    for (int m_idx = 0; m_idx < d; ++m_idx) { // Add dv
                        this->T.t[0].b[i][j].ver.input[m_idx] += this->T.t[0].b[i][j].dv[m_idx];
                    }

                    // Use this->T.l (transformer's MLP layers) instead of attention's column sum 'l'
                    this->T.t[0].b[i][j].hor.forward(d, this->T.l);
                    this->T.t[0].b[i][j].ver.forward(d, this->T.l);

                    // AND gate for the final output
                    std::vector<float> relu_hor_output = ReLU(this->T.t[0].b[i][j].hor.output);
                    std::vector<float> relu_ver_output = ReLU(this->T.t[0].b[i][j].ver.output);

                    for(size_t m_idx=0; m_idx < this->T.t[0].b[i][j].EH.size(); ++m_idx) {
                        this->T.t[0].b[i][j].EH[m_idx] += relu_hor_output[m_idx];
                    }

                    // Align EV update with forward.cpp: update all relevant rows
                    for (int token_iter = 0; token_iter < count; ++token_iter) {
                        std::vector<float> ev_row_to_update = this->T.t[0].b[i][j].EV(token_iter); // Get current row
                        for(size_t m_idx=0; m_idx < ev_row_to_update.size(); ++m_idx)
                            ev_row_to_update[m_idx] += relu_ver_output[m_idx];
                        setRow(this->T.t[0].b[i][j].EV, token_iter, ev_row_to_update); // Set updated row
                    }
                }
                // Accumulate EH from the last attention head of the partial attention layer into block's EH (or transformer's otok)
                // Assuming this->T.otok is the final output token embedding for the transformer
                if (i == 0) 
                    this->T.otok.assign(d, 0.0f); // Reset for this token step
                for(size_t m_idx=0; m_idx < d; ++m_idx) {
                    this->T.otok[m_idx] += this->T.t[0].b[i][y-1].EH[m_idx];
                }
            }
            computeOutput(this->T.otok, this->T.embeddings, this->T.vocabsize, this->T.indexForToken); // Use transformer's otok

            setRow(this->T.tokenEmbed, this->T.currentTokenCount, this->T.embeddings(this->T.indexForToken));
            this->T.mTokens[this->T.currentTokenCount] = this->T.tokens[this->T.indexForToken];
            std::cout << this->T.mTokens[this->T.currentTokenCount] << " ";
            this->T.currentTokenCount += 1;
            // check for local context
            if(this->T.currentTokenCount%CONTEXT_WIN == 0) {
                // set vertical context retention for next blocks
                for(int i = 0; i < x; i++) {
                    for(int j = 0; j < y; j++) {
                        for(int k = 0; k < CONTEXT_WIN; k++) {
                            this->T.EVuse[i][j][k] = this->T.t[0].b[i][j].EV(k);
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
            fflush(this->chat);
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
