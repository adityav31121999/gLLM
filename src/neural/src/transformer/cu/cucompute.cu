
// compute kernels and functions
#include "include/attention.hpp"    // EMBEDDING, SCALING and CONTEXT_WIN, etc.
#include "include/block.hpp"
#include "include/transformer.hpp"
#include <maths.hpp>
#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

// Helper macro for CUDA error checking
#define CUDA_CHECK(call) do {       \
    cudaError_t err = call;         \
    if (err != cudaSuccess) {       \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        /* Consider more robust cleanup here, maybe RAII */         \
        throw std::runtime_error(cudaGetErrorString(err));          \
    } \
} while (0)


// Define thread block dimensions (tune these based on your GPU architecture)
#define THREADS_PER_BLOCK_X 16      // or 32
#define THREADS_PER_BLOCK_Y 16      // or 32


/**
 * @brief compute KdotQ of specific parallel (or column of block vector b) using CUDA
 * kernels. This optimized version allocates large buffers on the GPU once, copies data 
 * in batches, launches kernels using offsets into these buffers, copies results back, 
 * and then frees. Mirrors the logic of the C++ parallelKdotQs function.
 * @param promptCount number of new tokens in the prompt being processed in this step.
 * @param currentTokenCount total number of tokens processed *before* this step across all blocks.
 * @param blockCount 1-based index of the current block being processed.
 * @param column 0-based index of the parallel (column in the block's attention grid) to compute.
 * @param isSelf true for self-attention, false for cross-attention.
 * @param inTraining true if in training mode, false if in inference mode.
 */
void transformer::cuParallelKdotQs(int& promptCount, int& currentTokenCount, int& blockCount, int& column, bool& isSelf,
    bool& inTraining) 
{
    // --- Basic Sanity Checks ---
    if (blockCount < 1 || blockCount > m) {
        fprintf(stderr, "Error in cuParallelKdotQs: Invalid blockCount %d (max %d)\n", blockCount, m);
        return;
    }
    if (column < 0 || column >= y) {
        fprintf(stderr, "Error in cuParallelKdotQs: Invalid column %d (max %d)\n", column, y-1);
        throw std::out_of_range("Invalid column index");
    }

    // initiate a block that reference the current block operations
    block& current_block = (inTraining == 1) ? t[blockCount - 1] : t[0]; // 0-based index for vector t
    if (x <= 0) {
        fprintf(stderr, "Warning in cuParallelKdotQs: Invalid number of parallels (layers) x = %d. No heads to process.\n", x);
        return; // No heads to process
    }
    if (promptCount < 0) {
         fprintf(stderr, "Warning in cuParallelKdotQs: promptCount is negative (%d). Setting to 0.\n", promptCount);
         return;
    }


    // --- Pre-computation and Setup ---
    const float inv_scaling = 1.0f / sqrtf(EMBEDDING);       // SCALING defined in attention.hpp
    const int embedding_dim = d;                    // 'd' is transformer's embedding dimension
    const int context_win_size = n;                 // 'n' is context window per head from transformer params
    const int kdotq_full_width = context_win_size;  // Max width/height of KdotQ matrix per head
    const int num_heads_in_parallel = x;            // Number of heads in this column (layers)

    // --- Data Sizes Per Head (in elements) ---
    // Ensure context_win_size and embedding_dim are positive
    if (context_win_size <= 0 || embedding_dim <= 0) {
         fprintf(stderr, "Error: Invalid dimensions (context_win_size=%d, embedding_dim=%d)\n", context_win_size, embedding_dim);
         return;
    }
    const size_t k_q_ev_head_elems = static_cast<size_t>(context_win_size) * embedding_dim;    // n * d
    const size_t kdotq_head_elems = static_cast<size_t>(context_win_size) * context_win_size;  // n * n
    const size_t qkcache_head_elems = static_cast<size_t>(embedding_dim) * embedding_dim;      // d * d

    // --- Total Data Sizes for the Parallel (in elements) ---
    const size_t total_k_elems = num_heads_in_parallel * k_q_ev_head_elems;     // x * n * d
    const size_t total_q_elems = num_heads_in_parallel * k_q_ev_head_elems;     // x * n * d
    const size_t total_evp_elems = num_heads_in_parallel * k_q_ev_head_elems;   // x * n * d
    const size_t total_m_elems = num_heads_in_parallel * qkcache_head_elems;    // x * d * d
    const size_t total_kdotq_elems = num_heads_in_parallel * kdotq_head_elems;  // x * n * n

    // --- Total Memory Comsumption (in bytes) ---
    const size_t total_k_bytes = total_k_elems * sizeof(float);
    const size_t total_q_bytes = total_q_elems * sizeof(float);
    const size_t total_evp_bytes = total_evp_elems * sizeof(float);
    const size_t total_m_bytes = total_m_elems * sizeof(float);
    const size_t total_kdotq_bytes = total_kdotq_elems * sizeof(float);

    // --- Device Pointers (Large Buffers) ---
    float* d_all_kdotq = nullptr;
    float* d_all_keys = nullptr;
    float* d_all_querys = nullptr;
    float* d_all_M = nullptr;
    float* d_all_EVp = nullptr;
    float* d_transformer_tokenEmbed_flat = nullptr;     // Shared, single copy
    float* d_block_tokForBlock_flat = nullptr;          // Shared, single copy

    // --- Host-side Aggregated Data Buffers ---
    std::vector<float> h_all_keys_flat;
    std::vector<float> h_all_querys_flat;
    std::vector<float> h_all_M_flat;
    std::vector<float> h_all_EVp_flat;
    std::vector<float> h_all_kdotq_flat;                // Allocate later if needed for copy-back
    std::vector<float> h_transformer_tokenEmbed_flat;   // Flattened global embeddings
    std::vector<float> h_block_tokForBlock_flat;        // Flattened block-local embeddings

    // --- Temporary Host Buffers for Flattening ---
    std::vector<float> temp_flat_buffer;

    try {
        // --- Allocate Large Device Buffers ---
        CUDA_CHECK(cudaMalloc(&d_all_kdotq, total_kdotq_bytes));
        // Initialize KdotQ to 0 or NaN? Let's initialize to 0 for safety.
        CUDA_CHECK(cudaMemset(d_all_kdotq, 0, total_kdotq_bytes));

        if (inTraining) {
            CUDA_CHECK(cudaMalloc(&d_all_keys, total_k_bytes));
            CUDA_CHECK(cudaMalloc(&d_all_querys, total_q_bytes));
            h_all_keys_flat.reserve(total_k_elems);
            h_all_querys_flat.reserve(total_q_elems);
        } 
        else {
            // Inference
            CUDA_CHECK(cudaMalloc(&d_all_M, total_m_bytes));
            h_all_M_flat.reserve(total_m_elems);

            if (blockCount == 1) {
                // Need global tokenEmbed
                h_transformer_tokenEmbed_flat = flatten(this->tokenEmbed);
                if (!h_transformer_tokenEmbed_flat.empty()) {
                    CUDA_CHECK(cudaMalloc(&d_transformer_tokenEmbed_flat, h_transformer_tokenEmbed_flat.size() * sizeof(float)));
                    CUDA_CHECK(cudaMemcpy(d_transformer_tokenEmbed_flat, h_transformer_tokenEmbed_flat.data(), h_transformer_tokenEmbed_flat.size() * sizeof(float), cudaMemcpyHostToDevice));
                } 
                else {
                    fprintf(stderr, "Warning: Global tokenEmbed is empty during Block 1 Inference. Cannot proceed.\n");
                    // If tokenEmbed is essential and empty, we likely cannot compute anything.
                    return;
                }
            }
            else {
                // Block N > 1
                // Need block-local tokForBlock and EVp from previous block
                h_block_tokForBlock_flat = flatten(current_block.tokForBlock);
                if (!h_block_tokForBlock_flat.empty()) {
                    CUDA_CHECK(cudaMalloc(&d_block_tokForBlock_flat, h_block_tokForBlock_flat.size() * sizeof(float)));
                    CUDA_CHECK(cudaMemcpy(d_block_tokForBlock_flat, h_block_tokForBlock_flat.data(), h_block_tokForBlock_flat.size() * sizeof(float), cudaMemcpyHostToDevice));
                }
                else {
                    fprintf(stderr, "Warning: Block-local tokForBlock is empty during Block N > 1 Inference.\n");
                    // If tokForBlock is needed and empty, might be an error or just the start of the block.
                    // Allocate a dummy buffer? Or rely on kernel checks? Let's allocate a small dummy. 
                    CUDA_CHECK(cudaMalloc(&d_block_tokForBlock_flat, 1 * sizeof(float))); // Allocate 1 float
                    CUDA_CHECK(cudaMemset(d_block_tokForBlock_flat, 0, 1 * sizeof(float))); // Set to 0
                }

                CUDA_CHECK(cudaMalloc(&d_all_EVp, total_evp_bytes));
                h_all_EVp_flat.reserve(total_evp_elems);
            }
        }

        // --- Pack Data on Host ---
        block* prev_block_ptr = (blockCount > 1) ? ((inTraining == 1) ? &t[blockCount - 2] : &t[0]): nullptr;

        for (int i = 0; i < num_heads_in_parallel; ++i) {
            // Check if head exists (safety check)
            if (i >= current_block.b.size() || column >= current_block.b[i].size()) {
                fprintf(stderr, "Error: Head index (%d, %d) out of bounds for current block.\n", i, column);
                return;
            }
            attention& head = current_block.b[i][column];

            if (inTraining) {
                // for keys
                temp_flat_buffer = flatten(head.K);
                if (temp_flat_buffer.size() != k_q_ev_head_elems) {
                     fprintf(stderr, "Warning: Head (%d, %d) K size mismatch (Expected %zu, Got %zu). Resizing/Padding host buffer.\n", i, column, k_q_ev_head_elems, temp_flat_buffer.size());
                     temp_flat_buffer.resize(k_q_ev_head_elems, 0.0f); // Pad with 0 if short, truncate if long
                }
                h_all_keys_flat.insert(h_all_keys_flat.end(), temp_flat_buffer.begin(), temp_flat_buffer.end());
                // for queries
                temp_flat_buffer = flatten(head.Q);
                if (temp_flat_buffer.size() != k_q_ev_head_elems) {
                    fprintf(stderr, "Warning: Head (%d, %d) Q size mismatch (Expected %zu, Got %zu). Resizing/Padding host buffer.\n", i, column, k_q_ev_head_elems, temp_flat_buffer.size());
                    temp_flat_buffer.resize(k_q_ev_head_elems, 0.0f);
                }
                h_all_querys_flat.insert(h_all_querys_flat.end(), temp_flat_buffer.begin(), temp_flat_buffer.end());
            } 
            else {
                // Inference
                temp_flat_buffer = flatten(head.qkCache); // head.qkCache is a 'mat'
                if (temp_flat_buffer.size() != qkcache_head_elems) {
                    fprintf(stderr, "Warning: Head (%d, %d) M (qkCache) size mismatch (Expected %zu, Got %zu). Resizing/Padding host buffer.\n", i, column, qkcache_head_elems, temp_flat_buffer.size());
                    temp_flat_buffer.resize(qkcache_head_elems, 0.0f);
                }
                h_all_M_flat.insert(h_all_M_flat.end(), temp_flat_buffer.begin(), temp_flat_buffer.end());

                if (blockCount > 1) {
                    if (!prev_block_ptr) { // Should not happen based on blockCount check, but safety first
                        return;
                    }
                    // Ensure previous block has the required head structure
                    if (i >= prev_block_ptr->b.size() || column >= prev_block_ptr->b[i].size()) {
                        fprintf(stderr, "Error: Head index (%d, %d) out of bounds for previous block.\n", i, column);
                        throw std::out_of_range("Head index out of bounds for previous block");
                    }
                    attention& prev_head = prev_block_ptr->b[i][column];
                    // for EV of head of previous block with same indices
                    temp_flat_buffer = flatten(prev_head.EV);
                    if (temp_flat_buffer.size() != k_q_ev_head_elems) {
                        fprintf(stderr, "Warning: Head (%d, %d) EVp size mismatch (Expected %zu, Got %zu). Resizing/Padding host buffer.\n", i, column, k_q_ev_head_elems, temp_flat_buffer.size());
                        temp_flat_buffer.resize(k_q_ev_head_elems, 0.0f);
                    }
                    h_all_EVp_flat.insert(h_all_EVp_flat.end(), temp_flat_buffer.begin(), temp_flat_buffer.end());
                }
            }
        } // End of host packing loop

        // --- Batch Copy Host -> Device ---
        if (inTraining) {
            if (h_all_keys_flat.size() != total_k_elems || h_all_querys_flat.size() != total_q_elems) {
                fprintf(stderr, "Error: Packed host K/Q size mismatch after loop (K: %zu vs %zu, Q: %zu vs %zu).\n",
                        h_all_keys_flat.size(), total_k_elems, h_all_querys_flat.size(), total_q_elems);
                return;
            }
            CUDA_CHECK(cudaMemcpy(d_all_keys, h_all_keys_flat.data(), total_k_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_all_querys, h_all_querys_flat.data(), total_q_bytes, cudaMemcpyHostToDevice));
        }
        else { 
            // Inference
            if (h_all_M_flat.size() != total_m_elems) {
                fprintf(stderr, "Error: Packed host M size mismatch after loop (%zu vs %zu).\n", h_all_M_flat.size(), total_m_elems);
                return;
            }
            CUDA_CHECK(cudaMemcpy(d_all_M, h_all_M_flat.data(), total_m_bytes, cudaMemcpyHostToDevice));
            if (blockCount > 1) {
                if (h_all_EVp_flat.size() != total_evp_elems) {
                    fprintf(stderr, "Error: Packed host EVp size mismatch after loop (%zu vs %zu).\n", h_all_EVp_flat.size(), total_evp_elems);
                    return;
                } 
                CUDA_CHECK(cudaMemcpy(d_all_EVp, h_all_EVp_flat.data(), total_evp_bytes, cudaMemcpyHostToDevice));
            }
            // d_transformer_tokenEmbed_flat and d_block_tokForBlock_flat already copied if needed
        }

        // --- Loop, Calculate Offsets, and Launch Kernels ---
        dim3 threadsPerBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y); // Defined earlier
        dim3 numBlocks;

        // Calculate context-related variables once before the loop if they are constant for all heads
        int current_tokens_in_window = 0;
        int num_queries_eff = 0;
        int num_keys_eff = 0;
        int prompt_start_index = 0;
        int context_len = 0;
        int effective_prompt_len = 0;
        int tokens_processed_in_prev_blocks = 0;
        int tokens_in_block_before_prompt = 0;
        int prompt_start_index_in_block = 0;
        int context_len_in_block = 0;

        if (inTraining) {
            // Effective number of tokens currently in the window *before* adding the prompt
            current_tokens_in_window = currentTokenCount % context_win_size;
            // Handle edge case: if exactly at the boundary, the window is full from previous step
            if (current_tokens_in_window == 0 && currentTokenCount > 0) {
                // If it's the start of a *new* block (blockCount > 1), the window starts empty
                if (blockCount > 1 && currentTokenCount >= context_win_size * (blockCount - 1)) {
                    current_tokens_in_window = 0;
                }
                else {
                    // Otherwise, it means the window was full
                    current_tokens_in_window = context_win_size;
                }
            }

            // Calculate effective Q/K lengths for this step within the window limit
            num_queries_eff = std::min(current_tokens_in_window + promptCount, context_win_size);
            num_keys_eff = num_queries_eff; // For both self/cross in training, Q and K span the same updated context

            // Calculate grid dimensions based on effective sizes
            numBlocks.x = (num_keys_eff + threadsPerBlock.x - 1) / threadsPerBlock.x;
            numBlocks.y = (num_queries_eff + threadsPerBlock.y - 1) / threadsPerBlock.y;
            numBlocks.z = 1;
        }
        else { // Inference
            if (blockCount == 1) {
                prompt_start_index = currentTokenCount; // Index where the new prompt starts globally
                context_len = std::min(currentTokenCount + promptCount, context_win_size); // Total tokens in window after prompt
                // Effective prompt length considering window limit
                effective_prompt_len = std::min(promptCount, std::max(0, context_win_size - prompt_start_index));

                // Grid dimensions depend on the context length (keys/j) and effective prompt length (queries/i)
                numBlocks.x = (context_len + threadsPerBlock.x - 1) / threadsPerBlock.x;
                numBlocks.y = (effective_prompt_len + threadsPerBlock.y - 1) / threadsPerBlock.y;
                numBlocks.z = 1;
            }
            else { // Block N > 1
                tokens_processed_in_prev_blocks = (blockCount - 1) * context_win_size;
                // Tokens already in this block's window before the current prompt
                tokens_in_block_before_prompt = std::max(0, std::min(currentTokenCount - tokens_processed_in_prev_blocks, context_win_size));
                prompt_start_index_in_block = tokens_in_block_before_prompt;
                // Total relevant tokens in this block's window after adding prompt
                context_len_in_block = std::min(tokens_in_block_before_prompt + promptCount, context_win_size);
                // Effective prompt length considering window limit
                effective_prompt_len = std::min(promptCount, std::max(0, context_win_size - prompt_start_index_in_block));

                // Grid dimensions
                numBlocks.x = (context_len_in_block + threadsPerBlock.x - 1) / threadsPerBlock.x;
                numBlocks.y = (effective_prompt_len + threadsPerBlock.y - 1) / threadsPerBlock.y;
                numBlocks.z = 1;
            }
        }

        // start KdotQ computation for column 'column'
        for (int i = 0; i < num_heads_in_parallel; ++i) {
            // Calculate offsets into the large device buffers for this head
            float* d_kdotq_head = d_all_kdotq + i * kdotq_head_elems;
            float* d_keys_head = d_all_keys + i * k_q_ev_head_elems;     // Only valid if inTraining
            float* d_querys_head = d_all_querys + i * k_q_ev_head_elems; // Only valid if inTraining
            float* d_M_head = d_all_M + i * qkcache_head_elems;         // Only valid if !inTraining
            float* d_EVp_head = d_all_EVp + i * k_q_ev_head_elems;       // Only valid if !inTraining && blockCount > 1

            // --- Select Kernel and Launch ---
            if (inTraining) {
                 if (num_queries_eff > 0 && num_keys_eff > 0) { // Only launch if there's work
                    if (isSelf) {
                        kernelKdotQforSelf_train<<<numBlocks, threadsPerBlock>>>(
                            d_kdotq_head, d_keys_head, d_querys_head, num_queries_eff, num_keys_eff,
                            kdotq_full_width, embedding_dim, inv_scaling);
                    }
                    else {
                        kernelKdotQforCross_train<<<numBlocks, threadsPerBlock>>>(
                            d_kdotq_head, d_keys_head, d_querys_head, num_queries_eff, num_keys_eff,
                            kdotq_full_width, embedding_dim, inv_scaling);
                    }
                    CUDA_CHECK(cudaGetLastError()); // Check kernel launch error immediately
                 }
            }
            else { // Inference Mode
                if (effective_prompt_len > 0) { // Only launch if the prompt has effect in this window
                    if (blockCount == 1) {
                        if (d_transformer_tokenEmbed_flat != nullptr && d_M_head != nullptr) {
                            if (isSelf) {
                                kernelKdotQ_Block1_Self_Inference<<<numBlocks, threadsPerBlock>>>(
                                    d_kdotq_head, d_transformer_tokenEmbed_flat, d_M_head, prompt_start_index, effective_prompt_len,
                                    context_len, kdotq_full_width, embedding_dim, inv_scaling);
                            }
                            else {
                                kernelKdotQ_Block1_Cross_Inference<<<numBlocks, threadsPerBlock>>>(
                                    d_kdotq_head, d_transformer_tokenEmbed_flat, d_M_head, prompt_start_index, effective_prompt_len,
                                    context_len, kdotq_full_width, embedding_dim, inv_scaling);
                            }
                            CUDA_CHECK(cudaGetLastError());
                        } 
                        else {
                            fprintf(stderr, "Warning: Skipping Block 1 inference kernel for head %d due to null input pointers.\n", i);
                        }
                    } 
                    else { // Block N > 1
                        if (d_block_tokForBlock_flat != nullptr && d_EVp_head != nullptr && d_M_head != nullptr) {
                            if (isSelf) {
                                kernelKdotQ_BlockN_Self_Inference<<<numBlocks, threadsPerBlock>>>(
                                    d_kdotq_head, d_block_tokForBlock_flat, d_EVp_head, d_M_head, prompt_start_index_in_block, effective_prompt_len,
                                    context_len_in_block, kdotq_full_width, embedding_dim, inv_scaling);
                            }
                            else {
                                kernelKdotQ_BlockN_Cross_Inference<<<numBlocks, threadsPerBlock>>>(
                                    d_kdotq_head, d_block_tokForBlock_flat, d_EVp_head, d_M_head, prompt_start_index_in_block, effective_prompt_len,
                                    context_len_in_block, kdotq_full_width, embedding_dim, inv_scaling);
                            }
                            CUDA_CHECK(cudaGetLastError());
                        } 
                        else {
                             fprintf(stderr, "Warning: Skipping Block N>1 inference kernel for head %d due to null input pointers (tok=%p, EVp=%p, M=%p).\n",
                                    i, (void*)d_block_tokForBlock_flat, (void*)d_EVp_head, (void*)d_M_head);
                        }
                    }
                } // End if effective_prompt_len > 0
            } // End Inference Mode
        } // End loop over heads (i)

        // --- Batch Copy Result Device -> Host ---
        // Allocate host buffer now that we know the size
        h_all_kdotq_flat.resize(total_kdotq_elems);
        CUDA_CHECK(cudaMemcpy(h_all_kdotq_flat.data(), d_all_kdotq, total_kdotq_bytes, cudaMemcpyDeviceToHost));

        // --- Unpack Results into Attention Heads ---
        auto it_kdotq = h_all_kdotq_flat.begin();
        for (int i = 0; i < num_heads_in_parallel; ++i) {
            attention& head = current_block.b[i][column];

            // Create a temporary vector view or copy the segment
            std::vector<float> head_kdotq_flat(it_kdotq, it_kdotq + kdotq_head_elems);
            it_kdotq += kdotq_head_elems;

            // Unflatten into the head's KdotQ matrix
            // Ensure head.KdotQ is pre-sized correctly or handle resize in unflatten
            // Assuming head.KdotQ is std::vector<std::vector<float>>
            // Resize head.KdotQ if necessary before unflattening
            if (head.KdotQ.size() != context_win_size || (context_win_size > 0 && head.KdotQ[0].size() != context_win_size)) {
                head.KdotQ.assign(context_win_size, std::vector<float>(context_win_size, 0.0f));
            }
            unflatten(head_kdotq_flat, head.KdotQ, context_win_size, context_win_size);
        }

    } catch (const std::exception& e) {
        fprintf(stderr, "Runtime error during cuParallelKdotQs: %s\n", e.what());
        // Cleanup allocated memory before re-throwing or returning
        cudaFree(d_all_kdotq); // Safe to call on nullptr
        cudaFree(d_all_keys);
        cudaFree(d_all_querys);
        cudaFree(d_all_M);
        cudaFree(d_all_EVp);
        cudaFree(d_transformer_tokenEmbed_flat);
        cudaFree(d_block_tokForBlock_flat);
        return; // Re-throw the exception
    }

    // --- Free All Large Device Buffers ---
    cudaFree(d_all_kdotq); // Safe to call on nullptr
    cudaFree(d_all_keys);
    cudaFree(d_all_querys);
    cudaFree(d_all_M);
    cudaFree(d_all_EVp);
    cudaFree(d_transformer_tokenEmbed_flat);
    cudaFree(d_block_tokForBlock_flat);

    // Optional: Synchronize device if needed, but likely handled elsewhere.
    // CUDA_CHECK(cudaDeviceSynchronize());
}
