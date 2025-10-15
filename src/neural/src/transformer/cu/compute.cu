#ifdef USE_CUDA
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
 * @param sequence1Count number of new tokens in the sequence1 being processed in this step.
 * @param currentTokenCount total number of tokens processed *before* this step across all blocks.
 * @param blockCount 1-based index of the current block being processed.
 * @param column 0-based index of the parallel (column in the block's attention grid) to compute.
 * @param isSelf true for self-attention, false for cross-attention.
 * @param inTraining true if in training mode, false if in inference mode.
 */
void transformer::cuParallelKdotQs(int& sequence1Count, int& currentTokenCount, int& blockCount, int& column, bool& isSelf,
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
    block& current_block = (inTraining == 1) ? blocks[blockCount - 1] : blocks[0]; // 0-based index for vector t
    if (x <= 0) {
        fprintf(stderr, "Warning in cuParallelKdotQs: Invalid number of parallels (layers) x = %d. No heads to process.\n", x);
        return; // No heads to process
    }
    if (sequence1Count < 0) {
        fprintf(stderr, "Warning in cuParallelKdotQs: sequence1Count is negative (%d). Setting to 0.\n", sequence1Count);
        return;
    }


    // --- Pre-computation and Setup ---
    const float inv_scaling = 1.0f / sqrtf(EMBEDDING);       // SCALING defined in attention.hpp
    const int embedding_dim = EMBEDDING;                    // 'd' is transformer's embedding dimension
    const int context_win_size = CONTEXT_WIN;                 // 'n' is context window per head from transformer params
    const int kdotq_full_width = CONTEXT_WIN;  // Max width/height of KdotQ matrix per head
    const int num_heads_in_parallel = NUMBER_OF_PA;            // Number of heads in this column (layers)

    // --- Data Sizes Per Head (in elements) ---
    // Ensure context_win_size and embedding_dim are positive
    if (context_win_size <= 0 || embedding_dim <= 0) {
        fprintf(stderr, "Error: Invalid dimensions (context_win_size=%d, embedding_dim=%d)\n", context_win_size, embedding_dim);
        return;
    }
    const size_t k_q_ev_head_elems = static_cast<size_t>(context_win_size) * CONTEXT_WIN;        // n * h
    const size_t kdotq_head_elems = static_cast<size_t>(context_win_size) * context_win_size;   // n * n
    const size_t qkcache_head_elems = static_cast<size_t>(embedding_dim) * embedding_dim;       // d * d

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
                if (!tokenEmbed.mapped_data) {
                    fprintf(stderr, "Error: Global tokenEmbed mat is not mapped.\n"); return;
                }
                h_transformer_tokenEmbed_flat = flatten(tokenEmbed); // tokenEmbed is mat
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
                if (!current_block.tokForBlock.mapped_data) {
                    fprintf(stderr, "Error: Block-local tokForBlock mat is not mapped.\n"); return;
                }
                h_block_tokForBlock_flat = flatten(current_block.tokForBlock); // current_block.tokForBlock is mat
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
        block* prev_block_ptr = (blockCount > 1) ? ((inTraining == 1) ? &blocks[blockCount - 2] : &blocks[0]): nullptr;

        for (int i = 0; i < num_heads_in_parallel; ++i) {
            // Check if head exists (safety check)
            if (i >= current_block.b.size() || column >= current_block.b[i].size()) {
                fprintf(stderr, "Error: Head index (%d, %d) out of bounds for current block.\n", i, column);
                return;
            }
            attention& head = current_block.b[i][column];

            if (inTraining) {
                if (!head.K.mapped_data || !head.Q.mapped_data) {
                    fprintf(stderr, "Error: Head (%d, %d) K or Q mat not mapped.\n", i, column); return;
                }
                // for keys
                temp_flat_buffer = flatten(head.K); // head.K is mat
                if (temp_flat_buffer.size() != static_cast<size_t>(head.K.row) * head.K.col) { // Check against mat's actual size
                    fprintf(stderr, "Warning: Head (%d, %d) K flatten size mismatch (Expected %d, Got %zu). Resizing/Padding host buffer.\n", i, column, head.K.row * head.K.col, temp_flat_buffer.size());
                    temp_flat_buffer.resize(k_q_ev_head_elems, 0.0f); // Pad with 0 if short, truncate if long
                }
                h_all_keys_flat.insert(h_all_keys_flat.end(), temp_flat_buffer.begin(), temp_flat_buffer.end());
                // for queries
                temp_flat_buffer = flatten(head.Q); // head.Q is mat
                if (temp_flat_buffer.size() != static_cast<size_t>(head.Q.row) * head.Q.col) {
                    fprintf(stderr, "Warning: Head (%d, %d) Q flatten size mismatch (Expected %d, Got %zu). Resizing/Padding host buffer.\n", i, column, head.Q.row * head.Q.col, temp_flat_buffer.size());
                    temp_flat_buffer.resize(k_q_ev_head_elems, 0.0f);
                }
                h_all_querys_flat.insert(h_all_querys_flat.end(), temp_flat_buffer.begin(), temp_flat_buffer.end());
            } 
            else {
                // Inference
                if (!head.qkCache.mapped_data) {
                    fprintf(stderr, "Error: Head (%d, %d) qkCache mat not mapped.\n", i, column); return;
                }
                temp_flat_buffer = flatten(head.qkCache); // head.qkCache is a 'mat'
                // qkcache_head_elems is d*d, head.qkCache.row * head.qkCache.col should also be d*d
                if (static_cast<size_t>(head.qkCache.row) != embedding_dim || static_cast<size_t>(head.qkCache.col) != embedding_dim) {
                    fprintf(stderr, "Warning: Head (%d, %d) M (qkCache) dimension mismatch (Expected %dx%d, Got %dx%d).\n",
                            i, column, embedding_dim, embedding_dim, head.qkCache.row, head.qkCache.col);
                }
                if (temp_flat_buffer.size() != static_cast<size_t>(head.qkCache.row) * head.qkCache.col) {
                    fprintf(stderr, "Warning: Head (%d, %d) M (qkCache) flatten size mismatch (Expected %d, Got %zu). Resizing/Padding host buffer.\n", i, column, head.qkCache.row * head.qkCache.col, temp_flat_buffer.size());
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
                    if (!prev_head.EV.mapped_data) {
                        fprintf(stderr, "Error: Prev_head (%d, %d) EV mat not mapped.\n", i, column); return;
                    }
                    // for EV of head of previous block with same indices
                    temp_flat_buffer = flatten(prev_head.EV); // prev_head.EV is mat
                    if (static_cast<size_t>(prev_head.EV.row) != context_win_size || static_cast<size_t>(prev_head.EV.col) != embedding_dim) {
                         fprintf(stderr, "Warning: Head (%d, %d) EVp dimension mismatch (Expected %dx%d, Got %dx%d).\n",
                                 i, column, context_win_size, embedding_dim, prev_head.EV.row, prev_head.EV.col);
                    }
                    if (temp_flat_buffer.size() != static_cast<size_t>(prev_head.EV.row) * prev_head.EV.col) {
                        fprintf(stderr, "Warning: Head (%d, %d) EVp flatten size mismatch (Expected %d, Got %zu). Resizing/Padding host buffer.\n", i, column, prev_head.EV.row * prev_head.EV.col, temp_flat_buffer.size());
                        temp_flat_buffer.resize(k_q_ev_head_elems, 0.0f);
                    }
                    h_all_EVp_flat.insert(h_all_EVp_flat.end(), temp_flat_buffer.begin(), temp_flat_buffer.end());
                }
            }
        } // End of host packing loop

        // --- Batch Copy Host -> Device ---
        if (inTraining == 1) {
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
        int sequence1_start_index = 0;
        int context_len = 0;
        int effective_sequence1_len = 0;
        int tokens_processed_in_prev_blocks = 0;
        int tokens_in_block_before_sequence1 = 0;
        int sequence1_start_index_in_block = 0;
        int context_len_in_block = 0;

        if (inTraining) {
            // Effective number of tokens currently in the window *before* adding the sequence1
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
            num_queries_eff = std::min(current_tokens_in_window + sequence1Count, context_win_size);
            num_keys_eff = num_queries_eff; // For both self/cross in training, Q and K span the same updated context

            // Calculate grid dimensions based on effective sizes
            numBlocks.x = (num_keys_eff + threadsPerBlock.x - 1) / threadsPerBlock.x;
            numBlocks.y = (num_queries_eff + threadsPerBlock.y - 1) / threadsPerBlock.y;
            numBlocks.z = 1;
        }
        else { // Inference
            if (blockCount == 1) {
                sequence1_start_index = currentTokenCount; // Index where the new sequence1 starts globally
                context_len = std::min(currentTokenCount + sequence1Count, context_win_size); // Total tokens in window after sequence1
                // Effective sequence1 length considering window limit
                effective_sequence1_len = std::min(sequence1Count, std::max(0, context_win_size - sequence1_start_index));

                // Grid dimensions depend on the context length (keys/j) and effective sequence1 length (queries/i)
                numBlocks.x = (context_len + threadsPerBlock.x - 1) / threadsPerBlock.x;
                numBlocks.y = (effective_sequence1_len + threadsPerBlock.y - 1) / threadsPerBlock.y;
                numBlocks.z = 1;
            }
            else { // Block N > 1
                tokens_processed_in_prev_blocks = (blockCount - 1) * context_win_size;
                // Tokens already in this block's window before the current sequence1
                tokens_in_block_before_sequence1 = std::max(0, std::min(currentTokenCount - tokens_processed_in_prev_blocks, context_win_size));
                sequence1_start_index_in_block = tokens_in_block_before_sequence1;
                // Total relevant tokens in this block's window after adding sequence1
                context_len_in_block = std::min(tokens_in_block_before_sequence1 + sequence1Count, context_win_size);
                // Effective sequence1 length considering window limit
                effective_sequence1_len = std::min(sequence1Count, std::max(0, context_win_size - sequence1_start_index_in_block));

                // Grid dimensions
                numBlocks.x = (context_len_in_block + threadsPerBlock.x - 1) / threadsPerBlock.x;
                numBlocks.y = (effective_sequence1_len + threadsPerBlock.y - 1) / threadsPerBlock.y;
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
                if (effective_sequence1_len > 0) { // Only launch if the sequence1 has effect in this window
                    if (blockCount == 1) {
                        if (d_transformer_tokenEmbed_flat != nullptr && d_M_head != nullptr) {
                            if (isSelf) {
                                kernelKdotQ_Block1_Selfi<<<numBlocks, threadsPerBlock>>>(
                                    d_kdotq_head, d_transformer_tokenEmbed_flat, d_M_head, sequence1_start_index, effective_sequence1_len,
                                    context_len, kdotq_full_width, embedding_dim, inv_scaling);
                            }
                            else {
                                kernelKdotQ_Block1_Crossi<<<numBlocks, threadsPerBlock>>>(
                                    d_kdotq_head, d_transformer_tokenEmbed_flat, d_M_head, sequence1_start_index, effective_sequence1_len,
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
                                kernelKdotQ_BlockN_Selfi<<<numBlocks, threadsPerBlock>>>(
                                    d_kdotq_head, d_block_tokForBlock_flat, d_EVp_head, d_M_head, sequence1_start_index_in_block, effective_sequence1_len,
                                    context_len_in_block, kdotq_full_width, embedding_dim, inv_scaling);
                            }
                            else {
                                kernelKdotQ_BlockN_Crossi<<<numBlocks, threadsPerBlock>>>(
                                    d_kdotq_head, d_block_tokForBlock_flat, d_EVp_head, d_M_head, sequence1_start_index_in_block, effective_sequence1_len,
                                    context_len_in_block, kdotq_full_width, embedding_dim, inv_scaling);
                            }
                            CUDA_CHECK(cudaGetLastError());
                        } 
                        else {
                             fprintf(stderr, "Warning: Skipping Block N>1 inference kernel for head %d due to null input pointers (tok=%p, EVp=%p, M=%p).\n",
                                    i, (void*)d_block_tokForBlock_flat, (void*)d_EVp_head, (void*)d_M_head);
                        }
                    }
                } // End if effective_sequence1_len > 0
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

            // head.KdotQ is a mat. Ensure it's mapped and correctly sized before direct memory copy.
            if (!head.KdotQ.mapped_data || head.KdotQ.row != context_win_size || head.KdotQ.col != context_win_size) {
                fprintf(stderr, "Error: Head (%d, %d) KdotQ mat is not mapped or incorrectly sized for unflatten (Expected %dx%d, Got %dx%d).\n", i, column, context_win_size, context_win_size, head.KdotQ.row, head.KdotQ.col);
                continue; // Skip unflattening for this head
            }
            // Directly copy data from head_kdotq_flat to head.KdotQ.mapped_data
            // Ensure head_kdotq_flat has the expected number of elements
            if (head_kdotq_flat.size() == kdotq_head_elems) { // kdotq_head_elems is context_win_size * context_win_size
                memcpy(head.KdotQ.mapped_data, head_kdotq_flat.data(), kdotq_head_elems * sizeof(float));
            } else {
                fprintf(stderr, "Error: Head (%d, %d) KdotQ flat data size mismatch for direct copy (Expected %zu, Got %zu).\n", i, column, kdotq_head_elems, head_kdotq_flat.size());
            }
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

#endif