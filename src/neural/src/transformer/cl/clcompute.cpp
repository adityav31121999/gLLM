#ifdef USE_OPENCL
#include "include/transformer.hpp"
#include <maths.hpp>
#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>
#include <map>
#include <cstdio>
#include <cfloat>

#define WORKGROUP_SIZE_X 16
#define WORKGROUP_SIZE_Y 16

/**
 * @brief compute KdotQ of specific parallel (or column of block vector b) using OpenCL kernels. 
 * This optimized version allocates large buffers on the device once, copies data in batches, 
 * launches kernels using offsets into these buffers, copies results back, and then frees. Mirrors 
 * the logic of the C++ parallelKdotQs and CUDA cuParallelKdotQs functions.
 * @param promptCount number of new tokens in the prompt being processed in this step.
 * @param currentTokenCount total number of tokens processed *before* this step across all blocks.
 * @param blockCount 1-based index of the current block being processed.
 * @param column 0-based index of the parallel (column in the block's attention grid) to compute.
 * @param isSelf true for self-attention, false for cross-attention.
 * @param inTraining true if in training mode, false if in inference mode.
 * @note Assumes cl_context, cl_queue, and cl_kernels are accessible (e.g., member variables).
 */
void transformer::clParallelKdotQs(int& promptCount, int& currentTokenCount, int& blockCount, int& column, bool& isSelf,
    bool& inTraining)
{
    // --- Basic Sanity Checks ---
    if (blockCount < 1 || blockCount > m) {
        fprintf(stderr, "Error in clParallelKdotQs: Invalid blockCount %d (max %d)\n", blockCount, m);
        return;
    }
    if (column < 0 || column >= y) {
        fprintf(stderr, "Error in clParallelKdotQs: Invalid column %d (max %d)\n", column, y - 1);
        throw std::out_of_range("Invalid column index");
    }

    // initiate a block that reference the current block operations
    block& current_block = (inTraining == 1) ? t[blockCount - 1] : t[0]; // 0-based index for vector t
    if (x <= 0) {
        fprintf(stderr, "Warning in clParallelKdotQs: Invalid number of parallels (layers) x = %d. No heads to process.\n", x);
        return; // No heads to process
    }
    if (promptCount < 0) {
        fprintf(stderr, "Warning in clParallelKdotQs: promptCount is negative (%d). Setting to 0.\n", promptCount);
        // Decide if we should proceed with promptCount = 0 or return. Let's return for safety.
        return;
    }

    // --- Pre-computation and Setup ---
    const float inv_scaling = 1.0f / sqrtf(static_cast<float>(EMBEDDING)); // Use static_cast for clarity
    const cl_int embedding_dim = d;                    // 'd' is transformer's embedding dimension
    const cl_int context_win_size = n;                 // 'n' is context window per head from transformer params
    const cl_int kdotq_full_width = context_win_size;  // Max width/height of KdotQ matrix per head
    const int num_heads_in_parallel = x;            // Number of heads in this column (layers)

    // --- Data Sizes Per Head (in elements) ---
    if (context_win_size <= 0 || embedding_dim <= 0) {
        fprintf(stderr, "Error: Invalid dimensions (context_win_size=%d, embedding_dim=%d)\n", context_win_size, embedding_dim);
        return;
    }
    const size_t k_q_ev_head_elems = static_cast<size_t>(context_win_size) * MATHEIGHTS;        // n * d
    const size_t kdotq_head_elems = static_cast<size_t>(context_win_size) * context_win_size;   // n * n
    const size_t qkcache_head_elems = static_cast<size_t>(embedding_dim) * embedding_dim;       // d * d

    // --- Total Data Sizes for the Parallel (in elements) ---
    const size_t total_k_elems = num_heads_in_parallel * k_q_ev_head_elems;     // x * n * d
    const size_t total_q_elems = num_heads_in_parallel * k_q_ev_head_elems;     // x * n * d
    const size_t total_evp_elems = num_heads_in_parallel * k_q_ev_head_elems;   // x * n * d
    const size_t total_m_elems = num_heads_in_parallel * qkcache_head_elems;    // x * d * d
    const size_t total_kdotq_elems = num_heads_in_parallel * kdotq_head_elems;  // x * n * n

    // --- Total Memory Consumption (in bytes) ---
    const size_t total_k_bytes = total_k_elems * sizeof(cl_float);
    const size_t total_q_bytes = total_q_elems * sizeof(cl_float);
    const size_t total_evp_bytes = total_evp_elems * sizeof(cl_float);
    const size_t total_m_bytes = total_m_elems * sizeof(cl_float);
    const size_t total_kdotq_bytes = total_kdotq_elems * sizeof(cl_float);

    // --- Host-side Aggregated Data Buffers ---
    std::vector<float> h_all_keys_flat;
    std::vector<float> h_all_querys_flat;
    std::vector<float> h_all_M_flat;
    std::vector<float> h_all_EVp_flat;
    std::vector<float> h_all_kdotq_flat;                // Allocate later if needed for copy-back
    std::vector<float> h_transformer_tokenEmbed_flat;   // Flattened global embeddings
    std::vector<float> h_block_tokForBlock_flat;        // Flattened block-local embeddings
    std::vector<float> temp_flat_buffer;                // Temporary Host Buffers for Flattening

    // --- OpenCL Device Buffers (using cl::Buffer RAII wrapper) ---
    cl::Buffer d_all_kdotq;
    cl::Buffer d_all_keys;
    cl::Buffer d_all_querys;
    cl::Buffer d_all_M;
    cl::Buffer d_all_EVp;
    cl::Buffer d_transformer_tokenEmbed_flat; // Shared, single copy
    cl::Buffer d_block_tokForBlock_flat;      // Shared, single copy

    cl_int cl_err; // For OpenCL error codes
    std::vector<cl::CommandQueue> clQueues; 

    try {
        if (num_heads_in_parallel > 0) {
            clQueues.reserve(num_heads_in_parallel);
            for (int head_idx_q = 0; head_idx_q < num_heads_in_parallel; ++head_idx_q) {
                clQueues.emplace_back(this->clcontext.context, this->clcontext.device, 0, &cl_err);
                CL_CHECK(cl_err);
            }
        }
        // --- Allocate Large Device Buffers ---
        // Use cl_context (assumed member variable or accessible)
        d_all_kdotq = cl::Buffer(this->clcontext.context, CL_MEM_READ_WRITE, total_kdotq_bytes);
        // Initialize KdotQ to 0 using the main context queue.
        cl_float zero = 0.0f;
        CL_CHECK(this->clcontext.queue.enqueueFillBuffer(d_all_kdotq, zero, 0, total_kdotq_bytes));
        // Ensure fill is complete before proceeding (optional, depends on queue properties)
        // CL_CHECK(cl_queue.finish());

        if (inTraining) {
            d_all_keys = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY, total_k_bytes);
            d_all_querys = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY, total_q_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
            h_all_keys_flat.reserve(total_k_elems);
            h_all_querys_flat.reserve(total_q_elems);
        }
        else {
            // Inference
            d_all_M = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY, total_m_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
            h_all_M_flat.reserve(total_m_elems);

            if (blockCount == 1) {
                // Need global tokenEmbed
                h_transformer_tokenEmbed_flat = flatten(this->tokenEmbed);
                if (!h_transformer_tokenEmbed_flat.empty()) {
                    d_transformer_tokenEmbed_flat = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                               h_transformer_tokenEmbed_flat.size() * sizeof(cl_float),
                                                               h_transformer_tokenEmbed_flat.data(), &cl_err); CL_CHECK(cl_err);
                } 
                else {
                    fprintf(stderr, "Warning: Global tokenEmbed is empty during Block 1 Inference. Cannot proceed.\n");
                    return; // Cannot compute without embeddings
                }
            } 
            else {
                // Block N > 1
                // Need block-local tokForBlock and EVp from previous block
                h_block_tokForBlock_flat = flatten(current_block.tokForBlock);
                if (!h_block_tokForBlock_flat.empty()) {
                     d_block_tokForBlock_flat = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                          h_block_tokForBlock_flat.size() * sizeof(cl_float),
                                                          h_block_tokForBlock_flat.data(), &cl_err); CL_CHECK(cl_err);
                }
                else {
                    fprintf(stderr, "Warning: Block-local tokForBlock is empty during Block N > 1 Inference.\n");
                    // Allocate a small dummy buffer if needed by kernel logic, otherwise might be okay.
                    // Let's assume kernels handle empty inputs gracefully or checks prevent launch.
                    // If a buffer *must* exist, create a minimal one:
                    // cl_float dummy_val = 0.0f;
                    // d_block_tokForBlock_flat = cl::Buffer(cl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float), &dummy_val);
                }

                d_all_EVp = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY, total_evp_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
                h_all_EVp_flat.reserve(total_evp_elems);
            }
        }

        // --- Pack Data on Host ---
        block* prev_block_ptr = (blockCount > 1) ? ((inTraining == 1) ? &t[blockCount - 2] : &t[0]) : nullptr;

        for (int i = 0; i < num_heads_in_parallel; ++i) {
            // Check if head exists (safety check)
            if (i >= static_cast<int>(current_block.b.size()) || column >= static_cast<int>(current_block.b[i].size())) {
                fprintf(stderr, "Error: Head index (%d, %d) out of bounds for current block.\n", i, column);
                return; // Or throw
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
                    if (!prev_block_ptr) {
                        fprintf(stderr, "Error: prev_block_ptr is null for blockCount > 1.\n");
                        return; // Should not happen based on blockCount check
                    }
                    // Ensure previous block has the required head structure
                    if (i >= static_cast<int>(prev_block_ptr->b.size()) || column >= static_cast<int>(prev_block_ptr->b[i].size())) {
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
        // Use enqueueWriteBuffer. CL_TRUE makes it blocking.
        if (inTraining) {
            if (h_all_keys_flat.size() != total_k_elems || h_all_querys_flat.size() != total_q_elems) {
                fprintf(stderr, "Error: Packed host K/Q size mismatch after loop (K: %zu vs %zu, Q: %zu vs %zu).\n",
                        h_all_keys_flat.size(), total_k_elems, h_all_querys_flat.size(), total_q_elems);
                return;
            }
            CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_all_keys, CL_TRUE, 0, total_k_bytes, h_all_keys_flat.data()));
            CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_all_querys, CL_TRUE, 0, total_q_bytes, h_all_querys_flat.data()));
        } 
        else {
            // Inference
            if (h_all_M_flat.size() != total_m_elems) {
                fprintf(stderr, "Error: Packed host M size mismatch after loop (%zu vs %zu).\n", h_all_M_flat.size(), total_m_elems);
                return;
            }
            CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_all_M, CL_TRUE, 0, total_m_bytes, h_all_M_flat.data()));
            if (blockCount > 1) {
                 if (h_all_EVp_flat.size() != total_evp_elems) {
                    fprintf(stderr, "Error: Packed host EVp size mismatch after loop (%zu vs %zu).\n", h_all_EVp_flat.size(), total_evp_elems);
                    return;
                }
                CL_CHECK(this->clcontext.queue.enqueueWriteBuffer(d_all_EVp, CL_TRUE, 0, total_evp_bytes, h_all_EVp_flat.data()));
            }
            // d_transformer_tokenEmbed_flat and d_block_tokForBlock_flat already copied via CL_MEM_COPY_HOST_PTR
        }

        // Ensure all initial H->D transfers and buffer fills are complete before kernel launches
        CL_CHECK(this->clcontext.queue.finish());

        // --- Loop, Calculate Offsets, and Launch Kernels ---
        cl::NDRange local_work_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y);
        cl::NDRange global_work_size; // Will be set inside the loop based on effective dimensions

        // Calculate context-related variables once before the loop if they are constant for all heads
        cl_int current_tokens_in_window = 0;
        cl_int num_queries_eff = 0;
        cl_int num_keys_eff = 0;
        cl_int prompt_start_index = 0;
        cl_int context_len = 0;
        cl_int effective_prompt_len = 0;
        cl_int tokens_processed_in_prev_blocks = 0;
        cl_int tokens_in_block_before_prompt = 0;
        cl_int prompt_start_index_in_block = 0;
        cl_int context_len_in_block = 0;

        // Helper lambda to calculate rounded global size
        auto calculate_global_size = [](size_t total_x, size_t total_y) {
            size_t global_x = (total_x + WORKGROUP_SIZE_X - 1) / WORKGROUP_SIZE_X * WORKGROUP_SIZE_X;
            size_t global_y = (total_y + WORKGROUP_SIZE_Y - 1) / WORKGROUP_SIZE_Y * WORKGROUP_SIZE_Y;
            return cl::NDRange(global_x, global_y);
        };

        if (inTraining) {
            // Effective number of tokens currently in the window *before* adding the prompt
            current_tokens_in_window = currentTokenCount % context_win_size;
            if (current_tokens_in_window == 0 && currentTokenCount > 0) {
                if (blockCount > 1 && currentTokenCount >= static_cast<cl_int>(context_win_size) * (blockCount - 1)) { // Cast for comparison
                    current_tokens_in_window = 0;
                } 
                else {
                    current_tokens_in_window = context_win_size;
                }
            }
            num_queries_eff = (std::min)(current_tokens_in_window + promptCount, context_win_size); // Parenthesize std::min
            num_keys_eff = num_queries_eff;
            global_work_size = calculate_global_size(num_keys_eff, num_queries_eff);
        }
        else {
            if (blockCount == 1) {
                prompt_start_index = currentTokenCount;
                context_len = (std::min)(currentTokenCount + promptCount, context_win_size); // Parenthesize std::min
                effective_prompt_len = (std::min)(promptCount, (std::max)(0, context_win_size - prompt_start_index)); // Parenthesize std::min and std::max
                global_work_size = calculate_global_size(context_len, effective_prompt_len);
            } 
            else {
                tokens_processed_in_prev_blocks = (blockCount - 1) * context_win_size;
                tokens_in_block_before_prompt = (std::max)(0, (std::min)(currentTokenCount - tokens_processed_in_prev_blocks, context_win_size)); // Parenthesize std::min and std::max
                prompt_start_index_in_block = tokens_in_block_before_prompt;
                context_len_in_block = (std::min)(tokens_in_block_before_prompt + promptCount, context_win_size); // Parenthesize std::min
                effective_prompt_len = (std::min)(promptCount, (std::max)(0, context_win_size - prompt_start_index_in_block)); // Parenthesize std::min and std::max
                global_work_size = calculate_global_size(context_len_in_block, effective_prompt_len); // Parenthesize std::min and std::max
            }
        }

        for (int i = 0; i < num_heads_in_parallel; ++i) {
            int blk = (inTraining == 0) ? 0 : blockCount-1;
            // Ensure blk is valid for t
            if (blk < 0 || blk >= static_cast<int>(t.size())) {
                fprintf(stderr, "Error: Calculated block index blk=%d is out of bounds for transformer 't' (size %zu).\n", blk, t.size());
                throw std::out_of_range("Block index out of bounds for transformer 't'");
            }
            attention& head_obj = t[blk].b[i][column];
            cl::CommandQueue& current_stream = (num_heads_in_parallel > 0) ? clQueues[i] : this->clcontext.queue; // Fallback, though loop shouldn't run if 0
            // Calculate byte offsets and sizes for the current head's data within aggregated buffers
            size_t kdotq_byte_offset    = i * kdotq_head_elems * sizeof(cl_float);
            size_t k_byte_offset        = i * k_q_ev_head_elems * sizeof(cl_float);
            size_t q_byte_offset        = i * k_q_ev_head_elems * sizeof(cl_float);
            size_t m_byte_offset        = i * qkcache_head_elems * sizeof(cl_float);
            size_t evp_byte_offset      = i * k_q_ev_head_elems * sizeof(cl_float);

            size_t kdotq_head_bytes     = kdotq_head_elems * sizeof(cl_float);
            size_t k_q_ev_head_bytes    = k_q_ev_head_elems * sizeof(cl_float);
            size_t qkcache_head_bytes   = qkcache_head_elems * sizeof(cl_float);

            // Create sub-buffer for KdotQ output for this head
            cl_buffer_region kdotq_region = {kdotq_byte_offset, kdotq_head_bytes};
            cl::Buffer d_kdotq_head = d_all_kdotq.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &kdotq_region, &cl_err); CL_CHECK(cl_err);

            cl::Kernel kernel; // Kernel object for this head's launch

            if (inTraining) {
                 if (num_queries_eff > 0 && num_keys_eff > 0) {
                    // Create sub-buffers for K and Q
                    cl_buffer_region k_region   = {k_byte_offset, k_q_ev_head_bytes};
                    cl::Buffer d_keys_head      = d_all_keys.createSubBuffer(CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &k_region, &cl_err); CL_CHECK(cl_err);

                    cl_buffer_region q_region   = {q_byte_offset, k_q_ev_head_bytes};
                    cl::Buffer d_querys_head    = d_all_querys.createSubBuffer(CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &q_region, &cl_err); CL_CHECK(cl_err);

                    if (isSelf) {
                        kernel = this->clcontext.kernels.at("kernelKdotQforSelf_train");
                        CL_CHECK(kernel.setArg(0, d_kdotq_head));  // Pass sub-buffer
                        CL_CHECK(kernel.setArg(1, d_keys_head));   // Pass sub-buffer
                        CL_CHECK(kernel.setArg(2, d_querys_head)); // Pass sub-buffer
                        CL_CHECK(kernel.setArg(3, static_cast<cl_int>(num_queries_eff)));
                        CL_CHECK(kernel.setArg(4, static_cast<cl_int>(num_keys_eff)));
                        CL_CHECK(kernel.setArg(5, static_cast<cl_int>(kdotq_full_width)));
                        CL_CHECK(kernel.setArg(6, static_cast<cl_int>(embedding_dim)));
                        CL_CHECK(kernel.setArg(7, inv_scaling));
                    } 
                    else { // Cross Attention
                        kernel = this->clcontext.kernels.at("kernelKdotQforCross_train");
                        CL_CHECK(kernel.setArg(0, d_kdotq_head));  // Pass sub-buffer
                        CL_CHECK(kernel.setArg(1, d_keys_head));   // Pass sub-buffer
                        CL_CHECK(kernel.setArg(2, d_querys_head)); // Pass sub-buffer
                        CL_CHECK(kernel.setArg(3, static_cast<cl_int>(num_queries_eff)));
                        CL_CHECK(kernel.setArg(4, static_cast<cl_int>(num_keys_eff)));
                        CL_CHECK(kernel.setArg(5, static_cast<cl_int>(kdotq_full_width)));
                        CL_CHECK(kernel.setArg(6, static_cast<cl_int>(embedding_dim)));
                        CL_CHECK(kernel.setArg(7, inv_scaling));
                    }
                    if (global_work_size[0] > 0 && global_work_size[1] > 0) { // Ensure global size is valid
                        CL_CHECK(current_stream.enqueueNDRangeKernel(kernel, cl::NullRange, global_work_size, local_work_size));
                    }
                 }
            } 
            else { // Inference Mode
                if (effective_prompt_len > 0) {
                    // Create sub-buffer for M
                    cl_buffer_region m_region   = {m_byte_offset, qkcache_head_bytes};
                    cl::Buffer d_M_head         = d_all_M.createSubBuffer(CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &m_region, &cl_err); CL_CHECK(cl_err);

                    if (blockCount == 1) {
                        if (isSelf) {
                            kernel = this->clcontext.kernels.at("kernelKdotQBlock1Self_Inference");
                            CL_CHECK(kernel.setArg(0, d_kdotq_head)); // Pass sub-buffer
                            CL_CHECK(kernel.setArg(1, d_transformer_tokenEmbed_flat)); // Shared buffer (no sub-buffer needed)
                            CL_CHECK(kernel.setArg(2, d_M_head));    // Pass sub-buffer: QK' cache
                            CL_CHECK(kernel.setArg(3, static_cast<cl_int>(prompt_start_index)));
                            CL_CHECK(kernel.setArg(4, static_cast<cl_int>(effective_prompt_len)));
                            CL_CHECK(kernel.setArg(5, static_cast<cl_int>(context_len)));
                            CL_CHECK(kernel.setArg(6, static_cast<cl_int>(kdotq_full_width)));
                            CL_CHECK(kernel.setArg(7, static_cast<cl_int>(embedding_dim)));
                            CL_CHECK(kernel.setArg(8, inv_scaling));
                        } 
                        else {
                            kernel = this->clcontext.kernels.at("kernelKdotQBlock1Cross_Inference");
                            CL_CHECK(kernel.setArg(0, d_kdotq_head)); // Pass sub-buffer
                            CL_CHECK(kernel.setArg(1, d_transformer_tokenEmbed_flat));
                            CL_CHECK(kernel.setArg(2, d_M_head));    // Pass sub-buffer: QK' cache
                            CL_CHECK(kernel.setArg(3, static_cast<cl_int>(prompt_start_index)));
                            CL_CHECK(kernel.setArg(4, static_cast<cl_int>(effective_prompt_len)));
                            CL_CHECK(kernel.setArg(5, static_cast<cl_int>(context_len)));
                            CL_CHECK(kernel.setArg(6, static_cast<cl_int>(kdotq_full_width)));
                            CL_CHECK(kernel.setArg(7, static_cast<cl_int>(embedding_dim)));
                            CL_CHECK(kernel.setArg(8, inv_scaling));
                        }
                    } 
                    else { // Block N > 1
                        // Create sub-buffer for EVp
                        cl_buffer_region evp_region = {evp_byte_offset, k_q_ev_head_bytes};
                        cl::Buffer d_EVp_head       = d_all_EVp.createSubBuffer(CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &evp_region, &cl_err); CL_CHECK(cl_err);
                        if (isSelf) {
                            kernel = this->clcontext.kernels.at("kernelKdotQBlockNSelf_Inference");
                            CL_CHECK(kernel.setArg(0, d_kdotq_head)); // Pass sub-buffer
                            CL_CHECK(kernel.setArg(1, d_block_tokForBlock_flat)); // Shared buffer
                            CL_CHECK(kernel.setArg(2, d_EVp_head));   // Pass sub-buffer
                            CL_CHECK(kernel.setArg(3, d_M_head));     // Pass sub-buffer: QK' cache
                            CL_CHECK(kernel.setArg(4, static_cast<cl_int>(prompt_start_index_in_block)));
                            CL_CHECK(kernel.setArg(5, static_cast<cl_int>(effective_prompt_len)));
                            CL_CHECK(kernel.setArg(6, static_cast<cl_int>(context_len_in_block)));
                            CL_CHECK(kernel.setArg(7, static_cast<cl_int>(kdotq_full_width)));
                            CL_CHECK(kernel.setArg(8, static_cast<cl_int>(embedding_dim)));
                            CL_CHECK(kernel.setArg(9, inv_scaling));
                        } 
                        else {
                            kernel = this->clcontext.kernels.at("kernelKdotQBlockNCross_Inference");
                            CL_CHECK(kernel.setArg(0, d_kdotq_head)); // Pass sub-buffer
                            CL_CHECK(kernel.setArg(1, d_block_tokForBlock_flat));
                            CL_CHECK(kernel.setArg(2, d_EVp_head));   // Pass sub-buffer
                            CL_CHECK(kernel.setArg(3, d_M_head));     // Pass sub-buffer: QK' cache
                            CL_CHECK(kernel.setArg(4, static_cast<cl_int>(prompt_start_index_in_block)));
                            CL_CHECK(kernel.setArg(5, static_cast<cl_int>(effective_prompt_len)));
                            CL_CHECK(kernel.setArg(6, static_cast<cl_int>(context_len_in_block)));
                            CL_CHECK(kernel.setArg(7, static_cast<cl_int>(kdotq_full_width)));
                            CL_CHECK(kernel.setArg(8, static_cast<cl_int>(embedding_dim)));
                            CL_CHECK(kernel.setArg(9, inv_scaling));
                        }
                    }
                    if (global_work_size[0] > 0 && global_work_size[1] > 0) { // Ensure global size is valid
                        CL_CHECK(current_stream.enqueueNDRangeKernel(kernel, cl::NullRange, global_work_size, local_work_size));
                    }
                }
            }
        }

        // Ensure all kernels are finished before reading back
        if (num_heads_in_parallel > 0) {
            for (int i = 0; i < num_heads_in_parallel; ++i) {
                CL_CHECK(clQueues[i].finish());
            }
        } else { // If no heads, still ensure the main queue is clear if any prior ops were on it.
            CL_CHECK(this->clcontext.queue.finish());
        }

        // --- Batch Copy Result Device -> Host ---
        h_all_kdotq_flat.resize(total_kdotq_elems); // Allocate host buffer
        CL_CHECK(this->clcontext.queue.enqueueReadBuffer(d_all_kdotq, CL_TRUE, 0, total_kdotq_bytes, h_all_kdotq_flat.data()));

        // --- Unpack Results into Attention Heads ---
        auto it_kdotq = h_all_kdotq_flat.begin();
        for (int i = 0; i < num_heads_in_parallel; ++i) {
             if (i >= static_cast<int>(current_block.b.size()) || column >= static_cast<int>(current_block.b[i].size())) {
                 // This should have been caught earlier, but double-check
                 fprintf(stderr, "Error: Head index (%d, %d) out of bounds during unpacking.\n", i, column);
                 continue;
             }
            attention& head = current_block.b[i][column];

            // Create a temporary vector view or copy the segment
            std::vector<float> head_kdotq_flat(it_kdotq, it_kdotq + kdotq_head_elems);
            it_kdotq += kdotq_head_elems;

            // Unflatten into the head's KdotQ matrix
            // Resize head.KdotQ if necessary before unflattening
            // head.KdotQ is a mat object, which should be initialized with correct dimensions
            // by the attention class constructor.
            if (head.KdotQ.row != context_win_size || head.KdotQ.col != context_win_size) {
                // This indicates a potential logic error in initialization or that context_win_size
                // has changed in a way not reflected in KdotQ. Re-initializing as a fallback.
                fprintf(stderr, "Warning: KdotQ dimensions mismatch for head (%d, %d) in clParallelKdotQs. Expected %dx%d, Got %dx%d. Re-initializing KdotQ.\n",
                        i, column, context_win_size, context_win_size, head.KdotQ.row, head.KdotQ.col);
                try {
                    // Re-assigning will call mat's move assignment or create a new mat and assign.
                    // This will create a new backing file for KdotQ.
                    head.KdotQ = mat(context_win_size, context_win_size);
                }
                catch (const std::exception& e) {
                    fprintf(stderr, "Error re-initializing KdotQ for head (%d, %d) in clParallelKdotQs: %s. Skipping update for this head.\n", i, column, e.what());
                    continue; // Skip to the next head
                }
            }

            // Ensure the memory for KdotQ is mapped
            if (!head.KdotQ.mapped_data) {
                fprintf(stderr, "Error: KdotQ.mapped_data is null for head (%d, %d) in clParallelKdotQs after ensuring dimensions. Skipping update.\n", i, column);
                continue; // Skip to the next head
            }

            size_t expected_elements = static_cast<size_t>(head.KdotQ.row) * head.KdotQ.col;
            if (head_kdotq_flat.size() != expected_elements) {
                fprintf(stderr, "Error: Size of flat data (%zu) from OpenCL kernel does not match KdotQ capacity (%zu) for head (%d, %d) in clParallelKdotQs. Skipping update.\n",
                        head_kdotq_flat.size(), expected_elements, i, column);
                continue; // Skip to the next head
            }
            
            size_t required_bytes_for_kdotq = expected_elements * sizeof(float);
            if (head.KdotQ.mapped_size < required_bytes_for_kdotq) {
                 fprintf(stderr, "Error: KdotQ.mapped_size (%zu) is insufficient for required bytes (%zu) for head (%d, %d) in clParallelKdotQs. Skipping update.\n",
                        head.KdotQ.mapped_size, required_bytes_for_kdotq, i, column);
                continue; // Skip to the next head
            }

            // Copy the flat data (already in row-major order) directly into the memory-mapped region of KdotQ
            memcpy(head.KdotQ.mapped_data, head_kdotq_flat.data(), required_bytes_for_kdotq);
        }
        d_all_kdotq = cl::Buffer();
        d_all_keys = cl::Buffer();
        d_all_querys = cl::Buffer();
        d_all_M = cl::Buffer();
        d_all_EVp = cl::Buffer();
        d_transformer_tokenEmbed_flat = cl::Buffer();
        d_block_tokForBlock_flat = cl::Buffer();
    }
    catch (const std::exception& e) { // Catches std::runtime_error from CL_CHECK and other std exceptions
        // Explicitly release all aggregate OpenCL buffers in case of an exception
        // before re-throwing or returning.
        d_all_kdotq = cl::Buffer();
        d_all_keys = cl::Buffer();
        d_all_querys = cl::Buffer();
        d_all_M = cl::Buffer();
        d_all_EVp = cl::Buffer();
        d_transformer_tokenEmbed_flat = cl::Buffer();
        d_block_tokForBlock_flat = cl::Buffer();
        fprintf(stderr, "Standard exception during clParallelKdotQs: %s\n", e.what());
        // cl::Buffer RAII will handle cleanup automatically
        return;
    }
}

#endif // USE_OPENCL
