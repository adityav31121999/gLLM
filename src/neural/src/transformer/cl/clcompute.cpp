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
 * @brief Compute KdotQ for all heads of blocks for training.
 * @param sequence1Count number of new tokens in the sequence1 being processed in this step.
 * @param currentTokenCount total number of tokens processed *before* this step across all blocks (Full context).
 * @param blockCount 1-based index of the current block being processed.
 * @param isSelf true for self-attention, false for cross-attention.
 * @param inTraining true if in training mode, false if in inference mode.
 * @note Assumes cl_context, cl_queue, and cl_kernels are accessible (e.g., member variables).
 */
void transformer::clKdotQ4Train(int& sequence1Count, int& currentTokenCount, int& blockCount, bool&  isSelf, bool&  inTraining)
{
    // --- Basic Sanity Checks ---
    if (blockCount < 1 || blockCount > m) {
        fprintf(stderr, "Error in clKdotQ4Train: Invalid blockCount %d (max %d)\n", blockCount, m);
        return;
    }
    if (sequence1Count < 0) {
        fprintf(stderr, "Warning in clKdotQ4Train: sequence1Count is negative (%d). Setting to 0.\n", sequence1Count);
        return;
    }

    // --- Pre-computation and Setup ---
    const float inv_scaling = 1.0f / sqrtf(static_cast<float>(EMBEDDING)); // Use static_cast for clarity
    size_t embed_dim = d;
    size_t context_win_size = n;
    size_t k_q_bytes = context_win_size * embed_dim * sizeof(float);
    size_t kdotq_bytes = context_win_size * context_win_size * sizeof(float);

    // initiate a block that reference the current block operations
    block& current_block = blocks[blockCount - 1];

    // Buffers will be created per-head inside the loop
    cl::Buffer d_KdotQ, d_K, d_Q;   // key, query, KdotQ buffers
    cl::Kernel kernel;              // kernel object for this launch

    cl::NDRange local_work_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y);
    cl::NDRange global_work_size;
    auto calculate_global_size = [](size_t total_x, size_t total_y) {
        size_t global_x = (total_x + WORKGROUP_SIZE_X - 1) / WORKGROUP_SIZE_X * WORKGROUP_SIZE_X;
        size_t global_y = (total_y + WORKGROUP_SIZE_Y - 1) / WORKGROUP_SIZE_Y * WORKGROUP_SIZE_Y;
        return cl::NDRange(global_x, global_y);
    };

    // Calculate context-related variables once before the loop if they are constant for all heads
    cl_int current_tokens_in_window = 0;
    cl_int num_queries_eff = 0;
    cl_int num_keys_eff = 0;

    // Effective number of tokens currently in the window *before* adding the sequence1
    current_tokens_in_window = currentTokenCount % context_win_size;
    if (current_tokens_in_window == 0 && currentTokenCount > 0) {
        if (blockCount > 1 && currentTokenCount >= static_cast<cl_int>(context_win_size) * (blockCount - 1)) { // Cast for comparison
            current_tokens_in_window = 0;
        } 
        else {
            current_tokens_in_window = context_win_size;
        }
    }
    num_queries_eff = std::min<float>(current_tokens_in_window + sequence1Count, context_win_size); // Parenthesize std::min
    num_keys_eff = num_queries_eff;
    global_work_size = calculate_global_size(num_keys_eff, num_queries_eff);

    try {
        //  ALL KDOTQ CALCULATIONS PER HEAD
        for(int i = 0; i < x; i++) {
            for(int j = 0; j < y; j++) {
                attention& head = current_block.b[i][j];

                // Create and write buffers for this specific head
                d_K = cl::Buffer(clcontext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, k_q_bytes, head.K.mapped_data);
                d_Q = cl::Buffer(clcontext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, k_q_bytes, head.Q.mapped_data);
                d_KdotQ = cl::Buffer(clcontext.context, CL_MEM_WRITE_ONLY, kdotq_bytes);

                if (isSelf)
                    kernel = clcontext.kernels.at("kernelKdotQforSelf_train");
                else
                    kernel = clcontext.kernels.at("kernelKdotQforCross_train");
                // kernel arguments
                CL_CHECK(kernel.setArg(0, d_KdotQ));
                CL_CHECK(kernel.setArg(1, d_K));
                CL_CHECK(kernel.setArg(2, d_Q));
                CL_CHECK(kernel.setArg(3, static_cast<cl_int>(num_queries_eff)));
                CL_CHECK(kernel.setArg(4, static_cast<cl_int>(num_keys_eff)));
                CL_CHECK(kernel.setArg(5, static_cast<cl_int>(context_win_size)));
                CL_CHECK(kernel.setArg(6, static_cast<cl_int>(embed_dim)));
                CL_CHECK(kernel.setArg(7, inv_scaling));

                if (global_work_size[0] > 0 && global_work_size[1] > 0) {
                    CL_CHECK(clcontext.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_work_size, local_work_size));
                }

                // Read the result back into the head's KdotQ matrix
                if (!head.KdotQ.mapped_data || head.KdotQ.row != context_win_size || head.KdotQ.col != context_win_size) {
                    fprintf(stderr, "Warning: KdotQ dimensions mismatch for head (%d, %d). Re-initializing.\n", i, j);
                    head.KdotQ = mat(context_win_size, context_win_size);
                }
                if (head.KdotQ.mapped_data) {
                    CL_CHECK(clcontext.queue.enqueueReadBuffer(d_KdotQ, CL_TRUE, 0, kdotq_bytes, head.KdotQ.mapped_data));
                } else {
                    fprintf(stderr, "Error: KdotQ for head (%d, %d) is not mapped. Cannot read result.\n", i, j);
                    return;
                }
                // Release buffers for this head
                d_KdotQ = cl::Buffer();
                d_K = cl::Buffer();
                d_Q = cl::Buffer();
            }
        }
        // Ensure all operations for this function are complete.
        clcontext.queue.finish();
    }
    catch (const std::out_of_range& oor) {
        fprintf(stderr, "Kernel not found in clKdotQ4Train: %s\n", oor.what());
        return;
    }
    catch (const std::exception& e) {
        fprintf(stderr, "Standard exception during clKdotQ4Train: %s\n", e.what());
        return;
    }
}


/**
 * @brief Compute KdotQ for all heads of blocks for inference.
 * @param sequence1Count number of new tokens in the sequence1 being processed in this step.
 * @param currentTokenCount total number of tokens processed *before* this step across all blocks (Full context).
 * @param blockCount 1-based index of the current block being processed.
 * @param isSelf true for self-attention, false for cross-attention.
 * @param inTraining true if in training mode, false if in inference mode.
 * @note Assumes cl_context, cl_queue, and cl_kernels are accessible (e.g., member variables).
 */
void transformer::clKdotQ4Infer(int& sequence1Count, int& currentTokenCount, int& blockCount, bool& isSelf, bool& inTraining)
{
    // --- Basic Sanity Checks ---
    if (blockCount < 1 || blockCount > m) {
        fprintf(stderr, "Error in clKdotQ4Infer: Invalid blockCount %d (max %d)\n", blockCount, m);
        return;
    }
    if (sequence1Count < 0) {
        fprintf(stderr, "Warning in clKdotQ4Infer: sequence1Count is negative (%d). Setting to 0.\n", sequence1Count);
        return;
    }

    // --- Pre-computation and Setup ---
    const float inv_scaling = 1.0f / sqrtf(static_cast<float>(EMBEDDING)); // Use static_cast for clarity
    size_t embed_dim = d;
    size_t context_win_size = n;
    size_t kdotq_val = context_win_size * context_win_size;
    size_t kdotq_bytes = kdotq_val * sizeof(float);
    size_t token_vals = context_win_size * embed_dim;
    size_t token_bytes = token_vals * sizeof(float);
    size_t qkcache_val = embed_dim * embed_dim;
    size_t qkcache_bytes = qkcache_val * sizeof(float);

    // initiate a block that reference the current block operations
    block& current_block = blocks[0];

    // Buffers will be created per-head inside the loop
    cl::Buffer d_KdotQ;             // key, query, KdotQ buffers
    cl::Buffer d_tokenEmbed, d_EVp; // token embedding and vertical retention buffers
    cl::Buffer d_M;                 // qkCache, Query and Key weights buffers
    cl::Kernel kernel;              // kernel object for this launch

    cl_int sequence1_start_index = 0;
    cl_int context_len = 0;
    cl_int effective_sequence1_len = 0;
    cl_int tokens_processed_in_prev_blocks = 0;
    cl_int tokens_in_block_before_sequence1 = 0;
    cl_int sequence1_start_index_in_block = 0;
    cl_int context_len_in_block = 0;

    cl::NDRange local_work_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y);
    cl::NDRange global_work_size; // Will be set inside the loop based on effective dimensions
    auto calculate_global_size = [](size_t total_x, size_t total_y) {
        size_t global_x = (total_x + WORKGROUP_SIZE_X - 1) / WORKGROUP_SIZE_X * WORKGROUP_SIZE_X;
        size_t global_y = (total_y + WORKGROUP_SIZE_Y - 1) / WORKGROUP_SIZE_Y * WORKGROUP_SIZE_Y;
        return cl::NDRange(global_x, global_y);
    };

    // inference
    if (blockCount == 1) {
        sequence1_start_index = currentTokenCount;
        context_len = std::min<size_t>(currentTokenCount + sequence1Count, context_win_size);
        effective_sequence1_len = std::min<size_t>(sequence1Count, std::max<size_t>(0, context_win_size - sequence1_start_index)); // Parenthesize std::min and std::max
        global_work_size = calculate_global_size(context_len, effective_sequence1_len);
    } 
    else {
        tokens_processed_in_prev_blocks = (blockCount - 1) * context_win_size;
        tokens_in_block_before_sequence1 = std::max<size_t>(0, std::min<size_t>(currentTokenCount - tokens_processed_in_prev_blocks, context_win_size)); // Parenthesize std::min and std::max
        sequence1_start_index_in_block = tokens_in_block_before_sequence1;
        context_len_in_block = std::min<size_t>(tokens_in_block_before_sequence1 + sequence1Count, context_win_size); // Parenthesize std::min
        effective_sequence1_len = std::min<size_t>(sequence1Count, std::max<size_t>(0, context_win_size - sequence1_start_index_in_block));
        global_work_size = calculate_global_size(context_len_in_block, effective_sequence1_len);
    }

    try {
        // get embeddings + positions
        if(blockCount == 1) {
            d_tokenEmbed = cl::Buffer(clcontext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, token_bytes, embedPlusPos.mapped_data);
        }
        else if (blockCount > 1) {
            size_t fromHereInTokenEmbed = static_cast<size_t>(CONTEXT_WIN) * (blockCount - 1) * sizeof(float);
            const float* host_src_ptr = embedPlusPos.mapped_data + (fromHereInTokenEmbed / sizeof(float));
            d_tokenEmbed = cl::Buffer(clcontext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, token_bytes, embedPlusPos.mapped_data);
        }
        else {
            fprintf(stderr, "Error in clKdotQ4Infer: Invalid blockCount %d (max %d)\n", blockCount, m);
            return;
        }

        //  ALL KDOTQ CALCULATIONS PER HEAD
        for(int i = 0; i < x; i++) {
            for(int j = 0; j < y; j++) {
                attention& head = current_block.b[i][j];
                d_KdotQ = cl::Buffer(clcontext.context, CL_MEM_WRITE_ONLY, kdotq_bytes);
                d_M = cl::Buffer(clcontext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, qkcache_bytes, head.qkCache.mapped_data);
                if (effective_sequence1_len > 0) {
                    if (blockCount == 1) {
                        if (isSelf)
                            kernel = clcontext.kernels.at("kernelKdotQBlock1Selfi");
                        else
                            kernel = clcontext.kernels.at("kernelKdotQBlock1Crossi");
                        // kernel arguments
                        CL_CHECK(kernel.setArg(0, d_KdotQ));
                        CL_CHECK(kernel.setArg(1, d_tokenEmbed));
                        CL_CHECK(kernel.setArg(2, d_M));
                        CL_CHECK(kernel.setArg(3, static_cast<cl_int>(sequence1_start_index)));
                        CL_CHECK(kernel.setArg(4, static_cast<cl_int>(effective_sequence1_len)));
                        CL_CHECK(kernel.setArg(5, static_cast<cl_int>(context_len)));
                        CL_CHECK(kernel.setArg(6, static_cast<cl_int>(context_win_size)));
                        CL_CHECK(kernel.setArg(7, static_cast<cl_int>(embed_dim)));
                        CL_CHECK(kernel.setArg(8, inv_scaling));
                    }
                    else {
                        attention& prev_head = blocks[blockCount - 2].b[i][j];
                        d_EVp = cl::Buffer(clcontext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, token_bytes, prev_head.EV.mapped_data);
                        if (isSelf)
                            kernel = clcontext.kernels.at("kernelKdotQBlockNSelfi");
                        else
                            kernel = clcontext.kernels.at("kernelKdotQBlockNCrossi");
                        // kernel arguments
                        CL_CHECK(kernel.setArg(0, d_KdotQ));
                        CL_CHECK(kernel.setArg(1, d_tokenEmbed));
                        CL_CHECK(kernel.setArg(2, d_EVp));
                        CL_CHECK(kernel.setArg(3, d_M));
                        CL_CHECK(kernel.setArg(4, static_cast<cl_int>(sequence1_start_index_in_block)));
                        CL_CHECK(kernel.setArg(5, static_cast<cl_int>(effective_sequence1_len)));
                        CL_CHECK(kernel.setArg(6, static_cast<cl_int>(context_len_in_block)));
                        CL_CHECK(kernel.setArg(7, static_cast<cl_int>(context_win_size)));
                        CL_CHECK(kernel.setArg(8, static_cast<cl_int>(embed_dim)));
                        CL_CHECK(kernel.setArg(9, inv_scaling));
                        d_EVp = cl::Buffer();
                    }
                    if (global_work_size[0] > 0 && global_work_size[1] > 0) { // Ensure global size is valid
                        CL_CHECK(clcontext.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_work_size, local_work_size));
                    }

                    // Read the result back into the head's KdotQ matrix
                    if (!head.KdotQ.mapped_data || head.KdotQ.row != context_win_size || head.KdotQ.col != context_win_size) {
                        fprintf(stderr, "Warning: KdotQ dimensions mismatch for head (%d, %d). Re-initializing.\n", i, j);
                        head.KdotQ = mat(context_win_size, context_win_size);
                    }
                    if (head.KdotQ.mapped_data) {
                        CL_CHECK(clcontext.queue.enqueueReadBuffer(d_KdotQ, CL_TRUE, 0, kdotq_bytes, head.KdotQ.mapped_data));
                    } else {
                        fprintf(stderr, "Error: KdotQ for head (%d, %d) is not mapped. Cannot read result.\n", i, j);
                    }
                }
                d_KdotQ = cl::Buffer();
                d_M = cl::Buffer();
            }
        }
        d_tokenEmbed = cl::Buffer();
        // Ensure all operations for this function are complete.
        clcontext.queue.finish();
    }
    catch (const std::out_of_range& oor) {
        fprintf(stderr, "Kernel not found in clKdotQ4Infer: %s\n", oor.what());
        return;
    }
    catch (const std::exception& e) {
        fprintf(stderr, "Standard exception during clKdotQ4Infer: %s\n", e.what());
        return;
    }
}

#endif // USE_OPENCL
