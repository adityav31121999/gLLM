// Add this to a new file, e.g., block/cl/kdotqcl.cpp
#ifdef USE_OPENCL

#include "include/block.hpp" // Includes attention.hpp -> mlp.hpp -> maths.hpp
#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <map>
#include <CL/cl.hpp> // Or <CL/cl.h>

// Assume these OpenCL utilities are defined elsewhere (as provided previously)
extern void CL_CHECK(cl_int err, const char* file, int line);
#define CL_CHECK(err) CL_CHECK(err, __FILE__, __LINE__)
extern cl_mem cl_create_buffer(cl_context context, cl_mem_flags flags, size_t size, void* host_ptr, cl_int& err);
extern void cl_write_buffer(cl_command_queue queue, cl_mem buffer, size_t size, const void* ptr, cl_bool blocking = CL_TRUE);
extern void cl_read_buffer(cl_command_queue queue, cl_mem buffer, size_t size, void* ptr, cl_bool blocking = CL_TRUE);
extern void cl_fill_buffer(cl_command_queue queue, cl_mem buffer, const void* pattern, size_t pattern_size, size_t offset, size_t size);
extern void cl_set_kernel_arg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void* arg_value);
extern void cl_enqueue_nd_range_kernel(cl_command_queue queue, cl_kernel kernel, cl_uint work_dim, const size_t* global_work_offset, const size_t* global_work_size, const size_t* local_work_size);
extern void cl_finish(cl_command_queue queue);
extern void cl_release_mem_object(cl_mem memobj);


/**
 * @brief OpenCL: Computes KdotQ in parallel for a column during TRAINING using K and Q matrices.
 *        Uses kernelKdotQforSelf_train or kernelKdotQforCross_train.
 *        Matches signature: void block::clParallelKdotQ(cl_context context, cl_command_queue queue, std::map<std::string, cl_kernel>& kernels, int& columnNumber, int& blockNumber, int& tokenCount, int& promptCount, bool isSelfAttention);
 *
 * @param context OpenCL context.
 * @param queue OpenCL command queue.
 * @param kernels Map of compiled OpenCL kernels.
 * @param columnNumber Index of the column of attention heads.
 * @param blockNumber 1-based index of the current block.
 * @param tokenCount Global token count *before* adding the prompt.
 * @param promptCount Number of new tokens in the current step.
 * @param isSelfAttention True for self-attention, false for cross-attention.
 */
void block::clParallelKdotQ(int& columnNumber, int& blockNumber, int& tokenCount, int& promptCount, bool isSelfAttention)
{
    if (promptCount <= 0) return;
    if (columnNumber < 0 || columnNumber >= this->y) {
        throw std::out_of_range("clParallelKdotQ: columnNumber out of range.");
    }

    cl_context context;
    cl_command_queue queue;
    cl_kernel kernels;
    
    // Use MATHEIGHTS for K/Q dimension based on forward pass logic
    const int key_query_dim = MATHEIGHTS;
    const float inv_scaling = 1.0f / std::sqrt(static_cast<float>(key_query_dim)); // Scale by key/query dim
    const int num_heads_in_column = this->x;
    cl_int err;

    std::vector<std::vector<float>> flat_kdotq_results(num_heads_in_column);
    std::vector<size_t> kdotq_total_sizes(num_heads_in_column);
    std::vector<cl_mem> d_kdotq_buffers(num_heads_in_column, nullptr); // Store buffer handles for later read/release
    std::vector<cl_mem> d_keys_buffers(num_heads_in_column, nullptr);   // Store temporary buffers for cleanup
    std::vector<cl_mem> d_querys_buffers(num_heads_in_column, nullptr); // Store temporary buffers for cleanup

    try {
        for (int i = 0; i < num_heads_in_column; ++i) {
            attention& head = b[i][columnNumber];

            // --- Context Calculation ---
            int context_len_total = tokenCount + promptCount;
            int block_start_token_index = (blockNumber - 1) * CONTEXT_WIN;
            int context_len_block = std::min(context_len_total - block_start_token_index, CONTEXT_WIN);
            context_len_block = std::max(0, context_len_block);

            int num_keys_eff = context_len_block;
            int num_queries_eff = context_len_block;
            int kdotq_rows = context_len_block;
            int kdotq_cols = context_len_block;

            // Resize/Zero host KdotQ buffer
            if (head.KdotQ.size() != static_cast<size_t>(kdotq_rows) || (kdotq_rows > 0 && head.KdotQ[0].size() != static_cast<size_t>(kdotq_cols))) {
                head.KdotQ.assign(kdotq_rows, std::vector<float>(kdotq_cols, 0.0f));
            } else {
                for(auto& row : head.KdotQ) std::fill(row.begin(), row.end(), 0.0f);
            }

            kdotq_total_sizes[i] = static_cast<size_t>(kdotq_rows) * kdotq_cols;
            size_t kdotq_size_bytes = kdotq_total_sizes[i] * sizeof(float);

            if (kdotq_size_bytes == 0) continue;

            // --- Validate K/Q Data ---
            // Use key_query_dim (MATHEIGHTS) for K/Q column check
            if (head.K.size() < static_cast<size_t>(num_keys_eff) || head.Q.size() < static_cast<size_t>(num_queries_eff) || (num_keys_eff > 0 &&
                (head.K[0].size() != static_cast<size_t>(key_query_dim) || head.Q[0].size() != static_cast<size_t>(key_query_dim))))
            {
                std::cerr << "Warning: Training K/Q size mismatch for head (" << i << ", " << columnNumber
                          << "). K=(" << head.K.size() << "x" << (head.K.empty() ? 0 : head.K[0].size())
                          << "), Q=(" << head.Q.size() << "x" << (head.Q.empty() ? 0 : head.Q[0].size())
                          << "), Expected K/Q rows >= " << num_keys_eff << ", cols = " << key_query_dim << ". Skipping." << std::endl;
                continue;
            }

            // --- GPU Allocation & Copy ---
            cl_mem d_kdotq = nullptr, d_keys = nullptr, d_querys = nullptr;
            // K/Q size uses key_query_dim (MATHEIGHTS)
            size_t keys_size_bytes = static_cast<size_t>(num_keys_eff) * key_query_dim * sizeof(float);
            size_t querys_size_bytes = static_cast<size_t>(num_queries_eff) * key_query_dim * sizeof(float);

            d_kdotq = cl_create_buffer(context, CL_MEM_WRITE_ONLY, kdotq_size_bytes, nullptr, err); CL_CHECK(err);
            d_keys = cl_create_buffer(context, CL_MEM_READ_ONLY, keys_size_bytes, nullptr, err); CL_CHECK(err);
            d_querys = cl_create_buffer(context, CL_MEM_READ_ONLY, querys_size_bytes, nullptr, err); CL_CHECK(err);
            d_kdotq_buffers[i] = d_kdotq; // Store for later read
            d_keys_buffers[i] = d_keys;   // Store for cleanup
            d_querys_buffers[i] = d_querys; // Store for cleanup

            float zero_pattern = 0.0f;
            cl_fill_buffer(queue, d_kdotq, &zero_pattern, sizeof(float), 0, kdotq_size_bytes); // Zero init

            // Flatten only the relevant part of K/Q
            std::vector<float> flat_K, flat_Q;
            flat_K.reserve(num_keys_eff * key_query_dim);
            flat_Q.reserve(num_queries_eff * key_query_dim);
            for(int r=0; r<num_keys_eff; ++r) flat_K.insert(flat_K.end(), head.K[r].begin(), head.K[r].end());
            for(int r=0; r<num_queries_eff; ++r) flat_Q.insert(flat_Q.end(), head.Q[r].begin(), head.Q[r].end());


            if (flat_K.size() * sizeof(float) < keys_size_bytes || flat_Q.size() * sizeof(float) < querys_size_bytes) {
                std::cerr << "Warning: Flattened K/Q size insufficient for head (" << i << ", " << columnNumber << "). Skipping." << std::endl;
                // Don't release yet, let the main cleanup handle it
                continue;
            }

            cl_write_buffer(queue, d_keys, keys_size_bytes, flat_K.data(), CL_FALSE); // Non-blocking write
            cl_write_buffer(queue, d_querys, querys_size_bytes, flat_Q.data(), CL_FALSE); // Non-blocking write

            // --- Kernel Launch ---
            cl_kernel kdotq_kernel = nullptr;
            if (isSelfAttention) {
                kdotq_kernel = this->kernels.at("kernelKdotQforSelf_train");
            } else {
                kdotq_kernel = this->kernels.at("kernelKdotQforCross_train");
            }

            size_t local_work_size[2] = { 16, 16 };
            size_t global_work_size[2] = {
                (static_cast<size_t>(num_keys_eff) + local_work_size[0] - 1) / local_work_size[0] * local_work_size[0],
                (static_cast<size_t>(num_queries_eff) + local_work_size[1] - 1) / local_work_size[1] * local_work_size[1]
            };

            cl_int cl_num_queries_eff = num_queries_eff;
            cl_int cl_num_keys_eff = num_keys_eff;
            cl_int cl_kdotq_cols = kdotq_cols;
            // Kernel expects key_query_dim (MATHEIGHTS) as the dimension for dot product
            cl_int cl_key_query_dim_arg = key_query_dim;

            cl_set_kernel_arg(kdotq_kernel, 0, sizeof(cl_mem), &d_kdotq);
            cl_set_kernel_arg(kdotq_kernel, 1, sizeof(cl_mem), &d_keys);
            cl_set_kernel_arg(kdotq_kernel, 2, sizeof(cl_mem), &d_querys);
            cl_set_kernel_arg(kdotq_kernel, 3, sizeof(cl_int), &cl_num_queries_eff);
            cl_set_kernel_arg(kdotq_kernel, 4, sizeof(cl_int), &cl_num_keys_eff);
            cl_set_kernel_arg(kdotq_kernel, 5, sizeof(cl_int), &cl_kdotq_cols);
            cl_set_kernel_arg(kdotq_kernel, 6, sizeof(cl_int), &cl_key_query_dim_arg); // Use MATHEIGHTS
            cl_set_kernel_arg(kdotq_kernel, 7, sizeof(cl_float), &inv_scaling);

            cl_enqueue_nd_range_kernel(queue, kdotq_kernel, 2, nullptr, global_work_size, local_work_size);

            // --- Enqueue Read Back (Asynchronous) ---
            flat_kdotq_results[i].resize(kdotq_total_sizes[i]);
            cl_read_buffer(queue, d_kdotq, kdotq_size_bytes, flat_kdotq_results[i].data(), CL_FALSE); // Non-blocking read

            // Note: Temporary buffers d_keys, d_querys, d_kdotq will be released after clFinish
        }

        // --- Sync & Unflatten ---
        cl_finish(queue); // Wait for all enqueued operations for all heads

        for (int i = 0; i < num_heads_in_column; ++i) {
             // Release temporary buffers for this head
            if (d_keys_buffers[i]) cl_release_mem_object(d_keys_buffers[i]);
            if (d_querys_buffers[i]) cl_release_mem_object(d_querys_buffers[i]);

            if (d_kdotq_buffers[i] == nullptr) continue; // Skip heads that had errors or zero size

            attention& head = b[i][columnNumber];
            int kdotq_rows = head.KdotQ.size();
            int kdotq_cols = (kdotq_rows > 0) ? head.KdotQ[0].size() : 0;

            if (kdotq_total_sizes[i] > 0 && flat_kdotq_results[i].size() == kdotq_total_sizes[i]) {
                unflatten(flat_kdotq_results[i], head.KdotQ, kdotq_rows, kdotq_cols);
            }
            cl_release_mem_object(d_kdotq_buffers[i]); // Release the kdotq buffer now
        }
    } catch (const std::exception& e) {
        std::cerr << "OpenCL Exception in clParallelKdotQ (Training): " << e.what() << std::endl;
        // Cleanup any remaining valid buffer handles
        for (int i = 0; i < num_heads_in_column; ++i) {
            if (d_kdotq_buffers[i]) cl_release_mem_object(d_kdotq_buffers[i]);
            if (d_keys_buffers[i]) cl_release_mem_object(d_keys_buffers[i]);
            if (d_querys_buffers[i]) cl_release_mem_object(d_querys_buffers[i]);
        }
        throw;
    }
}


/**
 * @brief OpenCL: Computes KdotQ in parallel for a column during INFERENCE for BLOCK 1.
 *        Uses global tokenEmbed and head.qkCache (M).
 *        Uses kernelKdotQ_Block1_Self_Inference or kernelKdotQ_Block1_Cross_Inference.
 *        Matches signature: void block::clParallelUseKdotQ(cl_context context, cl_command_queue queue, std::map<std::string, cl_kernel>& kernels, const std::vector<std::vector<float>>& tokenEmbed, int& columnNumber, int& tokenCount, int& promptCount, bool isSelfAttention);
 *
 * @param context OpenCL context.
 * @param queue OpenCL command queue.
 * @param kernels Map of compiled OpenCL kernels.
 * @param tokenEmbed Global token embeddings (Host). Should contain full context.
 * @param columnNumber Index of the column of attention heads.
 * @param tokenCount Global token count *before* adding the prompt.
 * @param promptCount Number of new tokens in the current step.
 * @param isSelfAttention True for self-attention, false for cross-attention.
 */
void block::clParallelUseKdotQ(const std::vector<std::vector<float>>& tokenEmbed, int& columnNumber, 
    int& tokenCount, int& promptCount, bool isSelfAttention)
{
    if (promptCount <= 0) return;
    if (columnNumber < 0 || columnNumber >= this->y) {
        throw std::out_of_range("clParallelUseKdotQ (Block 1): columnNumber out of range.");
    }

    cl_context context;
    cl_command_queue queue;
    cl_kernel kernels;
    
    const int embedding_dim = EMBEDDING;
    const float inv_scaling = 1.0f / std::sqrt(static_cast<float>(embedding_dim)); // Scale by embedding_dim for inference
    const int num_heads_in_column = this->x;
    cl_int err;

    std::vector<std::vector<float>> flat_kdotq_results(num_heads_in_column);
    std::vector<size_t> kdotq_total_sizes(num_heads_in_column);
    std::vector<cl_mem> d_kdotq_buffers(num_heads_in_column, nullptr);
    std::vector<cl_mem> d_M_buffers(num_heads_in_column, nullptr); // Store M buffers for cleanup

    // --- Pre-computation / Validation ---
    int context_len_total = tokenCount + promptCount;
    int prompt_start_index_global = tokenCount;

    if (tokenEmbed.empty() || tokenEmbed.size() < static_cast<size_t>(context_len_total) || (context_len_total > 0 && tokenEmbed[0].size() != static_cast<size_t>(embedding_dim))) {
        throw std::runtime_error("Invalid tokenEmbed for Block 1 inference.");
    }
    std::vector<float> flat_tokenEmbed = flatten(tokenEmbed);
    size_t embed_size_bytes = static_cast<size_t>(context_len_total) * embedding_dim * sizeof(float);
    if (flat_tokenEmbed.size() * sizeof(float) < embed_size_bytes) {
        throw std::runtime_error("Flattened tokenEmbed size error for Block 1.");
    }

    // Allocate and copy global tokenEmbed once
    cl_mem d_tokenEmbed_global = nullptr;
    d_tokenEmbed_global = cl_create_buffer(context, CL_MEM_READ_ONLY, embed_size_bytes, nullptr, err);
    if (err != CL_SUCCESS) { CL_CHECK(err); return; } // Early exit on allocation failure
    // Use blocking write for this shared resource before the loop starts
    cl_write_buffer(queue, d_tokenEmbed_global, embed_size_bytes, flat_tokenEmbed.data(), CL_TRUE);

    try {
        for (int i = 0; i < num_heads_in_column; ++i) {
            attention& head = b[i][columnNumber];

            // --- Context Calculation (Block 1 specific) ---
            int context_len_block = std::min(context_len_total, CONTEXT_WIN);
            context_len_block = std::max(0, context_len_block);

            int num_queries_eff = promptCount;
            int num_keys_eff = context_len_total;
            int kdotq_rows = context_len_block;
            int kdotq_cols = context_len_block;

            // Resize/Zero host KdotQ buffer
            if (head.KdotQ.size() != static_cast<size_t>(kdotq_rows) || (kdotq_rows > 0 && head.KdotQ[0].size() != static_cast<size_t>(kdotq_cols))) {
                head.KdotQ.assign(kdotq_rows, std::vector<float>(kdotq_cols, 0.0f));
            } else {
                for(auto& row : head.KdotQ) std::fill(row.begin(), row.end(), 0.0f);
            }

            kdotq_total_sizes[i] = static_cast<size_t>(kdotq_rows) * kdotq_cols;
            size_t kdotq_size_bytes = kdotq_total_sizes[i] * sizeof(float);

            if (kdotq_size_bytes == 0) continue;

            // --- Validate M (qkCache) ---
            if (head.qkCache.row != embedding_dim || head.qkCache.col != embedding_dim) {
                std::cerr << "Warning: Invalid qkCache dimensions for head (" << i << ", " << columnNumber << ") in Block 1. Skipping." << std::endl;
                continue;
            }
            std::vector<float> flat_M = flatten(head.qkCache);
            size_t M_size_bytes = static_cast<size_t>(embedding_dim) * embedding_dim * sizeof(float);
            if (flat_M.size() * sizeof(float) != M_size_bytes) {
                std::cerr << "Warning: Flattened qkCache size mismatch head (" << i << ", " << columnNumber << ") Block 1. Skipping." << std::endl;
                continue;
            }

            // --- GPU Allocation & Copy (M only) ---
            cl_mem d_kdotq = nullptr, d_M = nullptr;
            d_kdotq = cl_create_buffer(context, CL_MEM_WRITE_ONLY, kdotq_size_bytes, nullptr, err); CL_CHECK(err);
            d_M = cl_create_buffer(context, CL_MEM_READ_ONLY, M_size_bytes, nullptr, err); CL_CHECK(err);
            d_kdotq_buffers[i] = d_kdotq;
            d_M_buffers[i] = d_M; // Store for cleanup

            float zero_pattern = 0.0f;
            cl_fill_buffer(queue, d_kdotq, &zero_pattern, sizeof(float), 0, kdotq_size_bytes);
            cl_write_buffer(queue, d_M, M_size_bytes, flat_M.data(), CL_FALSE);

            // --- Kernel Launch ---
            cl_kernel kdotq_kernel = nullptr;
            if (isSelfAttention) {
                kdotq_kernel = this->kernels.at("kernelKdotQ_Block1_Self_Inference");
            } else {
                kdotq_kernel = this->kernels.at("kernelKdotQ_Block1_Cross_Inference");
            }

            size_t local_work_size[2] = { 16, 16 };
            // Grid covers prompt rows and full context columns
            size_t global_work_size[2] = {
                (static_cast<size_t>(num_keys_eff) + local_work_size[0] - 1) / local_work_size[0] * local_work_size[0],
                (static_cast<size_t>(num_queries_eff) + local_work_size[1] - 1) / local_work_size[1] * local_work_size[1]
            };

            cl_int cl_prompt_start_index = prompt_start_index_global;
            cl_int cl_prompt_len = promptCount;
            cl_int cl_context_len = context_len_total;
            cl_int cl_kdotq_cols = kdotq_cols; // Width of output buffer
            cl_int cl_embedding_dim_arg = embedding_dim;

            cl_set_kernel_arg(kdotq_kernel, 0, sizeof(cl_mem), &d_kdotq);
            cl_set_kernel_arg(kdotq_kernel, 1, sizeof(cl_mem), &d_tokenEmbed_global); // Use global embed buffer
            cl_set_kernel_arg(kdotq_kernel, 2, sizeof(cl_mem), &d_M);
            cl_set_kernel_arg(kdotq_kernel, 3, sizeof(cl_int), &cl_prompt_start_index);
            cl_set_kernel_arg(kdotq_kernel, 4, sizeof(cl_int), &cl_prompt_len);
            cl_set_kernel_arg(kdotq_kernel, 5, sizeof(cl_int), &cl_context_len);
            cl_set_kernel_arg(kdotq_kernel, 6, sizeof(cl_int), &cl_kdotq_cols);
            cl_set_kernel_arg(kdotq_kernel, 7, sizeof(cl_int), &cl_embedding_dim_arg);
            cl_set_kernel_arg(kdotq_kernel, 8, sizeof(cl_float), &inv_scaling);

            cl_enqueue_nd_range_kernel(queue, kdotq_kernel, 2, nullptr, global_work_size, local_work_size);

            // --- Enqueue Read Back (Asynchronous) ---
            flat_kdotq_results[i].resize(kdotq_total_sizes[i]);
            cl_read_buffer(queue, d_kdotq, kdotq_size_bytes, flat_kdotq_results[i].data(), CL_FALSE);

            // Note: d_M and d_kdotq released after clFinish
        }

        // --- Sync & Unflatten ---
        cl_finish(queue); // Wait for all heads in the column

        for (int i = 0; i < num_heads_in_column; ++i) {
            if (d_M_buffers[i]) cl_release_mem_object(d_M_buffers[i]); // Release M buffer

            if (d_kdotq_buffers[i] == nullptr) continue; // Skip heads with errors/zero size

            attention& head = b[i][columnNumber];
            int kdotq_rows = head.KdotQ.size();
            int kdotq_cols = (kdotq_rows > 0) ? head.KdotQ[0].size() : 0;

            if (kdotq_total_sizes[i] > 0 && flat_kdotq_results[i].size() == kdotq_total_sizes[i]) {
                unflatten(flat_kdotq_results[i], head.KdotQ, kdotq_rows, kdotq_cols);
            }
            cl_release_mem_object(d_kdotq_buffers[i]); // Release kdotq buffer
        }

    } catch (const std::exception& e) {
        std::cerr << "OpenCL Exception in clParallelUseKdotQ (Block 1): " << e.what() << std::endl;
        for (int i = 0; i < num_heads_in_column; ++i) {
            if (d_kdotq_buffers[i]) cl_release_mem_object(d_kdotq_buffers[i]);
            if (d_M_buffers[i]) cl_release_mem_object(d_M_buffers[i]);
        }
        // d_tokenEmbed_global needs release outside the loop
        if (d_tokenEmbed_global) cl_release_mem_object(d_tokenEmbed_global);
        throw;
    }

    // Release the global token embed buffer
    if (d_tokenEmbed_global) cl_release_mem_object(d_tokenEmbed_global);
}


/**
 * @brief OpenCL: Computes KdotQ in parallel for a column during INFERENCE for BLOCK N > 1.
 *        Uses block-local tokForBlock, previous block's EVp, and head.qkCache (M).
 *        Uses kernelKdotQ_BlockN_Self_Inference or kernelKdotQ_BlockN_Cross_Inference.
 *        Matches signature: void block::clParallelUseKdotQ(cl_context context, cl_command_queue queue, std::map<std::string, cl_kernel>& kernels, const std::vector<std::vector<std::vector<float>>>& EVp, int& columnNumber, int& blockNumber, int& tokenCount, int& promptCount, bool isSelfAttention);
 *
 * @param context OpenCL context.
 * @param queue OpenCL command queue.
 * @param kernels Map of compiled OpenCL kernels.
 * @param EVp Vertical retention vectors from the previous block (Host). Structure: EVp[head_idx][token_idx][embedding_dim].
 * @param columnNumber Index of the column of attention heads.
 * @param blockNumber 1-based index of the current block (must be > 1).
 * @param tokenCount Global token count *before* adding the prompt.
 * @param promptCount Number of new tokens in the current step.
 * @param isSelfAttention True for self-attention, false for cross-attention.
 */
void block::clParallelUseKdotQ(const std::vector<std::vector<std::vector<float>>>& EVp, int& columnNumber, int& blockNumber, 
    int& tokenCount, int& promptCount, bool isSelfAttention)
{
    if (blockNumber <= 1) {
         throw std::invalid_argument("clParallelUseKdotQ (Block N): blockNumber must be > 1.");
    }
    if (promptCount <= 0) return;
    if (columnNumber < 0 || columnNumber >= this->y) {
        throw std::out_of_range("clParallelUseKdotQ (Block N): columnNumber out of range.");
    }

    cl_context context;
    cl_command_queue queue;
    cl_kernel kernels;
    
    const int embedding_dim = EMBEDDING;
    const float inv_scaling = 1.0f / std::sqrt(static_cast<float>(embedding_dim)); // Scale by embedding_dim for inference
    const int num_heads_in_column = this->x;
    cl_int err;

    std::vector<std::vector<float>> flat_kdotq_results(num_heads_in_column);
    std::vector<size_t> kdotq_total_sizes(num_heads_in_column);
    std::vector<cl_mem> d_kdotq_buffers(num_heads_in_column, nullptr);
    std::vector<cl_mem> d_EVp_buffers(num_heads_in_column, nullptr); // Store EVp buffers for cleanup
    std::vector<cl_mem> d_M_buffers(num_heads_in_column, nullptr);   // Store M buffers for cleanup

    // --- Pre-computation / Validation ---
    int context_len_total = tokenCount + promptCount;
    int block_start_token_index = (blockNumber - 1) * CONTEXT_WIN;
    int context_len_block = std::min(context_len_total - block_start_token_index, CONTEXT_WIN);
    context_len_block = std::max(0, context_len_block);
    int prompt_start_index_global = tokenCount;
    int prompt_start_index_in_block = std::max(0, prompt_start_index_global - block_start_token_index);
    prompt_start_index_in_block = std::min(prompt_start_index_in_block, CONTEXT_WIN); // Clamp to block window

    if (this->tokForBlock.empty() || this->tokForBlock.size() < static_cast<size_t>(context_len_block) ||
        (context_len_block > 0 && this->tokForBlock[0].size() != static_cast<size_t>(embedding_dim))) {
        throw std::runtime_error("Invalid tokForBlock for Block N inference.");
    }
    std::vector<float> flat_tokForBlock = flatten(this->tokForBlock);
    size_t tok_size_bytes = static_cast<size_t>(context_len_block) * embedding_dim * sizeof(float);
    if (flat_tokForBlock.size() * sizeof(float) < tok_size_bytes) {
        throw std::runtime_error("Flattened tokForBlock size error for Block N.");
    }
    if (EVp.size() < static_cast<size_t>(num_heads_in_column)) {
        throw std::runtime_error("Invalid EVp structure for Block N inference.");
    }

    // Allocate and copy tokForBlock once
    cl_mem d_tokForBlock_global = nullptr;
    d_tokForBlock_global = cl_create_buffer(context, CL_MEM_READ_ONLY, tok_size_bytes, nullptr, err);
    if (err != CL_SUCCESS) { CL_CHECK(err); return; }
    // Use blocking write for this shared resource
    cl_write_buffer(queue, d_tokForBlock_global, tok_size_bytes, flat_tokForBlock.data(), CL_TRUE);

    try {
        for (int i = 0; i < num_heads_in_column; ++i) {
            attention& head = b[i][columnNumber];

            // --- Context Calculation (Block N specific) ---
            int num_queries_eff = promptCount;
            int num_keys_eff = context_len_block;
            int kdotq_rows = context_len_block;
            int kdotq_cols = context_len_block;

            // Resize/Zero host KdotQ buffer
            if (head.KdotQ.size() != static_cast<size_t>(kdotq_rows) || (kdotq_rows > 0 && head.KdotQ[0].size() != static_cast<size_t>(kdotq_cols))) {
                 head.KdotQ.assign(kdotq_rows, std::vector<float>(kdotq_cols, 0.0f));
            } else {
                for(auto& row : head.KdotQ) std::fill(row.begin(), row.end(), 0.0f);
            }

            kdotq_total_sizes[i] = static_cast<size_t>(kdotq_rows) * kdotq_cols;
            size_t kdotq_size_bytes = kdotq_total_sizes[i] * sizeof(float);

            if (kdotq_size_bytes == 0) continue;

            // --- Validate M (qkCache) and EVp[i] ---
            if (head.qkCache.row != embedding_dim || head.qkCache.col != embedding_dim) {
                std::cerr << "Warning: Invalid qkCache dimensions for head (" << i << ", " << columnNumber << ") in Block " << blockNumber << ". Skipping." << std::endl;
                continue;
            }
            if (EVp[i].empty() || EVp[i].size() < static_cast<size_t>(context_len_block) || (context_len_block > 0 && EVp[i][0].size() != static_cast<size_t>(embedding_dim))) {
                std::cerr << "Warning: Invalid EVp data for head (" << i << ", " << columnNumber << ") in Block " << blockNumber << ". Skipping." << std::endl;
                continue;
            }

            std::vector<float> flat_M = flatten(head.qkCache);
            // Flatten only the relevant part of EVp[i]
            std::vector<float> flat_EVp;
            flat_EVp.reserve(context_len_block * embedding_dim);
            for(int r=0; r<context_len_block; ++r) flat_EVp.insert(flat_EVp.end(), EVp[i][r].begin(), EVp[i][r].end());

            size_t M_size_bytes = static_cast<size_t>(embedding_dim) * embedding_dim * sizeof(float);
            size_t evp_size_bytes = static_cast<size_t>(context_len_block) * embedding_dim * sizeof(float);

            if (flat_M.size() * sizeof(float) != M_size_bytes || flat_EVp.size() * sizeof(float) < evp_size_bytes) {
                std::cerr << "Warning: Flattened qkCache/EVp size mismatch head (" << i << ", " << columnNumber << ") Block " << blockNumber << ". Skipping." << std::endl;
                continue;
            }

            // --- GPU Allocation & Copy (EVp, M) ---
            cl_mem d_kdotq = nullptr, d_EVp = nullptr, d_M = nullptr;
            d_kdotq = cl_create_buffer(context, CL_MEM_WRITE_ONLY, kdotq_size_bytes, nullptr, err); CL_CHECK(err);
            d_EVp = cl_create_buffer(context, CL_MEM_READ_ONLY, evp_size_bytes, nullptr, err); CL_CHECK(err);
            d_M = cl_create_buffer(context, CL_MEM_READ_ONLY, M_size_bytes, nullptr, err); CL_CHECK(err);
            d_kdotq_buffers[i] = d_kdotq;
            d_EVp_buffers[i] = d_EVp; // Store for cleanup
            d_M_buffers[i] = d_M;     // Store for cleanup

            float zero_pattern = 0.0f;
            cl_fill_buffer(queue, d_kdotq, &zero_pattern, sizeof(float), 0, kdotq_size_bytes);
            cl_write_buffer(queue, d_EVp, evp_size_bytes, flat_EVp.data(), CL_FALSE);
            cl_write_buffer(queue, d_M, M_size_bytes, flat_M.data(), CL_FALSE);

            // --- Kernel Launch ---
            cl_kernel kdotq_kernel = nullptr;
            if (isSelfAttention) {
                kdotq_kernel = this->kernels.at("kernelKdotQ_BlockN_Self_Inference");
            } else {
                kdotq_kernel = this->kernels.at("kernelKdotQ_BlockN_Cross_Inference");
            }

            size_t local_work_size[2] = { 16, 16 };
            // Grid covers prompt rows and block context columns
            size_t global_work_size[2] = {
                (static_cast<size_t>(num_keys_eff) + local_work_size[0] - 1) / local_work_size[0] * local_work_size[0],
                (static_cast<size_t>(num_queries_eff) + local_work_size[1] - 1) / local_work_size[1] * local_work_size[1]
            };

            cl_int cl_prompt_start_index_in_block = prompt_start_index_in_block;
            cl_int cl_prompt_len = promptCount;
            cl_int cl_context_len_block = context_len_block;
            cl_int cl_kdotq_cols = kdotq_cols; // Width of output buffer
            cl_int cl_embedding_dim_arg = embedding_dim;

            cl_set_kernel_arg(kdotq_kernel, 0, sizeof(cl_mem), &d_kdotq);
            cl_set_kernel_arg(kdotq_kernel, 1, sizeof(cl_mem), &d_tokForBlock_global); // Use global block token buffer
            cl_set_kernel_arg(kdotq_kernel, 2, sizeof(cl_mem), &d_EVp);
            cl_set_kernel_arg(kdotq_kernel, 3, sizeof(cl_mem), &d_M);
            cl_set_kernel_arg(kdotq_kernel, 4, sizeof(cl_int), &cl_prompt_start_index_in_block);
            cl_set_kernel_arg(kdotq_kernel, 5, sizeof(cl_int), &cl_prompt_len);
            cl_set_kernel_arg(kdotq_kernel, 6, sizeof(cl_int), &cl_context_len_block);
            cl_set_kernel_arg(kdotq_kernel, 7, sizeof(cl_int), &cl_kdotq_cols);
            cl_set_kernel_arg(kdotq_kernel, 8, sizeof(cl_int), &cl_embedding_dim_arg);
            cl_set_kernel_arg(kdotq_kernel, 9, sizeof(cl_float), &inv_scaling);

            cl_enqueue_nd_range_kernel(queue, kdotq_kernel, 2, nullptr, global_work_size, local_work_size);

            // --- Enqueue Read Back (Asynchronous) ---
            flat_kdotq_results[i].resize(kdotq_total_sizes[i]);
            cl_read_buffer(queue, d_kdotq, kdotq_size_bytes, flat_kdotq_results[i].data(), CL_FALSE);

            // Note: d_EVp, d_M, d_kdotq released after clFinish
        }

        // --- Sync & Unflatten ---
        cl_finish(queue); // Wait for all heads in the column

        for (int i = 0; i < num_heads_in_column; ++i) {
            if (d_EVp_buffers[i]) cl_release_mem_object(d_EVp_buffers[i]); // Release EVp buffer
            if (d_M_buffers[i]) cl_release_mem_object(d_M_buffers[i]);     // Release M buffer

            if (d_kdotq_buffers[i] == nullptr) continue; // Skip heads with errors/zero size

            attention& head = b[i][columnNumber];
            int kdotq_rows = head.KdotQ.size();
            int kdotq_cols = (kdotq_rows > 0) ? head.KdotQ[0].size() : 0;

            if (kdotq_total_sizes[i] > 0 && flat_kdotq_results[i].size() == kdotq_total_sizes[i]) {
                unflatten(flat_kdotq_results[i], head.KdotQ, kdotq_rows, kdotq_cols);
            }
            cl_release_mem_object(d_kdotq_buffers[i]); // Release kdotq buffer
        }

    } catch (const std::exception& e) {
        std::cerr << "OpenCL Exception in clParallelUseKdotQ (Block N): " << e.what() << std::endl;
        for (int i = 0; i < num_heads_in_column; ++i) {
            if (d_kdotq_buffers[i]) cl_release_mem_object(d_kdotq_buffers[i]);
            if (d_EVp_buffers[i]) cl_release_mem_object(d_EVp_buffers[i]);
            if (d_M_buffers[i]) cl_release_mem_object(d_M_buffers[i]);
        }
        // d_tokForBlock_global needs release outside the loop
        if (d_tokForBlock_global) cl_release_mem_object(d_tokForBlock_global);
        throw;
    }

    // Release the global block token buffer
    if (d_tokForBlock_global) cl_release_mem_object(d_tokForBlock_global);
}

#endif // USE_OPENCL
