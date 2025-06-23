
#ifdef USE_OPENCL
#if defined(_WIN64)
    #define CL_HPP_ENABLE_EXCEPTIONS
    #define CL_HPP_TARGET_OPENCL_VERSION 300
    // For Windows, use the older/common cl.hpp
    #include <CL/cl.hpp>
#elif defined(__linux__)
    #define CL_HPP_TARGET_OPENCL_VERSION 300
    #include <CL/opencl.hpp>
#endif
#include "include/block.hpp"
#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <maths.hpp>

/**
 * @brief OpenCL: Computes KdotQ in parallel for a column during TRAINING using K and Q matrices.
 *        Uses kernelKdotQforSelf_train or kernelKdotQforCross_train.
 *        Matches signature: void block::clParallelKdotQ(cl_context context, cl_command_queue queue, std::map<std::string, cl_kernel>& kernels, int& columnNumber, int& blockNumber, int& tokenCount, int& promptCount, bool isSelfAttention);
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

    // Use MATHEIGHTS for K/Q dimension based on forward pass logic
    const int key_query_dim = MATHEIGHTS;
    const float inv_scaling = 1.0f / std::sqrt(static_cast<float>(key_query_dim)); // Scale by key/query dim
    const int num_heads_in_column = this->x;

    std::vector<cl::Buffer> d_kdotq_buffers(num_heads_in_column); // Store cl::Buffer objects

    cl_int cl_err; // For OpenCL error codes

    try {
        for (int i = 0; i < num_heads_in_column; ++i) {
            attention& head = b[i][columnNumber];

            // --- Context Calculation ---
            int context_len_total = tokenCount + promptCount;
            int block_start_token_index = (blockNumber - 1) * CONTEXT_WIN;
            int context_len_block = (std::min)(context_len_total - block_start_token_index, CONTEXT_WIN); // Parenthesize std::min
            context_len_block = (std::max)(0, context_len_block); // Parenthesize std::max

            int num_keys_eff = context_len_block;
            int num_queries_eff = context_len_block;
            int kdotq_rows = context_len_block;
            int kdotq_cols = context_len_block;

            // Ensure KdotQ mat is correctly dimensioned
            if (head.KdotQ.row != kdotq_rows || head.KdotQ.col != kdotq_cols) {
                head.KdotQ = mat(kdotq_rows, kdotq_cols); // Reinitializes with a new temp file
            }
            // Zeroing KdotQ.mapped_data can be done here if needed before GPU fill,
            // but GPU fill is generally preferred.

            size_t kdotq_elements = static_cast<size_t>(kdotq_rows) * kdotq_cols;
            size_t kdotq_size_bytes = kdotq_elements * sizeof(float);

            if (kdotq_size_bytes == 0) continue;

            // --- Validate K/Q Data ---
            // Use key_query_dim (MATHEIGHTS) for K/Q column check
            if (head.K.row < num_keys_eff || head.Q.row < num_queries_eff ||
                head.K.col != key_query_dim || head.Q.col != key_query_dim)
            {
                std::cerr << "Warning: Training K/Q size mismatch for head (" << i << ", " << columnNumber
                          << "). K=(" << head.K.row << "x" << head.K.col
                          << "), Q=(" << head.Q.row << "x" << head.Q.col
                          << "), Expected K/Q rows >= " << num_keys_eff << ", cols = " << key_query_dim << ". Skipping." << std::endl;
                continue;
            }

            // K/Q size uses key_query_dim (MATHEIGHTS)
            size_t keys_size_bytes = static_cast<size_t>(num_keys_eff) * key_query_dim * sizeof(float);
            size_t querys_size_bytes = static_cast<size_t>(num_queries_eff) * key_query_dim * sizeof(float);
            
            cl::Buffer d_kdotq(this->clcontext.context, CL_MEM_WRITE_ONLY, kdotq_size_bytes, nullptr, &cl_err); CL_CHECK(cl_err); // Output buffer
            cl::Buffer d_keys(this->clcontext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, keys_size_bytes, head.K.mapped_data, &cl_err); CL_CHECK(cl_err);
            cl::Buffer d_querys(this->clcontext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, querys_size_bytes, head.Q.mapped_data, &cl_err); CL_CHECK(cl_err);
            d_kdotq_buffers[i] = d_kdotq; // Store for later read
            // No need to store d_keys/d_querys separately for release, cl::Buffer handles it

            // Zero initialize the output buffer
            float zero_pattern = 0.0f;
            CL_CHECK(this->clcontext.queue.enqueueFillBuffer(d_kdotq, zero_pattern, 0, kdotq_size_bytes));

            // --- Kernel Launch ---
            cl::Kernel kdotq_kernel;
            if (isSelfAttention) {
                kdotq_kernel = this->clcontext.kernels.at("kernelKdotQforSelf_train");
            } else {
                kdotq_kernel = this->clcontext.kernels.at("kernelKdotQforCross_train");
            }

            cl::NDRange local_work_size(16, 16);
            cl::NDRange global_work_size(
                (static_cast<size_t>(num_keys_eff) + local_work_size[0] - 1) / local_work_size[0] * local_work_size[0],
                (static_cast<size_t>(num_queries_eff) + local_work_size[1] - 1) / local_work_size[1] * local_work_size[1]);

            cl_int cl_num_queries_eff = num_queries_eff;
            cl_int cl_num_keys_eff = num_keys_eff;
            cl_int cl_kdotq_cols = kdotq_cols;
            // Kernel expects key_query_dim (MATHEIGHTS) as the dimension for dot product
            cl_int cl_key_query_dim_arg = key_query_dim;

            CL_CHECK(kdotq_kernel.setArg(0, d_kdotq));
            CL_CHECK(kdotq_kernel.setArg(1, d_keys));
            CL_CHECK(kdotq_kernel.setArg(2, d_querys));
            CL_CHECK(kdotq_kernel.setArg(3, cl_num_queries_eff));
            CL_CHECK(kdotq_kernel.setArg(4, cl_num_keys_eff));
            CL_CHECK(kdotq_kernel.setArg(5, cl_kdotq_cols));
            CL_CHECK(kdotq_kernel.setArg(6, cl_key_query_dim_arg)); // Use MATHEIGHTS
            CL_CHECK(kdotq_kernel.setArg(7, inv_scaling));
            CL_CHECK(this->clcontext.queue.enqueueNDRangeKernel(kdotq_kernel, cl::NullRange, global_work_size, local_work_size));

            // --- Enqueue Read Back (Asynchronous) ---
            CL_CHECK(this->clcontext.queue.enqueueReadBuffer(d_kdotq, CL_FALSE, 0, kdotq_size_bytes, head.KdotQ.mapped_data)); // Read directly to mat

            // Note: cl::Buffer d_keys, d_querys, d_kdotq will be released automatically when they go out of scope after clFinish
        }

        // --- Sync & Unflatten ---
        CL_CHECK(this->clcontext.queue.finish()); // Wait for all enqueued operations for all heads

        // Data is already read into head.KdotQ.mapped_data
        // d_kdotq_buffers[i] will be released by RAII.
    } catch (const std::exception& e) {
        std::cerr << "OpenCL Exception in clParallelKdotQ (Training): " << e.what() << std::endl;
        throw;
    }
}


/**
 * @brief OpenCL: Computes KdotQ in parallel for a column during INFERENCE for BLOCK 1.
 *        Uses global tokenEmbed and head.qkCache (M).
 *        Uses kernelKdotQ_Block1_Self_Inference or kernelKdotQ_Block1_Cross_Inference.
 *        Matches signature: void block::clParallelUseKdotQ(cl_context context, cl_command_queue queue, std::map<std::string, cl_kernel>& kernels, const std::vector<std::vector<float>>& tokenEmbed, int& columnNumber, int& tokenCount, int& promptCount, bool isSelfAttention);
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

    const int embedding_dim = EMBEDDING;
    const float inv_scaling = 1.0f / std::sqrt(static_cast<float>(embedding_dim)); // Scale by embedding_dim for inference
    const int num_heads_in_column = this->x;

    // Use cl::Buffer for automatic memory management
    std::vector<cl::Buffer> d_kdotq_buffers(num_heads_in_column); // Use cl::Buffer

    cl_int cl_err; // For OpenCL error codes

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
    cl::Buffer d_tokenEmbed_global;
    try {
        d_tokenEmbed_global = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, embed_size_bytes, flat_tokenEmbed.data(), &cl_err); CL_CHECK(cl_err);

        for (int i = 0; i < num_heads_in_column; ++i) {
            attention& head = b[i][columnNumber];

            // --- Context Calculation (Block 1 specific) ---
            int context_len_block = (std::min)(context_len_total, CONTEXT_WIN); // Parenthesize std::min
            context_len_block = (std::max)(0, context_len_block); // Parenthesize std::max

            int num_queries_eff = promptCount;
            int num_keys_eff = context_len_total;
            int kdotq_rows = context_len_block;
            int kdotq_cols = context_len_block;

            // Ensure KdotQ mat is correctly dimensioned
            if (head.KdotQ.row != kdotq_rows || head.KdotQ.col != kdotq_cols) {
                head.KdotQ = mat(kdotq_rows, kdotq_cols);
            }

            size_t kdotq_elements = static_cast<size_t>(kdotq_rows) * kdotq_cols;
            size_t kdotq_size_bytes = kdotq_elements * sizeof(float);


            if (kdotq_size_bytes == 0) continue;

            // --- Validate M (qkCache) ---
            if (head.qkCache.row != embedding_dim || head.qkCache.col != embedding_dim) {
                std::cerr << "Warning: Invalid qkCache dimensions for head (" << i << ", " << columnNumber << ") in Block 1. Skipping." << std::endl;
                continue;
            }
            size_t M_size_bytes = static_cast<size_t>(embedding_dim) * embedding_dim * sizeof(float);

            // --- GPU Allocation & Copy (M only) ---
            cl::Buffer d_kdotq(this->clcontext.context, CL_MEM_WRITE_ONLY, kdotq_size_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
            cl::Buffer d_M(this->clcontext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, M_size_bytes, head.qkCache.mapped_data, &cl_err); CL_CHECK(cl_err);
            d_kdotq_buffers[i] = d_kdotq;
            // No need to store d_M for cleanup, cl::Buffer handles it

            float zero_pattern = 0.0f;
            CL_CHECK(this->clcontext.queue.enqueueFillBuffer(d_kdotq, zero_pattern, 0, kdotq_size_bytes));
            // Write for d_M handled by CL_MEM_COPY_HOST_PTR

            // --- Kernel Launch ---
            cl::Kernel kdotq_kernel;
            if (isSelfAttention) {
                kdotq_kernel = this->clcontext.kernels.at("kernelKdotQ_Block1_Self_Inference");
            } else {
                kdotq_kernel = this->clcontext.kernels.at("kernelKdotQ_Block1_Cross_Inference");
            }

            cl::NDRange local_work_size(16, 16);
            cl::NDRange global_work_size(
                (static_cast<size_t>(num_keys_eff) + local_work_size[0] - 1) / local_work_size[0] * local_work_size[0],
                (static_cast<size_t>(num_queries_eff) + local_work_size[1] - 1) / local_work_size[1] * local_work_size[1]);

            cl_int cl_prompt_start_index = prompt_start_index_global;
            cl_int cl_prompt_len = promptCount;
            cl_int cl_context_len = context_len_total;
            cl_int cl_kdotq_cols = kdotq_cols; // Width of output buffer
            cl_int cl_embedding_dim_arg = embedding_dim;

            CL_CHECK(kdotq_kernel.setArg(0, d_kdotq));
            CL_CHECK(kdotq_kernel.setArg(1, d_tokenEmbed_global)); // Use global embed buffer
            CL_CHECK(kdotq_kernel.setArg(2, d_M));
            CL_CHECK(kdotq_kernel.setArg(3, cl_prompt_start_index));
            CL_CHECK(kdotq_kernel.setArg(4, cl_prompt_len));
            CL_CHECK(kdotq_kernel.setArg(5, cl_context_len));
            CL_CHECK(kdotq_kernel.setArg(6, cl_kdotq_cols));
            CL_CHECK(kdotq_kernel.setArg(7, cl_embedding_dim_arg));
            CL_CHECK(kdotq_kernel.setArg(8, inv_scaling));

            // Convert C-style arrays to cl::NDRange
            CL_CHECK(this->clcontext.queue.enqueueNDRangeKernel(kdotq_kernel, cl::NullRange, global_work_size, local_work_size));

            // --- Enqueue Read Back (Asynchronous) ---
            CL_CHECK(this->clcontext.queue.enqueueReadBuffer(d_kdotq, CL_FALSE, 0, kdotq_size_bytes, head.KdotQ.mapped_data));

            // Note: d_M and d_kdotq released automatically by cl::Buffer destructor after clFinish
        }

        // --- Sync & Unflatten ---
        CL_CHECK(this->clcontext.queue.finish()); // Wait for all heads in the column
        // Data is already read into head.KdotQ.mapped_data
        // d_kdotq_buffers[i] will be released by RAII.
    } catch (const std::exception& e) {
        std::cerr << "OpenCL Exception in clParallelUseKdotQ (Block 1): " << e.what() << std::endl;
        throw;
    }
}


/**
 * @brief OpenCL: Computes KdotQ in parallel for a column during INFERENCE for BLOCK N > 1.
 *        Uses block-local tokForBlock, previous block's EVp, and head.qkCache (M).
 *        Uses kernelKdotQ_BlockN_Self_Inference or kernelKdotQ_BlockN_Cross_Inference.
 *        Matches signature: void block::clParallelUseKdotQ(cl_context context, cl_command_queue queue, std::map<std::string, cl_kernel>& kernels, const std::vector<std::vector<std::vector<float>>>& EVp, int& columnNumber, int& blockNumber, int& tokenCount, int& promptCount, bool isSelfAttention);
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

    const int embedding_dim = EMBEDDING;
    const float inv_scaling = 1.0f / std::sqrt(static_cast<float>(embedding_dim)); // Scale by embedding_dim for inference
    const int num_heads_in_column = this->x;

    // Use cl::Buffer for automatic memory management
    std::vector<cl::Buffer> d_kdotq_buffers(num_heads_in_column); // Use cl::Buffer

    cl_int cl_err; // For OpenCL error codes

    // --- Pre-computation / Validation ---
    int context_len_total = tokenCount + promptCount;
    int block_start_token_index = (blockNumber - 1) * CONTEXT_WIN;
    int context_len_block = (std::min)(context_len_total - block_start_token_index, CONTEXT_WIN); // Parenthesize std::min
    context_len_block = (std::max)(0, context_len_block); // Parenthesize std::max
    int prompt_start_index_global = tokenCount;
    int prompt_start_index_in_block = (std::max)(0, prompt_start_index_global - block_start_token_index); // Parenthesize std::max
    prompt_start_index_in_block = (std::min)(prompt_start_index_in_block, CONTEXT_WIN); // Clamp to block window, parenthesize std::min

    if (this->tokForBlock.mapped_data == nullptr || this->tokForBlock.row < context_len_block ||
        (context_len_block > 0 && this->tokForBlock.col != embedding_dim)) {
        throw std::runtime_error("Invalid tokForBlock for Block N inference.");
    }
    size_t tok_size_bytes = static_cast<size_t>(context_len_block) * embedding_dim * sizeof(float);
    if (EVp.size() < static_cast<size_t>(num_heads_in_column)) {
        throw std::runtime_error("Invalid EVp structure for Block N inference.");
    }

    // Allocate and copy tokForBlock once
    cl::Buffer d_tokForBlock_global;
    try {
        d_tokForBlock_global = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, tok_size_bytes, this->tokForBlock.mapped_data, &cl_err); CL_CHECK(cl_err);

        for (int i = 0; i < num_heads_in_column; ++i) {
            attention& head = b[i][columnNumber];

            // --- Context Calculation (Block N specific) ---
            int num_queries_eff = promptCount;
            int num_keys_eff = context_len_block;
            int kdotq_rows = context_len_block;
            int kdotq_cols = context_len_block;

            // Ensure KdotQ mat is correctly dimensioned
            if (head.KdotQ.row != kdotq_rows || head.KdotQ.col != kdotq_cols) {
                 head.KdotQ = mat(kdotq_rows, kdotq_cols);
            }

            size_t kdotq_elements = static_cast<size_t>(kdotq_rows) * kdotq_cols;
            size_t kdotq_size_bytes = kdotq_elements * sizeof(float);

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

            // Flatten only the relevant part of EVp[i]
            std::vector<float> flat_EVp;
            flat_EVp.reserve(context_len_block * embedding_dim);
            for(int r=0; r<context_len_block; ++r) flat_EVp.insert(flat_EVp.end(), EVp[i][r].begin(), EVp[i][r].end());

            size_t M_size_bytes = static_cast<size_t>(embedding_dim) * embedding_dim * sizeof(float);
            size_t evp_size_bytes = static_cast<size_t>(context_len_block) * embedding_dim * sizeof(float);

            // --- GPU Allocation & Copy (EVp, M) ---
            cl::Buffer d_kdotq, d_EVp, d_M;
            d_kdotq = cl::Buffer(this->clcontext.context, CL_MEM_WRITE_ONLY, kdotq_size_bytes, nullptr, &cl_err); CL_CHECK(cl_err);
            d_EVp = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, evp_size_bytes, flat_EVp.data(), &cl_err); CL_CHECK(cl_err);
            d_M = cl::Buffer(this->clcontext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, M_size_bytes, head.qkCache.mapped_data, &cl_err); CL_CHECK(cl_err);
            d_kdotq_buffers[i] = d_kdotq;
            // No need to store d_EVp/d_M for cleanup

            float zero_pattern = 0.0f;
            CL_CHECK(this->clcontext.queue.enqueueFillBuffer(d_kdotq, zero_pattern, 0, kdotq_size_bytes));
            // Writes for d_EVp, d_M handled by CL_MEM_COPY_HOST_PTR

            // --- Kernel Launch ---
            cl::Kernel kdotq_kernel;
            if (isSelfAttention) {
                kdotq_kernel = this->clcontext.kernels.at("kernelKdotQ_BlockN_Self_Inference");
            } else {
                kdotq_kernel = this->clcontext.kernels.at("kernelKdotQ_BlockN_Cross_Inference");
            }

            cl::NDRange local_work_size(16, 16);
            cl::NDRange global_work_size(
                (static_cast<size_t>(num_keys_eff) + local_work_size[0] - 1) / local_work_size[0] * local_work_size[0],
                (static_cast<size_t>(num_queries_eff) + local_work_size[1] - 1) / local_work_size[1] * local_work_size[1]);

            cl_int cl_prompt_start_index_in_block = prompt_start_index_in_block;
            cl_int cl_prompt_len = promptCount;
            cl_int cl_context_len_block = context_len_block;
            cl_int cl_kdotq_cols = kdotq_cols; // Width of output buffer
            cl_int cl_embedding_dim_arg = embedding_dim;

            CL_CHECK(kdotq_kernel.setArg(0, d_kdotq));
            CL_CHECK(kdotq_kernel.setArg(1, d_tokForBlock_global)); // Use global block token buffer
            CL_CHECK(kdotq_kernel.setArg(2, d_EVp));
            CL_CHECK(kdotq_kernel.setArg(3, d_M));
            CL_CHECK(kdotq_kernel.setArg(4, cl_prompt_start_index_in_block));
            CL_CHECK(kdotq_kernel.setArg(5, cl_prompt_len));
            CL_CHECK(kdotq_kernel.setArg(6, cl_context_len_block));
            CL_CHECK(kdotq_kernel.setArg(7, cl_kdotq_cols));
            CL_CHECK(kdotq_kernel.setArg(8, cl_embedding_dim_arg));
            CL_CHECK(kdotq_kernel.setArg(9, inv_scaling));

            CL_CHECK(this->clcontext.queue.enqueueNDRangeKernel(kdotq_kernel, cl::NullRange, global_work_size, local_work_size));

            // --- Enqueue Read Back (Asynchronous) ---
            CL_CHECK(this->clcontext.queue.enqueueReadBuffer(d_kdotq, CL_FALSE, 0, kdotq_size_bytes, head.KdotQ.mapped_data));

            // Note: d_EVp, d_M, d_kdotq released automatically by cl::Buffer destructor after clFinish
        }

        // --- Sync & Unflatten ---
        CL_CHECK(this->clcontext.queue.finish()); // Wait for all heads in the column
        // Data is already read into head.KdotQ.mapped_data
        // d_kdotq_buffers[i] will be released by RAII.
    } catch (const std::exception& e) {
        std::cerr << "OpenCL Exception in clParallelUseKdotQ (Block N): " << e.what() << std::endl;
        throw;
    }
}

#endif // USE_OPENCL
