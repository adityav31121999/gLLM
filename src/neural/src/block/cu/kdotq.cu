
#include "include/attention.hpp"
#include "include/block.hpp"
#include <cuda_runtime.h>

// check for error caused in CUDA kernels and operations
#define CUDA_CHECK(condition)                                                          \
do {                                                                                   \
    cudaError_t error = condition;                                                     \
    if (error != cudaSuccess) {                                                        \
        fprintf(stderr, "CUDA error: %s:%d '%s' failed '%s'\n", __FILE__, __LINE__,    \
                #condition, cudaGetErrorString(error));                                \
        exit(EXIT_FAILURE);                                                            \
    }                                                                                  \
} while (0)


/**
 * @brief Computes KdotQ in parallel for a column during TRAINING using K and Q matrices.
 *        Uses kernelKdotQforSelf_train or kernelKdotQforCross_train.
 *
 * @param columnNumber Index of the column of attention heads.
 * @param blockNumber 1-based index of the current block.
 * @param tokenCount Global token count *before* adding the prompt.
 * @param promptCount Number of new tokens in the current step.
 * @param isSelfAttention True for self-attention, false for cross-attention.
 */
void block::cuParallelKdotQ(int& columnNumber, int& blockNumber, int& tokenCount, int& promptCount, 
    bool isSelfAttention)
{
    if (promptCount <= 0) return;
    if (columnNumber < 0 || columnNumber >= this->y) {
        throw std::out_of_range("cuParallelKdotQ: columnNumber out of range.");
    }

    const int embedding_dim = EMBEDDING;
    const float inv_scaling = 1.0f / std::sqrt(static_cast<float>(embedding_dim));
    const int num_heads_in_column = this->x;

    std::vector<cudaStream_t> streams(num_heads_in_column);

    for (int i = 0; i < num_heads_in_column; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    for (int i = 0; i < num_heads_in_column; ++i) {
        attention& head = b[i][columnNumber];
        cudaStream_t current_stream = streams[i];

        // --- Context Calculation ---
        int context_len_total = tokenCount + promptCount;
        int block_start_token_index = (blockNumber - 1) * CONTEXT_WIN;
        int context_len_block = std::min(context_len_total - block_start_token_index, CONTEXT_WIN);
        context_len_block = std::max(0, context_len_block);

        // For training, K and Q should hold the history for the block's window
        int num_keys_eff = context_len_block;
        int num_queries_eff = context_len_block; // Training computes the full matrix (or triangle)

        int kdotq_rows = context_len_block;
        int kdotq_cols = context_len_block;

        // head.KdotQ is a mat, assumed to be CONTEXT_WIN x CONTEXT_WIN.
        // We operate on the active kdotq_rows x kdotq_cols sub-matrix.
        // Its mapped_data should be zeroed out for the active region if necessary,
        // but cudaMemsetAsync on d_kdotq handles zeroing the device buffer.
        // The host KdotQ mat will be overwritten by cudaMemcpyAsync.
        size_t kdotq_active_elements = static_cast<size_t>(kdotq_rows) * kdotq_cols;
        size_t kdotq_size_bytes = kdotq_active_elements * sizeof(float);

        if (kdotq_size_bytes == 0) continue; // Nothing to compute

        // --- Validate K/Q Data ---
        if (!head.K.mapped_data || !head.Q.mapped_data ||
            head.K.row < num_keys_eff || head.Q.row < num_queries_eff ||
            head.K.col != embedding_dim || head.Q.col != embedding_dim)
        {
            fprintf(stderr, "Error: Training K/Q mat properties invalid for head (%d, %d). K_rows=%d Q_rows=%d Expected_rows=%d Emb=%d K_cols=%d Q_cols=%d\n",
                    i, columnNumber, head.K.row, head.Q.row, num_keys_eff, embedding_dim, head.K.col, head.Q.col);
            continue; // Skip this head
        }

        // --- GPU Allocation & Copy ---
        float *d_kdotq = nullptr, *d_keys = nullptr, *d_querys = nullptr;
        // We only need to copy the effective number of keys/queries for this block's context
        size_t keys_active_bytes = static_cast<size_t>(num_keys_eff) * embedding_dim * sizeof(float);
        size_t querys_active_bytes = static_cast<size_t>(num_queries_eff) * embedding_dim * sizeof(float);

        CUDA_CHECK(cudaMallocAsync(&d_kdotq, kdotq_size_bytes, current_stream));
        CUDA_CHECK(cudaMemsetAsync(d_kdotq, 0, kdotq_size_bytes, current_stream)); // Zero init
        CUDA_CHECK(cudaMallocAsync(&d_keys, keys_active_bytes, current_stream));
        CUDA_CHECK(cudaMallocAsync(&d_querys, querys_active_bytes, current_stream));

        CUDA_CHECK(cudaMemcpyAsync(d_keys, head.K.mapped_data, keys_active_bytes, cudaMemcpyHostToDevice, current_stream));
        CUDA_CHECK(cudaMemcpyAsync(d_querys, head.Q.mapped_data, querys_active_bytes, cudaMemcpyHostToDevice, current_stream));

        // --- Kernel Launch ---
        const dim3 blockDim(16, 16);
        const dim3 gridDim((num_keys_eff + blockDim.x - 1) / blockDim.x,
                           (num_queries_eff + blockDim.y - 1) / blockDim.y);

        if (isSelfAttention) {
            kernelKdotQforSelf_train<<<gridDim, blockDim, 0, current_stream>>>(
                d_kdotq, d_keys, d_querys, num_queries_eff, num_keys_eff,
                kdotq_cols, embedding_dim, inv_scaling);
        }
        else {
            kernelKdotQforCross_train<<<gridDim, blockDim, 0, current_stream>>>(
                d_kdotq, d_keys, d_querys, num_queries_eff, num_keys_eff,
                kdotq_cols, embedding_dim, inv_scaling);
        }
        CUDA_CHECK(cudaGetLastError());

        // --- Copy Back & Free ---
        // Copy to the beginning of head.KdotQ.mapped_data, covering the active region
        CUDA_CHECK(cudaMemcpyAsync(head.KdotQ.mapped_data, d_kdotq, kdotq_size_bytes, cudaMemcpyDeviceToHost, current_stream));

        CUDA_CHECK(cudaFreeAsync(d_keys, current_stream));
        CUDA_CHECK(cudaFreeAsync(d_querys, current_stream));
        CUDA_CHECK(cudaFreeAsync(d_kdotq, current_stream));
    }

    // --- Sync & Unflatten ---
    for (int i = 0; i < num_heads_in_column; ++i) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
}


/**
 * @brief Computes KdotQ in parallel for a column during INFERENCE for BLOCK 1.
 *        Uses global tokenEmbed and head.qkCache (M).
 *        Uses kernelKdotQ_Block1_Self_Inference or kernelKdotQ_Block1_Cross_Inference.
 *
 * @param tokenEmbed Global token embeddings (Host). Should contain full context.
 * @param columnNumber Index of the column of attention heads.
 * @param tokenCount Global token count *before* adding the prompt.
 * @param promptCount Number of new tokens in the current step.
 * @param isSelfAttention True for self-attention, false for cross-attention.
 */
void block::cuParallelUseKdotQ(const std::vector<std::vector<float>>& tokenEmbed,
                              int& columnNumber, int& tokenCount, int& promptCount, bool isSelfAttention)
{
    // This overload is specifically for Block 1 Inference
    // const int blockNumber = 1; // Implicitly Block 1

    if (promptCount <= 0) 
        return;
    if (columnNumber < 0 || columnNumber >= this->y) {
        throw std::out_of_range("cuParallelUseKdotQ (Block 1): columnNumber out of range.");
    }

    const int embedding_dim = EMBEDDING;
    const float inv_scaling = 1.0f / std::sqrt(static_cast<float>(embedding_dim));
    const int num_heads_in_column = this->x;

    std::vector<cudaStream_t> streams(num_heads_in_column);

    for (int i = 0; i < num_heads_in_column; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    // --- Pre-computation / Validation ---
    int context_len_total = tokenCount + promptCount; // Full context length needed for Block 1
    int prompt_start_index_global = tokenCount;

    // Validate tokenEmbed size once before the loop
    if (tokenEmbed.empty() || tokenEmbed.size() < context_len_total || (context_len_total > 0 && tokenEmbed[0].size() != embedding_dim)) {
        fprintf(stderr, "Error: Invalid tokenEmbed provided to cuParallelUseKdotQ (Block 1). Need %d rows, %d emb; Have %zu rows\n",
                context_len_total, embedding_dim, tokenEmbed.size());
        // Cleanup streams if created
        for (int i = 0; i < num_heads_in_column; ++i) cudaStreamDestroy(streams[i]);
        throw std::runtime_error("Invalid tokenEmbed for Block 1 inference.");
    }
    std::vector<float> flat_tokenEmbed = flatten(tokenEmbed);
    size_t embed_size_bytes = (size_t)context_len_total * embedding_dim * sizeof(float);
    if (flat_tokenEmbed.size() * sizeof(float) < embed_size_bytes) {
        fprintf(stderr, "Error: Flattened tokenEmbed size insufficient for Block 1.\n");
        for (int i = 0; i < num_heads_in_column; ++i) cudaStreamDestroy(streams[i]);
        throw std::runtime_error("Flattened tokenEmbed size error.");
    }

    for (int i = 0; i < num_heads_in_column; ++i) {
        attention& head = b[i][columnNumber];
        cudaStream_t current_stream = streams[i];

        // --- Context Calculation (Block 1 specific) ---
        // KdotQ buffer size depends on the block window, even if calculation uses global context
        int context_len_block = std::min(context_len_total, CONTEXT_WIN); // Block 1 window size
        context_len_block = std::max(0, context_len_block);

        int num_queries_eff = promptCount; // Rows to compute correspond to the prompt
        int num_keys_eff = context_len_total; // Keys span the full global context for Block 1

        int kdotq_rows = context_len_block; // Buffer sized for the window
        int kdotq_cols = context_len_block; // Buffer width for the window

        // head.KdotQ is a mat, assumed to be CONTEXT_WIN x CONTEXT_WIN.
        size_t kdotq_active_elements = static_cast<size_t>(kdotq_rows) * kdotq_cols;
        size_t kdotq_size_bytes = kdotq_active_elements * sizeof(float);

        if (kdotq_size_bytes == 0) continue;

        // --- Validate M (qkCache) ---
        if (!head.qkCache.mapped_data || head.qkCache.row != embedding_dim || head.qkCache.col != embedding_dim) {
            fprintf(stderr, "Error: Invalid qkCache dimensions for head (%d, %d) in Block 1.\n", i, columnNumber);
            continue; // Skip this head
        }
        size_t M_size_bytes = (size_t)embedding_dim * embedding_dim * sizeof(float);

        // --- GPU Allocation & Copy ---
        float *d_kdotq = nullptr, *d_tokenEmbed = nullptr, *d_M = nullptr;
        CUDA_CHECK(cudaMallocAsync(&d_kdotq, kdotq_size_bytes, current_stream));
        CUDA_CHECK(cudaMemsetAsync(d_kdotq, 0, kdotq_size_bytes, current_stream));
        // Note: flat_tokenEmbed is created outside the loop from the input std::vector<std::vector<float>>
        // embed_size_bytes is also calculated outside the loop
        // This d_tokenEmbed is for the *full* tokenEmbed, not just a part of it.
        CUDA_CHECK(cudaMallocAsync(&d_tokenEmbed, embed_size_bytes, current_stream));
        CUDA_CHECK(cudaMallocAsync(&d_M, M_size_bytes, current_stream));

        CUDA_CHECK(cudaMemcpyAsync(d_tokenEmbed, flat_tokenEmbed.data(), embed_size_bytes, cudaMemcpyHostToDevice, current_stream)); // flat_tokenEmbed is from input
        CUDA_CHECK(cudaMemcpyAsync(d_M, head.qkCache.mapped_data, M_size_bytes, cudaMemcpyHostToDevice, current_stream));

        // --- Kernel Launch ---
        const dim3 blockDim(16, 16);
        // Grid covers prompt rows and full context columns
        const dim3 gridDim((num_keys_eff + blockDim.x - 1) / blockDim.x,
                           (num_queries_eff + blockDim.y - 1) / blockDim.y);

        if (isSelfAttention) {
            kernelKdotQ_Block1_Self_Inference<<<gridDim, blockDim, 0, current_stream>>>(
                d_kdotq, d_tokenEmbed, d_M, // Pass device pointer
                prompt_start_index_global, promptCount, context_len_total,
                kdotq_cols, // Width of the output buffer (window size)
                embedding_dim, inv_scaling);
        }
        else {
            kernelKdotQ_Block1_Cross_Inference<<<gridDim, blockDim, 0, current_stream>>>(
                d_kdotq, d_tokenEmbed, d_M, // Pass device pointer
                prompt_start_index_global, promptCount, context_len_total,
                kdotq_cols, // Width of the output buffer (window size)
                embedding_dim, inv_scaling);
        }
        CUDA_CHECK(cudaGetLastError());

        // --- Copy Back & Free ---        
        CUDA_CHECK(cudaMemcpyAsync(head.KdotQ.mapped_data, d_kdotq, kdotq_size_bytes, cudaMemcpyDeviceToHost, current_stream));

        CUDA_CHECK(cudaFreeAsync(d_tokenEmbed, current_stream));
        CUDA_CHECK(cudaFreeAsync(d_M, current_stream));
        CUDA_CHECK(cudaFreeAsync(d_kdotq, current_stream));
    }

    // --- Sync & Unflatten ---
    for (int i = 0; i < num_heads_in_column; ++i) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
}


/**
 * @brief Computes KdotQ in parallel for a column during INFERENCE for BLOCK N > 1.
 *        Uses block-local tokForBlock, previous block's EVp, and head.qkCache (M).
 *        Uses kernelKdotQ_BlockN_Self_Inference or kernelKdotQ_BlockN_Cross_Inference.
 *
 * @param EVp Vertical retention vectors from the previous block (Host). Structure: EVp[head_idx][token_idx][embedding_dim].
 * @param columnNumber Index of the column of attention heads.
 * @param blockNumber 1-based index of the current block (must be > 1).
 * @param tokenCount Global token count *before* adding the prompt.
 * @param promptCount Number of new tokens in the current step.
 * @param isSelfAttention True for self-attention, false for cross-attention.
 */
void block::cuParallelUseKdotQ(const std::vector<std::vector<std::vector<float>>>& EVp,
                              int& columnNumber, int& blockNumber, int& tokenCount, int& promptCount, bool isSelfAttention)
{
    // This overload is specifically for Block N > 1 Inference
    if (blockNumber <= 1) {
         throw std::invalid_argument("cuParallelUseKdotQ (Block N): blockNumber must be > 1.");
    }
    if (promptCount <= 0) 
        return;
    if (columnNumber < 0 || columnNumber >= this->y) {
        throw std::out_of_range("cuParallelUseKdotQ (Block N): columnNumber out of range.");
    }

    const int embedding_dim = EMBEDDING;
    const float inv_scaling = 1.0f / std::sqrt(static_cast<float>(embedding_dim));
    const int num_heads_in_column = this->x;

    std::vector<cudaStream_t> streams(num_heads_in_column);

    for (int i = 0; i < num_heads_in_column; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    // --- Pre-computation / Validation ---
    int context_len_total = tokenCount + promptCount;
    int block_start_token_index = (blockNumber - 1) * CONTEXT_WIN;
    int context_len_block = std::min(context_len_total - block_start_token_index, CONTEXT_WIN);
    context_len_block = std::max(0, context_len_block);
    int prompt_start_index_global = tokenCount;
    int prompt_start_index_in_block = std::max(0, prompt_start_index_global - block_start_token_index);
    prompt_start_index_in_block = std::min(prompt_start_index_in_block, CONTEXT_WIN);

    // Validate tokForBlock size once
    if (!this->tokForBlock.mapped_data || this->tokForBlock.row < context_len_block || this->tokForBlock.col != embedding_dim)
    {
        fprintf(stderr, "Error: Invalid tokForBlock for Block %d inference. Need %d rows, %d emb; Have %d rows, %d cols\n",
                blockNumber, context_len_block, embedding_dim, this->tokForBlock.row, this->tokForBlock.col);
        for (int i = 0; i < num_heads_in_column; ++i) cudaStreamDestroy(streams[i]);
        throw std::runtime_error("Invalid tokForBlock for Block N inference.");
    }
    // tok_active_bytes is the amount of data from tokForBlock to be used by the kernel
    // It corresponds to the first context_len_block rows.
    size_t tok_active_bytes = static_cast<size_t>(context_len_block) * embedding_dim * sizeof(float);

    // Validate EVp structure minimally (more checks inside loop)
    if (EVp.size() < num_heads_in_column) {
        fprintf(stderr, "Error: EVp has insufficient heads for Block %d inference. Need %d, Have %zu\n",
                blockNumber, num_heads_in_column, EVp.size());
        for (int i = 0; i < num_heads_in_column; ++i) cudaStreamDestroy(streams[i]);
        throw std::runtime_error("Invalid EVp structure for Block N inference.");
    }


    for (int i = 0; i < num_heads_in_column; ++i) {
        attention& head = b[i][columnNumber];
        cudaStream_t current_stream = streams[i];

        // --- Context Calculation (Block N specific) ---
        int num_queries_eff = promptCount; // Rows to compute correspond to the prompt
        int num_keys_eff = context_len_block; // Keys span the block's window

        int kdotq_rows = context_len_block; // Buffer sized for the window
        int kdotq_cols = context_len_block; // Buffer width for the window

        // head.KdotQ is a mat, assumed to be CONTEXT_WIN x CONTEXT_WIN.
        size_t kdotq_active_elements = static_cast<size_t>(kdotq_rows) * kdotq_cols;
        size_t kdotq_size_bytes = kdotq_active_elements * sizeof(float);

        if (kdotq_size_bytes == 0) continue;

        // --- Validate M (qkCache) and EVp[i] ---
        if (!head.qkCache.mapped_data || head.qkCache.row != embedding_dim || head.qkCache.col != embedding_dim) {
            fprintf(stderr, "Error: Invalid qkCache dimensions for head (%d, %d) in Block %d.\n", i, columnNumber, blockNumber);
            continue; // Skip this head
        }
        // EVp[i] is std::vector<std::vector<float>>
        if (EVp[i].empty() || EVp[i].size() < context_len_block || (context_len_block > 0 && EVp[i][0].size() != embedding_dim)) {
            fprintf(stderr, "Error: Invalid EVp data for head (%d, %d) in Block %d. Need %d rows, %d emb; Have %zu rows\n",
                    i, columnNumber, blockNumber, context_len_block, embedding_dim, EVp[i].size());
            continue; // Skip this head
        }

        std::vector<float> flat_EVp = flatten(EVp[i]); // Flatten EV for this specific head
        size_t M_size_bytes = (size_t)embedding_dim * embedding_dim * sizeof(float);
        size_t evp_active_bytes = (size_t)context_len_block * embedding_dim * sizeof(float);

        if (flat_EVp.size() * sizeof(float) < evp_active_bytes) { // Check only flat_EVp
            fprintf(stderr, "Error: Flattened EVp size mismatch head (%d, %d) Block %d.\n", i, columnNumber, blockNumber);
            continue;
        }

        // --- GPU Allocation & Copy ---
        float *d_kdotq = nullptr, *d_tokForBlock = nullptr, *d_EVp = nullptr, *d_M = nullptr;
        CUDA_CHECK(cudaMallocAsync(&d_kdotq, kdotq_size_bytes, current_stream));
        CUDA_CHECK(cudaMemsetAsync(d_kdotq, 0, kdotq_size_bytes, current_stream));
        CUDA_CHECK(cudaMallocAsync(&d_tokForBlock, tok_active_bytes, current_stream));
        CUDA_CHECK(cudaMallocAsync(&d_EVp, evp_active_bytes, current_stream));
        CUDA_CHECK(cudaMallocAsync(&d_M, M_size_bytes, current_stream));

        CUDA_CHECK(cudaMemcpyAsync(d_tokForBlock, this->tokForBlock.mapped_data, tok_active_bytes, cudaMemcpyHostToDevice, current_stream));
        CUDA_CHECK(cudaMemcpyAsync(d_EVp, flat_EVp.data(), evp_active_bytes, cudaMemcpyHostToDevice, current_stream));
        CUDA_CHECK(cudaMemcpyAsync(d_M, head.qkCache.mapped_data, M_size_bytes, cudaMemcpyHostToDevice, current_stream));

        // --- Kernel Launch ---
        const dim3 blockDim(16, 16);
        // Grid covers prompt rows and block context columns
        const dim3 gridDim((num_keys_eff + blockDim.x - 1) / blockDim.x,
                           (num_queries_eff + blockDim.y - 1) / blockDim.y);

        if (isSelfAttention) {
            kernelKdotQ_BlockN_Self_Inference<<<gridDim, blockDim, 0, current_stream>>>(
                d_kdotq, d_tokForBlock, d_EVp, d_M,
                prompt_start_index_in_block, promptCount, context_len_block,
                kdotq_cols, // Width of the output buffer
                embedding_dim, inv_scaling);
        } 
        else {
            kernelKdotQ_BlockN_Cross_Inference<<<gridDim, blockDim, 0, current_stream>>>(
                d_kdotq, d_tokForBlock, d_EVp, d_M,
                prompt_start_index_in_block, promptCount, context_len_block,
                kdotq_cols, // Width of the output buffer
                embedding_dim, inv_scaling);
        }
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpyAsync(head.KdotQ.mapped_data, d_kdotq, kdotq_size_bytes, cudaMemcpyDeviceToHost, current_stream));

        CUDA_CHECK(cudaFreeAsync(d_tokForBlock, current_stream));
        CUDA_CHECK(cudaFreeAsync(d_EVp, current_stream));
        CUDA_CHECK(cudaFreeAsync(d_M, current_stream));
        CUDA_CHECK(cudaFreeAsync(d_kdotq, current_stream));
    }

    // --- Sync & Unflatten ---
    for (int i = 0; i < num_heads_in_column; ++i) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
}
