#include "include/attention.hpp"
#include "include/block.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
#include <string>
#include <cmath>     // For std::sqrt, std::max, std::min
#include <numeric>   // For std::iota (if needed elsewhere)
#include <algorithm> // For std::min, std::max

// Helper macro for CUDA error checking
#define CUDA_CHECK(call) do {       \
    cudaError_t err = call;         \
    if (err != cudaSuccess) {       \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        /* Consider more robust cleanup here, maybe RAII */         \
        throw std::runtime_error(cudaGetErrorString(err));          \
    } \
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
    if (columnNumber < 0 || columnNumber >= this->y) {
        throw std::out_of_range("cuParallelKdotQ: columnNumber " + std::to_string(columnNumber) + " is out of range.");
    }

    const int num_heads_in_column = this->x;
    if (num_heads_in_column == 0) return;

    const int d_embedding = EMBEDDING;
    const int kdotq_dim = CONTEXT_WIN;  // KdotQ is CONTEXT_WIN x CONTEXT_WIN
    const float inv_scaling = 1.0f / SCALING; // SCALING is std::sqrt(EMBEDDING)

    // --- Effective token counts for K, Q, KdotQ computation ---
    int context_len_total = tokenCount + promptCount;
    int block_start_token_index = (blockNumber - 1) * CONTEXT_WIN;
    int context_len_block = std::min(context_len_total - block_start_token_index, CONTEXT_WIN);
    context_len_block = std::max(0, context_len_block);
    int num_keys_eff = context_len_block;
    int num_queries_eff = context_len_block; // Training computes the full matrix (or triangle)

    // int kdotq_rows = context_len_block;
    // int kdotq_cols = context_len_block;

    // head.KdotQ is a mat, assumed to be CONTEXT_WIN x CONTEXT_WIN.
    if (promptCount <= 0 || context_len_block <= 0) { // If no new tokens or no context in block, KdotQ is effectively zero.
        for (int layer_idx = 0; layer_idx < num_heads_in_column; ++layer_idx) {
            attention &head_cpu = this->b[layer_idx][columnNumber];
            if (head_cpu.KdotQ.mapped_data) { // Ensure KdotQ is allocated
                // Zero out the relevant part or the whole matrix.
                // For simplicity, zeroing the whole CONTEXT_WIN x CONTEXT_WIN buffer.
                std::fill_n(head_cpu.KdotQ.mapped_data, static_cast<size_t>(head_cpu.KdotQ.row) * head_cpu.KdotQ.col, 0.0f);
            }
        }
        return;
    }

    size_t k_bytes_ph = static_cast<size_t>(kdotq_dim) * d_embedding * sizeof(float); // K is kdotq_dim x d_embedding
    size_t q_bytes_ph = static_cast<size_t>(kdotq_dim) * d_embedding * sizeof(float); // Q is kdotq_dim x d_embedding
    size_t kdotq_bytes_ph = static_cast<size_t>(kdotq_dim) * kdotq_dim * sizeof(float);

    float *agg_d_K = nullptr, *agg_d_Q = nullptr, *agg_d_KdotQ = nullptr;
    std::vector<cudaStream_t> streams(num_heads_in_column);
    std::vector<HeadDevicePointers> head_gpu_data(num_heads_in_column);

    try
    {
        CUDA_CHECK(cudaMalloc(&agg_d_K, num_heads_in_column * k_bytes_ph));
        CUDA_CHECK(cudaMalloc(&agg_d_Q, num_heads_in_column * q_bytes_ph));
        CUDA_CHECK(cudaMalloc(&agg_d_KdotQ, num_heads_in_column * kdotq_bytes_ph));

        for (int layer_idx = 0; layer_idx < num_heads_in_column; ++layer_idx)
        {
            CUDA_CHECK(cudaStreamCreateWithFlags(&streams[layer_idx], cudaStreamNonBlocking));
            cudaStream_t current_stream = streams[layer_idx];
            attention &head_cpu = this->b[layer_idx][columnNumber];
            HeadDevicePointers &current_head_pointers = head_gpu_data[layer_idx];

            current_head_pointers.d_K = agg_d_K + layer_idx * (k_bytes_ph / sizeof(float));
            current_head_pointers.d_Q = agg_d_Q + layer_idx * (q_bytes_ph / sizeof(float));
            current_head_pointers.d_KdotQ = agg_d_KdotQ + layer_idx * (kdotq_bytes_ph / sizeof(float));

            if (!head_cpu.K.mapped_data || !head_cpu.Q.mapped_data || !head_cpu.KdotQ.mapped_data) {
                 throw std::runtime_error("K, Q, or KdotQ mapped_data is null for head [" + std::to_string(layer_idx) + "][" + std::to_string(columnNumber) + "]");
            }
            if (head_cpu.K.row != kdotq_dim || head_cpu.K.col != d_embedding ||
                head_cpu.Q.row != kdotq_dim || head_cpu.Q.col != d_embedding ||
                head_cpu.KdotQ.row != kdotq_dim || head_cpu.KdotQ.col != kdotq_dim) {
                throw std::runtime_error("Dimension mismatch for K, Q, or KdotQ for head [" + std::to_string(layer_idx) + "][" + std::to_string(columnNumber) + "]. "
                                         "Expected K/Q: " + std::to_string(kdotq_dim) + "x" + std::to_string(d_embedding) +
                                         ", KdotQ: " + std::to_string(kdotq_dim) + "x" + std::to_string(kdotq_dim));
            }

            CUDA_CHECK(cudaMemcpyAsync(current_head_pointers.d_K, head_cpu.K.mapped_data, k_bytes_ph, cudaMemcpyHostToDevice, current_stream));
            CUDA_CHECK(cudaMemcpyAsync(current_head_pointers.d_Q, head_cpu.Q.mapped_data, q_bytes_ph, cudaMemcpyHostToDevice, current_stream));
            CUDA_CHECK(cudaMemsetAsync(current_head_pointers.d_KdotQ, 0, kdotq_bytes_ph, current_stream));

            dim3 threadsPerBlock(16, 16); 
            dim3 numBlocks((num_keys_eff + threadsPerBlock.x - 1) / threadsPerBlock.x,
                           (num_queries_eff + threadsPerBlock.y - 1) / threadsPerBlock.y);
            
            if (isSelfAttention) {
                kernelKdotQforSelf_train<<<numBlocks, threadsPerBlock, 0, current_stream>>>(
                    current_head_pointers.d_KdotQ, current_head_pointers.d_K, current_head_pointers.d_Q,
                    num_queries_eff, num_keys_eff, kdotq_dim, d_embedding, inv_scaling);
            } else {
                kernelKdotQforCross_train<<<numBlocks, threadsPerBlock, 0, current_stream>>>(
                    current_head_pointers.d_KdotQ, current_head_pointers.d_K, current_head_pointers.d_Q,
                    num_queries_eff, num_keys_eff, kdotq_dim, d_embedding, inv_scaling);
            }
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(head_cpu.KdotQ.mapped_data, current_head_pointers.d_KdotQ, kdotq_bytes_ph, cudaMemcpyDeviceToHost, current_stream));
        }

        for (int layer_idx = 0; layer_idx < num_heads_in_column; ++layer_idx)
        {
            CUDA_CHECK(cudaStreamSynchronize(streams[layer_idx]));
            CUDA_CHECK(cudaStreamDestroy(streams[layer_idx]));
        }
    }
    catch (const std::exception &e)
    {
        for (int k = 0; k < num_heads_in_column; ++k) { if (streams[k]) cudaStreamDestroy(streams[k]); }
        cudaFree(agg_d_K); cudaFree(agg_d_Q); cudaFree(agg_d_KdotQ);
        throw std::runtime_error(std::string("cuParallelKdotQ failed: ") + e.what());
    }
    cudaFree(agg_d_K); cudaFree(agg_d_Q); cudaFree(agg_d_KdotQ);
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
    if (columnNumber < 0 || columnNumber >= this->y) {
        throw std::out_of_range("cuParallelUseKdotQ (Block 1): columnNumber " + std::to_string(columnNumber) + " is out of range.");
    }

    const int num_heads_in_column = this->x;
    if (num_heads_in_column == 0) return;

    const int d_embedding = EMBEDDING;
    const int kdotq_dim = CONTEXT_WIN;  // KdotQ output dimensions
    const float inv_scaling = 1.0f / SCALING;

    int context_len_total = tokenCount + promptCount; // Full context length needed for Block 1
    int prompt_start_index_global = tokenCount;

    if (promptCount <= 0 || context_len_total <= 0) {
         for (int layer_idx = 0; layer_idx < num_heads_in_column; ++layer_idx) {
            attention &head_cpu = this->b[layer_idx][columnNumber];
            if (head_cpu.KdotQ.mapped_data) {
                std::fill_n(head_cpu.KdotQ.mapped_data, static_cast<size_t>(head_cpu.KdotQ.row) * head_cpu.KdotQ.col, 0.0f);
            }
        }
        return;
    }
    
    std::vector<float> flat_tokenEmbed_host;
    flatten2DVector(tokenEmbed, flat_tokenEmbed_host, context_len_total, d_embedding);
    size_t token_embed_bytes = flat_tokenEmbed_host.size() * sizeof(float);
    float *d_tokenEmbed_flat = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tokenEmbed_flat, token_embed_bytes));
    CUDA_CHECK(cudaMemcpy(d_tokenEmbed_flat, flat_tokenEmbed_host.data(), token_embed_bytes, cudaMemcpyHostToDevice));

    size_t m_qkcache_bytes_ph = static_cast<size_t>(d_embedding) * d_embedding * sizeof(float); // M is d_embedding x d_embedding
    size_t kdotq_bytes_ph = static_cast<size_t>(kdotq_dim) * kdotq_dim * sizeof(float);

    float *agg_d_M_qkCache = nullptr, *agg_d_KdotQ = nullptr;
    std::vector<cudaStream_t> streams(num_heads_in_column);

    try {
        CUDA_CHECK(cudaMalloc(&agg_d_M_qkCache, num_heads_in_column * m_qkcache_bytes_ph));
        CUDA_CHECK(cudaMalloc(&agg_d_KdotQ, num_heads_in_column * kdotq_bytes_ph));

        for (int layer_idx = 0; layer_idx < num_heads_in_column; ++layer_idx) {
            CUDA_CHECK(cudaStreamCreateWithFlags(&streams[layer_idx], cudaStreamNonBlocking));
            cudaStream_t current_stream = streams[layer_idx];
            attention &head_cpu = this->b[layer_idx][columnNumber];

            float* current_d_M_qkCache = agg_d_M_qkCache + layer_idx * (m_qkcache_bytes_ph / sizeof(float));
            float* current_d_KdotQ = agg_d_KdotQ + layer_idx * (kdotq_bytes_ph / sizeof(float));

            if (!head_cpu.qkCache.mapped_data || !head_cpu.KdotQ.mapped_data) {
                 throw std::runtime_error("qkCache or KdotQ mapped_data is null for head [" + std::to_string(layer_idx) + "][" + std::to_string(columnNumber) + "]");
            }
            if (head_cpu.qkCache.row != d_embedding || head_cpu.qkCache.col != d_embedding ||
                head_cpu.KdotQ.row != kdotq_dim || head_cpu.KdotQ.col != kdotq_dim) {
                throw std::runtime_error("Dimension mismatch for qkCache or KdotQ for head [" + std::to_string(layer_idx) + "][" + std::to_string(columnNumber) + "]");
            }

            CUDA_CHECK(cudaMemcpyAsync(current_d_M_qkCache, head_cpu.qkCache.mapped_data, m_qkcache_bytes_ph, cudaMemcpyHostToDevice, current_stream));
            CUDA_CHECK(cudaMemsetAsync(current_d_KdotQ, 0, kdotq_bytes_ph, current_stream));

            int kernel_prompt_len = promptCount; 
            int kernel_context_len = context_len_total; 

            dim3 threadsPerBlock(16, 16); 
            dim3 numBlocks((kernel_context_len + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                           (kernel_prompt_len + threadsPerBlock.y - 1) / threadsPerBlock.y);

            if (isSelfAttention) {
                kernelKdotQ_Block1_Self_Inference<<<numBlocks, threadsPerBlock, 0, current_stream>>>(
                    current_d_KdotQ, d_tokenEmbed_flat, current_d_M_qkCache,
                    prompt_start_index_global, kernel_prompt_len, kernel_context_len, 
                    kdotq_dim, d_embedding, inv_scaling);
            } else {
                kernelKdotQ_Block1_Cross_Inference<<<numBlocks, threadsPerBlock, 0, current_stream>>>(
                    current_d_KdotQ, d_tokenEmbed_flat, current_d_M_qkCache,
                    prompt_start_index_global, kernel_prompt_len, kernel_context_len,
                    kdotq_dim, d_embedding, inv_scaling);
            }
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(head_cpu.KdotQ.mapped_data, current_d_KdotQ, kdotq_bytes_ph, cudaMemcpyDeviceToHost, current_stream));
        }

        for (int layer_idx = 0; layer_idx < num_heads_in_column; ++layer_idx) {
            CUDA_CHECK(cudaStreamSynchronize(streams[layer_idx]));
            CUDA_CHECK(cudaStreamDestroy(streams[layer_idx]));
        }
    }
    catch (const std::exception &e) {
        for (int k = 0; k < num_heads_in_column; ++k) { if (streams[k]) cudaStreamDestroy(streams[k]); }
        cudaFree(d_tokenEmbed_flat); cudaFree(agg_d_M_qkCache); cudaFree(agg_d_KdotQ);
        throw std::runtime_error(std::string("cuParallelUseKdotQ (Block 1) failed: ") + e.what());
    }
    cudaFree(d_tokenEmbed_flat); cudaFree(agg_d_M_qkCache); cudaFree(agg_d_KdotQ);
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
         throw std::out_of_range("cuParallelUseKdotQ (Block N): blockNumber must be > 1.");
    }
    if (columnNumber < 0 || columnNumber >= this->y) {
        throw std::out_of_range("cuParallelUseKdotQ (Block N): columnNumber " + std::to_string(columnNumber) + " is out of range.");
    }

    const int num_heads_in_column = this->x;
    if (num_heads_in_column == 0) return;

    const int d_embedding = EMBEDDING;
    const int kdotq_dim = CONTEXT_WIN;  // KdotQ output dimensions
    const float inv_scaling = 1.0f / SCALING;

    int context_len_total = tokenCount + promptCount;
    int block_start_token_index = (blockNumber - 1) * CONTEXT_WIN;
    int context_len_block = std::min(context_len_total - block_start_token_index, CONTEXT_WIN);
    context_len_block = std::max(0, context_len_block);

    int prompt_start_index_global = tokenCount;
    int prompt_start_index_in_block = std::max(0, prompt_start_index_global - block_start_token_index);
    prompt_start_index_in_block = std::min(prompt_start_index_in_block, CONTEXT_WIN);

    // current_block_token_count is the number of tokens in this->tokForBlock
    int current_block_token_count = this->tokForBlock.row; 

    if (promptCount <= 0 || (isSelfAttention && current_block_token_count <= 0)) {
         for (int layer_idx = 0; layer_idx < num_heads_in_column; ++layer_idx) {
            attention &head_cpu = this->b[layer_idx][columnNumber];
            if (head_cpu.KdotQ.mapped_data) {
                std::fill_n(head_cpu.KdotQ.mapped_data, static_cast<size_t>(head_cpu.KdotQ.row) * head_cpu.KdotQ.col, 0.0f);
            }
        }
        return;
    }

    float *d_tokForBlock_flat = nullptr;
    size_t tokForBlock_bytes = 0;
    if (current_block_token_count > 0) {
        if (!this->tokForBlock.mapped_data || this->tokForBlock.col != d_embedding) {
            throw std::runtime_error("tokForBlock data invalid or dimension mismatch.");
        }
        tokForBlock_bytes = static_cast<size_t>(this->tokForBlock.row) * d_embedding * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_tokForBlock_flat, tokForBlock_bytes));
        CUDA_CHECK(cudaMemcpy(d_tokForBlock_flat, this->tokForBlock.mapped_data, tokForBlock_bytes, cudaMemcpyHostToDevice));
    }

    size_t m_qkcache_bytes_ph = static_cast<size_t>(d_embedding) * d_embedding * sizeof(float);
    size_t kdotq_bytes_ph = static_cast<size_t>(kdotq_dim) * kdotq_dim * sizeof(float);
    size_t max_evp_rows = CONTEXT_WIN; 
    size_t evp_flat_bytes_ph_max = max_evp_rows * d_embedding * sizeof(float);

    float *agg_d_M_qkCache = nullptr, *agg_d_KdotQ = nullptr, *agg_d_EVp_head_flat = nullptr;
    std::vector<cudaStream_t> streams(num_heads_in_column);

    try {
        CUDA_CHECK(cudaMalloc(&agg_d_M_qkCache, num_heads_in_column * m_qkcache_bytes_ph));
        CUDA_CHECK(cudaMalloc(&agg_d_KdotQ, num_heads_in_column * kdotq_bytes_ph));
        CUDA_CHECK(cudaMalloc(&agg_d_EVp_head_flat, num_heads_in_column * evp_flat_bytes_ph_max));

        for (int layer_idx = 0; layer_idx < num_heads_in_column; ++layer_idx) {
            CUDA_CHECK(cudaStreamCreateWithFlags(&streams[layer_idx], cudaStreamNonBlocking));
            cudaStream_t current_stream = streams[layer_idx];
            attention &head_cpu = this->b[layer_idx][columnNumber];

            if (static_cast<size_t>(layer_idx) >= EVp.size()) { // EVp is [head_idx_in_col][token_idx][embed_dim]
                throw std::out_of_range("EVp access out of bounds for head index " + std::to_string(layer_idx));
            }
            const auto& EVp_head_specific_host = EVp[layer_idx];
            int num_ev_rows_from_prev = EVp_head_specific_host.size();
            
            std::vector<float> flat_EVp_head_host;
            float* current_d_EVp_head = agg_d_EVp_head_flat + layer_idx * (evp_flat_bytes_ph_max / sizeof(float));
            size_t actual_evp_bytes_for_head = 0;

            if (num_ev_rows_from_prev > 0) {
                if (EVp_head_specific_host[0].size() != static_cast<size_t>(d_embedding)) {
                     throw std::runtime_error("EVp embedding dimension mismatch for head [" + std::to_string(layer_idx) + "][" + std::to_string(columnNumber) + "]");
                }
                flatten2DVector(EVp_head_specific_host, flat_EVp_head_host, num_ev_rows_from_prev, d_embedding);
                actual_evp_bytes_for_head = flat_EVp_head_host.size() * sizeof(float);
                CUDA_CHECK(cudaMemcpyAsync(current_d_EVp_head, flat_EVp_head_host.data(), actual_evp_bytes_for_head, cudaMemcpyHostToDevice, current_stream));
            } else if (!isSelfAttention && promptCount > 0) { // Cross attention with prompt needs EVp for queries
                 // If EVp is empty for cross-attention but promptCount > 0, this is an issue if queries are from EVp.
                 // The original kdotq.cu logic implies queries are promptCount from EVp.
                 // If num_ev_rows_from_prev is 0, then kernel_prompt_len for cross will be 0.
            }

            float* current_d_M_qkCache = agg_d_M_qkCache + layer_idx * (m_qkcache_bytes_ph / sizeof(float));
            float* current_d_KdotQ = agg_d_KdotQ + layer_idx * (kdotq_bytes_ph / sizeof(float));

            if (!head_cpu.qkCache.mapped_data || !head_cpu.KdotQ.mapped_data) {
                 throw std::runtime_error("qkCache or KdotQ mapped_data is null for head [" + std::to_string(layer_idx) + "][" + std::to_string(columnNumber) + "]");
            }
            if (head_cpu.qkCache.row != d_embedding || head_cpu.qkCache.col != d_embedding ||
                head_cpu.KdotQ.row != kdotq_dim || head_cpu.KdotQ.col != kdotq_dim) {
                throw std::runtime_error("Dimension mismatch for qkCache or KdotQ for head [" + std::to_string(layer_idx) + "][" + std::to_string(columnNumber) + "]");
            }

            CUDA_CHECK(cudaMemcpyAsync(current_d_M_qkCache, head_cpu.qkCache.mapped_data, m_qkcache_bytes_ph, cudaMemcpyHostToDevice, current_stream));
            CUDA_CHECK(cudaMemsetAsync(current_d_KdotQ, 0, kdotq_bytes_ph, current_stream));

            int kernel_prompt_len = promptCount; // Number of queries to compute, as per original logic
            int kernel_context_len_in_block = current_block_token_count; // Keys from tokForBlock

            // If cross-attention, EVp is the source of query embeddings, tokForBlock for key embeddings.
            // The kernel's `prompt_start_index_in_block` refers to an offset in the query source (EVp or tokForBlock).
            // The kernel's `prompt_len` is the number of queries.
            // The kernel's `context_len_in_block` is the number of keys from tokForBlock.
            // The original kdotq.cu uses `prompt_start_index_in_block` (global based) and `promptCount` for kernel's `prompt_len`.
            // This implies for cross-attention, queries are `promptCount` items starting at `prompt_start_index_in_block` from `d_EVp`.
            // And for self-attention, queries are `promptCount` items starting at `prompt_start_index_in_block` from `d_tokForBlock`.

            if (kernel_prompt_len == 0 || (!isSelfAttention && num_ev_rows_from_prev == 0) || (isSelfAttention && kernel_context_len_in_block == 0) ) {
                 // No computation needed if no queries or no keys for self-attention.
            } else {
                dim3 threadsPerBlock(16, 16);
                dim3 numBlocks((kernel_context_len_in_block + threadsPerBlock.x - 1) / threadsPerBlock.x,
                               (kernel_prompt_len + threadsPerBlock.y - 1) / threadsPerBlock.y);
                if (numBlocks.x == 0 && kernel_context_len_in_block > 0) numBlocks.x = 1; // Ensure at least one block if there are keys
                if (numBlocks.y == 0 && kernel_prompt_len > 0) numBlocks.y = 1;       // Ensure at least one block if there are queries

                if (numBlocks.x > 0 && numBlocks.y > 0) {
                    if (isSelfAttention) {
                        kernelKdotQ_BlockN_Self_Inference<<<numBlocks, threadsPerBlock, 0, current_stream>>>(
                            current_d_KdotQ, d_tokForBlock_flat, current_d_EVp_head, current_d_M_qkCache,
                            prompt_start_index_in_block, kernel_prompt_len, kernel_context_len_in_block, 
                            kdotq_dim, d_embedding, inv_scaling);
                    } else { // Cross Attention
                        kernelKdotQ_BlockN_Cross_Inference<<<numBlocks, threadsPerBlock, 0, current_stream>>>(
                            current_d_KdotQ, d_tokForBlock_flat, current_d_EVp_head, current_d_M_qkCache,
                            prompt_start_index_in_block, kernel_prompt_len, kernel_context_len_in_block,
                            kdotq_dim, d_embedding, inv_scaling);
                    }
                    CUDA_CHECK(cudaGetLastError());
                }
            }
            CUDA_CHECK(cudaMemcpyAsync(head_cpu.KdotQ.mapped_data, current_d_KdotQ, kdotq_bytes_ph, cudaMemcpyDeviceToHost, current_stream));
        }

        for (int layer_idx = 0; layer_idx < num_heads_in_column; ++layer_idx) {
            CUDA_CHECK(cudaStreamSynchronize(streams[layer_idx]));
            CUDA_CHECK(cudaStreamDestroy(streams[layer_idx]));
        }
    }
    catch (const std::exception &e) {
        for (int k = 0; k < num_heads_in_column; ++k) { if (streams[k]) cudaStreamDestroy(streams[k]); }
        if (d_tokForBlock_flat) cudaFree(d_tokForBlock_flat);
        cudaFree(agg_d_M_qkCache); cudaFree(agg_d_KdotQ); cudaFree(agg_d_EVp_head_flat);
        throw std::runtime_error(std::string("cuParallelUseKdotQ (Block N) failed: ") + e.what());
    }
    if (d_tokForBlock_flat) cudaFree(d_tokForBlock_flat);
    cudaFree(agg_d_M_qkCache); cudaFree(agg_d_KdotQ); cudaFree(agg_d_EVp_head_flat);
}
