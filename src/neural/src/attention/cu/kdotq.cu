#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "include/attention.hpp"
#include <vector>
#include <iostream>
#include <stdexcept>
#include <numeric>
#include <cmath>
#include <maths.hpp>
#include <string>

/**
 * @brief CUDA kernel to compute the product of a token matrix and a transposed weight matrix.
 *        Calculates KQoutputMatrix = tokenMatrix * KQmatrix^T.
 *        Each thread computes one element of the output matrix.
 * @param[in]  tokenMatrix    Device pointer to the token matrix (row-major, size tokenCount x dim).
 * @param[in]  KQmatrix       Device pointer to the weight Q/K matrix (row-major, size height x dim).
 * @param[out] KQoutputMatrix Device pointer to the output query/key matrix (row-major, size tokenCount x height).
 * @param[in]  tokenCount     Number of rows in tokenMatrix (and KQoutputMatrix).
 * @param[in]  dim            The embedding dimension (columns in tokenMatrix and KQmatrix).
 * @param[in]  height         Number of rows in KQmatrix (columns in KQoutputMatrix).
 */
__global__ void computeKQall(const float* tokenMatrix, const float* KQmatrix,
    float* KQoutputMatrix, int tokenCount, int dim, int height)
{
    // Global thread ID for the row of the output matrix (and tokenMatrix)
    int token_idx = blockIdx.y * blockDim.y + threadIdx.y;
    // Global thread ID for the column of the output matrix (and row of KQmatrix)
    int kq_height_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check
    if (token_idx < tokenCount && kq_height_idx < height) {
        const float* token_vec = tokenMatrix + token_idx * dim;
        const float* kq_matrix_row = KQmatrix + kq_height_idx * dim;

        float dot_product = cuComputeDot(token_vec, kq_matrix_row, dim);
        KQoutputMatrix[token_idx * height + kq_height_idx] = dot_product;
    }
}

/**------------------------------------TRAINING------------------------------------**/

/**
 * @brief CUDA kernel for calculating the scaled KdotQ matrix for self-attention during training.
 *        Computes KdotQ[i][j] = dot(Q[i], K[j]) / SCALING for j <= i (causal masking).
 * @param[out] d_kdotq         Device pointer to the output KdotQ matrix (row-major, size num_queries_eff x kdotq_width).
 * @param[in]  d_keys          Device pointer to the Key vectors (K matrix, row-major, size num_keys_eff x embedding_dim).
 * @param[in]  d_querys        Device pointer to the Query vectors (Q matrix, row-major, size num_queries_eff x embedding_dim).
 * @param[in]  num_queries_eff The number of query rows (i) to compute.
 * @param[in]  num_keys_eff    The number of key columns (j) available.
 * @param[in]  kdotq_width     The total width (number of columns) of the d_kdotq buffer, used for indexing.
 * @param[in]  embedding_dim   The dimension of each key and query vector.
 * @param[in]  inv_scaling     The inverse scaling factor (1.0f / sqrt(embedding_dim)).
 */
__global__ void kernelKdotQforSelf_train(float* d_kdotq, const float* d_keys, const float* d_querys, int num_queries_eff,
    int num_keys_eff, int kdotq_width, int embedding_dim, float inv_scaling)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Global key index (column)
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Global query index (row)

    // Check boundaries and apply causal mask (j <= i)
    if (i < num_queries_eff && j < num_keys_eff && j <= i) {
        const float* q_vec = d_querys + i * embedding_dim;
        const float* k_vec = d_keys + j * embedding_dim;

        float dot_product = cuComputeDot(q_vec, k_vec, embedding_dim);

        int kdotq_index = i * kdotq_width + j;
        d_kdotq[kdotq_index] = dot_product * inv_scaling;
    }
    // Implicitly, elements where j > i remain uninitialized or zero if buffer was zeroed.
}


/**
 * @brief CUDA kernel for calculating the scaled KdotQ matrix for cross-attention during training.
 *        Computes KdotQ[i][j] = dot(Q[i], K[j]) / SCALING for all i, j.
 * @param[out] d_kdotq         Device pointer to the output KdotQ matrix (row-major, size num_queries_eff x kdotq_width).
 * @param[in]  d_keys          Device pointer to the Key vectors (K matrix, row-major, size num_keys_eff x embedding_dim).
 * @param[in]  d_querys        Device pointer to the Query vectors (Q matrix, row-major, size num_queries_eff x embedding_dim).
 * @param[in]  num_queries_eff The number of query rows (i) to compute.
 * @param[in]  num_keys_eff    The number of key columns (j) to compute.
 * @param[in]  kdotq_width     The total width (number of columns) of the d_kdotq buffer, used for indexing.
 * @param[in]  embedding_dim   The dimension of each key and query vector.
 * @param[in]  inv_scaling     The inverse scaling factor (1.0f / sqrt(embedding_dim)).
 */
__global__ void kernelKdotQforCross_train(float* d_kdotq, const float* d_keys, const float* d_querys, int num_queries_eff,
    int num_keys_eff, int kdotq_width, int embedding_dim, float inv_scaling)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Global key index (column)
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Global query index (row)

    // Check boundaries (no causal mask for cross-attention)
    if (i < num_queries_eff && j < num_keys_eff) {
        const float* q_vec = d_querys + i * embedding_dim;
        const float* k_vec = d_keys + j * embedding_dim;

        float dot_product = cuComputeDot(q_vec, k_vec, embedding_dim);

        int kdotq_index = i * kdotq_width + j;
        d_kdotq[kdotq_index] = dot_product * inv_scaling;
    }
}

/**------------------------------------INFERENCE------------------------------------**/

/**
 * @brief CUDA kernel for Block 1 SELF-ATTENTION KdotQ calculation during inference.
 *        Computes KdotQ[i][j] = (tokenEmbed[i] * M * tokenEmbed[j]^T) / SCALING for j <= i,
 *        where 'i' corresponds to new sequence1 tokens and 'j' spans the full context.
 * @param[out] d_kdotq         Device pointer to the output KdotQ matrix (row-major, potentially sparse).
 * @param[in]  d_tokenEmbed    Device pointer to token embeddings for the *entire* context (row-major, context_len x embedding_dim).
 * @param[in]  d_M             Device pointer to the precomputed QK' matrix M (row-major, embedding_dim x embedding_dim).
 * @param[in]  sequence1_start_index Index of the first token of the new sequence1 within d_tokenEmbed.
 * @param[in]  sequence1_len      Number of tokens in the new sequence1.
 * @param[in]  context_len     Total number of tokens currently in the context (sequence1_start_index + sequence1_len).
 * @param[in]  kdotq_width     The total width (number of columns) of the d_kdotq buffer (usually context_len).
 * @param[in]  embedding_dim   The dimension of token vectors and the matrix M.
 * @param[in]  inv_scaling     The inverse scaling factor (1.0f / sqrt(embedding_dim)).
 */
__global__ void kernelKdotQ_Block1_Self_Inference(float* d_kdotq, const float* d_tokenEmbed, const float* d_M,
    int sequence1_start_index, int sequence1_len, int context_len, int kdotq_width, int embedding_dim, float inv_scaling)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;        // Global key index (column) spanning full context
    int i_offset = blockIdx.y * blockDim.y + threadIdx.y; // Query index offset relative to the start of the sequence1
    int i = sequence1_start_index + i_offset;                // Global query index (row) within the sequence1 range

    // Check boundaries and apply causal mask (j <= i)
    // Only compute rows 'i' corresponding to the new sequence1.
    if (i_offset < sequence1_len && j < context_len && j <= i) {
        const float* q_vec = d_tokenEmbed + i * embedding_dim; // Query vector from sequence1
        const float* k_vec = d_tokenEmbed + j * embedding_dim; // Key vector from full context

        // Compute quadratic form: q_vec * M * k_vec^T
        float dot_product = cuComputeDotvmv(q_vec, k_vec, d_M, embedding_dim);

        int kdotq_index = i * kdotq_width + j;
        d_kdotq[kdotq_index] = dot_product * inv_scaling;
    }
}


/**
 * @brief CUDA kernel for Block 1 CROSS-ATTENTION KdotQ calculation during inference.
 *        Computes KdotQ[i][j] = (tokenEmbed[i] * M * tokenEmbed[j]^T) / SCALING for all j,
 *        where 'i' corresponds to new sequence1 tokens and 'j' spans the full context.
 * @param[out] d_kdotq         Device pointer to the output KdotQ matrix (row-major, potentially sparse).
 * @param[in]  d_tokenEmbed    Device pointer to token embeddings for the *entire* context (row-major, context_len x embedding_dim).
 * @param[in]  d_M             Device pointer to the precomputed QK' matrix M (row-major, embedding_dim x embedding_dim).
 * @param[in]  sequence1_start_index Index of the first token of the new sequence1 within d_tokenEmbed.
 * @param[in]  sequence1_len      Number of tokens in the new sequence1.
 * @param[in]  context_len     Total number of tokens currently in the context (sequence1_start_index + sequence1_len).
 * @param[in]  kdotq_width     The total width (number of columns) of the d_kdotq buffer (usually context_len).
 * @param[in]  embedding_dim   The dimension of token vectors and the matrix M.
 * @param[in]  inv_scaling     The inverse scaling factor (1.0f / sqrt(embedding_dim)).
 */
__global__ void kernelKdotQ_Block1_Cross_Inference(float* d_kdotq, const float* d_tokenEmbed, const float* d_M,
    int sequence1_start_index, int sequence1_len, int context_len, int kdotq_width, int embedding_dim, float inv_scaling)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;        // Global key index (column) spanning full context
    int i_offset = blockIdx.y * blockDim.y + threadIdx.y; // Query index offset relative to the start of the sequence1
    int i = sequence1_start_index + i_offset;                // Global query index (row) within the sequence1 range

    // Check boundaries (no causal mask for cross-attention)
    // Only compute rows 'i' corresponding to the new sequence1.
    if (i_offset < sequence1_len && j < context_len) {
        const float* q_vec = d_tokenEmbed + i * embedding_dim; // Query vector from sequence1
        const float* k_vec = d_tokenEmbed + j * embedding_dim; // Key vector from full context

        // Compute quadratic form: q_vec * M * k_vec^T
        float dot_product = cuComputeDotvmv(q_vec, k_vec, d_M, embedding_dim);

        int kdotq_index = i * kdotq_width + j;
        d_kdotq[kdotq_index] = dot_product * inv_scaling;
    }
}


/**
 * @brief CUDA kernel for Block N (N > 1) SELF-ATTENTION KdotQ calculation during inference.
 *        Computes KdotQ[i][j] = (tokForBlock[i] * M * EVp[j]^T) / SCALING for j <= i,
 *        where 'i' corresponds to new sequence1 tokens within the block's window,
 *        and 'j' spans the relevant context (from previous block's EV) within the block's window.
 * @param[out] d_kdotq         Device pointer to the output KdotQ matrix for this block's window (row-major).
 * @param[in]  d_tokForBlock   Device pointer to token embeddings relevant to this block's window (row-major, context_len_in_block x embedding_dim).
 * @param[in]  d_EVp           Device pointer to the previous block's output vectors (EV') relevant to this block's window (row-major, context_len_in_block x embedding_dim).
 * @param[in]  d_M             Device pointer to the precomputed QK' matrix M (row-major, embedding_dim x embedding_dim).
 * @param[in]  sequence1_start_index_in_block Index of the first sequence1 token *within this block's window*.
 * @param[in]  sequence1_len      Number of tokens in the new sequence1.
 * @param[in]  context_len_in_block Total number of relevant tokens in this block's window.
 * @param[in]  kdotq_width     The total width (number of columns) of the d_kdotq buffer (usually context_len_in_block).
 * @param[in]  embedding_dim   The dimension of token vectors and the matrix M.
 * @param[in]  inv_scaling     The inverse scaling factor (1.0f / sqrt(embedding_dim)).
 */
__global__ void kernelKdotQ_BlockN_Self_Inference(float* d_kdotq, const float* d_tokForBlock, const float* d_EVp,
    const float* d_M, int sequence1_start_index_in_block, int sequence1_len, int context_len_in_block, int kdotq_width,
    int embedding_dim, float inv_scaling)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;        // Key index (column) within the block's window
    int i_offset = blockIdx.y * blockDim.y + threadIdx.y; // Query index offset relative to the start of the sequence1
    int i = sequence1_start_index_in_block + i_offset;       // Query index (row) within the block's window

    // Check boundaries and apply causal mask (j <= i) within the block window
    // Only compute rows 'i' corresponding to the new sequence1 within this block.
    if (i_offset < sequence1_len && j < context_len_in_block && j <= i) {
        const float* q_vec = d_tokForBlock + i * embedding_dim; // Query vector from block's tokens
        const float* k_vec = d_EVp + j * embedding_dim;         // Key vector from previous block's EV

        // Compute quadratic form: q_vec * M * k_vec^T
        float dot_product = cuComputeDotvmv(q_vec, k_vec, d_M, embedding_dim);

        int kdotq_index = i * kdotq_width + j;
        d_kdotq[kdotq_index] = dot_product * inv_scaling;
    }
}


/**
 * @brief CUDA kernel for Block N (N > 1) CROSS-ATTENTION KdotQ calculation during inference.
 *        Computes KdotQ[i][j] = (tokForBlock[i] * M * EVp[j]^T) / SCALING for all j,
 *        where 'i' corresponds to new sequence1 tokens within the block's window,
 *        and 'j' spans the relevant context (from previous block's EV) within the block's window.
 * @param[out] d_kdotq         Device pointer to the output KdotQ matrix for this block's window (row-major).
 * @param[in]  d_tokForBlock   Device pointer to token embeddings relevant to this block's window (row-major, context_len_in_block x embedding_dim).
 * @param[in]  d_EVp           Device pointer to the previous block's output vectors (EV') relevant to this block's window (row-major, context_len_in_block x embedding_dim).
 * @param[in]  d_M             Device pointer to the precomputed QK' matrix M (row-major, embedding_dim x embedding_dim).
 * @param[in]  sequence1_start_index_in_block Index of the first sequence1 token *within this block's window*.
 * @param[in]  sequence1_len      Number of tokens in the new sequence1.
 * @param[in]  context_len_in_block Total number of relevant tokens in this block's window.
 * @param[in]  kdotq_width     The total width (number of columns) of the d_kdotq buffer (usually context_len_in_block).
 * @param[in]  embedding_dim   The dimension of token vectors and the matrix M.
 * @param[in]  inv_scaling     The inverse scaling factor (1.0f / sqrt(embedding_dim)).
 */
__global__ void kernelKdotQ_BlockN_Cross_Inference(float* d_kdotq, const float* d_tokForBlock, const float* d_EVp,
    const float* d_M, int sequence1_start_index_in_block, int sequence1_len, int context_len_in_block, int kdotq_width,
    int embedding_dim, float inv_scaling)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;        // Key index (column) within the block's window
    int i_offset = blockIdx.y * blockDim.y + threadIdx.y; // Query index offset relative to the start of the sequence1
    int i = sequence1_start_index_in_block + i_offset;       // Query index (row) within the block's window

    // Check boundaries (no causal mask for cross-attention)
    // Only compute rows 'i' corresponding to the new sequence1 within this block.
    if (i_offset < sequence1_len && j < context_len_in_block) {
        const float* q_vec = d_tokForBlock + i * embedding_dim; // Query vector from block's tokens
        const float* k_vec = d_EVp + j * embedding_dim;         // Key vector from previous block's EV

        // Compute quadratic form: q_vec * M * k_vec^T
        float dot_product = cuComputeDotvmv(q_vec, k_vec, d_M, embedding_dim);

        int kdotq_index = i * kdotq_width + j;
        d_kdotq[kdotq_index] = dot_product * inv_scaling;
    }
}
#endif