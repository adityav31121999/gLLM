
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <numeric>
#include <cmath>
#include <maths.hpp>

/**
 * @brief compute key or query vector
 * @param[in] tokenEmbed embedding of token for which key or query vector need to be calculated
 * @param[in] matrix key or query matrix
 * @param[out] KorQ Key or Query vector
 * @param[in] dim embedding dimension
 * @param[in] height key and query matrix rows
 */
__device__ void cuComputeKorQ(const float* tokenEmbed, const float* matrix, float* KorQ, int dim, int height) {
    for (int i = 0; i < height; ++i) {
        // ith row of matrix
        const float* matrix_row_i = matrix + i * dim;
        for (int j = 0; j < dim; ++j) {
            // dot product of vec1 with ith row of matrix
            KorQ[i] += tokenEmbed[j] * matrix_row_i[j];
        }
    }
}

/**
 * @brief Computes the dot product of two vectors residing in global memory.
 * @param vec1 Pointer to the first vector.
 * @param vec2 Pointer to the second vector.
 * @param dim The dimension of the vectors.
 * @return The dot product of vec1 and vec2.
 */
__device__ float compute_dot_product(const float* vec1, const float* vec2, int dim) {
    float dot_product = 0.0f;
    for (int k = 0; k < dim; ++k) {
        dot_product += vec1[k] * vec2[k];
    }
    return dot_product;
}


/**
 * @brief Computes the dot product of vec1 * matrix * vec2.
 * @param vec1 Pointer to the first vector (row vector, size dim).
 * @param vec2 Pointer to the second vector (column vector, size dim).
 * @param matrix Pointer to the matrix (row-major, dim x dim).
 * @param dim The dimension of the vectors and the matrix.
 * @return The scalar result of vec1 * matrix * vec2.
 */
__device__ float compute_dot_product(const float* vec1, const float* vec2, const float* matrix, int dim)
{
    float final_dot_product = 0.0f;
    for (int i = 0; i < dim; ++i) {
        // inner_sum = vec1.matrix[i]
        float inner_sum = 0.0f;
        // ith row of matrix
        const float* matrix_row_i = matrix + i * dim;

        for (int j = 0; j < dim; ++j) {
            // dot product of vec1 with ith row of matrix
            inner_sum += vec1[j] * matrix_row_i[j];
        }

        // vec1 x matrix[i] x vec2
        final_dot_product += inner_sum * vec2[i];
    }
    return final_dot_product;
}

/**
 * @brief compute prediction by inner products of EH and the rows of embeddings
 * @param EH horizontal retention vector (output from the last block)
 * @param embeddings token embeddings matrix (row-major: voc x dim)
 * @param dim embedding dimension (size of EH and columns of embeddings)
 * @param voc vocabulary size (rows of embeddings)
 * @return The index of the token embedding with the highest dot product (predicted token index).
 * @note it is assumed in this function that the case of "all dot products being zero" will
 *      not occur
 */
__device__ int compute_prediction(const float* EH, const float* embeddings, int dim, int voc) {
    // for empty vocabulary / embeddings
    if (voc <= 0 || embeddings == nullptr) {
        return -1;
    }
    // Initialize with the smallest possible float value
    float max_dot_product = -FLT_MAX;
    int predicted_index = 0;
    for (int i = 0; i < voc; ++i) {
        // pointer to ith token embedding row
        const float* current_embedding_row = embeddings + i * dim;
        float current_dot_product = compute_dot_product(EH, current_embedding_row, dim);
        // update index if new maximum dot product is available
        if (current_dot_product > max_dot_product) {
            max_dot_product = current_dot_product;
            predicted_index = i;
        }
    }
    return predicted_index;
}

/**------------------------------------TRAINING------------------------------------**/

/**
 * @brief kernel for calculating KdotQ using keys and queries for self attention (Q[i] dot K[j] where j <= i)
 * @param[out] d_kdotq         Device pointer to hold dot products (KdotQ matrix, row-major)
 * @param[in]  d_keys          Device pointer to keys (K matrix, row-major)
 * @param[in]  d_querys        Device pointer to queries (Q matrix, row-major)
 * @param[in]  num_queries_eff Number of query rows (i) to compute for in this launch.
 * @param[in]  num_keys_eff    Number of key columns (j) to compute for in this launch.
 * @param[in]  kdotq_width     Total width (number of columns) of the d_kdotq buffer for indexing.
 * @param[in]  embedding_dim   Dimension of each key/query vector.
 * @param[in]  inv_scaling     Value of 1.0f / SCALING.
*/
__global__ void kernelKdotQforSelf_train(float* d_kdotq, const float* d_keys, const float* d_querys, int num_queries_eff, 
    int num_keys_eff, int kdotq_width, int embedding_dim, float inv_scaling)
{
    // Calculate the global row (query index i) and column (key index j) for this thread
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Key index (column)
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Query index (row)

    // Boundary check AND self-attention causal mask (j <= i)
    if (i < num_queries_eff && j < num_keys_eff && j <= i) {
        // Pointers to the start of the i-th query vector and j-th key vector
        const float* q_vec = d_querys + i * embedding_dim;
        const float* k_vec = d_keys + j * embedding_dim;

        // Compute dot product using the device function
        float dot_product = compute_dot_product(q_vec, k_vec, embedding_dim);

        // Calculate the index in the flattened d_kdotq array (row-major)
        int kdotq_index = i * kdotq_width + j;

        // Store the scaled result
        d_kdotq[kdotq_index] = dot_product * inv_scaling;
    }
}


/**
 * @brief kernel for calculating KdotQ using keys and queries for cross attention (Q[i] dot K[j] for all i, j)
 * @param[out] d_kdotq         Device pointer to hold dot products (KdotQ matrix, row-major)
 * @param[in]  d_keys          Device pointer to keys (K matrix, row-major)
 * @param[in]  d_querys        Device pointer to queries (Q matrix, row-major)
 * @param[in]  num_queries_eff Number of query rows (i) to compute for in this launch.
 * @param[in]  num_keys_eff    Number of key columns (j) to compute for in this launch.
 * @param[in]  kdotq_width     Total width (number of columns) of the d_kdotq buffer for indexing.
 * @param[in]  embedding_dim   Dimension of each key/query vector.
 * @param[in]  inv_scaling     Value of 1.0f / SCALING.
 */
__global__ void kernelKdotQforCross_train(float* d_kdotq, const float* d_keys, const float* d_querys, int num_queries_eff,
    int num_keys_eff, int kdotq_width, int embedding_dim, float inv_scaling)
{
    // Calculate the global column (key index j) and row (query index i) for this thread
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Key index (column)
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Query index (row) // <<< CORRECTED LINE

    // Boundary check (no causal mask for cross-attention)
    if (i < num_queries_eff && j < num_keys_eff) {

        // Pointers to the start of the i-th query vector and j-th key vector
        const float* q_vec = d_querys + i * embedding_dim;
        const float* k_vec = d_keys + j * embedding_dim;

        // Compute dot product using the device function
        float dot_product = compute_dot_product(q_vec, k_vec, embedding_dim);

        // Calculate the index in the flattened d_kdotq array (row-major)
        int kdotq_index = i * kdotq_width + j;

        // Store the scaled result
        d_kdotq[kdotq_index] = dot_product * inv_scaling;
    }
}

/**------------------------------------INFERENCE------------------------------------**/

/**
 * @brief INFERENCE kernel for Block 1 SELF-ATTENTION KdotQ.
 * Computes KdotQ[i][j] = (tokenEmbed[i] * M * tokenEmbed[j]) / SCALING for j <= i,
 * where i and j are the index of attention score of KdotQ grid.
 * @param[out] d_kdotq         Device pointer to output KdotQ (potentially sparse, only computes needed values).
 * @param[in]  d_tokenEmbed    Device pointer to token embeddings for the *entire* context (row-major).
 * @param[in]  d_M             Device pointer to the QK' matrix M (row-major, dim x dim).
 * @param[in]  prompt_start_index Index of the first token of the new prompt in d_tokenEmbed.
 * @param[in]  prompt_len      Number of tokens in the new prompt.
 * @param[in]  context_len     Total number of tokens in the context (prompt_start_index + prompt_len).
 * @param[in]  kdotq_width     Total width (number of columns) of the d_kdotq buffer for indexing (usually context_len).
 * @param[in]  embedding_dim   Dimension of each token vector and the matrix M.
 * @param[in]  inv_scaling     Value of 1.0f / SCALING.
 */
__global__ void kernelKdotQ_Block1_Self_Inference(float* d_kdotq, const float* d_tokenEmbed, const float* d_M, 
    int prompt_start_index, int prompt_len, int context_len, int kdotq_width, int embedding_dim, float inv_scaling)
{
    // Calculate the global key index (j) and the offset for the query index (i_offset)
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Key index (column)
    // this is same condition for cpp with promptCount and currentTokenCount
    int i_offset = blockIdx.y * blockDim.y + threadIdx.y; // Query index offset within the prompt
    int i = prompt_start_index + i_offset;

    // forms lower triangle with all elements of diagonal and elements left to diagonal
    if (i_offset < prompt_len && j < context_len && j <= i) {

        // Pointers to the start of the i-th query vector and j-th key vector
        const float* q_vec = d_tokenEmbed + i * embedding_dim;
        const float* k_vec = d_tokenEmbed + j * embedding_dim;

        // Compute dot product: q_vec * M * k_vec
        float dot_product = compute_dot_product(q_vec, k_vec, d_M, embedding_dim);

        // Calculate the index in the flattened d_kdotq array (row-major)
        // We are computing rows corresponding to the new prompt (i)
        // against columns corresponding to the full context (j)
        int kdotq_index = i * kdotq_width + j;

        // Store the scaled result
        d_kdotq[kdotq_index] = dot_product * inv_scaling;
    }
}
  

/**
 * @brief INFERENCE kernel for Block 1 CROSS-ATTENTION KdotQ.
 * Computes KdotQ[i][j] = (tokenEmbed[i] * M * tokenEmbed[j]) / SCALING for all j,
 * where i corresponds to the new prompt tokens and j spans the entire context.
 * @param[out] d_kdotq         Device pointer to output KdotQ (potentially sparse).
 * @param[in]  d_tokenEmbed    Device pointer to token embeddings for the *entire* context (row-major).
 * @param[in]  d_M             Device pointer to the QK' matrix M (row-major, dim x dim).
 * @param[in]  prompt_start_index Index of the first token of the new prompt in d_tokenEmbed.
 * @param[in]  prompt_len      Number of tokens in the new prompt.
 * @param[in]  context_len     Total number of tokens in the context (prompt_start_index + prompt_len).
 * @param[in]  kdotq_width     Total width (number of columns) of the d_kdotq buffer for indexing (usually context_len).
 * @param[in]  embedding_dim   Dimension of each token vector and the matrix M.
 * @param[in]  inv_scaling     Value of 1.0f / SCALING.
 */
__global__ void kernelKdotQ_Block1_Cross_Inference(float* d_kdotq, const float* d_tokenEmbed, const float* d_M,
    int prompt_start_index, int prompt_len, int context_len, int kdotq_width, int embedding_dim, float inv_scaling)
{
    // Calculate the global key index (j) and the offset for the query index (i_offset)
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Key index (column)
    int i_offset = blockIdx.y * blockDim.y + threadIdx.y; // Query index offset within the prompt

    // Calculate the absolute query index (i)
    int i = prompt_start_index + i_offset;

    // Boundary checks:
    // 1. Query index 'i' must be within the new prompt range.
    // 2. Key index 'j' must be within the total context.
    // NO causal mask for cross-attention.
    if (i_offset < prompt_len && j < context_len) {

        // Pointers to the start of the i-th query vector and j-th key vector
        const float* q_vec = d_tokenEmbed + i * embedding_dim;
        const float* k_vec = d_tokenEmbed + j * embedding_dim;

        // Compute dot product: q_vec * M * k_vec
        float dot_product = compute_dot_product(q_vec, k_vec, d_M, embedding_dim);

        // Calculate the index in the flattened d_kdotq array (row-major)
        int kdotq_index = i * kdotq_width + j;

        // Store the scaled result
        d_kdotq[kdotq_index] = dot_product * inv_scaling;
    }
}


/**
 * @brief INFERENCE kernel for Block N (N > 1) SELF-ATTENTION KdotQ.
 * Computes KdotQ[i][j] = (tokForBlock[i] * M * EVp[j]) / SCALING for j <= i,
 * where i corresponds to the new prompt tokens within the block's window,
 * and j spans the relevant context within the block's window.
 * @param[out] d_kdotq         Device pointer to output KdotQ for this block's window.
 * @param[in]  d_tokForBlock   Device pointer to token embeddings for this block's context window (row-major).
 * @param[in]  d_EVp           Device pointer to previous block's output vectors (EV') for this block's context window (row-major).
 * @param[in]  d_M             Device pointer to the QK' matrix M (row-major, dim x dim).
 * @param[in]  prompt_start_index_in_block Index of the first token of the new prompt *within the block's window*.
 * @param[in]  prompt_len      Number of tokens in the new prompt.
 * @param[in]  context_len_in_block Total number of relevant tokens in this block's window.
 * @param[in]  kdotq_width     Total width (number of columns) of the d_kdotq buffer for indexing (usually context_len_in_block).
 * @param[in]  embedding_dim   Dimension of each token vector and the matrix M.
 * @param[in]  inv_scaling     Value of 1.0f / SCALING.
 */
__global__ void kernelKdotQ_BlockN_Self_Inference(float* d_kdotq, const float* d_tokForBlock, const float* d_EVp, 
    const float* d_M, int prompt_start_index_in_block, int prompt_len, int context_len_in_block, int kdotq_width, 
    int embedding_dim, float inv_scaling)
{
    // Calculate the key index (j) and query index offset (i_offset) *within the block's window*
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Key index (column) in block window
    int i_offset = blockIdx.y * blockDim.y + threadIdx.y; // Query index offset within the prompt

    // Calculate the query index (i) *within the block's window*
    int i = prompt_start_index_in_block + i_offset;

    // Boundary checks:
    // 1. Query index 'i' must be within the new prompt range within the block's window.
    // 2. Key index 'j' must be within the relevant context length of the block's window.
    // 3. Self-attention causal mask: j <= i (indices relative to block window).
    if (i_offset < prompt_len && j < context_len_in_block && j <= i) {

        // Pointers to the start of the i-th query vector (from tokForBlock) and j-th key vector (from EVp)
        const float* q_vec = d_tokForBlock + i * embedding_dim;
        const float* k_vec = d_EVp + j * embedding_dim;

        // Compute dot product: q_vec * M * k_vec
        float dot_product = compute_dot_product(q_vec, k_vec, d_M, embedding_dim);

        // Calculate the index in the flattened d_kdotq array (row-major, relative to block window)
        int kdotq_index = i * kdotq_width + j;

        // Store the scaled result
        d_kdotq[kdotq_index] = dot_product * inv_scaling;
    }
}


/**
 * @brief INFERENCE kernel for Block N (N > 1) CROSS-ATTENTION KdotQ.
 * Computes KdotQ[i][j] = (tokForBlock[i] * M * EVp[j]) / SCALING for all j,
 * where i corresponds to the new prompt tokens within the block's window,
 * and j spans the relevant context within the block's window.
 * @param[out] d_kdotq         Device pointer to output KdotQ for this block's window.
 * @param[in]  d_tokForBlock   Device pointer to token embeddings for this block's context window (row-major).
 * @param[in]  d_EVp           Device pointer to previous block's output vectors (EV') for this block's context window (row-major).
 * @param[in]  d_M             Device pointer to the QK' matrix M (row-major, dim x dim).
 * @param[in]  prompt_start_index_in_block Index of the first token of the new prompt *within the block's window*.
 * @param[in]  prompt_len      Number of tokens in the new prompt.
 * @param[in]  context_len_in_block Total number of relevant tokens in this block's window.
 * @param[in]  kdotq_width     Total width (number of columns) of the d_kdotq buffer for indexing (usually context_len_in_block).
 * @param[in]  embedding_dim   Dimension of each token vector and the matrix M.
 * @param[in]  inv_scaling     Value of 1.0f / SCALING.
 */
__global__ void kernelKdotQ_BlockN_Cross_Inference(float* d_kdotq, const float* d_tokForBlock, const float* d_EVp, 
    const float* d_M, int prompt_start_index_in_block, int prompt_len, int context_len_in_block, int kdotq_width, 
    int embedding_dim, float inv_scaling)
{
    // Calculate the key index (j) and query index offset (i_offset) *within the block's window*
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Key index (column) in block window
    int i_offset = blockIdx.y * blockDim.y + threadIdx.y; // Query index offset within the prompt

    // Calculate the query index (i) *within the block's window*
    int i = prompt_start_index_in_block + i_offset;

    // Boundary checks:
    // 1. Query index 'i' must be within the new prompt range within the block's window.
    // 2. Key index 'j' must be within the relevant context length of the block's window.
    // NO causal mask for cross-attention.
    if (i_offset < prompt_len && j < context_len_in_block) {

        // Pointers to the start of the i-th query vector (from tokForBlock) and j-th key vector (from EVp)
        const float* q_vec = d_tokForBlock + i * embedding_dim;
        const float* k_vec = d_EVp + j * embedding_dim;

        // Compute dot product: q_vec * M * k_vec
        float dot_product = compute_dot_product(q_vec, k_vec, d_M, embedding_dim);

        // Calculate the index in the flattened d_kdotq array (row-major, relative to block window)
        int kdotq_index = i * kdotq_width + j;

        // Store the scaled result
        d_kdotq[kdotq_index] = dot_product * inv_scaling;
    }
}

/**
 * @brief CUDA kernel to compute row sums (k) and column sums (l) of the attention head matrix,
 *        respecting the self-attention mask if applicable. Mirrors C++ loops exactly.
 * Output: d_row_sums[i] = sum_{j=0}^{limit-1}(head[i][j]), d_col_sums[i] = sum_{j=0}^{limit-1}(head[j][i])
 * where limit = isSelfAttention ? i : num_tokens.
 * @param d_head Input head matrix (num_tokens x num_tokens)
 * @param d_row_sums Output row sums (size num_tokens)
 * @param d_col_sums Output column sums (size num_tokens)
 * @param num_tokens Dimension of the head matrix (tokenCount or count)
 * @param isSelfAttention Flag to apply masking
 */
__global__ void computeHeadSumsMaskedKernel(const float* d_head, float* d_row_sums, float* d_col_sums, int num_tokens, 
    bool isSelfAttention)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Parallelize over token index 'i'
    if (i < num_tokens) {
        float row_sum_k = 0.0f;
        float col_sum_l = 0.0f;

        // Determine the limit for summation based on C++ logic
        int limit = isSelfAttention ? i : num_tokens;

        // Calculate row sum (k) for token i
        for (int j = 0; j < limit; ++j) {
            row_sum_k += d_head[i * num_tokens + j];
        }

        // Calculate column sum (l) for token i (using same limit as C++)
        for (int j = 0; j < limit; ++j) {
             // Summing head[j][i] where j < limit
             col_sum_l += d_head[j * num_tokens + i];
        }

        d_row_sums[i] = row_sum_k;
        d_col_sums[i] = col_sum_l;
    }
}


/**
 * @brief CUDA kernel to accumulate weighted K and Q vectors based on head sums.
 * d_dh_accum = sum_i (d_row_sums[i] * d_K[i])
 * d_dv_accum = sum_i (d_col_sums[i] * d_Q[i])
 * Parallelized over the h_dim (MATHEIGHTS). Uses atomicAdd for safety.
 * @param d_row_sums Input row sums (size num_tokens)
 * @param d_col_sums Input column sums (size num_tokens)
 * @param d_K Input K matrix (num_tokens x h_dim)
 * @param d_Q Input Q matrix (num_tokens x h_dim)
 * @param d_dh_accum Output accumulated dh vector (size h_dim) - MUST be zero-initialized
 * @param d_dv_accum Output accumulated dv vector (size h_dim) - MUST be zero-initialized
 * @param num_tokens Number of tokens to iterate over (tokenCount or count)
 * @param h_dim Dimension MATHEIGHTS
 */
__global__ void accumulateWeightedVectorsKernel(const float* d_row_sums, const float* d_col_sums,
                                                const float* d_K, const float* d_Q, float* d_dh_accum,
                                                float* d_dv_accum, int num_tokens, int h_dim) 
{
    int h_idx = blockIdx.x * blockDim.x + threadIdx.x; // Index into the h_dim (MATHEIGHTS) dimension

    if (h_idx < h_dim) {
        float total_dh = 0.0f;
        float total_dv = 0.0f;

        // Iterate through each token i
        for (int i = 0; i < num_tokens; ++i) {
            // Get K[i][h_idx] and Q[i][h_idx] (Row-major access)
            float k_i_h = d_K[i * h_dim + h_idx];
            float q_i_h = d_Q[i * h_dim + h_idx];

            // Accumulate: dh += row_sum[i] * K[i][h_idx]
            total_dh += d_row_sums[i] * k_i_h;
            // Accumulate: dv += col_sum[i] * Q[i][h_idx]
            total_dv += d_col_sums[i] * q_i_h;
        }

        // Atomically add the computed sum for this h_idx
        atomicAdd((float*)&d_dh_accum[h_idx], total_dh);
        atomicAdd((float*)&d_dv_accum[h_idx], total_dv);
    }
}
