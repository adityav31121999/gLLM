
#include "include/transformer.hpp"

#ifdef USE_OPENCL

#include <CL/cl.hpp>
#include <cfloat>

/**
 * @brief Computes the dot product of two vectors residing in global memory.
 * @param vec1 Pointer to the first vector in global memory.
 * @param vec2 Pointer to the second vector in global memory.
 * @param dim The dimension of the vectors.
 * @return The dot product of vec1 and vec2.
 */
inline float compute_dot_product(__global const float* vec1, __global const float* vec2, int dim) 
{
    float dot_product = 0.0f;
    for (int k = 0; k < dim; ++k) {
        dot_product += vec1[k] * vec2[k];
    }
    return dot_product;
}

/**
 * @brief Computes the dot product of vec1 * matrix * vec2.
 * @param vec1 Pointer to the first vector (row vector, size dim) in global memory.
 * @param vec2 Pointer to the second vector (column vector, size dim) in global memory.
 * @param matrix Pointer to the matrix (row-major, dim x dim) in global memory.
 * @param dim The dimension of the vectors and the matrix.
 * @return The scalar result of vec1 * matrix * vec2.
 */
inline float compute_dot_product(__global const float* vec1, __global const float* vec2, __global const float* matrix,
    int dim)
{
    float final_dot_product = 0.0f;
    for (int i = 0; i < dim; ++i) {
        // inner_sum = vec1 . matrix_row_i
        float inner_sum = 0.0f;
        // ith row of matrix
        __global const float* matrix_row_i = matrix + i * dim;

        for (int j = 0; j < dim; ++j) {
            // dot product of vec1 with ith row of matrix
            inner_sum += vec1[j] * matrix_row_i[j];
        }

        // (vec1 . matrix_row_i) * vec2[i]
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
int compute_prediction(__global const float* EH, __global const float* embeddings, int dim, int voc) 
{
    // for empty embeddings
    if (voc <= 0) {
        return -1;
    }
    // Initialize with the smallest possible float value
    float max_dot_product = -FLT_MAX;
    int predicted_index = 0;
    for (int i = 0; i < voc; ++i) {
        // pointer to ith token embedding row
        __global const float* current_embedding_row = embeddings + i * dim;
        float current_dot_product = compute_dot_product_ocl(EH, current_embedding_row, dim);
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
 * @brief OpenCL kernel for calculating KdotQ using keys and queries for self attention (Q[i] dot K[j] where j <= i)
 * @param[out] d_kdotq         Global pointer to hold dot products (KdotQ matrix, row-major)
 * @param[in]  d_keys          Global pointer to keys (K matrix, row-major)
 * @param[in]  d_querys        Global pointer to queries (Q matrix, row-major)
 * @param[in]  num_queries_eff Number of query rows (i) to compute for in this launch.
 * @param[in]  num_keys_eff    Number of key columns (j) to compute for in this launch.
 * @param[in]  kdotq_width     Total width (number of columns) of the d_kdotq buffer for indexing.
 * @param[in]  embedding_dim   Dimension of each key/query vector.
 * @param[in]  inv_scaling     Value of 1.0f / SCALING.
*/
__kernel void kernelKdotQforSelf_train(__global float* d_kdotq, __global const float* d_keys, __global const float* d_querys,
                                int num_queries_eff, int num_keys_eff, int kdotq_width, int embedding_dim, float inv_scaling)
{
    // Calculate the global row (query index i) and column (key index j) for this work-item
    int j = get_global_id(0); // Key index (column)
    int i = get_global_id(1); // Query index (row)

    // Boundary check AND self-attention causal mask (j <= i)
    if (i < num_queries_eff && j < num_keys_eff && j <= i) {
        // Pointers to the start of the i-th query vector and j-th key vector
        __global const float* q_vec = d_querys + i * embedding_dim;
        __global const float* k_vec = d_keys + j * embedding_dim;
        float dot_product = compute_dot_product_ocl(q_vec, k_vec, embedding_dim);

        // Calculate the index in the flattened d_kdotq array (row-major)
        int kdotq_index = i * kdotq_width + j;
        d_kdotq[kdotq_index] = dot_product * inv_scaling;
    }
}

/**
 * @brief OpenCL kernel for calculating KdotQ using keys and queries for cross attention (Q[i] dot K[j] for all i, j)
 * @param[out] d_kdotq         Global pointer to hold dot products (KdotQ matrix, row-major)
 * @param[in]  d_keys          Global pointer to keys (K matrix, row-major)
 * @param[in]  d_querys        Global pointer to queries (Q matrix, row-major)
 * @param[in]  num_queries_eff Number of query rows (i) to compute for in this launch.
 * @param[in]  num_keys_eff    Number of key columns (j) to compute for in this launch.
 * @param[in]  kdotq_width     Total width (number of columns) of the d_kdotq buffer for indexing.
 * @param[in]  embedding_dim   Dimension of each key/query vector.
 * @param[in]  inv_scaling     Value of 1.0f / SCALING.
 */
__kernel void kernelKdotQforCross_train(__global float* d_kdotq, __global const float* d_keys, __global const float* d_querys,
                                int num_queries_eff, int num_keys_eff, int kdotq_width, int embedding_dim, float inv_scaling)
{
    // Calculate the global column (key index j) and row (query index i) for this work-item
    int j = get_global_id(0); // Key index (column)
    int i = get_global_id(1); // Query index (row)

    // Boundary check (no causal mask for cross-attention)
    if (i < num_queries_eff && j < num_keys_eff) {
        // Pointers to the start of the i-th query vector and j-th key vector
        __global const float* q_vec = d_querys + i * embedding_dim;
        __global const float* k_vec = d_keys + j * embedding_dim;
        float dot_product = compute_dot_product_ocl(q_vec, k_vec, embedding_dim);
        // Calculate the index in the flattened d_kdotq array (row-major)
        int kdotq_index = i * kdotq_width + j;
        d_kdotq[kdotq_index] = dot_product * inv_scaling;
    }
}

/**------------------------------------INFERENCE------------------------------------**/

/**
 * @brief INFERENCE OpenCL kernel for Block 1 SELF-ATTENTION KdotQ.
 * Computes KdotQ[i][j] = (tokenEmbed[i] * M * tokenEmbed[j]) / SCALING for j <= i,
 * where i and j are the index of attention score of KdotQ grid.
 * @param[out] d_kdotq         Global pointer to output KdotQ (potentially sparse, only computes needed values).
 * @param[in]  d_tokenEmbed    Global pointer to token embeddings for the *entire* context (row-major).
 * @param[in]  d_M             Global pointer to the QK' matrix M (row-major, dim x dim).
 * @param[in]  prompt_start_index Index of the first token of the new prompt in d_tokenEmbed.
 * @param[in]  prompt_len      Number of tokens in the new prompt.
 * @param[in]  context_len     Total number of tokens in the context (prompt_start_index + prompt_len).
 * @param[in]  kdotq_width     Total width (number of columns) of the d_kdotq buffer for indexing (usually context_len).
 * @param[in]  embedding_dim   Dimension of each token vector and the matrix M.
 * @param[in]  inv_scaling     Value of 1.0f / SCALING.
 */
__kernel void kernelKdotQBlock1Self_Inference(__global float* d_kdotq, __global const float* d_tokenEmbed, __global const float* d_M,
                            int prompt_start_index, int prompt_len, int context_len, int kdotq_width,int embedding_dim, float inv_scaling)
{
    // Calculate the global key index (j) and the offset for the query index (i_offset)
    int j = get_global_id(0);
    int i_offset = get_global_id(1);
    int i = prompt_start_index + i_offset;
    // Boundary checks for self masking
    if (i_offset < prompt_len && j < context_len && j <= i) {
        // Pointers to the start of the i-th query vector and j-th key vector
        __global const float* q_vec = d_tokenEmbed + i * embedding_dim;
        __global const float* k_vec = d_tokenEmbed + j * embedding_dim;
        float dot_product = compute_dot_product_mat_ocl(q_vec, k_vec, d_M, embedding_dim);
        // index in flatten array
        int kdotq_index = i * kdotq_width + j;
        d_kdotq[kdotq_index] = dot_product * inv_scaling;
    }
}

/**
 * @brief INFERENCE OpenCL kernel for Block 1 CROSS-ATTENTION KdotQ.
 * Computes KdotQ[i][j] = (tokenEmbed[i] * M * tokenEmbed[j]) / SCALING for all j,
 * where i corresponds to the new prompt tokens and j spans the entire context.
 * @param[out] d_kdotq         Global pointer to output KdotQ (potentially sparse).
 * @param[in]  d_tokenEmbed    Global pointer to token embeddings for the *entire* context (row-major).
 * @param[in]  d_M             Global pointer to the QK' matrix M (row-major, dim x dim).
 * @param[in]  prompt_start_index Index of the first token of the new prompt in d_tokenEmbed.
 * @param[in]  prompt_len      Number of tokens in the new prompt.
 * @param[in]  context_len     Total number of tokens in the context (prompt_start_index + prompt_len).
 * @param[in]  kdotq_width     Total width (number of columns) of the d_kdotq buffer for indexing (usually context_len).
 * @param[in]  embedding_dim   Dimension of each token vector and the matrix M.
 * @param[in]  inv_scaling     Value of 1.0f / SCALING.
 */
__kernel void kernelKdotQBlock1Cross_Inference(__global float* d_kdotq, __global const float* d_tokenEmbed, __global const float* d_M,
                            int prompt_start_index, int prompt_len, int context_len, int kdotq_width, int embedding_dim, float inv_scaling)
{
    // Calculate the global key index (j) and the offset for the query index (i_offset)
    int j = get_global_id(0);
    int i_offset = get_global_id(1);
    int i = prompt_start_index + i_offset;
    if (i_offset < prompt_len && j < context_len) {
        // Pointers to the start of the i-th query vector and j-th key vector
        __global const float* q_vec = d_tokenEmbed + i * embedding_dim;
        __global const float* k_vec = d_tokenEmbed + j * embedding_dim;
        float dot_product = compute_dot_product_mat_ocl(q_vec, k_vec, d_M, embedding_dim);
        // Calculate the index in the flattened d_kdotq array (row-major)
        int kdotq_index = i * kdotq_width + j;
        d_kdotq[kdotq_index] = dot_product * inv_scaling;
    }
}

/**
 * @brief INFERENCE OpenCL kernel for Block N (N > 1) SELF-ATTENTION KdotQ.
 * Computes KdotQ[i][j] = (tokForBlock[i] * M * EVp[j]) / SCALING for j <= i,
 * where i corresponds to the new prompt tokens within the block's window,
 * and j spans the relevant context within the block's window.
 * @param[out] d_kdotq         Global pointer to output KdotQ for this block's window.
 * @param[in]  d_tokForBlock   Global pointer to token embeddings for this block's context window (row-major).
 * @param[in]  d_EVp           Global pointer to previous block's output vectors (EV') for this block's context window (row-major).
 * @param[in]  d_M             Global pointer to the QK' matrix M (row-major, dim x dim).
 * @param[in]  prompt_start_index_in_block Index of the first token of the new prompt *within the block's window*.
 * @param[in]  prompt_len      Number of tokens in the new prompt.
 * @param[in]  context_len_in_block Total number of relevant tokens in this block's window.
 * @param[in]  kdotq_width     Total width (number of columns) of the d_kdotq buffer for indexing (usually context_len_in_block).
 * @param[in]  embedding_dim   Dimension of each token vector and the matrix M.
 * @param[in]  inv_scaling     Value of 1.0f / SCALING.
 */
__kernel void kernelKdotQBlockNSelf_Inference(__global float* d_kdotq, __global const float* d_tokForBlock, __global const float* d_EVp,
                            __global const float* d_M, int prompt_start_index_in_block, int prompt_len, int context_len_in_block,
                            int kdotq_width, int embedding_dim, float inv_scaling)
{
    // Calculate the key index (j) and query index offset (i_offset) *within the block's window*
    int j = get_global_id(0);
    int i_offset = get_global_id(1);
    int i = prompt_start_index_in_block + i_offset;
    // self masking and boundary checks
    if (i_offset < prompt_len && j < context_len_in_block && j <= i) {
        // Pointers to the start of the i-th query vector (from tokForBlock) and j-th key vector (from EVp)
        __global const float* q_vec = d_tokForBlock + i * embedding_dim;
        __global const float* k_vec = d_EVp + j * embedding_dim;
        float dot_product = compute_dot_product_mat_ocl(q_vec, k_vec, d_M, embedding_dim);
        // Calculate the index in the flattened d_kdotq array (row-major, relative to block window)
        int kdotq_index = i * kdotq_width + j;
        d_kdotq[kdotq_index] = dot_product * inv_scaling;
    }
}

/**
 * @brief INFERENCE OpenCL kernel for Block N (N > 1) CROSS-ATTENTION KdotQ.
 * Computes KdotQ[i][j] = (tokForBlock[i] * M * EVp[j]) / SCALING for all j,
 * where i corresponds to the new prompt tokens within the block's window,
 * and j spans the relevant context within the block's window.
 * @param[out] d_kdotq         Global pointer to output KdotQ for this block's window.
 * @param[in]  d_tokForBlock   Global pointer to token embeddings for this block's context window (row-major).
 * @param[in]  d_EVp           Global pointer to previous block's output vectors (EV') for this block's context window (row-major).
 * @param[in]  d_M             Global pointer to the QK' matrix M (row-major, dim x dim).
 * @param[in]  prompt_start_index_in_block Index of the first token of the new prompt *within the block's window*.
 * @param[in]  prompt_len      Number of tokens in the new prompt.
 * @param[in]  context_len_in_block Total number of relevant tokens in this block's window.
 * @param[in]  kdotq__width     Total width (number of columns) of the d_kdotq buffer for indexing (usually context_len_in_block).
 * @param[in]  embedding_dim   Dimension of each token vector and the matrix M.
 * @param[in]  inv_scaling     Value of 1.0f / SCALING.
 */
__kernel void kernelKdotQBlockNCross_Inference(__global float* d_kdotq, __global const float* d_tokForBlock, __global const float* d_EVp,
                                __global const float* d_M, int prompt_start_index_in_block, int prompt_len, int context_len_in_block,
                                int kdotq_width, int embedding_dim, float inv_scaling)
{
    // Calculate the key index (j) and query index offset (i_offset) *within the block's window*
    int j = get_global_id(0);
    int i_offset = get_global_id(1);
    int i = prompt_start_index_in_block + i_offset;
    if (i_offset < prompt_len && j < context_len_in_block) {
        // Pointers to the start of the i-th query vector (from tokForBlock) and j-th key vector (from EVp)
        __global const float* q_vec = d_tokForBlock + i * embedding_dim;
        __global const float* k_vec = d_EVp + j * embedding_dim;
        float dot_product = compute_dot_product_mat_ocl(q_vec, k_vec, d_M, embedding_dim);
        // Calculate the index in the flattened d_kdotq array (row-major, relative to block window)
        int kdotq_index = i * kdotq_width + j;
        d_kdotq[kdotq_index] = dot_product * inv_scaling;
    }
}

#endif
