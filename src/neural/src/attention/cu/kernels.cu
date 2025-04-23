
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <numeric>
#include <cmath>
#include <maths.hpp>
#include <string>

// --- Helper Kernels ---

/**
 * @brief Converts a 2D std::vector into a flattened 1D std::vector.
 * @param[in] vec2d The 2D vector to be flattened. Assumes consistent column sizes.
 * @return A 1D vector containing the elements of vec2d in row-major order.
 * @throw std::runtime_error If the input 2D vector has inconsistent column sizes.
 */
std::vector<float> flatten(const std::vector<std::vector<float>>& vec2d) {
    if (vec2d.empty()) return {};
    size_t rows = vec2d.size();
    size_t cols = vec2d[0].size();
    std::vector<float> flat(rows * cols);
    for (size_t i = 0; i < rows; ++i) {
        if (vec2d[i].size() != cols) {
            throw std::runtime_error("Inconsistent column size in flatten");
        }
        // Use memcpy for potentially faster copying of contiguous memory blocks
        memcpy(flat.data() + i * cols, vec2d[i].data(), cols * sizeof(float));
    }
    return flat;
}


/**
 * @brief Flattens a mat object into a 1D vector (row-major).
 * @param matrix The input mat object. Assumes mat has member `a` which is std::vector<std::vector<float>>.
 * @return A 1D vector containing the flattened data.
 */
std::vector<float> flatten(const mat& matrix) {
    // Assuming mat class has a member 'a' of type std::vector<std::vector<float>>
    return flatten(matrix.a);
}


/**
 * @brief Converts a 1D std::vector into a 2D std::vector with specified dimensions.
 * @param[in] flat The 1D vector to be unflattened.
 * @param[out] vec2d The resulting 2D vector. Will be resized.
 * @param[in] rows The desired number of rows for the 2D vector.
 * @param[in] cols The desired number of columns for the 2D vector.
 * @throw std::runtime_error If the number of elements in flat is not equal to rows * cols.
 */
void unflatten(const std::vector<float>& flat, std::vector<std::vector<float>>& vec2d, size_t rows, size_t cols) {
    if (flat.size() != rows * cols) {
        throw std::runtime_error("Size mismatch in unflatten: flat size " + std::to_string(flat.size()) + " != rows*cols " + std::to_string(rows*cols));
    }
    vec2d.resize(rows);
    for (size_t i = 0; i < rows; ++i) {
        vec2d[i].resize(cols);
        // Use memcpy for potentially faster copying of contiguous memory blocks
        memcpy(vec2d[i].data(), flat.data() + i * cols, cols * sizeof(float));
    }
}

/**------------------------------------MULTIPLICATION------------------------------------**/

/**
 * @brief CUDA kernel for element-wise vector multiplication. Multiplies `target_and_output` by `factor` in place.
 * @param[in,out] target_and_output Device pointer to the vector to be multiplied (and store the result).
 * @param[in] factor Device pointer to the vector acting as the multiplier.
 * @param[in] size The number of elements in the vectors.
 */
__global__ void kernelElementwiseMultiply(float* target_and_output, const float* factor, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        target_and_output[idx] *= factor[idx];
    }
}

/**
 * @brief CUDA device function to compute a key or query vector by multiplying a token embedding with a matrix.
 *        KorQ = tokenEmbed * matrix^T (effectively, as matrix is row-major)
 * @param[in] tokenEmbed Device pointer to the token embedding vector (size dim).
 * @param[in] matrix Device pointer to the key or query matrix (row-major, height x dim).
 * @param[out] KorQ Device pointer to the resulting Key or Query vector (size height). Must be zero-initialized before calling.
 * @param[in] dim The embedding dimension (columns of matrix, size of tokenEmbed).
 * @param[in] height The number of rows in the key/query matrix (size of KorQ).
 */
__device__ void cuComputeKorQ(const float* tokenEmbed, const float* matrix, float* KorQ, int dim, int height) {
    // This computes KorQ[i] = dot(tokenEmbed, matrix_row_i)
    for (int i = 0; i < height; ++i) {
        const float* matrix_row_i = matrix + i * dim;
        float dot_product = 0.0f; // Accumulate dot product locally
        for (int j = 0; j < dim; ++j) {
            dot_product += tokenEmbed[j] * matrix_row_i[j];
        }
        KorQ[i] = dot_product; // Assign the computed dot product
    }
}

/**
 * @brief CUDA device function to compute the dot product of two vectors.
 * @param[in] vec1 Device pointer to the first vector.
 * @param[in] vec2 Device pointer to the second vector.
 * @param[in] dim The dimension (number of elements) of the vectors.
 * @return The scalar dot product of vec1 and vec2.
 */
__device__ float compute_dot_product(const float* vec1, const float* vec2, int dim) {
    float dot_product = 0.0f;
    for (int k = 0; k < dim; ++k) {
        dot_product += vec1[k] * vec2[k];
    }
    return dot_product;
}


/**
 * @brief CUDA device function to compute the quadratic form vec1 * matrix * vec2^T.
 * @param[in] vec1 Device pointer to the first vector (treated as a row vector, size dim).
 * @param[in] vec2 Device pointer to the second vector (treated as a column vector, size dim).
 * @param[in] matrix Device pointer to the matrix (row-major, dim x dim).
 * @param[in] dim The dimension of the vectors and the square matrix.
 * @return The scalar result of vec1 * matrix * vec2^T.
 */
__device__ float compute_dot_product(const float* vec1, const float* vec2, const float* matrix, int dim)
{
    float final_dot_product = 0.0f;
    // This computes (vec1 * matrix) * vec2^T
    for (int i = 0; i < dim; ++i) { // Iterate over rows of matrix (and elements of vec2)
        float inner_sum = 0.0f; // Represents element i of (vec1 * matrix)
        const float* matrix_row_i = matrix + i * dim;
        // Compute dot product of vec1 with i-th row of matrix
        for (int j = 0; j < dim; ++j) {
            inner_sum += vec1[j] * matrix_row_i[j];
        }
        // Multiply the result by the corresponding element of vec2 and accumulate
        final_dot_product += inner_sum * vec2[i];
    }
    return final_dot_product;
}

/**
 * @brief CUDA device function to compute the predicted token index by finding the highest dot product
 *        between a vector (EH) and rows of an embedding matrix.
 * @param[in] EH Device pointer to the horizontal retention vector (size dim).
 * @param[in] embeddings Device pointer to the token embeddings matrix (row-major: voc x dim).
 * @param[in] dim The embedding dimension (size of EH and columns of embeddings).
 * @param[in] voc The vocabulary size (number of rows in embeddings).
 * @return The index of the token embedding with the highest dot product. Returns -1 if voc <= 0 or embeddings is null.
 * @note Assumes that the case of "all dot products being exactly equal" is handled implicitly by returning the first max index found.
 * @note Assumes FLT_MAX is available (usually via <cfloat> or CUDA includes).
 */
__device__ int compute_prediction(const float* EH, const float* embeddings, int dim, int voc) {
    if (voc <= 0 || embeddings == nullptr) {
        return -1; // Handle invalid input
    }
    float max_dot_product = -FLT_MAX; // Initialize with the smallest possible float value
    int predicted_index = 0; // Default to the first token

    for (int i = 0; i < voc; ++i) {
        const float* current_embedding_row = embeddings + i * dim;
        float current_dot_product = compute_dot_product(EH, current_embedding_row, dim);

        if (current_dot_product > max_dot_product) {
            max_dot_product = current_dot_product;
            predicted_index = i;
        }
    }
    return predicted_index;
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

        float dot_product = compute_dot_product(q_vec, k_vec, embedding_dim);

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

        float dot_product = compute_dot_product(q_vec, k_vec, embedding_dim);

        int kdotq_index = i * kdotq_width + j;
        d_kdotq[kdotq_index] = dot_product * inv_scaling;
    }
}

/**------------------------------------INFERENCE------------------------------------**/

/**
 * @brief CUDA kernel for Block 1 SELF-ATTENTION KdotQ calculation during inference.
 *        Computes KdotQ[i][j] = (tokenEmbed[i] * M * tokenEmbed[j]^T) / SCALING for j <= i,
 *        where 'i' corresponds to new prompt tokens and 'j' spans the full context.
 * @param[out] d_kdotq         Device pointer to the output KdotQ matrix (row-major, potentially sparse).
 * @param[in]  d_tokenEmbed    Device pointer to token embeddings for the *entire* context (row-major, context_len x embedding_dim).
 * @param[in]  d_M             Device pointer to the precomputed QK' matrix M (row-major, embedding_dim x embedding_dim).
 * @param[in]  prompt_start_index Index of the first token of the new prompt within d_tokenEmbed.
 * @param[in]  prompt_len      Number of tokens in the new prompt.
 * @param[in]  context_len     Total number of tokens currently in the context (prompt_start_index + prompt_len).
 * @param[in]  kdotq_width     The total width (number of columns) of the d_kdotq buffer (usually context_len).
 * @param[in]  embedding_dim   The dimension of token vectors and the matrix M.
 * @param[in]  inv_scaling     The inverse scaling factor (1.0f / sqrt(embedding_dim)).
 */
__global__ void kernelKdotQ_Block1_Self_Inference(float* d_kdotq, const float* d_tokenEmbed, const float* d_M,
    int prompt_start_index, int prompt_len, int context_len, int kdotq_width, int embedding_dim, float inv_scaling)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;        // Global key index (column) spanning full context
    int i_offset = blockIdx.y * blockDim.y + threadIdx.y; // Query index offset relative to the start of the prompt
    int i = prompt_start_index + i_offset;                // Global query index (row) within the prompt range

    // Check boundaries and apply causal mask (j <= i)
    // Only compute rows 'i' corresponding to the new prompt.
    if (i_offset < prompt_len && j < context_len && j <= i) {
        const float* q_vec = d_tokenEmbed + i * embedding_dim; // Query vector from prompt
        const float* k_vec = d_tokenEmbed + j * embedding_dim; // Key vector from full context

        // Compute quadratic form: q_vec * M * k_vec^T
        float dot_product = compute_dot_product(q_vec, k_vec, d_M, embedding_dim);

        int kdotq_index = i * kdotq_width + j;
        d_kdotq[kdotq_index] = dot_product * inv_scaling;
    }
}


/**
 * @brief CUDA kernel for Block 1 CROSS-ATTENTION KdotQ calculation during inference.
 *        Computes KdotQ[i][j] = (tokenEmbed[i] * M * tokenEmbed[j]^T) / SCALING for all j,
 *        where 'i' corresponds to new prompt tokens and 'j' spans the full context.
 * @param[out] d_kdotq         Device pointer to the output KdotQ matrix (row-major, potentially sparse).
 * @param[in]  d_tokenEmbed    Device pointer to token embeddings for the *entire* context (row-major, context_len x embedding_dim).
 * @param[in]  d_M             Device pointer to the precomputed QK' matrix M (row-major, embedding_dim x embedding_dim).
 * @param[in]  prompt_start_index Index of the first token of the new prompt within d_tokenEmbed.
 * @param[in]  prompt_len      Number of tokens in the new prompt.
 * @param[in]  context_len     Total number of tokens currently in the context (prompt_start_index + prompt_len).
 * @param[in]  kdotq_width     The total width (number of columns) of the d_kdotq buffer (usually context_len).
 * @param[in]  embedding_dim   The dimension of token vectors and the matrix M.
 * @param[in]  inv_scaling     The inverse scaling factor (1.0f / sqrt(embedding_dim)).
 */
__global__ void kernelKdotQ_Block1_Cross_Inference(float* d_kdotq, const float* d_tokenEmbed, const float* d_M,
    int prompt_start_index, int prompt_len, int context_len, int kdotq_width, int embedding_dim, float inv_scaling)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;        // Global key index (column) spanning full context
    int i_offset = blockIdx.y * blockDim.y + threadIdx.y; // Query index offset relative to the start of the prompt
    int i = prompt_start_index + i_offset;                // Global query index (row) within the prompt range

    // Check boundaries (no causal mask for cross-attention)
    // Only compute rows 'i' corresponding to the new prompt.
    if (i_offset < prompt_len && j < context_len) {
        const float* q_vec = d_tokenEmbed + i * embedding_dim; // Query vector from prompt
        const float* k_vec = d_tokenEmbed + j * embedding_dim; // Key vector from full context

        // Compute quadratic form: q_vec * M * k_vec^T
        float dot_product = compute_dot_product(q_vec, k_vec, d_M, embedding_dim);

        int kdotq_index = i * kdotq_width + j;
        d_kdotq[kdotq_index] = dot_product * inv_scaling;
    }
}


/**
 * @brief CUDA kernel for Block N (N > 1) SELF-ATTENTION KdotQ calculation during inference.
 *        Computes KdotQ[i][j] = (tokForBlock[i] * M * EVp[j]^T) / SCALING for j <= i,
 *        where 'i' corresponds to new prompt tokens within the block's window,
 *        and 'j' spans the relevant context (from previous block's EV) within the block's window.
 * @param[out] d_kdotq         Device pointer to the output KdotQ matrix for this block's window (row-major).
 * @param[in]  d_tokForBlock   Device pointer to token embeddings relevant to this block's window (row-major, context_len_in_block x embedding_dim).
 * @param[in]  d_EVp           Device pointer to the previous block's output vectors (EV') relevant to this block's window (row-major, context_len_in_block x embedding_dim).
 * @param[in]  d_M             Device pointer to the precomputed QK' matrix M (row-major, embedding_dim x embedding_dim).
 * @param[in]  prompt_start_index_in_block Index of the first prompt token *within this block's window*.
 * @param[in]  prompt_len      Number of tokens in the new prompt.
 * @param[in]  context_len_in_block Total number of relevant tokens in this block's window.
 * @param[in]  kdotq_width     The total width (number of columns) of the d_kdotq buffer (usually context_len_in_block).
 * @param[in]  embedding_dim   The dimension of token vectors and the matrix M.
 * @param[in]  inv_scaling     The inverse scaling factor (1.0f / sqrt(embedding_dim)).
 */
__global__ void kernelKdotQ_BlockN_Self_Inference(float* d_kdotq, const float* d_tokForBlock, const float* d_EVp,
    const float* d_M, int prompt_start_index_in_block, int prompt_len, int context_len_in_block, int kdotq_width,
    int embedding_dim, float inv_scaling)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;        // Key index (column) within the block's window
    int i_offset = blockIdx.y * blockDim.y + threadIdx.y; // Query index offset relative to the start of the prompt
    int i = prompt_start_index_in_block + i_offset;       // Query index (row) within the block's window

    // Check boundaries and apply causal mask (j <= i) within the block window
    // Only compute rows 'i' corresponding to the new prompt within this block.
    if (i_offset < prompt_len && j < context_len_in_block && j <= i) {
        const float* q_vec = d_tokForBlock + i * embedding_dim; // Query vector from block's tokens
        const float* k_vec = d_EVp + j * embedding_dim;         // Key vector from previous block's EV

        // Compute quadratic form: q_vec * M * k_vec^T
        float dot_product = compute_dot_product(q_vec, k_vec, d_M, embedding_dim);

        int kdotq_index = i * kdotq_width + j;
        d_kdotq[kdotq_index] = dot_product * inv_scaling;
    }
}


/**
 * @brief CUDA kernel for Block N (N > 1) CROSS-ATTENTION KdotQ calculation during inference.
 *        Computes KdotQ[i][j] = (tokForBlock[i] * M * EVp[j]^T) / SCALING for all j,
 *        where 'i' corresponds to new prompt tokens within the block's window,
 *        and 'j' spans the relevant context (from previous block's EV) within the block's window.
 * @param[out] d_kdotq         Device pointer to the output KdotQ matrix for this block's window (row-major).
 * @param[in]  d_tokForBlock   Device pointer to token embeddings relevant to this block's window (row-major, context_len_in_block x embedding_dim).
 * @param[in]  d_EVp           Device pointer to the previous block's output vectors (EV') relevant to this block's window (row-major, context_len_in_block x embedding_dim).
 * @param[in]  d_M             Device pointer to the precomputed QK' matrix M (row-major, embedding_dim x embedding_dim).
 * @param[in]  prompt_start_index_in_block Index of the first prompt token *within this block's window*.
 * @param[in]  prompt_len      Number of tokens in the new prompt.
 * @param[in]  context_len_in_block Total number of relevant tokens in this block's window.
 * @param[in]  kdotq_width     The total width (number of columns) of the d_kdotq buffer (usually context_len_in_block).
 * @param[in]  embedding_dim   The dimension of token vectors and the matrix M.
 * @param[in]  inv_scaling     The inverse scaling factor (1.0f / sqrt(embedding_dim)).
 */
__global__ void kernelKdotQ_BlockN_Cross_Inference(float* d_kdotq, const float* d_tokForBlock, const float* d_EVp,
    const float* d_M, int prompt_start_index_in_block, int prompt_len, int context_len_in_block, int kdotq_width,
    int embedding_dim, float inv_scaling)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;        // Key index (column) within the block's window
    int i_offset = blockIdx.y * blockDim.y + threadIdx.y; // Query index offset relative to the start of the prompt
    int i = prompt_start_index_in_block + i_offset;       // Query index (row) within the block's window

    // Check boundaries (no causal mask for cross-attention)
    // Only compute rows 'i' corresponding to the new prompt within this block.
    if (i_offset < prompt_len && j < context_len_in_block) {
        const float* q_vec = d_tokForBlock + i * embedding_dim; // Query vector from block's tokens
        const float* k_vec = d_EVp + j * embedding_dim;         // Key vector from previous block's EV

        // Compute quadratic form: q_vec * M * k_vec^T
        float dot_product = compute_dot_product(q_vec, k_vec, d_M, embedding_dim);

        int kdotq_index = i * kdotq_width + j;
        d_kdotq[kdotq_index] = dot_product * inv_scaling;
    }
}

/**
 * @brief CUDA kernel to compute row sums and column sums of the attention head matrix,
 *        applying causal masking for self-attention if specified.
 *        Row sum k[i] = sum_{j=0}^{limit-1}(head[i][j])
 *        Col sum l[i] = sum_{j=0}^{limit-1}(head[j][i])
 *        where limit = isSelfAttention ? i : num_tokens.
 * @param[in] d_head Device pointer to the input head matrix (row-major, num_tokens x num_tokens).
 * @param[out] d_row_sums Device pointer to store the computed row sums (size num_tokens).
 * @param[out] d_col_sums Device pointer to store the computed column sums (size num_tokens).
 * @param[in] num_tokens The dimension of the square head matrix (number of tokens).
 * @param[in] isSelfAttention Boolean flag; if true, applies causal masking (j < limit = i).
 */
__global__ void computeHeadSumsMaskedKernel(const float* d_head, float* d_row_sums, float* d_col_sums, int num_tokens,
    bool isSelfAttention)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Parallelize over token index 'i'

    if (i < num_tokens) {
        float row_sum_k = 0.0f;
        float col_sum_l = 0.0f;

        // Determine the upper limit for summation based on attention type
        // For self-attention, sum up to (but not including) the current token index 'i' for causal masking?
        // The C++ comment says limit = isSelfAttention ? i : num_tokens. Let's assume it means sum up to j=i-1.
        // Re-reading C++ comment: limit = isSelfAttention ? i : num_tokens. This means sum up to j=i-1 for self-attention.
        // Let's implement the comment's logic: sum up to j = limit-1.
        int limit = isSelfAttention ? i + 1 : num_tokens; // If self-attention, limit is i+1 (sum j=0 to i)
        if (limit > num_tokens) limit = num_tokens; // Ensure limit doesn't exceed bounds

        // Calculate row sum (k) for token i: sum head[i][j] for j < limit
        for (int j = 0; j < limit; ++j) {
             // Apply self-attention mask: only sum if j <= i
             if (!isSelfAttention || j <= i) {
                row_sum_k += d_head[i * num_tokens + j];
             }
        }

        // Calculate column sum (l) for token i: sum head[j][i] for j < limit
        for (int j = 0; j < limit; ++j) {
             // Apply self-attention mask: only sum if i <= j (for head[j][i], this means j >= i)
             // Wait, the C++ comment implies the same limit 'j < limit' for both sums. Let's stick to that.
             // This means col_sum_l[i] = sum_{j=0}^{limit-1} head[j][i]
             // If self-attention, limit = i+1, so col_sum_l[i] = sum_{j=0}^{i} head[j][i]
             // This seems consistent with typical attention backprop needs.
             col_sum_l += d_head[j * num_tokens + i];
        }


        d_row_sums[i] = row_sum_k;
        d_col_sums[i] = col_sum_l;
    }
}


/**
 * @brief CUDA kernel to accumulate weighted Key and Query vectors based on head row/column sums.
 *        Computes:
 *        d_dh_accum += sum_i (d_row_sums[i] * d_K[i])
 *        d_dv_accum += sum_i (d_col_sums[i] * d_Q[i])
 *        Uses atomicAdd for safe accumulation across threads working on the same h_idx.
 * @param[in] d_row_sums Device pointer to the precomputed row sums (size num_tokens).
 * @param[in] d_col_sums Device pointer to the precomputed column sums (size num_tokens).
 * @param[in] d_K Device pointer to the Key matrix (row-major, num_tokens x h_dim).
 * @param[in] d_Q Device pointer to the Query matrix (row-major, num_tokens x h_dim).
 * @param[in,out] d_dh_accum Device pointer to the accumulated dh vector (size h_dim). MUST be zero-initialized before kernel launch.
 * @param[in,out] d_dv_accum Device pointer to the accumulated dv vector (size h_dim). MUST be zero-initialized before kernel launch.
 * @param[in] num_tokens The number of tokens (rows in K/Q, size of sum vectors).
 * @param[in] h_dim The dimension of the Key/Query vectors (e.g., MATHEIGHTS).
 */
__global__ void accumulateWeightedVectorsKernel(const float* d_row_sums, const float* d_col_sums,
                                                const float* d_K, const float* d_Q, float* d_dh_accum,
                                                float* d_dv_accum, int num_tokens, int h_dim)
{
    int h_idx = blockIdx.x * blockDim.x + threadIdx.x; // Parallelize over the h_dim dimension

    if (h_idx < h_dim) {
        float total_dh_for_h_idx = 0.0f;
        float total_dv_for_h_idx = 0.0f;

        // Iterate through each token i
        for (int i = 0; i < num_tokens; ++i) {
            // Access K[i][h_idx] and Q[i][h_idx] using row-major indexing
            float k_i_h = d_K[i * h_dim + h_idx];
            float q_i_h = d_Q[i * h_dim + h_idx];

            // Accumulate dh contribution for this h_idx
            total_dh_for_h_idx += d_row_sums[i] * k_i_h;
            // Accumulate dv contribution for this h_idx
            total_dv_for_h_idx += d_col_sums[i] * q_i_h;
        }

        // Atomically add the computed sums for this h_idx to the global accumulators
        atomicAdd(&d_dh_accum[h_idx], total_dh_for_h_idx);
        atomicAdd(&d_dv_accum[h_idx], total_dv_for_h_idx);
    }
}

/**------------------------------------BACKPROP------------------------------------**/

/**
 * @brief CUDA kernel to compute gradients w.r.t. MLP inputs (dh, dv) for the *first head* scenario.
 *        Extracts gradients from the first column (input neuron 0) of the first layer's gradient weights (gweights[0]).
 *        grad_dh[i] = hor_gweights[0][i][0]
 *        grad_dv[i] = ver_gweights[0][i][0]
 * @param[in] d_hor_gweights0 Device pointer to the horizontal MLP's first layer gradient weights (row-major, embedding_dim x embedding_dim). Can be null.
 * @param[in] d_ver_gweights0 Device pointer to the vertical MLP's first layer gradient weights (row-major, embedding_dim x embedding_dim). Can be null.
 * @param[out] grad_dh Device pointer to store the computed gradient w.r.t. horizontal input (size embedding_dim).
 * @param[out] grad_dv Device pointer to store the computed gradient w.r.t. vertical input (size embedding_dim).
 * @param[in] embedding_dim The dimension of the embedding and MLP layers.
 */
__global__ void kernelComputeGradDhDv_1stHead(const float* d_hor_gweights0, const float* d_ver_gweights0,
    float* grad_dh, float* grad_dv, int embedding_dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Corresponds to the output neuron index 'i' of the first layer

    if (i < embedding_dim) {
        // Calculate flat index for gweights[0][i][0] assuming row-major [output_neuron][input_neuron]
        int gweight_idx = i * embedding_dim + 0; // Index for the weight connecting input 0 to output i

        if (d_hor_gweights0 != nullptr) {
            grad_dh[i] = d_hor_gweights0[gweight_idx];
        }
        else {
            grad_dh[i] = 0.0f; // Set to zero if horizontal MLP path is not active/provided
        }

        if (d_ver_gweights0 != nullptr) {
            grad_dv[i] = d_ver_gweights0[gweight_idx];
        }
        else {
            grad_dv[i] = 0.0f; // Set to zero if vertical MLP path is not active/provided
        }
    }
}


/**
 * @brief CUDA kernel for Step 1 of `cuBackward(expected)`: Compute initial gradients w.r.t. EH and a scaled version for EV.
 *        grad_eh = 2 * (eh - expected_h)
 *        grad_ev_scaled = grad_eh * 0.1
 * @param[in] eh Device pointer to the horizontal embedding vector EH (size embedding_dim).
 * @param[in] expected_h Device pointer to the target horizontal embedding vector (size embedding_dim).
 * @param[out] grad_eh Device pointer to store the gradient w.r.t. EH (size embedding_dim).
 * @param[out] grad_ev_scaled Device pointer to store the scaled gradient for the EV path (size embedding_dim).
 * @param[in] embedding_dim The dimension of the embedding vectors.
 */
__global__ void kernelComputeGradientsEH_EV(const float* eh, const float* expected_h,
                        float* grad_eh, float* grad_ev_scaled, int embedding_dim) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < embedding_dim) {
        // Assuming Mean Squared Error loss: dL/d(eh) = d/d(eh) [(eh - expected_h)^2] = 2 * (eh - expected_h)
        float grad = 2.0f * (eh[idx] - expected_h[idx]);
        grad_eh[idx] = grad;
        // Apply scaling factor for the vertical path gradient (as seen in C++ code)
        grad_ev_scaled[idx] = grad * 0.1f;
    }
}


/**
 * @brief CUDA kernel for Step 3 of `cuBackward(expected)`: Compute gradients w.r.t. MLP inputs (dh, dv)
 *        by summing the first layer's gradient weights across the input dimension.
 *        grad_dh[i] = sum_j (hor_gweights[0][j][i])  <- Note: Indexing seems reversed vs kernelComputeGradDhDv_1stHead
 *        grad_dv[i] = sum_j (ver_gweights[0][j][i])  <- Note: Indexing seems reversed vs kernelComputeGradDhDv_1stHead
 *        Assuming gweights are stored [output_neuron][input_neuron] (row-major).
 *        The kernel computes sum over input neurons 'j' for a given output neuron 'i'.
 * @param[in] d_hor_gweights0 Device pointer to the horizontal MLP's first layer gradient weights (row-major, embedding_dim x embedding_dim).
 * @param[in] d_ver_gweights0 Device pointer to the vertical MLP's first layer gradient weights (row-major, embedding_dim x embedding_dim).
 * @param[out] grad_dh Device pointer to store the computed gradient w.r.t. horizontal input (size embedding_dim).
 * @param[out] grad_dv Device pointer to store the computed gradient w.r.t. vertical input (size embedding_dim).
 * @param[in] embedding_dim The dimension of the embedding and MLP layers.
 */
__global__ void kernelComputeGradDhDv(const float* d_hor_gweights0, const float* d_ver_gweights0,
                                      float* grad_dh, float* grad_dv,
                                      int embedding_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Corresponds to the output neuron index 'i'

    if (i < embedding_dim) {
        float sum_dh = 0.0f;
        float sum_dv = 0.0f;
        // Sum gweights[0][i][j] over j (input dimension)
        // Flat index for gweights[0][i][j] is i * embedding_dim + j
        for (int j = 0; j < embedding_dim; ++j) {
            int gweight_idx = i * embedding_dim + j;
            sum_dh += d_hor_gweights0[gweight_idx];
            sum_dv += d_ver_gweights0[gweight_idx];
        }
        grad_dh[i] = sum_dh;
        grad_dv[i] = sum_dv;
    }
}


/**
 * @brief CUDA kernel for Step 4 of `cuBackward(expected)`: Compute intermediate values pre_MH and pre_MV.
 *        pre_mh[h] = sum_i ( sum_j(head[i][j]) * K[i][h] )
 *        pre_mv[h] = sum_i ( sum_j(head[j][i]) * Q[i][h] )
 * @param[in] head Device pointer to the attention head matrix (row-major, token_count x token_count).
 * @param[in] k Device pointer to the Key matrix K (row-major, token_count x mat_heights).
 * @param[in] q Device pointer to the Query matrix Q (row-major, token_count x mat_heights).
 * @param[out] pre_mh Device pointer to store the pre_MH vector (size mat_heights).
 * @param[out] pre_mv Device pointer to store the pre_MV vector (size mat_heights).
 * @param[in] token_count The number of tokens (dimension of head matrix, rows of K/Q).
 * @param[in] mat_heights The height dimension (columns of K/Q, size of pre_mh/pre_mv).
 * @note This implementation involves redundant calculations (sums over head). Consider optimizing with separate reduction kernels if performance is critical.
 */
__global__ void kernelComputePreMH_MV(const float* head, const float* k, const float* q,
                                      float* pre_mh, float* pre_mv,
                                      int token_count, int mat_heights) {
    int h = blockIdx.x * blockDim.x + threadIdx.x; // Parallelize over the mat_heights dimension

    if (h < mat_heights) {
        float mh_val_h = 0.0f;
        float mv_val_h = 0.0f;

        // Iterate over tokens 'i'
        for (int i = 0; i < token_count; ++i) {
            float sum_head_row_i = 0.0f; // Sum of row i of head
            float sum_head_col_i = 0.0f; // Sum of column i of head

            // Calculate sum of row i and column i of the head matrix
            for (int j = 0; j < token_count; ++j) {
                sum_head_row_i += head[i * token_count + j]; // head[i][j]
                sum_head_col_i += head[j * token_count + i]; // head[j][i]
            }

            // Access K[i][h] and Q[i][h] using row-major indexing
            float k_ih = k[i * mat_heights + h];
            float q_ih = q[i * mat_heights + h];

            // Accumulate results for dimension h
            mh_val_h += sum_head_row_i * k_ih;
            mv_val_h += sum_head_col_i * q_ih;
        }
        pre_mh[h] = mh_val_h;
        pre_mv[h] = mv_val_h;
    }
}


/**
 * @brief CUDA kernel for Step 4 of `cuBackward(expected)`: Compute gradients grad_MH and grad_MV via outer product.
 *        grad_mh[h][d] = pre_mh[h] * grad_dh[d]
 *        grad_mv[h][d] = pre_mv[h] * grad_dv[d]
 * @param[in] pre_mh Device pointer to the pre_MH vector (size mat_heights).
 * @param[in] pre_mv Device pointer to the pre_MV vector (size mat_heights).
 * @param[in] grad_dh Device pointer to the gradient w.r.t. horizontal input (size embedding_dim).
 * @param[in] grad_dv Device pointer to the gradient w.r.t. vertical input (size embedding_dim).
 * @param[out] grad_mh Device pointer to store the gradient w.r.t. MH (row-major, mat_heights x embedding_dim).
 * @param[out] grad_mv Device pointer to store the gradient w.r.t. MV (row-major, mat_heights x embedding_dim).
 * @param[in] mat_heights The height dimension.
 * @param[in] embedding_dim The embedding dimension.
 */
__global__ void kernelComputeGradMH_MV(const float* pre_mh, const float* pre_mv,
                                       const float* grad_dh, const float* grad_dv,
                                       float* grad_mh, float* grad_mv,
                                       int mat_heights, int embedding_dim) {
    int h = blockIdx.y * blockDim.y + threadIdx.y; // Row index (mat_heights)
    int d = blockIdx.x * blockDim.x + threadIdx.x; // Column index (embedding_dim)

    if (h < mat_heights && d < embedding_dim) {
        int idx = h * embedding_dim + d; // Flat index for row-major matrix
        grad_mh[idx] = pre_mh[h] * grad_dh[d];
        grad_mv[idx] = pre_mv[h] * grad_dv[d];
    }
}


/**
 * @brief CUDA kernel for Step 5 of `cuBackward(expected)`: Compute the gradient w.r.t. the attention head matrix.
 *        grad_head[i][j] = (sum_d (sum_h K[i][h]*MH[h][d]) * grad_dh[d]) + (sum_d (sum_h Q[j][h]*MV[h][d]) * grad_dv[d])
 *        This is equivalent to: grad_head[i][j] = dot(K[i]*MH, grad_dh) + dot(Q[j]*MV, grad_dv)
 * @param[in] k Device pointer to the Key matrix K (row-major, token_count x mat_heights).
 * @param[in] q Device pointer to the Query matrix Q (row-major, token_count x mat_heights).
 * @param[in] mh_a Device pointer to the MH matrix (row-major, mat_heights x embedding_dim).
 * @param[in] mv_a Device pointer to the MV matrix (row-major, mat_heights x embedding_dim).
 * @param[in] grad_dh Device pointer to the gradient w.r.t. horizontal input (size embedding_dim).
 * @param[in] grad_dv Device pointer to the gradient w.r.t. vertical input (size embedding_dim).
 * @param[out] grad_head Device pointer to store the gradient w.r.t. the head matrix (row-major, token_count x token_count).
 * @param[in] token_count The number of tokens.
 * @param[in] mat_heights The height dimension.
 * @param[in] embedding_dim The embedding dimension.
 */
__global__ void kernelComputeGradHead(const float* k, const float* q,
                                      const float* mh_a, const float* mv_a,
                                      const float* grad_dh, const float* grad_dv,
                                      float* grad_head,
                                      int token_count, int mat_heights, int embedding_dim) {
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row index for grad_head (token i)
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Column index for grad_head (token j)

    if (i < token_count && j < token_count) {
        float grad_dh_term_ij = 0.0f; // Contribution from the dh path for grad_head[i][j]
        float grad_dv_term_ij = 0.0f; // Contribution from the dv path for grad_head[i][j]

        // Calculate dot(K[i]*MH, grad_dh) and dot(Q[j]*MV, grad_dv)
        for (int d = 0; d < embedding_dim; ++d) {
            float k_mh_id = 0.0f; // K[i] * MH[:, d]
            float q_mv_jd = 0.0f; // Q[j] * MV[:, d]

            // Compute the dot products involving K[i]/Q[j] and columns of MH/MV
            for (int h = 0; h < mat_heights; ++h) {
                // Access K[i][h], Q[j][h], MH[h][d], MV[h][d] using row-major indexing
                k_mh_id += k[i * mat_heights + h] * mh_a[h * embedding_dim + d];
                q_mv_jd += q[j * mat_heights + h] * mv_a[h * embedding_dim + d];
            }
            // Accumulate the contribution for dimension d
            grad_dh_term_ij += k_mh_id * grad_dh[d];
            grad_dv_term_ij += q_mv_jd * grad_dv[d];
        }

        // Store the total gradient for head[i][j]
        grad_head[i * token_count + j] = grad_dh_term_ij + grad_dv_term_ij;
    }
}


/**
 * @brief CUDA kernel for Step 6 of `cuBackward(expected)`: Compute the gradient w.r.t. KdotQ (before scaling).
 *        Applies the derivative of the LOTA function element-wise.
 *        grad_kdotq = grad_head * lota_derivative / scaling_factor
 * @param[in] grad_head Device pointer to the gradient w.r.t. the attention head (size `size`).
 * @param[in] lota_derivative Device pointer to the precomputed derivative of the LOTA function (size `size`).
 * @param[out] grad_kdotq Device pointer to store the gradient w.r.t. KdotQ (size `size`).
 * @param[in] scaling_factor The scaling factor used in the forward pass (e.g., sqrt(embedding_dim)).
 * @param[in] size The total number of elements in the KdotQ/head matrices (token_count * token_count).
 */
__global__ void kernelComputeGradKdotQ_LOTA(const float* grad_head, const float* lota_derivative,
                                           float* grad_kdotq, float scaling_factor, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Apply the chain rule: dL/dKdotQ = dL/dHead * dHead/dKdotQ
        // dHead/dKdotQ includes the LOTA derivative and the inverse scaling factor
        grad_kdotq[idx] = grad_head[idx] * lota_derivative[idx] / scaling_factor;
    }
}


/**
 * @brief CUDA kernel for Step 7 of `cuBackward(expected)`: Compute gradients w.r.t. K and Q matrices.
 *        grad_K = grad_KdotQ * Q^T
 *        grad_Q = K^T * grad_KdotQ
 * @param[in] grad_kdotq Device pointer to the gradient w.r.t. KdotQ (row-major, token_count x token_count).
 * @param[in] k Device pointer to the Key matrix K (row-major, token_count x mat_heights).
 * @param[in] q Device pointer to the Query matrix Q (row-major, token_count x mat_heights).
 * @param[out] grad_k Device pointer to store the gradient w.r.t. K (row-major, token_count x mat_heights).
 * @param[out] grad_q Device pointer to store the gradient w.r.t. Q (row-major, token_count x mat_heights).
 * @param[in] token_count The number of tokens.
 * @param[in] mat_heights The height dimension.
 */
__global__ void kernelComputeGradK_Q(const float* grad_kdotq, const float* k, const float* q,
                                     float* grad_k, float* grad_q,
                                     int token_count, int mat_heights) {
    // Each thread computes one element of grad_k and one element of grad_q
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row index for grad_k (token i)
    int h = blockIdx.x * blockDim.x + threadIdx.x; // Column index for grad_k/grad_q (height h)

    if (i < token_count && h < mat_heights) {
        float sum_for_grad_k_ih = 0.0f; // Accumulator for grad_K[i][h]
        float sum_for_grad_q_ih = 0.0f; // Accumulator for grad_Q[i][h] (Note: 'i' here corresponds to 'j' in formula)

        // Calculate grad_K[i][h] = sum_j (grad_KdotQ[i][j] * Q[j][h])
        for (int j = 0; j < token_count; ++j) {
            sum_for_grad_k_ih += grad_kdotq[i * token_count + j] * q[j * mat_heights + h];
        }

        // Calculate grad_Q[i][h] = sum_j (K[j][h] * grad_KdotQ[j][i]) (Note: 'i' is column index of grad_KdotQ)
        // Equivalent to: grad_Q[token_idx][h] = sum_j (K[j][h] * grad_KdotQ[j][token_idx]) where token_idx = i
        for (int j = 0; j < token_count; ++j) {
            sum_for_grad_q_ih += k[j * mat_heights + h] * grad_kdotq[j * token_count + i];
        }

        // Store results using row-major indexing
        grad_k[i * mat_heights + h] = sum_for_grad_k_ih;
        grad_q[i * mat_heights + h] = sum_for_grad_q_ih;
    }
}


/**
 * @brief CUDA kernel for Step 8 of `cuBackward(expected)`: Compute gradients w.r.t. MK and MQ.
 *        grad_MK[h][d] = sum_i (grad_K[i][h] * X[i][d])
 *        grad_MQ[h][d] = sum_i (grad_Q[i][h] * X[i][d])
 * @param[in] grad_k Device pointer to the gradient w.r.t. K (row-major, token_count x mat_heights).
 * @param[in] grad_q Device pointer to the gradient w.r.t. Q (row-major, token_count x mat_heights).
 * @param[in] k Device pointer to the Key matrix (row-major, token_count x embedding_dim). **Likely should be original token embeddings X.**
 * @param[in] q Device pointer to the Query matrix (row-major, token_count x embedding_dim). **Likely should be original token embeddings X.**
 * @param[out] grad_mk Device pointer to store the gradient w.r.t. MK (row-major, mat_heights x embedding_dim).
 * @param[out] grad_mq Device pointer to store the gradient w.r.t. MQ (row-major, mat_heights x embedding_dim).
 * @param[in] token_count The number of tokens.
 * @param[in] mat_heights The height dimension.
 * @param[in] embedding_dim The embedding dimension.
 * @note There's a potential inconsistency in the source C++ regarding the dimensions/source of K and Q used in this step versus others.
 *       This kernel assumes `k` and `q` inputs are `token_count x embedding_dim` (likely the original token embeddings or equivalent).
 *       If `k` and `q` passed are `token_count x mat_heights`, this kernel's logic is incorrect based on the C++ formula.
 *       Using `kernelComputeGradMK_MQ_Simplified` might be preferred if original embeddings are available.
 */
__global__ void kernelComputeGradMK_MQ(const float* grad_k, const float* grad_q,
                                       const float* k, const float* q, // Assumed token_count x embedding_dim
                                       float* grad_mk, float* grad_mq,
                                       int token_count, int mat_heights, int embedding_dim) {
    int h = blockIdx.y * blockDim.y + threadIdx.y; // Row index for grad_mk/grad_mq (height h)
    int d = blockIdx.x * blockDim.x + threadIdx.x; // Column index for grad_mk/grad_mq (dimension d)

    if (h < mat_heights && d < embedding_dim) {
        float sum_mk_hd = 0.0f;
        float sum_mq_hd = 0.0f;

        // Sum over tokens 'i'
        for (int i = 0; i < token_count; ++i) {
            // Access grad_K[i][h], grad_Q[i][h] (from token_count x mat_heights matrices)
            float grad_k_ih = grad_k[i * mat_heights + h];
            float grad_q_ih = grad_q[i * mat_heights + h];

            // Access K[i][d], Q[i][d] (from assumed token_count x embedding_dim matrices)
            float k_id = k[i * embedding_dim + d];
            float q_id = q[i * embedding_dim + d];

            sum_mk_hd += grad_k_ih * k_id;
            sum_mq_hd += grad_q_ih * q_id;
        }

        // Store results using row-major indexing
        grad_mk[h * embedding_dim + d] = sum_mk_hd;
        grad_mq[h * embedding_dim + d] = sum_mq_hd;
    }
}


/**
 * @brief CUDA kernel for Steps 9 & 10 of `cuBackward(expected)`: Update attention weight matrices (MH, MV, MQ, MK) and embeddings (EH, EV).
 *        Applies gradients using a simple gradient descent step: weight -= learning_rate * gradient.
 *        EV update uses the scaled gradient `grad_ev_scaled` applied uniformly across the context window dimension.
 * @param[in,out] mh_a Device pointer to the MH matrix (row-major, mat_heights x embedding_dim). Updated in place.
 * @param[in,out] mv_a Device pointer to the MV matrix (row-major, mat_heights x embedding_dim). Updated in place.
 * @param[in,out] mq_a Device pointer to the MQ matrix (row-major, mat_heights x embedding_dim). Updated in place.
 * @param[in,out] mk_a Device pointer to the MK matrix (row-major, mat_heights x embedding_dim). Updated in place.
 * @param[in,out] eh Device pointer to the EH vector (size embedding_dim). Updated in place.
 * @param[in,out] ev Device pointer to the EV matrix (row-major, context_win x embedding_dim). Updated in place.
 * @param[in] grad_mh Device pointer to the gradient w.r.t. MH.
 * @param[in] grad_mv Device pointer to the gradient w.r.t. MV.
 * @param[in] grad_mq Device pointer to the gradient w.r.t. MQ.
 * @param[in] grad_mk Device pointer to the gradient w.r.t. MK.
 * @param[in] grad_eh Device pointer to the gradient w.r.t. EH.
 * @param[in] grad_ev_scaled Device pointer to the scaled gradient for EV update (size embedding_dim).
 * @param[in] learning_rate The learning rate for the gradient descent update.
 * @param[in] mat_heights The height dimension of the attention matrices.
 * @param[in] embedding_dim The embedding dimension.
 * @param[in] context_win The context window size (number of rows in EV).
 */
__global__ void kernelUpdateWeights_EH_EV(float* mh_a, float* mv_a, float* mq_a, float* mk_a,
                                          float* eh, float* ev,
                                          const float* grad_mh, const float* grad_mv,
                                          const float* grad_mq, const float* grad_mk,
                                          const float* grad_eh, const float* grad_ev_scaled,
                                          float learning_rate,
                                          int mat_heights, int embedding_dim, int context_win) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Global index for parallelization

    // Update MH, MV, MQ, MK matrices (Size: mat_heights * embedding_dim)
    int matrix_size = mat_heights * embedding_dim;
    if (idx < matrix_size) {
        mh_a[idx] -= learning_rate * grad_mh[idx];
        mv_a[idx] -= learning_rate * grad_mv[idx];
        mq_a[idx] -= learning_rate * grad_mq[idx];
        mk_a[idx] -= learning_rate * grad_mk[idx];
    }

    // Update EH vector (Size: embedding_dim)
    // Ensure this update only happens if idx is within the embedding_dim range
    // This might be inefficient if matrix_size >> embedding_dim. Consider separate kernels or grid sizes.
    if (idx < embedding_dim) {
        eh[idx] -= learning_rate * grad_eh[idx];
    }

    // Update EV matrix (Size: context_win * embedding_dim)
    // Apply the single scaled gradient vector grad_ev_scaled based on the column (embedding dim) index.
    int ev_size = context_win * embedding_dim;
     if (idx < ev_size) {
         int embed_idx = idx % embedding_dim; // Get the column index (0 to embedding_dim-1)
         ev[idx] -= learning_rate * grad_ev_scaled[embed_idx]; // Apply the corresponding scaled gradient
     }
}

// --- Kernels specific to second cuBackward overload (`cuBackward(expectedV)`) ---

/**
 * @brief CUDA kernel for Step 1 of `cuBackward(expectedV)`: Compute gradients w.r.t. EV.
 *        Calculates the full gradient `grad_ev_full`, sums it along the context window dimension
 *        to get `grad_ev_summed`, and scales the summed gradient for MLP input `grad_ev_scaled`.
 *        grad_ev_full[win][embed] = 2 * (ev[win][embed] - expected_v[win][embed])
 *        grad_ev_summed[embed] = sum_win (grad_ev_full[win][embed])
 *        grad_ev_scaled[embed] = grad_ev_summed[embed] * learning_rate (Note: C++ scales by LR here)
 * @param[in] ev Device pointer to the EV matrix (row-major, context_win x embedding_dim).
 * @param[in] expected_v Device pointer to the target EV matrix (row-major, context_win x embedding_dim).
 * @param[out] grad_ev_full Device pointer to store the full gradient w.r.t. EV (size context_win * embedding_dim).
 * @param[out] grad_ev_summed Device pointer to store the gradient summed over the context window (size embedding_dim).
 * @param[out] grad_ev_scaled Device pointer to store the summed gradient scaled by learning rate (size embedding_dim).
 * @param[in] learning_rate The learning rate (used for scaling the summed gradient).
 * @param[in] context_win The context window size (rows in EV).
 * @param[in] embedding_dim The embedding dimension (columns in EV).
 */
__global__ void kernelComputeGradientsEV_V(const float* ev, const float* expected_v,
                                           float* grad_ev_full, float* grad_ev_summed, float* grad_ev_scaled,
                                           float learning_rate,
                                           int context_win, int embedding_dim) {
    int embed_idx = blockIdx.x * blockDim.x + threadIdx.x; // Parallelize over embedding dimension

    if (embed_idx < embedding_dim) {
        float sum_grad_embed = 0.0f; // Accumulator for the sum along the context window

        // Iterate through the context window for the current embedding dimension
        for (int win_idx = 0; win_idx < context_win; ++win_idx) {
            int idx = win_idx * embedding_dim + embed_idx; // Flat index for ev[win_idx][embed_idx]
            // Assuming Mean Squared Error loss
            float grad = 2.0f * (ev[idx] - expected_v[idx]);
            grad_ev_full[idx] = grad; // Store the gradient for this specific element
            sum_grad_embed += grad;   // Accumulate the gradient for this embedding dimension
        }
        grad_ev_summed[embed_idx] = sum_grad_embed;
        // Scale the summed gradient by learning rate (as done in C++ step 1 for this path)
        grad_ev_scaled[embed_idx] = sum_grad_embed * learning_rate;
    }
}


/**
 * @brief CUDA kernel for Step 3 of `cuBackward(expectedV)`: Compute gradient w.r.t. ver MLP input (dv).
 *        Extracts the gradient from the first column (input neuron 0) of the vertical MLP's first layer gradient weights.
 *        grad_dv[i] = ver_gweights[0][i][0]
 * @param[in] d_ver_gweights0 Device pointer to the vertical MLP's first layer gradient weights (row-major, embedding_dim x embedding_dim).
 * @param[out] grad_dv Device pointer to store the computed gradient w.r.t. vertical input (size embedding_dim).
 * @param[in] embedding_dim The dimension of the embedding and MLP layers.
 */
__global__ void kernelComputeGradDv_V(const float* d_ver_gweights0, float* grad_dv, int embedding_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Corresponds to the output neuron index 'i' of the first layer

    if (i < embedding_dim) {
        // Calculate flat index for gweights[0][i][0] assuming row-major [output_neuron][input_neuron]
        int gweight_idx = i * embedding_dim + 0;
        grad_dv[i] = d_ver_gweights0[gweight_idx];
    }
}

/**
 * @brief CUDA kernel for Step 4 of `cuBackward(expectedV)`: Compute intermediate value pre_MV only.
 *        pre_mv[h] = sum_i ( sum_j(head[j][i]) * Q[i][h] )
 * @param[in] head Device pointer to the attention head matrix (row-major, token_count x token_count).
 * @param[in] q Device pointer to the Query matrix Q (row-major, token_count x mat_heights).
 * @param[out] pre_mv Device pointer to store the pre_MV vector (size mat_heights).
 * @param[in] token_count The number of tokens (dimension of head matrix, rows of Q).
 * @param[in] mat_heights The height dimension (columns of Q, size of pre_mv).
 * @note This implementation involves redundant calculations (sums over head columns). Consider optimizing if needed.
 */
__global__ void kernelComputePreMV_V(const float* head, const float* q,
                                     float* pre_mv,
                                     int token_count, int mat_heights) {
    int h = blockIdx.x * blockDim.x + threadIdx.x; // Parallelize over the mat_heights dimension

    if (h < mat_heights) {
        float mv_val_h = 0.0f;
        // Iterate over tokens 'i' (which corresponds to column index in C++ head sum)
        for (int i = 0; i < token_count; ++i) {
            float sum_head_col_i = 0.0f; // Sum of column i of head
            // Calculate sum of column i of the head matrix
            for (int j = 0; j < token_count; ++j) {
                sum_head_col_i += head[j * token_count + i]; // head[j][i]
            }
            // Access Q[i][h] using row-major indexing
            float q_ih = q[i * mat_heights + h];
            // Accumulate result for dimension h
            mv_val_h += sum_head_col_i * q_ih;
        }
        pre_mv[h] = mv_val_h;
    }
}


/**
 * @brief CUDA kernel for Step 4 of `cuBackward(expectedV)`: Compute gradient grad_MV only via outer product.
 *        grad_mv[h][d] = pre_mv[h] * grad_dv[d]
 * @param[in] pre_mv Device pointer to the pre_MV vector (size mat_heights).
 * @param[in] grad_dv Device pointer to the gradient w.r.t. vertical input (size embedding_dim).
 * @param[out] grad_mv Device pointer to store the gradient w.r.t. MV (row-major, mat_heights x embedding_dim).
 * @param[in] mat_heights The height dimension.
 * @param[in] embedding_dim The embedding dimension.
 */
__global__ void kernelComputeGradMV_V(const float* pre_mv, const float* grad_dv,
                                      float* grad_mv,
                                      int mat_heights, int embedding_dim) {
    int h = blockIdx.y * blockDim.y + threadIdx.y; // Row index (mat_heights)
    int d = blockIdx.x * blockDim.x + threadIdx.x; // Column index (embedding_dim)

    if (h < mat_heights && d < embedding_dim) {
        int idx = h * embedding_dim + d; // Flat index for row-major matrix
        grad_mv[idx] = pre_mv[h] * grad_dv[d];
    }
}

/**
 * @brief CUDA kernel for Step 5 of `cuBackward(expectedV)`: Compute the gradient w.r.t. the attention head matrix (dv path only).
 *        grad_head[i][j] = sum_d (sum_h Q[j][h]*MV[h][d]) * grad_dv[d]
 *        This is equivalent to: grad_head[i][j] = dot(Q[j]*MV, grad_dv)
 * @param[in] q Device pointer to the Query matrix Q (row-major, token_count x mat_heights).
 * @param[in] mv_a Device pointer to the MV matrix (row-major, mat_heights x embedding_dim).
 * @param[in] grad_dv Device pointer to the gradient w.r.t. vertical input (size embedding_dim).
 * @param[out] grad_head Device pointer to store the gradient w.r.t. the head matrix (row-major, token_count x token_count).
 * @param[in] token_count The number of tokens.
 * @param[in] mat_heights The height dimension.
 * @param[in] embedding_dim The embedding dimension.
 * @note The calculation for grad_head[i][j] only depends on 'j' (via Q[j]) and not 'i'. All elements in a column 'j' will receive the same gradient value.
 */
__global__ void kernelComputeGradHead_V(const float* q, const float* mv_a,
                                        const float* grad_dv, float* grad_head,
                                        int token_count, int mat_heights, int embedding_dim) {
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row index for grad_head (token i)
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Column index for grad_head (token j)

    if (i < token_count && j < token_count) {
        float grad_dv_term_j = 0.0f; // Contribution from the dv path for grad_head[any_i][j]

        // Calculate dot(Q[j]*MV, grad_dv)
        for (int d = 0; d < embedding_dim; ++d) {
            float q_mv_jd = 0.0f; // Q[j] * MV[:, d]
            // Compute the dot product involving Q[j] and column d of MV
            for (int h = 0; h < mat_heights; ++h) {
                // Access Q[j][h], MV[h][d] using row-major indexing
                q_mv_jd += q[j * mat_heights + h] * mv_a[h * embedding_dim + d];
            }
            // Accumulate the contribution for dimension d
            grad_dv_term_j += q_mv_jd * grad_dv[d];
        }

        // Store the gradient for head[i][j] (value depends only on j)
        grad_head[i * token_count + j] = grad_dv_term_j;
    }
}


/**
 * @brief CUDA kernel for Step 7 of `cuBackward(expectedV)`: Compute gradient w.r.t. Q matrix only.
 *        grad_Q = K^T * grad_KdotQ
 * @param[in] grad_kdotq Device pointer to the gradient w.r.t. KdotQ (row-major, token_count x token_count).
 * @param[in] k Device pointer to the Key matrix K (row-major, token_count x mat_heights).
 * @param[out] grad_q Device pointer to store the gradient w.r.t. Q (row-major, token_count x mat_heights).
 * @param[in] token_count The number of tokens.
 * @param[in] mat_heights The height dimension.
 */
__global__ void kernelComputeGradQ_V(const float* grad_kdotq, const float* k,
                                     float* grad_q,
                                     int token_count, int mat_heights) {
    // Each thread computes one element of grad_q
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Row index for grad_q (token j)
    int h = blockIdx.x * blockDim.x + threadIdx.x; // Column index for grad_q (height h)

    if (j < token_count && h < mat_heights) {
        float sum_for_grad_q_jh = 0.0f; // Accumulator for grad_Q[j][h]

        // Calculate grad_Q[j][h] = sum_i (K[i][h] * grad_KdotQ[i][j])
        for (int i = 0; i < token_count; ++i) {
            sum_for_grad_q_jh += k[i * mat_heights + h] * grad_kdotq[i * token_count + j];
        }

        // Store result using row-major indexing
        grad_q[j * mat_heights + h] = sum_for_grad_q_jh;
    }
}

/**
 * @brief CUDA kernel for Step 8 of `cuBackward(expectedV)`: Compute gradient w.r.t. MQ only.
 *        grad_MQ[h][d] = sum_i (grad_Q[i][h] * X[i][d])
 * @param[in] grad_q Device pointer to the gradient w.r.t. Q (row-major, token_count x mat_heights).
 * @param[in] q Device pointer to the Query matrix (row-major, token_count x embedding_dim). **Likely should be original token embeddings X.**
 * @param[out] grad_mq Device pointer to store the gradient w.r.t. MQ (row-major, mat_heights x embedding_dim).
 * @param[in] token_count The number of tokens.
 * @param[in] mat_heights The height dimension.
 * @param[in] embedding_dim The embedding dimension.
 * @note See note in `kernelComputeGradMK_MQ` regarding potential inconsistency of `q` input. Assumes `q` is `token_count x embedding_dim`.
 */
__global__ void kernelComputeGradMQ_V(const float* grad_q, const float* q, // Assumed token_count x embedding_dim
                                      float* grad_mq,
                                      int token_count, int mat_heights, int embedding_dim) {
    int h = blockIdx.y * blockDim.y + threadIdx.y; // Row index for grad_mq (height h)
    int d = blockIdx.x * blockDim.x + threadIdx.x; // Column index for grad_mq (dimension d)

    if (h < mat_heights && d < embedding_dim) {
        float sum_mq_hd = 0.0f;

        // Sum over tokens 'i' (C++ used 'j', using 'i' for clarity)
        for (int i = 0; i < token_count; ++i) {
            // Access grad_Q[i][h] (from token_count x mat_heights matrix)
            float grad_q_ih = grad_q[i * mat_heights + h];
            // Access Q[i][d] (from assumed token_count x embedding_dim matrix)
            float q_id = q[i * embedding_dim + d];
            sum_mq_hd += grad_q_ih * q_id;
        }

        // Store result using row-major indexing
        grad_mq[h * embedding_dim + d] = sum_mq_hd;
    }
}


/**
 * @brief CUDA kernel for Step 8 of `cuBackward(expectedV)`: Compute the correction term for the MK gradient.
 *        grad_MK_correction[h][d] = sum_i sum_j (-grad_MQ[h][d] * Q[j][h] * K[i][h])
 * @param[in] grad_mq Device pointer to the gradient w.r.t. MQ (row-major, mat_heights x embedding_dim).
 * @param[in] q Device pointer to the Query matrix Q (row-major, token_count x mat_heights).
 * @param[in] k Device pointer to the Key matrix K (row-major, token_count x mat_heights).
 * @param[out] grad_mk_correction Device pointer to store the MK correction term (row-major, mat_heights x embedding_dim).
 * @param[in] token_count The number of tokens.
 * @param[in] mat_heights The height dimension.
 * @param[in] embedding_dim The embedding dimension.
 * @note This kernel has O(token_count^2) complexity per thread, which can be very slow for large token counts. Consider optimization if performance is critical.
 */
__global__ void kernelComputeGradMKCorrection(const float* grad_mq, const float* q, const float* k,
                                              float* grad_mk_correction,
                                              int token_count, int mat_heights, int embedding_dim) {
    int h = blockIdx.y * blockDim.y + threadIdx.y; // Row index (mat_heights)
    int d = blockIdx.x * blockDim.x + threadIdx.x; // Column index (embedding_dim)

    if (h < mat_heights && d < embedding_dim) {
        float correction_sum_hd = 0.0f;
        float grad_mq_hd = grad_mq[h * embedding_dim + d]; // grad_MQ[h][d]

        // Double summation over tokens i and j
        for (int i = 0; i < token_count; ++i) {
            for (int j = 0; j < token_count; ++j) {
                // Access Q[j][h] and K[i][h] using row-major indexing
                float q_jh = q[j * mat_heights + h];
                float k_ih = k[i * mat_heights + h];
                correction_sum_hd -= grad_mq_hd * q_jh * k_ih; // Accumulate the negative term
            }
        }
        grad_mk_correction[h * embedding_dim + d] = correction_sum_hd;
    }
}


/**
 * @brief CUDA kernel for Steps 9 & 10 of `cuBackward(expectedV)`: Update attention weight matrices (MV, MQ, MK) and the EV matrix.
 *        Applies gradients using a simple gradient descent step. MK is updated using the correction term.
 *        EV is updated using the full, unscaled gradient `grad_ev_full`.
 * @param[in,out] mv_a Device pointer to the MV matrix (row-major, mat_heights x embedding_dim). Updated in place.
 * @param[in,out] mq_a Device pointer to the MQ matrix (row-major, mat_heights x embedding_dim). Updated in place.
 * @param[in,out] mk_a Device pointer to the MK matrix (row-major, mat_heights x embedding_dim). Updated in place using correction.
 * @param[in,out] ev Device pointer to the EV matrix (row-major, context_win x embedding_dim). Updated in place.
 * @param[in] grad_mv Device pointer to the gradient w.r.t. MV.
 * @param[in] grad_mq Device pointer to the gradient w.r.t. MQ.
 * @param[in] grad_mk_correction Device pointer to the correction term for the MK gradient.
 * @param[in] grad_ev_full Device pointer to the full gradient w.r.t. EV (size context_win * embedding_dim).
 * @param[in] learning_rate The learning rate for the gradient descent update.
 * @param[in] mat_heights The height dimension of the attention matrices.
 * @param[in] embedding_dim The embedding dimension.
 * @param[in] context_win The context window size (number of rows in EV).
 */
__global__ void kernelUpdateWeights_EV_V(float* mv_a, float* mq_a, float* mk_a, float* ev,
                                         const float* grad_mv, const float* grad_mq,
                                         const float* grad_mk_correction,
                                         const float* grad_ev_full,
                                         float learning_rate,
                                         int mat_heights, int embedding_dim, int context_win) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Global index for parallelization

    // Update MV, MQ, MK matrices (Size: mat_heights * embedding_dim)
    int matrix_size = mat_heights * embedding_dim;
    if (idx < matrix_size) {
        mv_a[idx] -= learning_rate * grad_mv[idx];
        mq_a[idx] -= learning_rate * grad_mq[idx];
        mk_a[idx] -= learning_rate * grad_mk_correction[idx]; // Apply correction gradient to MK
    }

    // Update EV matrix (Size: context_win * embedding_dim) using the full gradient
    int ev_size = context_win * embedding_dim;
    // Ensure this update only happens if idx is within the EV size range.
    // Might be inefficient if matrix_size is very different from ev_size.
    if (idx < ev_size) {
        ev[idx] -= learning_rate * grad_ev_full[idx];
    }
}


/**
 * @brief CUDA kernel to compute a simplified derivative for LOTA-like normalization (e.g., Softmax derivative approximation).
 *        lota_deriv_simple[i][j] = (sum_k(head[i][k]) - head[i][j]) / (sum_k(head[i][k]))^2
 * @param[in] head Device pointer to the attention head matrix (row-major, token_count x token_count).
 * @param[in] row_sums Device pointer to the precomputed row sums of the head matrix (size token_count).
 * @param[out] lota_deriv_simple Device pointer to store the computed simple derivative (row-major, token_count x token_count).
 * @param[in] token_count The number of tokens (dimension of the head matrix).
 */
__global__ void kernelComputeSimpleLOTAder(const float* head, const float* row_sums, float* lota_deriv_simple, int token_count)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < token_count && col < token_count) {
        float sum = row_sums[row]; // Get the precomputed sum for the current row
        int idx = row * token_count + col;
        if (sum > 1e-9f) { // Avoid division by zero or very small numbers
            float val = head[idx];
            lota_deriv_simple[idx] = (sum - val) / (sum * sum);
        }
        else {
            lota_deriv_simple[idx] = 0.0f; // Assign zero derivative if sum is too small
        }
    }
}

/**
 * @brief Naive CUDA kernel for row-wise sum reduction of a matrix.
 * @param[in] matrix Device pointer to the input matrix (row-major, rows x cols).
 * @param[out] sums Device pointer to store the row sums (size rows).
 * @param[in] rows The number of rows in the matrix.
 * @param[in] cols The number of columns in the matrix.
 * @note This is a basic implementation. For better performance on large matrices, consider using optimized reduction techniques (e.g., shared memory, CUB library).
 */
__global__ void kernelRowSum(const float* matrix, float* sums, int rows, int cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x; // Parallelize over rows

    if (row < rows) {
        float current_sum = 0.0f;
        // Sum elements across the columns for the current row
        for (int j = 0; j < cols; ++j) {
            current_sum += matrix[row * cols + j];
        }
        sums[row] = current_sum;
    }
}


/**
 * @brief CUDA kernel for Step 8 (Simplified): Compute gradients w.r.t. MK and MQ using embedding vectors.
 *        grad_MK[h][d] = sum_i (grad_K[i][h] * K_embed[i][d])
 *        grad_MQ[h][d] = sum_i (grad_Q[i][h] * Q_embed[i][d])
 * @param[in] grad_k Device pointer to the gradient w.r.t. K (row-major, token_count x mat_heights). Can be null.
 * @param[in] grad_q Device pointer to the gradient w.r.t. Q (row-major, token_count x mat_heights). Can be null.
 * @param[in] k_embed Device pointer to the Key embedding vectors (row-major, token_count x embedding_dim). Can be null.
 * @param[in] q_embed Device pointer to the Query embedding vectors (row-major, token_count x embedding_dim). Can be null.
 * @param[out] grad_mk Device pointer to store the gradient w.r.t. MK (row-major, mat_heights x embedding_dim).
 * @param[out] grad_mq Device pointer to store the gradient w.r.t. MQ (row-major, mat_heights x embedding_dim).
 * @param[in] token_count The number of tokens.
 * @param[in] mat_heights The height dimension.
 * @param[in] embedding_dim The embedding dimension.
 * @note This version correctly uses separate embedding vectors (k_embed, q_embed) for the calculation, resolving potential inconsistencies.
 */
__global__ void kernelComputeGradMK_MQ_Simplified(const float* grad_k, const float* grad_q,
        const float* k_embed, const float* q_embed, // Use embedding versions!
        float* grad_mk, float* grad_mq,
        int token_count, int mat_heights, int embedding_dim)
{
    int h = blockIdx.y * blockDim.y + threadIdx.y; // Row index for grad_mk/grad_mq (height h)
    int d = blockIdx.x * blockDim.x + threadIdx.x; // Column index for grad_mk/grad_mq (dimension d)

    if (h < mat_heights && d < embedding_dim) {
        float sum_mk_hd = 0.0f;
        float sum_mq_hd = 0.0f;

        // Sum over tokens 'i'
        for (int i = 0; i < token_count; ++i) {
            // Check if inputs are valid before accessing
            if (grad_k != nullptr && k_embed != nullptr) {
                // Access grad_K[i][h] and K_embed[i][d]
                sum_mk_hd += grad_k[i * mat_heights + h] * k_embed[i * embedding_dim + d];
            }
            if (grad_q != nullptr && q_embed != nullptr) {
                // Access grad_Q[i][h] and Q_embed[i][d]
                sum_mq_hd += grad_q[i * mat_heights + h] * q_embed[i * embedding_dim + d];
            }
        }
        // Store results using row-major indexing
        grad_mk[h * embedding_dim + d] = sum_mk_hd;
        grad_mq[h * embedding_dim + d] = sum_mq_hd;
    }
}


/**
 * @brief CUDA kernel for Step 9/10 of `cuBackward1stHead(expected)`: Update attention weights (MH, MV, MQ, MK) and conditionally EH.
 *        Applies gradients using gradient descent. EH update is controlled by a flag.
 * @param[in,out] mh_a Device pointer to the MH matrix. Updated in place.
 * @param[in,out] mv_a Device pointer to the MV matrix. Updated in place.
 * @param[in,out] mq_a Device pointer to the MQ matrix. Updated in place.
 * @param[in,out] mk_a Device pointer to the MK matrix. Updated in place.
 * @param[in,out] eh Device pointer to the EH vector. Updated in place if update_eh is true.
 * @param[in] grad_mh Device pointer to the gradient w.r.t. MH. Can be null.
 * @param[in] grad_mv Device pointer to the gradient w.r.t. MV. Can be null.
 * @param[in] grad_mq Device pointer to the gradient w.r.t. MQ. Can be null.
 * @param[in] grad_mk Device pointer to the gradient w.r.t. MK. Can be null.
 * @param[in] grad_eh Device pointer to the gradient w.r.t. EH. Can be null.
 * @param[in] learning_rate The learning rate.
 * @param[in] update_eh Boolean flag; if true, the EH vector is updated.
 * @param[in] mat_heights The height dimension of the attention matrices.
 * @param[in] embedding_dim The embedding dimension.
 */
__global__ void kernelUpdateWeights_1stHead_H(float* mh_a, float* mv_a, float* mq_a, float* mk_a,
    float* eh,
    const float* grad_mh, const float* grad_mv,
    const float* grad_mq, const float* grad_mk,
    const float* grad_eh,
    float learning_rate, bool update_eh,
    int mat_heights, int embedding_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Global index for parallelization

    // Update MH, MV, MQ, MK matrices (Size: mat_heights * embedding_dim)
    int matrix_size = mat_heights * embedding_dim;
    if (idx < matrix_size) {
        // Check if gradient pointers are valid before applying update
        if(grad_mh != nullptr)
            mh_a[idx] -= learning_rate * grad_mh[idx];
        if(grad_mv != nullptr)
            mv_a[idx] -= learning_rate * grad_mv[idx];
        if(grad_mq != nullptr)
            mq_a[idx] -= learning_rate * grad_mq[idx];
        if(grad_mk != nullptr)
            mk_a[idx] -= learning_rate * grad_mk[idx];
    }

    // Update EH vector (Size: embedding_dim) - Conditionally
    // Ensure this update only happens if idx is within the embedding_dim range and flag is set.
    if (update_eh && idx < embedding_dim) {
        if(grad_eh != nullptr) // Check if gradient pointer is valid
            eh[idx] -= learning_rate * grad_eh[idx];
    }
}


/**
 * @brief CUDA kernel for Step 9/10 of `cuBackward1stHead(expectedV)`: Update attention weights (MV, MQ, MK).
 *        Applies gradients using gradient descent. MK is updated using the correction term. No EH/EV update.
 * @param[in,out] mv_a Device pointer to the MV matrix. Updated in place.
 * @param[in,out] mq_a Device pointer to the MQ matrix. Updated in place.
 * @param[in,out] mk_a Device pointer to the MK matrix. Updated in place using correction.
 * @param[in] grad_mv Device pointer to the gradient w.r.t. MV. Can be null.
 * @param[in] grad_mq Device pointer to the gradient w.r.t. MQ. Can be null.
 * @param[in] grad_mk_correction Device pointer to the correction term for the MK gradient. Can be null.
 * @param[in] learning_rate The learning rate.
 * @param[in] mat_heights The height dimension of the attention matrices.
 * @param[in] embedding_dim The embedding dimension.
 */
__global__ void kernelUpdateWeights_1stHead_V(float* mv_a, float* mq_a, float* mk_a,
    const float* grad_mv, const float* grad_mq, const float* grad_mk_correction,
    float learning_rate, int mat_heights, int embedding_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Global index for parallelization

    // Update MV, MQ, MK matrices (Size: mat_heights * embedding_dim)
    int matrix_size = mat_heights * embedding_dim;
    if (idx < matrix_size) {
        // Check if gradient pointers are valid before applying update
        if(grad_mv != nullptr)
            mv_a[idx] -= learning_rate * grad_mv[idx];
        if(grad_mq != nullptr)
            mq_a[idx] -= learning_rate * grad_mq[idx];
        if(grad_mk_correction != nullptr)
            mk_a[idx] -= learning_rate * grad_mk_correction[idx]; // Use correction for MK
    }
    // No EH or EV update in this kernel
}


/**
 * @brief CUDA kernel for Step 9/10 of `cuBackward1stHead(expectedH, expectedV)`: Update attention weights (MH, MV, MQ, MK).
 *        Applies gradients using gradient descent. No EH/EV update.
 * @param[in,out] mh_a Device pointer to the MH matrix. Updated in place.
 * @param[in,out] mv_a Device pointer to the MV matrix. Updated in place.
 * @param[in,out] mq_a Device pointer to the MQ matrix. Updated in place.
 * @param[in,out] mk_a Device pointer to the MK matrix. Updated in place.
 * @param[in] grad_mh Device pointer to the gradient w.r.t. MH. Can be null.
 * @param[in] grad_mv Device pointer to the gradient w.r.t. MV. Can be null.
 * @param[in] grad_mq Device pointer to the gradient w.r.t. MQ. Can be null.
 * @param[in] grad_mk Device pointer to the gradient w.r.t. MK. Can be null.
 * @param[in] learning_rate The learning rate.
 * @param[in] mat_heights The height dimension of the attention matrices.
 * @param[in] embedding_dim The embedding dimension.
 */
__global__ void kernelUpdateWeights_1stHead_HV(float* mh_a, float* mv_a, float* mq_a, float* mk_a,
    const float* grad_mh, const float* grad_mv, const float* grad_mq, const float* grad_mk,
    float learning_rate, int mat_heights, int embedding_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Global index for parallelization

    // Update MH, MV, MQ, MK matrices (Size: mat_heights * embedding_dim)
    int matrix_size = mat_heights * embedding_dim;
    if (idx < matrix_size) {
        // Check if gradient pointers are valid before applying update
        if(grad_mh != nullptr)
            mh_a[idx] -= learning_rate * grad_mh[idx];
        if(grad_mv != nullptr)
            mv_a[idx] -= learning_rate * grad_mv[idx];
        if(grad_mq != nullptr)
            mq_a[idx] -= learning_rate * grad_mq[idx];
        if(grad_mk != nullptr)
            mk_a[idx] -= learning_rate * grad_mk[idx];
    }
    // No EH or EV update in this kernel
}


/**
 * @brief Simple CUDA kernel to update weights using gradient descent.
 *        weights_to_update -= learning_rate * gradients
 * @param[in,out] weights_to_update Device pointer to the weights/parameters to be updated.
 * @param[in] gradients Device pointer to the corresponding gradients.
 * @param[in] lr The learning rate.
 * @param[in] n_elements The total number of elements in the weights/gradients arrays.
 */
__global__ void kernelUpdateSimple(float* weights_to_update, const float* gradients, float lr, size_t n_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        if (gradients != nullptr) { // Check if gradient exists
             weights_to_update[idx] -= lr * gradients[idx];
        }
    }
};
