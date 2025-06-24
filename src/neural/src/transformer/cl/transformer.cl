// Helper macro for indexing flattened matrix (assuming row-major)
#define IDX(row, col, dim) ((row) * (dim) + (col))

// Enable extensions for atomics and potentially double precision (which might include float atomics)
// #pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
// #pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable // For double support

// Forward declarations to prevent implicit declaration warnings/errors
inline float compute_dot_product(__global const float* vec1, __global const float* vec2, int dim);
inline float compute_dot_product_mat(__global const float* vec1, __global const float* vec2, __global const float* matrix, int dim);
int compute_prediction(__global const float* EH, __global const float* embeddings, int dim, int voc);

inline float compute_dot_product(__global const float* vec1, __global const float* vec2, int dim) 
{
    float dot_product = 0.0f;
    for (int k = 0; k < dim; ++k) {
        dot_product += vec1[k] * vec2[k];
    }
    return dot_product;
}

inline float compute_dot_product_mat(__global const float* vec1, __global const float* vec2, __global const float* matrix,
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
        // Use the correctly named inline function
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


__kernel void kernelKdotQforSelf_train_transformer(__global float* d_kdotq, __global const float* d_keys, 
            __global const float* d_querys, int num_queries_eff, int num_keys_eff, int kdotq_width, 
            int embedding_dim, float inv_scaling)
{
    // Calculate the global row (query index i) and column (key index j) for this work-item
    int j = get_global_id(0); // Key index (column)
    int i = get_global_id(1); // Query index (row)

    // Boundary check AND self-attention causal mask (j <= i)
    if (i < num_queries_eff && j < num_keys_eff && j <= i) {
        // Pointers to the start of the i-th query vector and j-th key vector
        __global const float* q_vec = d_querys + i * embedding_dim;
        __global const float* k_vec = d_keys + j * embedding_dim;
        // Use the correctly named inline function
        float dot_product = compute_dot_product(q_vec, k_vec, embedding_dim);

        // Calculate the index in the flattened d_kdotq array (row-major)
        int kdotq_index = i * kdotq_width + j;
        d_kdotq[kdotq_index] = dot_product * inv_scaling;
    }
}

__kernel void kernelKdotQforCross_train_transformer(__global float* d_kdotq, __global const float* d_keys, 
            __global const float* d_querys, int num_queries_eff, int num_keys_eff, int kdotq_width, 
            int embedding_dim, float inv_scaling)
{
    // Calculate the global column (key index j) and row (query index i) for this work-item
    int j = get_global_id(0); // Key index (column)
    int i = get_global_id(1); // Query index (row)

    // Boundary check (no causal mask for cross-attention)
    if (i < num_queries_eff && j < num_keys_eff) {
        // Pointers to the start of the i-th query vector and j-th key vector
        __global const float* q_vec = d_querys + i * embedding_dim;
        __global const float* k_vec = d_keys + j * embedding_dim;
        // Use the correctly named inline function
        float dot_product = compute_dot_product(q_vec, k_vec, embedding_dim);
        // Calculate the index in the flattened d_kdotq array (row-major)
        int kdotq_index = i * kdotq_width + j;
        d_kdotq[kdotq_index] = dot_product * inv_scaling;
    }
}

/**------------------------------------INFERENCE------------------------------------**/

__kernel void kernelKdotQBlock1Self_Inference(__global float* d_kdotq, __global const float* d_tokenEmbed, 
            __global const float* d_M, int prompt_start_index, int prompt_len, int context_len, 
            int kdotq_width,int embedding_dim, float inv_scaling)
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
        // Use the correctly named inline function
        float dot_product = compute_dot_product_mat(q_vec, k_vec, d_M, embedding_dim);
        // index in flatten array
        int kdotq_index = i * kdotq_width + j;
        d_kdotq[kdotq_index] = dot_product * inv_scaling;
    }
}

__kernel void kernelKdotQBlock1Cross_Inference(__global float* d_kdotq, __global const float* d_tokenEmbed, 
            __global const float* d_M, int prompt_start_index, int prompt_len, int context_len, 
            int kdotq_width, int embedding_dim, float inv_scaling)
{
    // Calculate the global key index (j) and the offset for the query index (i_offset)
    int j = get_global_id(0);
    int i_offset = get_global_id(1);
    int i = prompt_start_index + i_offset;
    if (i_offset < prompt_len && j < context_len) {
        // Pointers to the start of the i-th query vector and j-th key vector
        __global const float* q_vec = d_tokenEmbed + i * embedding_dim;
        __global const float* k_vec = d_tokenEmbed + j * embedding_dim;
        // Use the correctly named inline function
        float dot_product = compute_dot_product_mat(q_vec, k_vec, d_M, embedding_dim);
        // Calculate the index in the flattened d_kdotq array (row-major)
        int kdotq_index = i * kdotq_width + j;
        d_kdotq[kdotq_index] = dot_product * inv_scaling;
    }
}

__kernel void kernelKdotQBlockNSelf_Inference(__global float* d_kdotq, __global const float* d_tokForBlock, 
            __global const float* d_EVp, __global const float* d_M, int prompt_start_index_in_block, 
            int prompt_len, int context_len_in_block, int kdotq_width, int embedding_dim, float inv_scaling)
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
        // Use the correctly named inline function
        float dot_product = compute_dot_product_mat(q_vec, k_vec, d_M, embedding_dim);
        // Calculate the index in the flattened d_kdotq array (row-major, relative to block window)
        int kdotq_index = i * kdotq_width + j;
        d_kdotq[kdotq_index] = dot_product * inv_scaling;
    }
}

__kernel void kernelKdotQBlockNCross_Inference(__global float* d_kdotq, __global const float* d_tokForBlock,
            __global const float* d_EVp, __global const float* d_M, int prompt_start_index_in_block, 
            int prompt_len, int context_len_in_block, int kdotq_width, int embedding_dim, float inv_scaling)
{
    // Calculate the key index (j) and query index offset (i_offset) *within the block's window*
    int j = get_global_id(0);
    int i_offset = get_global_id(1);
    int i = prompt_start_index_in_block + i_offset;
    if (i_offset < prompt_len && j < context_len_in_block) {
        // Pointers to the start of the i-th query vector (from tokForBlock) and j-th key vector (from EVp)
        __global const float* q_vec = d_tokForBlock + i * embedding_dim;
        __global const float* k_vec = d_EVp + j * embedding_dim;
        // Use the correctly named inline function
        float dot_product = compute_dot_product_mat(q_vec, k_vec, d_M, embedding_dim);
        // Calculate the index in the flattened d_kdotq array (row-major, relative to block window)
        int kdotq_index = i * kdotq_width + j;
        d_kdotq[kdotq_index] = dot_product * inv_scaling;
    }
}