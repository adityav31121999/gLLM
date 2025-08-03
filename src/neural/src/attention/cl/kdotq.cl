// Helper macro for indexing flattened matrix (assuming row-major)
#define IDX(row, col, dim) ((row) * (dim) + (col))

// Enable extensions for atomics and potentially double precision (which might include float atomics)
// #pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
// #pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable
// #pragma OPENCL EXTENSION cl_khr_fp64 : enable // For double support
// #pragma OPENCL EXTENSION cl_khr_float_atomics : enable // Not supported on target, using manual implementation

/**------------------------------------TRAINING------------------------------------**/

__kernel void kernelKdotQforSelf_train(__global float* d_kdotq, __global const float* d_keys, __global const float* d_querys,
                                       int num_queries_eff, int num_keys_eff, int kdotq_width, int embedding_dim, float inv_scaling)
{
    int j = get_global_id(0); // Global key index (column)
    int i = get_global_id(1); // Global query index (row)

    if (i < num_queries_eff && j < num_keys_eff && j <= i) {
        __global const float* q_vec = d_querys + i * embedding_dim;
        __global const float* k_vec = d_keys + j * embedding_dim;

        // --- Inlined dot product calculation ---
        float dot_product = 0.0f;
        for (int k = 0; k < embedding_dim; ++k) { // Use embedding_dim
            dot_product += q_vec[k] * k_vec[k];   // Use q_vec, k_vec
        }
        // --- End Inlined dot product ---

        int kdotq_index = i * kdotq_width + j;
        d_kdotq[kdotq_index] = dot_product * inv_scaling;
    }
}

__kernel void kernelKdotQforCross_train(__global float* d_kdotq, __global const float* d_keys, __global const float* d_querys,
                                        int num_queries_eff, int num_keys_eff, int kdotq_width, int embedding_dim, float inv_scaling)
{
    int j = get_global_id(0); // Global key index (column)
    int i = get_global_id(1); // Global query index (row)

    if (i < num_queries_eff && j < num_keys_eff) {
        __global const float* q_vec = d_querys + i * embedding_dim;
        __global const float* k_vec = d_keys + j * embedding_dim;

        // --- Inlined dot product calculation ---
        float dot_product = 0.0f;
        for (int k = 0; k < embedding_dim; ++k) { // Use embedding_dim
            dot_product += q_vec[k] * k_vec[k];   // Use q_vec, k_vec
        }
        // --- End Inlined dot product ---

        int kdotq_index = i * kdotq_width + j;
        d_kdotq[kdotq_index] = dot_product * inv_scaling;
    }
}


/**------------------------------------INFERENCE------------------------------------**/

__kernel void kernelKdotQ_Block1_Self_Inference(__global float* d_kdotq, __global const float* d_tokenEmbed, __global const float* d_M,
                                                int prompt_start_index, int prompt_len, int context_len, int kdotq_width,
                                                int embedding_dim, float inv_scaling)
{
    int j = get_global_id(0);
    int i_offset = get_global_id(1);
    int i = prompt_start_index + i_offset;

    if (i_offset < prompt_len && j < context_len && j <= i) {
        __global const float* q_vec = d_tokenEmbed + i * embedding_dim;
        __global const float* k_vec = d_tokenEmbed + j * embedding_dim;

        // This uses the quadratic form logic, not the simple vector dot product
        float final_dot_product = 0.0f;
        for (int row_idx = 0; row_idx < embedding_dim; ++row_idx) {
            float inner_sum = 0.0f;
            __global const float* matrix_row_i = d_M + row_idx * embedding_dim;
            for (int col_idx = 0; col_idx < embedding_dim; ++col_idx) {
                inner_sum += q_vec[col_idx] * matrix_row_i[col_idx];
            }
            final_dot_product += inner_sum * k_vec[row_idx]; // Note: k_vec is used like vec2 here
        }

        int kdotq_index = i * kdotq_width + j;
        d_kdotq[kdotq_index] = final_dot_product * inv_scaling;
    }
}

__kernel void kernelKdotQ_Block1_Cross_Inference(__global float* d_kdotq, __global const float* d_tokenEmbed, __global const float* d_M,
                                                 int prompt_start_index, int prompt_len, int context_len, int kdotq_width,
                                                 int embedding_dim, float inv_scaling)
{
    int j = get_global_id(0);
    int i_offset = get_global_id(1);
    int i = prompt_start_index + i_offset;

    if (i_offset < prompt_len && j < context_len) {
        __global const float* q_vec = d_tokenEmbed + i * embedding_dim;
        __global const float* k_vec = d_tokenEmbed + j * embedding_dim;

        // This uses the quadratic form logic, not the simple vector dot product
        float final_dot_product = 0.0f;
        for (int row_idx = 0; row_idx < embedding_dim; ++row_idx) {
            float inner_sum = 0.0f;
            __global const float* matrix_row_i = d_M + row_idx * embedding_dim;
            for (int col_idx = 0; col_idx < embedding_dim; ++col_idx) {
                inner_sum += q_vec[col_idx] * matrix_row_i[col_idx];
            }
            final_dot_product += inner_sum * k_vec[row_idx]; // Note: k_vec is used like vec2 here
        }

        int kdotq_index = i * kdotq_width + j;
        d_kdotq[kdotq_index] = final_dot_product * inv_scaling;
    }
}

__kernel void kernelKdotQ_BlockN_Self_Inference(__global float* d_kdotq, __global const float* d_tokForBlock, __global const float* d_EVp,
                                                __global const float* d_M, int prompt_start_index_in_block, int prompt_len,
                                                int context_len_in_block, int kdotq_width, int embedding_dim, float inv_scaling)
{
    int j = get_global_id(0);
    int i_offset = get_global_id(1);
    int i = prompt_start_index_in_block + i_offset;

    if (i_offset < prompt_len && j < context_len_in_block && j <= i) {
        __global const float* q_vec = d_tokForBlock + i * embedding_dim;
        __global const float* k_vec = d_EVp + j * embedding_dim; // EVp is used as the 'key' vector source

        // This uses the quadratic form logic, not the simple vector dot product
        float final_dot_product = 0.0f;
        for (int row_idx = 0; row_idx < embedding_dim; ++row_idx) {
            float inner_sum = 0.0f;
            __global const float* matrix_row_i = d_M + row_idx * embedding_dim;
            for (int col_idx = 0; col_idx < embedding_dim; ++col_idx) {
                inner_sum += q_vec[col_idx] * matrix_row_i[col_idx];
            }
            final_dot_product += inner_sum * k_vec[row_idx]; // Note: k_vec (from EVp) is used like vec2 here
        }

        int kdotq_index = i * kdotq_width + j;
        d_kdotq[kdotq_index] = final_dot_product * inv_scaling;
    }
}

__kernel void kernelKdotQ_BlockN_Cross_Inference(__global float* d_kdotq, __global const float* d_tokForBlock, __global const float* d_EVp,
                                                 __global const float* d_M, int prompt_start_index_in_block, int prompt_len,
                                                 int context_len_in_block, int kdotq_width, int embedding_dim, float inv_scaling)
{
    int j = get_global_id(0);
    int i_offset = get_global_id(1);
    int i = prompt_start_index_in_block + i_offset;

    if (i_offset < prompt_len && j < context_len_in_block) {
        __global const float* q_vec = d_tokForBlock + i * embedding_dim;
        __global const float* k_vec = d_EVp + j * embedding_dim; // EVp is used as the 'key' vector source

        // This uses the quadratic form logic, not the simple vector dot product
        float final_dot_product = 0.0f;
        for (int row_idx = 0; row_idx < embedding_dim; ++row_idx) {
            float inner_sum = 0.0f;
            __global const float* matrix_row_i = d_M + row_idx * embedding_dim;
            for (int col_idx = 0; col_idx < embedding_dim; ++col_idx) {
                inner_sum += q_vec[col_idx] * matrix_row_i[col_idx];
            }
            final_dot_product += inner_sum * k_vec[row_idx]; // Note: k_vec (from EVp) is used like vec2 here
        }

        int kdotq_index = i * kdotq_width + j;
        d_kdotq[kdotq_index] = final_dot_product * inv_scaling;
    }
}

__kernel void kernelComputeGradKdotQ_LOTA(__global const float* grad_head, __global const float* lota_derivative,
                                          __global float* grad_kdotq, float scaling_factor, int size)
{
    int idx = get_global_id(0);
    if (idx < size) {
        grad_kdotq[idx] = (fabs(scaling_factor) > 1e-9f) ? (grad_head[idx] * lota_derivative[idx] / scaling_factor) : 0.0f;
    }
}
