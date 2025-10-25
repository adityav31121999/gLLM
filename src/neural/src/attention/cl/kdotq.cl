/**------------------------------------KQ Calculation------------------------------------**/

__kernel void kernelCompute_single_kq_vector( __global const float* d_token_embedding,  // Input: token vector (size: embedding_dim)
                                            __global const float* d_projection_matrix, // Input: MQ or MK matrix (size: mat_heights x embedding_dim)
                                            __global float* d_output_kq_vector,     // Output: K or Q vector (size: mat_heights)
                                            int embedding_dim,
                                            int mat_heights)
{
    // This kernel is intended to be launched with a single global work-item.
    // It computes: d_output_kq_vector[i] = dot(d_token_embedding, d_projection_matrix_row_i)
    if (get_global_id(0) == 0) {
        for (int i = 0; i < mat_heights; ++i) {
            __global const float* matrix_row_i = d_projection_matrix + i * embedding_dim;
            float dot_product = 0.0f;
            for (int j = 0; j < embedding_dim; ++j) {
                dot_product += d_token_embedding[j] * matrix_row_i[j];
            }
            d_output_kq_vector[i] = dot_product;
        }
    }
}

__kernel void kernelComputeKQall(
    __global const float* tokenMatrix,  // token matrix or EV of size local_context x embedding_dim
    __global const float* KQmatrix,     // weight Q/K matrix of size mat_heights x embedding_dim
    __global float* KQoutputMatrix,     // query/key matrix of size local_context x mat_heights
    int tokenCount,                     // number of rows in tokenMatrix
    int dim,                            // embedding dimension (must be divisible by 4)
    int height                          // number of rows in KQmatrix
    )
{
    // Global work-item ID corresponds to the row index in tokenMatrix (token_idx)
    int token_idx = get_global_id(0);
    // Global work-item ID corresponds to the column index in KQoutputMatrix (kq_height_idx)
    int kq_height_idx = get_global_id(1);
    
    // Ensure dim is a multiple of 4 for float4 operations
    // This check should ideally be done on the host side.
    // If dim is not a multiple of 4, the behavior might be incorrect or lead to out-of-bounds access.
    // For simplicity, we assume dim is a multiple of 4 here.
    int dim_float4 = dim / 4;

    // Check bounds for the current token and KQ matrix height
    if (token_idx < tokenCount && kq_height_idx < height) {
        // Get the current token embedding vector
        __global const float4* current_token_embedding_f4 = (__global const float4*)(tokenMatrix + token_idx * dim);
        // Get the current row from the KQmatrix (projection matrix)
        __global const float4* current_kq_matrix_row_f4 = (__global const float4*)(KQmatrix + kq_height_idx * dim);

        // Calculate dot product: dot(current_token_embedding, current_kq_matrix_row)
        float dot_product = 0.0f;
        for (int j = 0; j < dim_float4; ++j) {
            float4 token_vec = current_token_embedding_f4[j];
            float4 kq_vec = current_kq_matrix_row_f4[j];
            dot_product += dot(token_vec, kq_vec);
        }

        KQoutputMatrix[token_idx * height + kq_height_idx] = dot_product;
    }
}

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

__kernel void kernelKdotQ_Block1_Selfi(__global float* d_kdotq, __global const float* d_tokenEmbed, __global const float* d_M,
                                                int sequence1_start_index, int sequence1_len, int context_len, int kdotq_width,
                                                int embedding_dim, float inv_scaling)
{
    int j = get_global_id(0);
    int i_offset = get_global_id(1);
    int i = sequence1_start_index + i_offset;

    if (i_offset < sequence1_len && j < context_len && j <= i) {
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

__kernel void kernelKdotQ_Block1_Crossi(__global float* d_kdotq, __global const float* d_tokenEmbed, __global const float* d_M,
                                                 int sequence1_start_index, int sequence1_len, int context_len, int kdotq_width,
                                                 int embedding_dim, float inv_scaling)
{
    int j = get_global_id(0);
    int i_offset = get_global_id(1);
    int i = sequence1_start_index + i_offset;

    if (i_offset < sequence1_len && j < context_len) {
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

__kernel void kernelKdotQ_BlockN_Selfi(__global float* d_kdotq, __global const float* d_tokForBlock, __global const float* d_EVp,
                                                __global const float* d_M, int sequence1_start_index_in_block, int sequence1_len,
                                                int context_len_in_block, int kdotq_width, int embedding_dim, float inv_scaling)
{
    int j = get_global_id(0);
    int i_offset = get_global_id(1);
    int i = sequence1_start_index_in_block + i_offset;

    if (i_offset < sequence1_len && j < context_len_in_block && j <= i) {
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

__kernel void kernelKdotQ_BlockN_Crossi(__global float* d_kdotq, __global const float* d_tokForBlock, __global const float* d_EVp,
                                                 __global const float* d_M, int sequence1_start_index_in_block, int sequence1_len,
                                                 int context_len_in_block, int kdotq_width, int embedding_dim, float inv_scaling)
{
    int j = get_global_id(0);
    int i_offset = get_global_id(1);
    int i = sequence1_start_index_in_block + i_offset;

    if (i_offset < sequence1_len && j < context_len_in_block) {
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

/// Lota derivatives

__kernel void kernelComputeGradKdotQ_LOTA(__global const float* grad_head, __global const float* lota_derivative,
                                          __global float* grad_kdotq, float scaling_factor, int row, int col, int size)
{
    int idx = get_global_id(0);
    if (idx < size) {
        grad_kdotq[idx] = (fabs(scaling_factor) > 1e-15f) ? (grad_head[idx] * lota_derivative[idx] / scaling_factor) : 0.0f;
    }
}


__kernel void kernelComputeSimpleLOTAder(__global const float* head, __global const float* row_sums,
                                         __global float* lota_deriv_simple, int token_count)
{
    int row = get_global_id(1);
    int col = get_global_id(0);
    if (row < token_count && col < token_count) {
        float sum = row_sums[row];
        int idx = row * token_count + col;
        lota_deriv_simple[idx] = (sum > 1e-15f) ? ((sum - head[idx]) / (sum * sum)) : 0.0f;
    }
}
