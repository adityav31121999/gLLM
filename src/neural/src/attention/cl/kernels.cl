
// --- Static helper cl_compute_dot_product_vec is NO LONGER NEEDED by these kernels ---
// --- It can be removed if no other kernels use it ---

__kernel void vectorAddKernel(__global const float* A, __global const float* B,
    __global float* C, int len)
{
    int idx = get_global_id(0);
    if (idx < len) {
        C[idx] = A[idx] + B[idx];
    }
}

/**
 * @brief OpenCL Kernel: Computes the dot product of two vectors.
 *        VERSION: Single Work-Item (Sequential).
 *        NOTE: Launch with global_size = 1, local_size = 1.
 *              Inefficient for large 'dim'.
 * @param[in] vec1 Device pointer to the first vector (__global const float*).
 * @param[in] vec2 Device pointer to the second vector (__global const float*).
 * @param[out] result Device pointer to store the scalar result (__global float*).
 * @param[in] dim The dimension (number of elements) of the vectors.
 */
__kernel void kernelVecDotVec(
    __global const float* vec1,
    __global const float* vec2,
    __global float* result,
    int dim)
{
    // Execute only on the first global work-item
    if (get_global_id(0) == 0) {
        float dot_product = 0.0f;
        for (int k = 0; k < dim; ++k) {
            dot_product += vec1[k] * vec2[k];
        }
        *result = dot_product;
    }
}

/**
 * @brief OpenCL Kernel: Computes a key or query vector by multiplying a token embedding with a matrix.
 *        KorQ = tokenEmbed * matrix^T (effectively, as matrix is row-major)
 *        Parallelized over the output vector 'KorQ'.
 * @param[in] tokenEmbed Device pointer to the token embedding vector (__global const float*).
 * @param[in] matrix Device pointer to the key or query matrix (__global const float*).
 * @param[out] KorQ Device pointer to the resulting Key or Query vector (__global float*).
 * @param[in] dim The embedding dimension.
 * @param[in] height The number of rows in the key/query matrix (size of KorQ).
 */
__kernel void kernelComputeKorQ(
    __global const float* tokenEmbed,
    __global const float* matrix,
    __global float* KorQ,
    int dim,
    int height)
{
    // Global work-item ID corresponds to the output vector index 'i'
    int i = get_global_id(0);

    // Check bounds for the output vector
    if (i < height) {
        // Calculate KorQ[i] = dot(tokenEmbed, matrix_row_i)
        __global const float* matrix_row_i = matrix + i * dim;
        float dot_product = 0.0f;
        for (int j = 0; j < dim; ++j) {
            dot_product += tokenEmbed[j] * matrix_row_i[j];
        }
        KorQ[i] = dot_product;
    }
}


/**
 * @brief OpenCL Kernel: Computes the quadratic form vec1 * matrix * vec2^T.
 *        NOTE: Implemented as a single work-item kernel for direct conversion.
 * @param[in] vec1 Device pointer to the first vector (__global const float*).
 * @param[in] vec2 Device pointer to the second vector (__global const float*).
 * @param[in] matrix Device pointer to the matrix (__global const float*).
 * @param[out] result Device pointer to store the scalar result (__global float*).
 * @param[in] dim The dimension of the vectors and the square matrix.
 */
__kernel void kernelDotvecmatvec(
    __global const float* vec1,
    __global const float* vec2,
    __global const float* matrix,
    __global float* result,
    int dim)
{
    // Execute only on the first global work-item
    if (get_global_id(0) == 0) {
        float final_dot_product = 0.0f;
        for (int i = 0; i < dim; ++i) {
            float inner_sum = 0.0f;
            __global const float* matrix_row_i = matrix + i * dim;
            for (int j = 0; j < dim; ++j) {
                inner_sum += vec1[j] * matrix_row_i[j];
            }
            final_dot_product += inner_sum * vec2[i];
        }
        *result = final_dot_product;
    }
}

/**
 * @brief OpenCL Kernel: Computes the predicted token index by finding the highest dot product.
 *        NOTE: Implemented as a single work-item kernel for direct conversion.
 *              (Dot product logic inlined)
 * @param[in] EH Device pointer to the horizontal retention vector (__global const float*).
 * @param[in] embeddings Device pointer to the token embeddings matrix (__global const float*).
 * @param[out] result_index Device pointer to store the resulting index (__global int*).
 * @param[in] dim The embedding dimension.
 * @param[in] voc The vocabulary size.
 */
__kernel void kernelComputePrediction(
    __global const float* EH,
    __global const float* embeddings,
    __global int* result_index,
    int dim,
    int voc)
{
    // Execute only on the first global work-item
    if (get_global_id(0) == 0) {
        if (voc <= 0 || embeddings == NULL) {
             *result_index = -1; // Indicate error or invalid input
             return;
        }

        float max_dot_product = -MAXFLOAT;
        int predicted_index = 0;

        for (int i = 0; i < voc; ++i) {
            __global const float* current_embedding_row = embeddings + i * dim;

            // --- Inlined dot product calculation ---
            float current_dot_product = 0.0f;
            for (int k = 0; k < dim; ++k) { // Use 'dim' as loop limit
                current_dot_product += EH[k] * current_embedding_row[k]; // Use EH, current_embedding_row
            }
            // --- End Inlined dot product ---

            if (current_dot_product > max_dot_product) {
                max_dot_product = current_dot_product;
                predicted_index = i;
            }
        }
        *result_index = predicted_index;
    }
}

// --- Kernels ---

/**------------------------------------MULTIPLICATION------------------------------------**/

/**
 * @brief OpenCL kernel for element-wise vector multiplication. Multiplies `target_and_output` by `factor` in place.
 * @param[in,out] target_and_output Device pointer (__global float*).
 * @param[in] factor Device pointer (__global const float*).
 * @param[in] size The number of elements.
 */
__kernel void kernelElementwiseMultiply(__global float* target_and_output, __global const float* factor, int size) {
    int idx = get_global_id(0);
    if (idx < size) {
        target_and_output[idx] *= factor[idx];
    }
}

/**------------------------------------TRAINING------------------------------------**/

/**
 * @brief OpenCL kernel for calculating the scaled KdotQ matrix for self-attention during training.
 *        Computes KdotQ[i][j] = dot(Q[i], K[j]) / SCALING for j <= i (causal masking).
 *        (Dot product logic inlined)
 * @param[out] d_kdotq         Device pointer (__global float*).
 * @param[in]  d_keys          Device pointer (__global const float*).
 * @param[in]  d_querys        Device pointer (__global const float*).
 * @param[in]  num_queries_eff Number of query rows.
 * @param[in]  num_keys_eff    Number of key columns.
 * @param[in]  kdotq_width     Total width of d_kdotq buffer.
 * @param[in]  embedding_dim   Dimension of key/query vectors.
 * @param[in]  inv_scaling     Inverse scaling factor.
 */
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

/**
 * @brief OpenCL kernel for calculating the scaled KdotQ matrix for cross-attention during training.
 *        Computes KdotQ[i][j] = dot(Q[i], K[j]) / SCALING for all i, j.
 *        (Dot product logic inlined)
 * @param[out] d_kdotq         Device pointer (__global float*).
 * @param[in]  d_keys          Device pointer (__global const float*).
 * @param[in]  d_querys        Device pointer (__global const float*).
 * @param[in]  num_queries_eff Number of query rows.
 * @param[in]  num_keys_eff    Number of key columns.
 * @param[in]  kdotq_width     Total width of d_kdotq buffer.
 * @param[in]  embedding_dim   Dimension of key/query vectors.
 * @param[in]  inv_scaling     Inverse scaling factor.
 */
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

// Kernels kernelKdotQ_Block1_Self_Inference, kernelKdotQ_Block1_Cross_Inference,
// kernelKdotQ_BlockN_Self_Inference, kernelKdotQ_BlockN_Cross_Inference
// already use kernelDotvecmatvec (which doesn't use the vector dot product)
// so they remain unchanged from the previous version.

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


/**
 * @brief OpenCL kernel to compute row sums and column sums of the attention head matrix,
 *        applying causal masking for self-attention if specified.
 * @param[in] d_head Device pointer (__global const float*).
 * @param[out] d_row_sums Device pointer (__global float*).
 * @param[out] d_col_sums Device pointer (__global float*).
 * @param[in] num_tokens Dimension of the square head matrix.
 * @param[in] isSelfAttention Int flag (0=false, 1=true).
 */
__kernel void computeHeadSumsMaskedKernel(__global const float* d_head, __global float* d_row_sums, __global float* d_col_sums,
                                          int num_tokens, int isSelfAttention) // Use int for bool
{
    int i = get_global_id(0); // Parallelize over token index 'i'

    if (i < num_tokens) {
        float row_sum_k = 0.0f;
        float col_sum_l = 0.0f;

        // Limit based on C++ comment: sum up to j = limit-1.
        // Corrected logic: For row sum, sum columns j up to limit. For col sum, sum rows j up to limit.
        int limit = num_tokens; // Limit for column sum is always num_tokens

        // Calculate row sum (k) for token i: sum head[i][j] for j < num_tokens, applying mask if needed
        for (int j = 0; j < num_tokens; ++j) {
             // Apply self-attention mask: only sum if j <= i
             if (isSelfAttention == 0 || j <= i) {
                row_sum_k += d_head[i * num_tokens + j];
             }
        }

        // Calculate column sum (l) for token i: sum head[j][i] for j < num_tokens, applying mask if needed
        for (int j = 0; j < num_tokens; ++j) {
             // Apply self-attention mask: only sum if i <= j (equivalent to j accessing row i in KdotQ)
             if (isSelfAttention == 0 || i <= j) { // Corrected mask for column sum
                col_sum_l += d_head[j * num_tokens + i];
             }
        }

        d_row_sums[i] = row_sum_k;
        d_col_sums[i] = col_sum_l;
    }
}

/**
 * @brief OpenCL kernel to accumulate weighted Key and Query vectors based on head row/column sums.
 *        Uses atomic_fetch_add for accumulation. Requires OpenCL 2.0+ or float atomic extensions.
 * @param[in] d_row_sums Device pointer (__global const float*).
 * @param[in] d_col_sums Device pointer (__global const float*).
 * @param[in] d_K Device pointer (__global const float*).
 * @param[in] d_Q Device pointer (__global const float*).
 * @param[in,out] d_dh_accum Device pointer (__global float*). MUST be zero-initialized.
 * @param[in,out] d_dv_accum Device pointer (__global float*). MUST be zero-initialized.
 * @param[in] num_tokens Number of tokens.
 * @param[in] h_dim Dimension of Key/Query vectors.
 */
__kernel void accumulateWeightedVectorsKernel(__global const float* d_row_sums, __global const float* d_col_sums,
                                              __global const float* d_K, __global const float* d_Q,
                                              __global float* d_dh_accum, __global float* d_dv_accum,
                                              int num_tokens, int h_dim)
{
    int h_idx = get_global_id(0); // Parallelize over the h_dim dimension

    if (h_idx < h_dim) {
        float total_dh_for_h_idx = 0.0f;
        float total_dv_for_h_idx = 0.0f;

        for (int i = 0; i < num_tokens; ++i) {
            float k_i_h = d_K[i * h_dim + h_idx];
            float q_i_h = d_Q[i * h_dim + h_idx];
            total_dh_for_h_idx += d_row_sums[i] * k_i_h;
            total_dv_for_h_idx += d_col_sums[i] * q_i_h;
        }

        // Atomically add the computed sums. Requires OpenCL 2.0+ or float atomic extensions.
        // Add volatile qualifier inside atomic call.
        atomic_fetch_add((volatile __global float*)&d_dh_accum[h_idx], total_dh_for_h_idx);
        atomic_fetch_add((volatile __global float*)&d_dv_accum[h_idx], total_dv_for_h_idx);
        // If float atomics are unavailable, this needs a different reduction strategy.
    }
}

/**------------------------------------BACKPROP------------------------------------**/

// The backpropagation kernels do not use the vector dot product helper,
// so they remain unchanged from the previous version.

__kernel void kernelComputeGradDhDv_1stHead(__global const float* d_hor_gweights0, __global const float* d_ver_gweights0,
                                            __global float* grad_dh, __global float* grad_dv, int embedding_dim)
{
    int i = get_global_id(0);
    if (i < embedding_dim) {
        int gweight_idx = i * embedding_dim + 0;
        grad_dh[i] = (d_hor_gweights0 != NULL) ? d_hor_gweights0[gweight_idx] : 0.0f;
        grad_dv[i] = (d_ver_gweights0 != NULL) ? d_ver_gweights0[gweight_idx] : 0.0f;
    }
}

__kernel void kernelComputeGradientsEH_EV(__global const float* eh, __global const float* expected_h,
                                          __global float* grad_eh, __global float* grad_ev_scaled, int embedding_dim)
{
    int idx = get_global_id(0);
    if (idx < embedding_dim) {
        float grad = 2.0f * (eh[idx] - expected_h[idx]);
        grad_eh[idx] = grad;
        grad_ev_scaled[idx] = grad * 0.1f;
    }
}

__kernel void kernelComputeGradDhDv(__global const float* d_hor_gweights0, __global const float* d_ver_gweights0,
                                    __global float* grad_dh, __global float* grad_dv, int embedding_dim)
{
    int i = get_global_id(0);
    if (i < embedding_dim) {
        float sum_dh = 0.0f;
        float sum_dv = 0.0f;
        for (int j = 0; j < embedding_dim; ++j) {
            int gweight_idx = i * embedding_dim + j;
            sum_dh += d_hor_gweights0[gweight_idx];
            sum_dv += d_ver_gweights0[gweight_idx];
        }
        grad_dh[i] = sum_dh;
        grad_dv[i] = sum_dv;
    }
}

__kernel void kernelComputePreMH_MV(__global const float* head, __global const float* k, __global const float* q,
                                    __global float* pre_mh, __global float* pre_mv,
                                    int token_count, int mat_heights)
{
    int h = get_global_id(0);
    if (h < mat_heights) {
        float mh_val_h = 0.0f;
        float mv_val_h = 0.0f;
        for (int i = 0; i < token_count; ++i) {
            float sum_head_row_i = 0.0f;
            float sum_head_col_i = 0.0f;
            for (int j = 0; j < token_count; ++j) {
                sum_head_row_i += head[i * token_count + j];
                sum_head_col_i += head[j * token_count + i];
            }
            mh_val_h += sum_head_row_i * k[i * mat_heights + h];
            mv_val_h += sum_head_col_i * q[i * mat_heights + h];
        }
        pre_mh[h] = mh_val_h;
        pre_mv[h] = mv_val_h;
    }
}

__kernel void kernelComputeGradMH_MV(__global const float* pre_mh, __global const float* pre_mv,
                                     __global const float* grad_dh, __global const float* grad_dv,
                                     __global float* grad_mh, __global float* grad_mv,
                                     int mat_heights, int embedding_dim)
{
    int h = get_global_id(1);
    int d = get_global_id(0);
    if (h < mat_heights && d < embedding_dim) {
        int idx = h * embedding_dim + d;
        grad_mh[idx] = pre_mh[h] * grad_dh[d];
        grad_mv[idx] = pre_mv[h] * grad_dv[d];
    }
}

__kernel void kernelComputeGradHead(__global const float* k, __global const float* q,
                                    __global const float* mh_a, __global const float* mv_a,
                                    __global const float* grad_dh, __global const float* grad_dv,
                                    __global float* grad_head,
                                    int token_count, int mat_heights, int embedding_dim)
{
    int i = get_global_id(1);
    int j = get_global_id(0);
    if (i < token_count && j < token_count) {
        float grad_dh_term_ij = 0.0f;
        float grad_dv_term_ij = 0.0f;
        for (int d = 0; d < embedding_dim; ++d) {
            float k_mh_id = 0.0f;
            float q_mv_jd = 0.0f;
            for (int h = 0; h < mat_heights; ++h) {
                k_mh_id += k[i * mat_heights + h] * mh_a[h * embedding_dim + d];
                q_mv_jd += q[j * mat_heights + h] * mv_a[h * embedding_dim + d];
            }
            grad_dh_term_ij += k_mh_id * grad_dh[d];
            grad_dv_term_ij += q_mv_jd * grad_dv[d];
        }
        grad_head[i * token_count + j] = grad_dh_term_ij + grad_dv_term_ij;
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

__kernel void kernelComputeGradK_Q(__global const float* grad_kdotq, __global const float* k, __global const float* q,
                                   __global float* grad_k, __global float* grad_q,
                                   int token_count, int mat_heights)
{
    int i = get_global_id(1);
    int h = get_global_id(0);
    if (i < token_count && h < mat_heights) {
        float sum_for_grad_k_ih = 0.0f;
        float sum_for_grad_q_ih = 0.0f;
        for (int j = 0; j < token_count; ++j) {
            sum_for_grad_k_ih += grad_kdotq[i * token_count + j] * q[j * mat_heights + h];
        }
        for (int j = 0; j < token_count; ++j) {
            sum_for_grad_q_ih += k[j * mat_heights + h] * grad_kdotq[j * token_count + i];
        }
        grad_k[i * mat_heights + h] = sum_for_grad_k_ih;
        grad_q[i * mat_heights + h] = sum_for_grad_q_ih;
    }
}

__kernel void kernelComputeGradMK_MQ(__global const float* grad_k, __global const float* grad_q,
                                     __global const float* k, __global const float* q,
                                     __global float* grad_mk, __global float* grad_mq,
                                     int token_count, int mat_heights, int embedding_dim)
{
    int h = get_global_id(1);
    int d = get_global_id(0);
    if (h < mat_heights && d < embedding_dim) {
        float sum_mk_hd = 0.0f;
        float sum_mq_hd = 0.0f;
        for (int i = 0; i < token_count; ++i) {
            sum_mk_hd += grad_k[i * mat_heights + h] * k[i * embedding_dim + d]; // k is token_count x embedding_dim
            sum_mq_hd += grad_q[i * mat_heights + h] * q[i * embedding_dim + d]; // q is token_count x embedding_dim
        }
        grad_mk[h * embedding_dim + d] = sum_mk_hd;
        grad_mq[h * embedding_dim + d] = sum_mq_hd;
    }
}

__kernel void kernelUpdateWeights_EH_EV(__global float* mh_a, __global float* mv_a, __global float* mq_a, __global float* mk_a,
                                        __global float* eh, __global float* ev,
                                        __global const float* grad_mh, __global const float* grad_mv,
                                        __global const float* grad_mq, __global const float* grad_mk,
                                        __global const float* grad_eh, __global const float* grad_ev_scaled,
                                        float learning_rate,
                                        int mat_heights, int embedding_dim, int context_win)
{
    int idx = get_global_id(0);
    int matrix_size = mat_heights * embedding_dim;
    if (idx < matrix_size) {
        mh_a[idx] -= learning_rate * grad_mh[idx];
        mv_a[idx] -= learning_rate * grad_mv[idx];
        mq_a[idx] -= learning_rate * grad_mq[idx];
        mk_a[idx] -= learning_rate * grad_mk[idx];
    }
    if (idx < embedding_dim) {
        eh[idx] -= learning_rate * grad_eh[idx];
    }
    int ev_size = context_win * embedding_dim;
    if (idx < ev_size) {
        int embed_idx = idx % embedding_dim;
        ev[idx] -= learning_rate * grad_ev_scaled[embed_idx];
    }
}

__kernel void kernelComputeGradientsEV_V(__global const float* ev, __global const float* expected_v,
                                         __global float* grad_ev_full, __global float* grad_ev_summed, __global float* grad_ev_scaled,
                                         float learning_rate,
                                         int context_win, int embedding_dim)
{
    int embed_idx = get_global_id(0);
    if (embed_idx < embedding_dim) {
        float sum_grad_embed = 0.0f;
        for (int win_idx = 0; win_idx < context_win; ++win_idx) {
            int idx = win_idx * embedding_dim + embed_idx;
            float grad = 2.0f * (ev[idx] - expected_v[idx]);
            grad_ev_full[idx] = grad;
            sum_grad_embed += grad;
        }
        grad_ev_summed[embed_idx] = sum_grad_embed;
        grad_ev_scaled[embed_idx] = sum_grad_embed * learning_rate;
    }
}

__kernel void kernelComputeGradDv_V(__global const float* d_ver_gweights0, __global float* grad_dv, int embedding_dim)
{
    int i = get_global_id(0);
    if (i < embedding_dim) {
        int gweight_idx = i * embedding_dim + 0;
        grad_dv[i] = d_ver_gweights0[gweight_idx];
    }
}

__kernel void kernelComputePreMV_V(__global const float* head, __global const float* q,
                                   __global float* pre_mv,
                                   int token_count, int mat_heights)
{
    int h = get_global_id(0);
    if (h < mat_heights) {
        float mv_val_h = 0.0f;
        for (int i = 0; i < token_count; ++i) {
            float sum_head_col_i = 0.0f;
            for (int j = 0; j < token_count; ++j) {
                sum_head_col_i += head[j * token_count + i];
            }
            mv_val_h += sum_head_col_i * q[i * mat_heights + h];
        }
        pre_mv[h] = mv_val_h;
    }
}

__kernel void kernelComputeGradMV_V(__global const float* pre_mv, __global const float* grad_dv,
                                    __global float* grad_mv,
                                    int mat_heights, int embedding_dim)
{
    int h = get_global_id(1);
    int d = get_global_id(0);
    if (h < mat_heights && d < embedding_dim) {
        int idx = h * embedding_dim + d;
        grad_mv[idx] = pre_mv[h] * grad_dv[d];
    }
}

__kernel void kernelComputeGradHead_V(__global const float* q, __global const float* mv_a,
                                      __global const float* grad_dv, __global float* grad_head,
                                      int token_count, int mat_heights, int embedding_dim)
{
    int i = get_global_id(1);
    int j = get_global_id(0);
    if (i < token_count && j < token_count) {
        float grad_dv_term_j = 0.0f;
        for (int d = 0; d < embedding_dim; ++d) {
            float q_mv_jd = 0.0f;
            for (int h = 0; h < mat_heights; ++h) {
                q_mv_jd += q[j * mat_heights + h] * mv_a[h * embedding_dim + d];
            }
            grad_dv_term_j += q_mv_jd * grad_dv[d];
        }
        grad_head[i * token_count + j] = grad_dv_term_j;
    }
}

__kernel void kernelComputeGradQ_V(__global const float* grad_kdotq, __global const float* k,
                                   __global float* grad_q,
                                   int token_count, int mat_heights)
{
    int j = get_global_id(1);
    int h = get_global_id(0);
    if (j < token_count && h < mat_heights) {
        float sum_for_grad_q_jh = 0.0f;
        for (int i = 0; i < token_count; ++i) {
            sum_for_grad_q_jh += k[i * mat_heights + h] * grad_kdotq[i * token_count + j];
        }
        grad_q[j * mat_heights + h] = sum_for_grad_q_jh;
    }
}

__kernel void kernelComputeGradMQ_V(__global const float* grad_q, __global const float* q,
                                    __global float* grad_mq,
                                    int token_count, int mat_heights, int embedding_dim)
{
    int h = get_global_id(1);
    int d = get_global_id(0);
    if (h < mat_heights && d < embedding_dim) {
        float sum_mq_hd = 0.0f;
        for (int i = 0; i < token_count; ++i) {
            sum_mq_hd += grad_q[i * mat_heights + h] * q[i * embedding_dim + d]; // q is token_count x embedding_dim
        }
        grad_mq[h * embedding_dim + d] = sum_mq_hd;
    }
}

__kernel void kernelComputeGradMKCorrection(__global const float* grad_mq, __global const float* q, __global const float* k,
                                            __global float* grad_mk_correction,
                                            int token_count, int mat_heights, int embedding_dim)
{
    int h = get_global_id(1);
    int d = get_global_id(0);
    if (h < mat_heights && d < embedding_dim) {
        float correction_sum_hd = 0.0f;
        float grad_mq_hd = grad_mq[h * embedding_dim + d];
        for (int i = 0; i < token_count; ++i) {
            for (int j = 0; j < token_count; ++j) {
                correction_sum_hd -= grad_mq_hd * q[j * mat_heights + h] * k[i * mat_heights + h]; // k is token_count x mat_heights
            }
        }
        grad_mk_correction[h * embedding_dim + d] = correction_sum_hd;
    }
}

__kernel void kernelUpdateWeights_EV_V(__global float* mv_a, __global float* mq_a, __global float* mk_a, __global float* ev,
                                       __global const float* grad_mv, __global const float* grad_mq,
                                       __global const float* grad_mk_correction,
                                       __global const float* grad_ev_full,
                                       float learning_rate,
                                       int mat_heights, int embedding_dim, int context_win)
{
    int idx = get_global_id(0);
    int matrix_size = mat_heights * embedding_dim;
    if (idx < matrix_size) {
        mv_a[idx] -= learning_rate * grad_mv[idx];
        mq_a[idx] -= learning_rate * grad_mq[idx];
        mk_a[idx] -= learning_rate * grad_mk_correction[idx];
    }
    int ev_size = context_win * embedding_dim;
    if (idx < ev_size) {
        ev[idx] -= learning_rate * grad_ev_full[idx];
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
        lota_deriv_simple[idx] = (sum > 1e-9f) ? ((sum - head[idx]) / (sum * sum)) : 0.0f;
    }
}

__kernel void kernelRowSum(__global const float* matrix, __global float* sums, int rows, int cols)
{
    int row = get_global_id(0);
    if (row < rows) {
        float current_sum = 0.0f;
        for (int j = 0; j < cols; ++j) {
            current_sum += matrix[row * cols + j];
        }
        sums[row] = current_sum;
    }
}

__kernel void kernelComputeGradMK_MQ_Simplified(__global const float* grad_k, __global const float* grad_q,
                                                __global const float* k_embed, __global const float* q_embed,
                                                __global float* grad_mk, __global float* grad_mq,
                                                int token_count, int mat_heights, int embedding_dim)
{
    int h = get_global_id(1);
    int d = get_global_id(0);
    if (h < mat_heights && d < embedding_dim) {
        float sum_mk_hd = 0.0f;
        float sum_mq_hd = 0.0f;
        for (int i = 0; i < token_count; ++i) {
            if (grad_k != NULL && k_embed != NULL) {
                sum_mk_hd += grad_k[i * mat_heights + h] * k_embed[i * embedding_dim + d];
            }
            if (grad_q != NULL && q_embed != NULL) {
                sum_mq_hd += grad_q[i * mat_heights + h] * q_embed[i * embedding_dim + d];
            }
        }
        grad_mk[h * embedding_dim + d] = sum_mk_hd;
        grad_mq[h * embedding_dim + d] = sum_mq_hd;
    }
}

__kernel void kernelUpdateWeights_1stHead_H(__global float* mh_a, __global float* mv_a, __global float* mq_a, __global float* mk_a,
                                            __global float* eh,
                                            __global const float* grad_mh, __global const float* grad_mv,
                                            __global const float* grad_mq, __global const float* grad_mk,
                                            __global const float* grad_eh,
                                            float learning_rate, int update_eh,
                                            int mat_heights, int embedding_dim)
{
    int idx = get_global_id(0);
    int matrix_size = mat_heights * embedding_dim;
    if (idx < matrix_size) {
        if(grad_mh != NULL) mh_a[idx] -= learning_rate * grad_mh[idx];
        if(grad_mv != NULL) mv_a[idx] -= learning_rate * grad_mv[idx];
        if(grad_mq != NULL) mq_a[idx] -= learning_rate * grad_mq[idx];
        if(grad_mk != NULL) mk_a[idx] -= learning_rate * grad_mk[idx];
    }
    if (update_eh != 0 && idx < embedding_dim) {
        if(grad_eh != NULL) eh[idx] -= learning_rate * grad_eh[idx];
    }
}

__kernel void kernelUpdateWeights_1stHead_V(__global float* mv_a, __global float* mq_a, __global float* mk_a,
                                            __global const float* grad_mv, __global const float* grad_mq,
                                            __global const float* grad_mk_correction,
                                            float learning_rate, int mat_heights, int embedding_dim)
{
    int idx = get_global_id(0);
    int matrix_size = mat_heights * embedding_dim;
    if (idx < matrix_size) {
        if(grad_mv != NULL) mv_a[idx] -= learning_rate * grad_mv[idx];
        if(grad_mq != NULL) mq_a[idx] -= learning_rate * grad_mq[idx];
        if(grad_mk_correction != NULL) mk_a[idx] -= learning_rate * grad_mk_correction[idx];
    }
}

__kernel void kernelUpdateWeights_1stHead_HV(__global float* mh_a, __global float* mv_a, __global float* mq_a, __global float* mk_a,
                                             __global const float* grad_mh, __global const float* grad_mv,
                                             __global const float* grad_mq, __global const float* grad_mk,
                                             float learning_rate, int mat_heights, int embedding_dim)
{
    int idx = get_global_id(0);
    int matrix_size = mat_heights * embedding_dim;
    if (idx < matrix_size) {
        if(grad_mh != NULL) mh_a[idx] -= learning_rate * grad_mh[idx];
        if(grad_mv != NULL) mv_a[idx] -= learning_rate * grad_mv[idx];
        if(grad_mq != NULL) mq_a[idx] -= learning_rate * grad_mq[idx];
        if(grad_mk != NULL) mk_a[idx] -= learning_rate * grad_mk[idx];
    }
}

__kernel void kernelUpdateSimple(__global float* weights_to_update, __global const float* gradients, float lr, size_t n_elements)
{
    int idx = get_global_id(0);
    if (idx < n_elements) {
        if (gradients != NULL) {
             weights_to_update[idx] -= lr * gradients[idx];
        }
    }
}
