// Helper function for sign (not directly available in OpenCL C without a custom implementation)
inline float sign_f(float x) {
    if (x > 0) return 1.0f;
    if (x < 0) return -1.0f;
    return 0.0f;
}

inline void atomic_add_float(volatile __global float *addr, float val) {
    union {
        unsigned int u32;
        float f32;
    } next, expected, current;

    current.f32 = *addr; // Read initial value

    do {
        expected.f32 = current.f32; // Expected value for CAS
        next.f32 = expected.f32 + val; // Calculate new value
        // Atomically compare expected.u32 with the value at addr.
        // If they match, replace the value at addr with next.u32.
        // Update current.u32 with the value that was previously at addr.
        current.u32 = atomic_cmpxchg((volatile __global unsigned int *)addr, expected.u32, next.u32);
    } while (current.u32 != expected.u32); // Loop if CAS failed (value changed by another thread)
}

__kernel void kernelElementwiseMultiply(__global float* target_and_output, __global const float* factor, int size) {
    int idx = get_global_id(0);
    if (idx < size) {
        target_and_output[idx] *= factor[idx];
    }
}


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

        // Assuming dim is a multiple of 4 for float4 operations
        int dim_float4 = dim / 4;
        __global const float4* EH_f4 = (__global const float4*)EH;

        for (int i = 0; i < voc; ++i) {
            __global const float4* current_embedding_row_f4 = (__global const float4*)(embeddings + i * dim);

            // --- Inlined dot product calculation ---
            float current_dot_product = 0.0f;
            for (int k = 0; k < dim_float4; ++k) {
                float4 eh_vec = EH_f4[k];
                float4 embed_vec = current_embedding_row_f4[k];
                current_dot_product += dot(eh_vec, embed_vec);
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

__kernel void kernelComputePredictionWithScores(
    __global const float* EH,
    __global const float* deEmbeddings,
    __global float* scores,
    __global int* result_index,
    int dim,
    int voc)
{
    // Execute only on the first global work-item
    if (get_global_id(0) == 0) {
        if (voc <= 0 || deEmbeddings == NULL) {
            *result_index = -1; // Indicate error or invalid input
            return;
        }

        float max_dot_product = -MAXFLOAT;
        int predicted_index = 0;

        // Assuming dim is a multiple of 4 for float4 operations
        int dim_float4 = dim / 4;
        __global const float4* EH_f4 = (__global const float4*)EH;

        for (int i = 0; i < voc; ++i) {
            __global const float4* current_deEmbedding_row_f4 = (__global const float4*)(deEmbeddings + i * dim);

            // --- Inlined dot product calculation ---
            float current_dot_product = 0.0f;
            for (int k = 0; k < dim_float4; ++k) {
                float4 eh_vec = EH_f4[k];
                float4 embed_vec = current_deEmbedding_row_f4[k];
                current_dot_product += dot(eh_vec, embed_vec);
            }
            scores[i] = current_dot_product;
            // --- End Inlined dot product ---

            if (current_dot_product > max_dot_product) {
                max_dot_product = current_dot_product;
                predicted_index = i;
            }
        }
        *result_index = predicted_index;
    }
}

// ----------------------------- ForProp ----------------------------- //

__kernel void accumulateEVRowsKernelCL(__global const float* d_EV, __global float* d_output,
    int num_rows, int col_size)
{
    int col_idx = get_global_id(0); // Each work-item computes one element of d_output

    if (col_idx < col_size) {
        float sum = 0.0f;
        for (int row_idx = 0; row_idx < num_rows; ++row_idx) {
            sum += d_EV[row_idx * col_size + col_idx];
        }
        d_output[col_idx] = sum;
    }
}

__kernel void updateEVRowsKernelCL(__global float* d_EV_rows, __global const float* d_vector_to_add,
    int num_rows_to_update, int num_cols)
{
    int row_idx = get_global_id(0); // Each work-item handles one row

    if (row_idx < num_rows_to_update) {
        for (int col_idx = 0; col_idx < num_cols; ++col_idx) {
            d_EV_rows[row_idx * num_cols + col_idx] += d_vector_to_add[col_idx];
        }
    }
}

__kernel void computeHeadSumsMaskedKernel(__global const float* d_head, __global float* d_row_sums, __global float* d_col_sums,
                                          int num_tokens, int isSelfAttention) // Use int for bool
{
    int i = get_global_id(0); // Parallelize over token index 'i'
 
    if (i < num_tokens) {
        float4 row_sum_k4 = (float4)(0.0f);
        float4 col_sum_l4 = (float4)(0.0f);
 
        int limit = (isSelfAttention != 0) ? (i + 1) : num_tokens;
        int limit_div_4 = limit / 4;
 
        // Calculate row sum (k) for token i: sum head[i][j] for j < num_tokens, applying mask if needed
        // Vectorized part
        for (int j = 0; j < limit_div_4; ++j) {
            row_sum_k4 += vload4(j, d_head + i * num_tokens);
        }
 
        // Scalar part for remaining elements
        float row_sum_k = row_sum_k4.s0 + row_sum_k4.s1 + row_sum_k4.s2 + row_sum_k4.s3;
        for (int j = limit_div_4 * 4; j < limit; ++j) {
            row_sum_k += d_head[i * num_tokens + j];
        }
 
        // Calculate column sum (l) for token i: sum head[j][i] for j < num_tokens, applying mask if needed
        // Vectorized part (unrolled reads)
        for (int j = 0; j < limit_div_4; ++j) {
            int j_base = j * 4;
            col_sum_l4.s0 += d_head[(j_base + 0) * num_tokens + i];
            col_sum_l4.s1 += d_head[(j_base + 1) * num_tokens + i];
            col_sum_l4.s2 += d_head[(j_base + 2) * num_tokens + i];
            col_sum_l4.s3 += d_head[(j_base + 3) * num_tokens + i];
        }
 
        // Scalar part for remaining elements
        float col_sum_l = col_sum_l4.s0 + col_sum_l4.s1 + col_sum_l4.s2 + col_sum_l4.s3;
        for (int j = limit_div_4 * 4; j < limit; ++j) {
            col_sum_l += d_head[j * num_tokens + i];
        }
 
        d_row_sums[i] = row_sum_k;
        d_col_sums[i] = col_sum_l;
    }
}

__kernel void accumulateWeightedVectorsKernel(__global const float* d_row_sums, __global const float* d_col_sums,
                                              __global const float* d_K, __global const float* d_Q,
                                              __global float* d_dh_accum, __global float* d_dv_accum,
                                              int num_tokens, int h_dim)
{
    int h_idx_base = get_global_id(0) * 4; // Each work-item handles 4 elements of h_dim

    if (h_idx_base < h_dim) {
        float4 total_dh4 = (float4)(0.0f);
        float4 total_dv4 = (float4)(0.0f);

        // Loop over tokens
        for (int i = 0; i < num_tokens; ++i) {
            // Load 4 elements from d_K and d_Q for the current token 'i'
            float4 k_i_h4 = vload4(0, d_K + i * h_dim + h_idx_base);
            float4 q_i_h4 = vload4(0, d_Q + i * h_dim + h_idx_base);

            // Accumulate weighted vectors using FMA (fused multiply-add) for performance
            total_dh4 = fma(d_row_sums[i], k_i_h4, total_dh4);
            total_dv4 = fma(d_col_sums[i], q_i_h4, total_dv4);
        }

        // Atomically add the 4 computed sums to the global accumulators
        atomic_add_float(&d_dh_accum[h_idx_base + 0], total_dh4.s0);
        atomic_add_float(&d_dh_accum[h_idx_base + 1], total_dh4.s1);
        atomic_add_float(&d_dh_accum[h_idx_base + 2], total_dh4.s2);
        atomic_add_float(&d_dh_accum[h_idx_base + 3], total_dh4.s3);

        atomic_add_float(&d_dv_accum[h_idx_base + 0], total_dv4.s0);
        atomic_add_float(&d_dv_accum[h_idx_base + 1], total_dv4.s1);
        atomic_add_float(&d_dv_accum[h_idx_base + 2], total_dv4.s2);
        atomic_add_float(&d_dv_accum[h_idx_base + 3], total_dv4.s3);
    }
}

/**------------------------------------BACKPROP------------------------------------**/

__kernel void kernelComputeGradpred(__global const float* predNorm, __global const float* oneHot,
                                    __global float* grad_pred, int vocab)
{
    int idx = get_global_id(0);
    if (idx < vocab) {
        float pNorm = predNorm[idx];
        float label = oneHot[idx];

        // The gradient of BCE loss w.r.t. the logits (pre-sigmoid input) is simply (prediction - label).
        // This is numerically stable.
        float grad = pNorm - label;
        grad_pred[idx] = grad;
    }
}

/*
 * @brief Computes the outer product of two vectors: grad = d_delta * otok^T.
 *        This operation calculates the gradient of the de-embedding weights.
 *        This is an optimized version using float4 for higher memory throughput.
 *        The operation is `grad[row][col] = d_delta[row] * otok[col]`.
 *
 * @note  The `dEmbedDim` must be divisible by 4. The host must launch this kernel
 *        with a global size where the second dimension is `dEmbedDim / 4`, and
 *        pass `dEmbedDim / 4` as the `dEmbedDim_div_4` argument.
 *
 * @param d_delta   Input gradient vector w.r.t. predictions, size `vocab`.
 * @param otok      Input final hidden state vector, size `dEmbedDim`.
 * @param grad      Output gradient matrix for de-embedding weights, size `vocab x dEmbedDim`.
 * @param vocab     The size of the `d_delta` vector (number of rows in `grad`).
 * @param dEmbedDim_div_4 The size of the `otok` vector divided by 4 (number of float4 columns in `grad`).
 */
__kernel void KernelComputeGradDeEmbeddings(__global const float* d_delta,
                                           __global const float4* otok,      // Read otok as float4
                                           __global float4* grad,          // Write grad as float4
                                           int vocab,
                                           int dEmbedDim_div_4) {
    int row = get_global_id(0);  // Corresponds to vocab dimension
    int col4 = get_global_id(1); // Corresponds to dEmbedDim dimension (in float4 units)

    if (row < vocab && col4 < dEmbedDim_div_4) {
        const float delta_val = d_delta[row];
        grad[row * dEmbedDim_div_4 + col4] = delta_val * otok[col4];
    }
}

// perform product of row vector and matrix and form a vector
__kernel void kernelGradForAttentionOutput(__global const float* d_deEmbed,
                                          __global const float* d_delta,
                                          __global float* grad4EH,
                                          int vocabSize, int dEmbedDim)
{
    int idx4 = get_global_id(0) * 4; // Each work item processes 4 elements
    if (idx4 < dEmbedDim) {
        float4 sum4 = (float4)(0.0f);
        // Process 4 elements at a time using float4
        for (int j = 0; j < vocabSize; ++j) {
            float4 deEmbed = vload4(0, &d_deEmbed[j * dEmbedDim + idx4]);
            sum4 = fma(d_delta[j], deEmbed, sum4);
        }
        // Store the 4 accumulated sums
        vstore4(sum4, 0, &grad4EH[idx4]);

        // Handle remaining elements if dEmbedDim is not a multiple of 4
        if (idx4 + 4 > dEmbedDim && idx4 < dEmbedDim) {
            int remaining = min(4, dEmbedDim - idx4);
            for (int k = 0; k < remaining; ++k) {
                float sum = 0.0f;
                for (int j = 0; j < vocabSize; ++j) {
                    sum += d_deEmbed[j * dEmbedDim + idx4 + k] * d_delta[j];
                }
                grad4EH[idx4 + k] = sum;
            }
        }
    }
}

__kernel void kernelComputeGradientsEH(__global const float* eh, __global const float* expected_h,
                                       __global float* grad_eh, int embedding_dim)
{
    int idx = get_global_id(0);
    if (idx < embedding_dim) {
        float pred = eh[idx];
        float label = expected_h[idx];

        // The gradient of BCE loss w.r.t. the logits (pre-sigmoid input) is simply (prediction - label).
        // This is numerically stable.
        float grad = pred - label;
        grad_eh[idx] = grad;
    }
}

__kernel void kernelComputeGradientsEHEVFromMSE(__global const float* eh, __global const float* expected_h,
                                          __global float* grad_eh, __global float* grad_ev_scaled, int embedding_dim)
{
    int idx = get_global_id(0);
    if (idx < embedding_dim) {
        float pred = eh[idx];
        float label = expected_h[idx];
        // The gradient of BCE loss w.r.t. the logits (pre-sigmoid input) is simply (prediction - label).
        // This is numerically stable and avoids the division by (pred * (1-pred)), which explodes when pred is near 0 or 1.
        float grad = pred - label;
        grad_eh[idx] = grad;
        grad_ev_scaled[idx] = grad * 0.01f;
    }
}

__kernel void kernelComputeGradientsEH_EV(__global const float* eh, __global const float* expected_h,
                                          __global float* grad_eh, __global float* grad_ev_scaled, int embedding_dim)
{
    int idx = get_global_id(0);
    if (idx < embedding_dim) {
        float pred = eh[idx];
        float label = expected_h[idx];
        // The gradient of BCE loss w.r.t. the logits (pre-sigmoid input) is simply (prediction - label).
        // This is numerically stable and avoids the division by (pred * (1-pred)), which explodes when pred is near 0 or 1.
        float grad = pred - label;
        grad_eh[idx] = grad;
        grad_ev_scaled[idx] = grad * 0.01f;
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

            float pred = ev[idx];
            float label = expected_v[idx];

            // The gradient of BCE loss w.r.t. the logits (pre-sigmoid input) is simply (prediction - label).
            // This is numerically stable.
            // Clamp pred to avoid division by zero or near-zero values in the denominator
            pred = fmin(fmax(pred, 1e-7f), 1.0f - 1e-7f);
            // Binary Cross Entropy gradient w.r.t. sigmoid output
            float grad = (pred - label);// / (pred * (1.0f - pred));
            grad_ev_full[idx] = grad;
            sum_grad_embed += grad;
        }
        grad_ev_summed[embed_idx] = sum_grad_embed;
        grad_ev_scaled[embed_idx] = sum_grad_embed * learning_rate;
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

__kernel void kernelComputePreMH_MV(__global const float* head, __global const float* k, __global const float* q,
                                    __global float* pre_mh, __global float* pre_mv,
                                    int token_count, int mat_heights)
{
    int h = get_global_id(0);
    if (h < mat_heights) {
        float mh_val_h = 0.0f;
        float mv_val_h = 0.0f;

        // This calculation determines how many full float4 vectors we can process.
        // Integer division automatically handles this. For example, if token_count is 127,
        // token_count_div_4 will be 31.
        int token_count_div_4 = token_count / 4;

        for (int i = 0; i < token_count; ++i) {
            float sum_head_row_i = 0.0f;
            float sum_head_col_i = 0.0f;

            // --- Vectorized Part ---
            // This loop processes the bulk of the data in chunks of 4.
            float4 sum_vec = (float4)(0.0f);
            for (int j = 0; j < token_count_div_4; ++j) {
                sum_vec += vload4(j, &head[i * token_count]);
            }
            sum_head_row_i = sum_vec.x + sum_vec.y + sum_vec.z + sum_vec.w;

            // --- Remainder Handling Part ---
            // This loop handles the leftover elements if token_count is not a multiple of 4.
            // It starts at the index right after the last element processed by the vectorized loop.
            // For example, if token_count is 127, this loop will start at j = 31 * 4 = 124
            // and run for j=124, j=125, and j=126.
            for (int j = token_count_div_4 * 4; j < token_count; ++j) {
                sum_head_row_i += head[i * token_count + j];
            }

            // Column sum remains scalar to avoid bad memory access patterns.
            for (int j = 0; j < token_count; ++j) {
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
    // Each work-item now handles a vector of 4 floats along the d-dimension.
    // The global work size for dimension 0 must be ceil(embedding_dim / 4.0).
    int d_vec = get_global_id(0);
    int h = get_global_id(1);

    int d = d_vec * 4; // Starting index for the float4

    if (h < mat_heights && d < embedding_dim) {
        int idx = h * embedding_dim + d;

        // Load scalar values once per work-item
        float pmh_h = pre_mh[h];
        float pmv_h = pre_mv[h];

        // Check boundary conditions to avoid reading past the buffer
        if (d + 3 < embedding_dim) {
            // Load 4 elements at once
            float4 grad_dh_vec = vload4(0, &grad_dh[d]);
            float4 grad_dv_vec = vload4(0, &grad_dv[d]);

            // Perform vectorized computation
            float4 grad_mh_vec = pmh_h * grad_dh_vec;
            float4 grad_mv_vec = pmv_h * grad_dv_vec;

            // Store 4 elements at once
            vstore4(grad_mh_vec, 0, &grad_mh[idx]);
            vstore4(grad_mv_vec, 0, &grad_mv[idx]);
        } else {
            // Handle the tail end of the data if embedding_dim is not a multiple of 4
            for (int i = 0; i < embedding_dim - d; ++i) {
                grad_mh[idx + i] = pmh_h * grad_dh[d + i];
                grad_mv[idx + i] = pmv_h * grad_dv[d + i];
            }
        }
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
        // Use float4 accumulators for vectorization over the embedding_dim
        float4 grad_dh_term_ij_vec = (float4)(0.0f);
        float4 grad_dv_term_ij_vec = (float4)(0.0f);

        int embedding_dim_div_4 = embedding_dim / 4;

        // Loops are reordered (h, then d) to make the d-loop innermost for vectorization
        for (int h = 0; h < mat_heights; ++h) {
            float k_h = k[i * mat_heights + h];
            float q_h = q[j * mat_heights + h];

            // Process 4 elements of the embedding dimension at a time
            for (int d_vec = 0; d_vec < embedding_dim_div_4; ++d_vec) {
                int d_idx = d_vec * 4;
                float4 mh_a_vec = vload4(0, &mh_a[h * embedding_dim + d_idx]);
                float4 mv_a_vec = vload4(0, &mv_a[h * embedding_dim + d_idx]);
                float4 grad_dh_vec = vload4(0, &grad_dh[d_idx]);
                float4 grad_dv_vec = vload4(0, &grad_dv[d_idx]);

                // Fused multiply-add for efficiency
                grad_dh_term_ij_vec = fma(k_h * mh_a_vec, grad_dh_vec, grad_dh_term_ij_vec);
                grad_dv_term_ij_vec = fma(q_h * mv_a_vec, grad_dv_vec, grad_dv_term_ij_vec);
            }
        }

        // Horizontally sum the vector accumulators to get the final scalar result
        float grad_dh_term_ij = grad_dh_term_ij_vec.x + grad_dh_term_ij_vec.y + grad_dh_term_ij_vec.z + grad_dh_term_ij_vec.w;
        float grad_dv_term_ij = grad_dv_term_ij_vec.x + grad_dv_term_ij_vec.y + grad_dv_term_ij_vec.z + grad_dv_term_ij_vec.w;

        // Handle the remainder if embedding_dim is not a multiple of 4
        int remainder_start = embedding_dim_div_4 * 4;
        for (int d = remainder_start; d < embedding_dim; ++d) {
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
                                          __global float* grad_kdotq, float scaling_factor, int row, int col, int size)
{
    int i = get_global_id(1); // Row index
    // Each work-item now handles a vector of 4 floats along the j-dimension.
    // The global work size for dimension 0 must be ceil(size / 4.0).
    int j_vec = get_global_id(0);
    int j = j_vec * 4; // Starting float index for this work-item

    if (i < row && j < size) {
        // Calculate the inverse scaling factor once to replace division with multiplication
        float inv_scaling_factor = (fabs(scaling_factor) > 1e-9f) ? (1.0f / scaling_factor) : 0.0f;
        int index = i * col + j;

        // --- Vectorized Path ---
        // Check if we can safely read and write a full float4 without going out of bounds.
        if (j + 3 < size) {
            // Load 4 elements at once
            float4 grad_head_vec = vload4(0, &grad_head[index]);
            float4 lota_derivative_vec = vload4(0, &lota_derivative[index]);

            // Perform the computation on the vector
            float4 result = grad_head_vec * lota_derivative_vec * inv_scaling_factor;

            // Store 4 elements at once
            vstore4(result, 0, &grad_kdotq[index]);
        }
        // --- Scalar Remainder Path ---
        // If we are at the edge, handle the remaining 1-3 elements individually.
        else {
            for (int k = 0; k < size - j; ++k) {
                grad_kdotq[index + k] = grad_head[index + k] * lota_derivative[index + k] * inv_scaling_factor;
            }
        }
    }
}


__kernel void kernelComputeGradK_Q(__global const float* grad_kdotq, __global const float* k, __global const float* q,
                                   __global float* grad_k, __global float* grad_q,
                                   int token_count, int mat_heights)
{
    int i = get_global_id(1);
    // Each work-item now computes 4 outputs along the h-dimension.
    // The global work size for dimension 0 must be ceil(mat_heights / 4.0).
    int h_vec = get_global_id(0);
    int h = h_vec * 4; // Starting h index for this work-item

    if (i < token_count && h < mat_heights) {
        // --- Vectorized Path ---
        if (h + 3 < mat_heights) {
            float4 sum_k = (float4)(0.0f); // Accumulator for grad_k
            float4 sum_q = (float4)(0.0f); // Accumulator for grad_q

            for (int j = 0; j < token_count; ++j) {
                // Load scalar values that are constant for this inner loop
                float grad_kdotq_ij = grad_kdotq[i * token_count + j]; // For grad_q
                float grad_kdotq_ji = grad_kdotq[j * token_count + i]; // For grad_k

                // Load 4 adjacent h values from k and q (coalesced access)
                float4 k_vec = vload4(0, &k[j * mat_heights + h]);
                float4 q_vec = vload4(0, &q[j * mat_heights + h]);

                // Fused multiply-add for performance
                sum_q = fma(grad_kdotq_ij, k_vec, sum_q);
                sum_k = fma(grad_kdotq_ji, q_vec, sum_k);
            }
            vstore4(sum_k, 0, &grad_k[i * mat_heights + h]);
            vstore4(sum_q, 0, &grad_q[i * mat_heights + h]);
        }
        // --- Scalar Remainder Path ---
        // Handle the case where mat_heights is not a multiple of 4
        else {
            for (int h_rem = h; h_rem < mat_heights; ++h_rem) {
                float sum_for_grad_k_ih = 0.0f;
                float sum_for_grad_q_ih = 0.0f;
                for (int j = 0; j < token_count; ++j) {
                    sum_for_grad_q_ih += grad_kdotq[i * token_count + j] * k[j * mat_heights + h_rem];
                    sum_for_grad_k_ih += grad_kdotq[j * token_count + i] * q[j * mat_heights + h_rem];
                }
                grad_k[i * mat_heights + h_rem] = sum_for_grad_k_ih;
                grad_q[i * mat_heights + h_rem] = sum_for_grad_q_ih;
            }
        }
    }
}


__kernel void kernelComputeGradMK_MQ(__global const float* grad_k, __global const float* grad_q,
                                     __global const float* k, __global const float* q,
                                     __global float* grad_mk, __global float* grad_mq,
                                     int token_count, int mat_heights, int embedding_dim)
{
    int h = get_global_id(1);
    // Each work-item now computes 4 outputs along the d-dimension.
    // The global work size for dimension 0 must be ceil(embedding_dim / 4.0).
    int d_vec = get_global_id(0);
    int d = d_vec * 4; // Starting d index for this work-item

    if (h < mat_heights && d < embedding_dim) {
        // --- Vectorized Path ---
        if (d + 3 < embedding_dim) {
            float4 sum_mk = (float4)(0.0f);
            float4 sum_mq = (float4)(0.0f);

            // Loop over the reduction dimension
            for (int i = 0; i < token_count; ++i) {
                // Load scalar values that are constant for this inner loop
                float grad_k_ih = grad_k[i * mat_heights + h];
                float grad_q_ih = grad_q[i * mat_heights + h];

                // Load 4 adjacent d values from k and q (coalesced access)
                float4 k_vec = vload4(0, &k[i * embedding_dim + d]);
                float4 q_vec = vload4(0, &q[i * embedding_dim + d]);

                sum_mk = fma(grad_k_ih, k_vec, sum_mk);
                sum_mq = fma(grad_q_ih, q_vec, sum_mq);
            }
            vstore4(sum_mk, 0, &grad_mk[h * embedding_dim + d]);
            vstore4(sum_mq, 0, &grad_mq[h * embedding_dim + d]);
        }
        // --- Scalar Remainder Path ---
        // Handle the case where embedding_dim is not a multiple of 4
        else {
            for (int d_rem = d; d_rem < embedding_dim; ++d_rem) {
                float sum_mk_hd = 0.0f;
                float sum_mq_hd = 0.0f;
                for (int i = 0; i < token_count; ++i) {
                    sum_mk_hd += grad_k[i * mat_heights + h] * k[i * embedding_dim + d_rem];
                    sum_mq_hd += grad_q[i * mat_heights + h] * q[i * embedding_dim + d_rem];
                }
                grad_mk[h * embedding_dim + d_rem] = sum_mk_hd;
                grad_mq[h * embedding_dim + d_rem] = sum_mq_hd;
            }
        }
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
    // Each work-item now handles a vector of 4 floats along the d-dimension.
    // The global work size for dimension 0 must be ceil(embedding_dim / 4.0).
    int d_vec = get_global_id(0);
    int d = d_vec * 4; // Starting float index for this work-item

    if (h < mat_heights && d < embedding_dim) {
        int idx = h * embedding_dim + d;

        // Load the scalar value once per work-item
        float pmv_h = pre_mv[h];

        // --- Vectorized Path ---
        // Check if we can safely process a full float4 without going out of bounds.
        if (d + 3 < embedding_dim) {
            // Load 4 elements from grad_dv
            float4 grad_dv_vec = vload4(0, &grad_dv[d]);
            // Perform vectorized computation
            float4 grad_mv_vec = pmv_h * grad_dv_vec;
            // Store 4 results at once
            vstore4(grad_mv_vec, 0, &grad_mv[idx]);
        }
        // --- Scalar Remainder Path ---
        // If we are near the end, process the remaining 1-3 elements individually.
        else {
            for (int i = 0; i < embedding_dim - d; ++i) {
                grad_mv[idx + i] = pmv_h * grad_dv[d + i];
            }
        }
    }
}

__kernel void kernelComputeGradHead_V(__global const float* q, __global const float* mv_a,
                                      __global const float* grad_dv, __global float* grad_head,
                                      int token_count, int mat_heights, int embedding_dim)
{
    int i = get_global_id(1);
    int j = get_global_id(0);

    if (i < token_count && j < token_count) {
        // Use a float4 accumulator to vectorize the reduction over embedding_dim
        float4 grad_dv_term_j_vec = (float4)(0.0f);
        int embedding_dim_div_4 = embedding_dim / 4;

        // Loop over h, the outer part of the reduction
        for (int h = 0; h < mat_heights; ++h) {
            float q_jh = q[j * mat_heights + h];

            // --- Vectorized Reduction over d ---
            // The d-loop is now innermost to allow for contiguous memory access
            for (int d_vec = 0; d_vec < embedding_dim_div_4; ++d_vec) {
                int d_idx = d_vec * 4;
                float4 mv_a_vec = vload4(0, &mv_a[h * embedding_dim + d_idx]);
                float4 grad_dv_vec = vload4(0, &grad_dv[d_idx]);
                // Fused multiply-add (fma) is more efficient
                grad_dv_term_j_vec = fma(q_jh * mv_a_vec, grad_dv_vec, grad_dv_term_j_vec);
            }
        }

        // Horizontally sum the vector accumulator to get the partial scalar result
        float grad_dv_term_j = grad_dv_term_j_vec.x + grad_dv_term_j_vec.y + grad_dv_term_j_vec.z + grad_dv_term_j_vec.w;

        // --- Scalar Remainder Path ---
        // Handle the tail end of the embedding_dim if it's not a multiple of 4
        int remainder_start = embedding_dim_div_4 * 4;
        for (int d = remainder_start; d < embedding_dim; ++d) {
            float q_mv_jd = 0.0f;
            for (int h = 0; h < mat_heights; ++h) {
                q_mv_jd += q[j * mat_heights + h] * mv_a[h * embedding_dim + d];
            }
            grad_dv_term_j += q_mv_jd * grad_dv[d];
        }

        // The result is identical for all `i` for a given `j`, as per the original logic
        grad_head[i * token_count + j] = grad_dv_term_j;
    }
}


__kernel void kernelComputeGradQ_V(__global const float* grad_kdotq, __global const float* k,
                                   __global float* grad_q, int token_count, int mat_heights)
{
    int j = get_global_id(1);
    // Each work-item now handles 4 elements along the h-dimension.
    // The global work size for dimension 0 must be ceil(mat_heights / 4.0).
    int h_vec = get_global_id(0);
    int h = h_vec * 4;

    if (j < token_count && h < mat_heights) {
        // --- Vectorized Path ---
        // Check if we can safely process a full float4 without going out of bounds.
        if (h + 3 < mat_heights) {
            float4 sum_vec = (float4)(0.0f);

            // Reduction loop
            for (int i = 0; i < token_count; ++i) {
                // Load the scalar value once for the inner loop
                float grad_val = grad_kdotq[i * token_count + j];
                // Load 4 contiguous elements from k (coalesced read)
                float4 k_vec = vload4(0, &k[i * mat_heights + h]);
                // Use fused multiply-add for better performance
                sum_vec = fma(k_vec, grad_val, sum_vec);
            }
            // Store 4 results at once (coalesced write)
            vstore4(sum_vec, 0, &grad_q[j * mat_heights + h]);
        }
        // --- Scalar Remainder Path ---
        // Handle the case where mat_heights is not a multiple of 4.
        else {
            for (int h_rem = h; h_rem < mat_heights; ++h_rem) {
                float sum_for_grad_q_jh = 0.0f;
                for (int i = 0; i < token_count; ++i) {
                    sum_for_grad_q_jh += k[i * mat_heights + h_rem] * grad_kdotq[i * token_count + j];
                }
                grad_q[j * mat_heights + h_rem] = sum_for_grad_q_jh;
            }
        }
    }
}


__kernel void kernelComputeGradMQ_V(__global const float* grad_q, __global const float* q,
                                    __global float* grad_mq,
                                    int token_count, int mat_heights, int embedding_dim)
{
    int h = get_global_id(0);
    // Each work-item now handles 4 elements along the d-dimension.
    // The global work size for dimension 1 must be ceil(embedding_dim / 4.0).
    int d_vec = get_global_id(1);
    int d = d_vec * 4;

    if (h < mat_heights && d < embedding_dim) {
        // --- Vectorized Path ---
        if (q != NULL && (d + 3 < embedding_dim)) {
            float4 sum_vec = (float4)(0.0f);

            for (int i = 0; i < token_count; ++i) {
                float grad_q_val = grad_q[i * mat_heights + h];
                // Coalesced read from the q buffer
                float4 q_vec = vload4(0, &q[i * embedding_dim + d]);
                sum_vec = fma(grad_q_val, q_vec, sum_vec);
            }
            // Coalesced write to the grad_mq buffer
            vstore4(sum_vec, 0, &grad_mq[h * embedding_dim + d]);
        }
        // --- Scalar Remainder Path ---
        // Handles both the q == NULL case and the remainder of embedding_dim.
        else {
            float sum_mq_hd = 0.0f;
            if (q != NULL) {
                for (int d_rem = d; d_rem < embedding_dim; ++d_rem) {
                    sum_mq_hd = 0.0f; // Reset for each element in the remainder
                    for (int i = 0; i < token_count; ++i) {
                        sum_mq_hd += grad_q[i * mat_heights + h] * q[i * embedding_dim + d_rem];
                    }
                    grad_mq[h * embedding_dim + d_rem] = sum_mq_hd;
                }
            } else { // If q is NULL, just write zero
                 for (int d_rem = d; d_rem < embedding_dim; ++d_rem) {
                    grad_mq[h * embedding_dim + d_rem] = 0.0f;
                 }
            }
        }
    }
}


__kernel void kernelComputeGradMKCorrection(__global const float* grad_mq, __global const float* q, __global const float* k,
                                            __global float* grad_mk_correction,
                                            int token_count, int mat_heights, int embedding_dim)
{
    int h = get_global_id(1);
    // Each work-item now handles 4 elements along the d-dimension.
    // The global work size for dimension 0 must be ceil(embedding_dim / 4.0).
    int d_vec = get_global_id(0);
    int d = d_vec * 4;

    if (h < mat_heights && d < embedding_dim) {
        // --- Step 1: Perform the O(N) reductions (algebraic simplification) ---
        // These sums are scalar and depend only on h, so they are calculated once per work-item.
        // Note: These loops have strided memory access and are not good candidates for vectorization.
        float sum_q_h = 0.0f;
        for (int j = 0; j < token_count; ++j) {
            sum_q_h += q[j * mat_heights + h];
        }

        float sum_k_h = 0.0f;
        for (int i = 0; i < token_count; ++i) {
            sum_k_h += k[i * mat_heights + h];
        }

        // Pre-calculate the combined term
        float correction_term = -sum_q_h * sum_k_h;

        // --- Step 2: Apply the correction in a vectorized manner over d ---

        // --- Vectorized Path ---
        if (d + 3 < embedding_dim) {
            // Load 4 contiguous elements from grad_mq
            float4 grad_mq_vec = vload4(0, &grad_mq[h * embedding_dim + d]);
            // Calculate the final result with a vector-scalar multiply
            float4 result_vec = grad_mq_vec * correction_term;
            // Store 4 results at once
            vstore4(result_vec, 0, &grad_mk_correction[h * embedding_dim + d]);
        }
        // --- Scalar Remainder Path ---
        // Handle the tail end where embedding_dim is not a multiple of 4.
        else {
            for (int d_rem = d; d_rem < embedding_dim; ++d_rem) {
                grad_mk_correction[h * embedding_dim + d_rem] = grad_mq[h * embedding_dim + d_rem] * correction_term;
            }
        }
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
    // Each work-item now handles 4 elements along the d-dimension.
    // The global work size for dimension 0 must be ceil(embedding_dim / 4.0).
    int d_vec = get_global_id(0);
    int d = d_vec * 4; // Starting float index for this work-item

    int h = get_global_id(1); // Corresponds to mat_heights (height-like)

    if (h < mat_heights && d < embedding_dim) {
        // --- Vectorized Path ---
        // This path is taken if we can safely read and write a full float4.
        if (d + 3 < embedding_dim) {
            float4 sum_mk_vec = (float4)(0.0f);
            float4 sum_mq_vec = (float4)(0.0f);

            // Main reduction loop over token_count
            for (int i = 0; i < token_count; ++i) {
                // Check for grad_mk computation
                if (grad_k != NULL && k_embed != NULL) {
                    float grad_k_val = grad_k[i * mat_heights + h];
                    float4 k_embed_vec = vload4(0, &k_embed[i * embedding_dim + d]);
                    sum_mk_vec = fma(grad_k_val, k_embed_vec, sum_mk_vec);
                }

                // Check for grad_mq computation
                if (grad_q != NULL && q_embed != NULL) {
                    float grad_q_val = grad_q[i * mat_heights + h];
                    float4 q_embed_vec = vload4(0, &q_embed[i * embedding_dim + d]);
                    sum_mq_vec = fma(grad_q_val, q_embed_vec, sum_mq_vec);
                }
            }

            // Store the final vector results
            vstore4(sum_mk_vec, 0, &grad_mk[h * embedding_dim + d]);
            vstore4(sum_mq_vec, 0, &grad_mq[h * embedding_dim + d]);
        }
        // --- Scalar Remainder Path ---
        // This path handles the last 1-3 elements if embedding_dim is not a multiple of 4.
        else {
            for (int d_rem = d; d_rem < embedding_dim; ++d_rem) {
                float sum_mk_hd = 0.0f;
                float sum_mq_hd = 0.0f;
                for (int i = 0; i < token_count; ++i) {
                    if (grad_k != NULL && k_embed != NULL) {
                        sum_mk_hd += grad_k[i * mat_heights + h] * k_embed[i * embedding_dim + d_rem];
                    }
                    if (grad_q != NULL && q_embed != NULL) {
                        sum_mq_hd += grad_q[i * mat_heights + h] * q_embed[i * embedding_dim + d_rem];
                    }
                }
                grad_mk[h * embedding_dim + d_rem] = sum_mk_hd;
                grad_mq[h * embedding_dim + d_rem] = sum_mq_hd;
            }
        }
    }
}
