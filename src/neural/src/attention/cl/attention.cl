// Helper macro for indexing flattened matrix (assuming row-major)
#define IDX(row, col, dim) ((row) * (dim) + (col))

// Enable extensions for atomics and potentially double precision (which might include float atomics)
// #pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
// #pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable
// #pragma OPENCL EXTENSION cl_khr_fp64 : enable // For double support
// #pragma OPENCL EXTENSION cl_khr_float_atomics : enable // Not supported on target, using manual implementation

// --- Static helper cl_compute_dot_product_vec is NO LONGER NEEDED by these kernels ---
// --- It can be removed if no other kernels use it ---

// Helper for sign function
inline float sign_f(float x) {
    return (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f);
}

__kernel void vectorAddKernel_attention(__global const float* A,
    __global const float* B,
    __global float* C, int len)
{
    int idx = get_global_id(0);
    if (idx < len) {
        C[idx] = A[idx] + B[idx];
    }
}

/**------------------------------------MULTIPLICATION------------------------------------**/

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


__kernel void kernelElementwiseMultiply(__global float* target_and_output, __global const float* factor, int size) {
    int idx = get_global_id(0);
    if (idx < size) {
        target_and_output[idx] *= factor[idx];
    }
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



// -----------------------------

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
             // Apply self-attention mask: only sum if j <= i (sum head[j][i] for j from 0 to i)
             if (isSelfAttention == 0 || j <= i) {
                col_sum_l += d_head[j * num_tokens + i];
             }
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
        // Use custom atomic_add_float since cl_khr_float_atomics is not guaranteed
        atomic_add_float(&d_dh_accum[h_idx], total_dh_for_h_idx);
        atomic_add_float(&d_dv_accum[h_idx], total_dv_for_h_idx);

        // If float atomics are unavailable, this needs a different reduction strategy.
    }
}

/**------------------------------------BACKPROP------------------------------------**/

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

__kernel void kernelComputeGradientsEH_EV(__global const float* eh, __global const float* expected_h,
                                          __global float* grad_eh, __global float* grad_ev_scaled, int embedding_dim)
{
    int idx = get_global_id(0);
    if (idx < embedding_dim) {
        float pred = eh[idx];
        float label = expected_h[idx];
        // The gradient of BCE loss w.r.t. the logits (pre-sigmoid input) is simply (prediction - label).
        // This is numerically stable and avoids the division by (pred * (1-pred)), which explodes when pred is near 0 or 1.
        float grad = pred - label; // Declaration was missing
        grad_eh[idx] = grad;
        grad_ev_scaled[idx] = grad * 0.1f;
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
            float grad = (pred - label) / (pred * (1.0f - pred));
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
        float sum_for_grad_k_ih = 0.0f; // Accumulator for grad_K[i][h] = sum_j(grad_kdotq[j][i] * Q[j][h])
        float sum_for_grad_q_ih = 0.0f; // Accumulator for grad_Q[i][h] = sum_j(grad_kdotq[i][j] * K[j][h])

        // Correctly calculate grad_Q[i][h] = sum_j (grad_kdotq[i][j] * K[j][h])
        for (int j = 0; j < token_count; ++j) {
            sum_for_grad_q_ih += grad_kdotq[i * token_count + j] * k[j * mat_heights + h];
        }

        // Correctly calculate grad_K[i][h] = sum_j (grad_kdotq[j][i] * Q[j][h])
        for (int j = 0; j < token_count; ++j) {
            sum_for_grad_k_ih += grad_kdotq[j * token_count + i] * q[j * mat_heights + h];
        }
        grad_k[i * mat_heights + h] = sum_for_grad_k_ih; // Store results using row-major indexing
        grad_q[i * mat_heights + h] = sum_for_grad_q_ih; // Store results using row-major indexing
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
                                   __global float* grad_q, int token_count, int mat_heights)
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
    int h = get_global_id(0); // Parallelize over mat_heights
    int d = get_global_id(1); // Parallelize over embedding_dim
    if (h < mat_heights && d < embedding_dim) {
        float sum_mq_hd = 0.0f;
        if (q != NULL) { // Check if q (q_embed) is not NULL
            for (int i = 0; i < token_count; ++i) {
                // The host should ensure 'q' (representing q_embed) is a valid buffer.
                sum_mq_hd += grad_q[i * mat_heights + h] * q[i * embedding_dim + d];
            }
        }
        grad_mq[h * embedding_dim + d] = sum_mq_hd;
    }
}

__kernel void kernelComputeGradMKCorrection(__global const float* grad_mq, __global const float* q, __global const float* k,
                                            __global float* grad_mk_correction,
                                            int token_count, int mat_heights, int embedding_dim) // grad_mq is mat_heights x embedding_dim
{
    // q and k are token_count x mat_heights
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
    int d = get_global_id(0); // Corresponds to embedding_dim (width-like)
    int h = get_global_id(1); // Corresponds to mat_heights (height-like)
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

__kernel void kernelUpdateSimple(__global float* weights_to_update, __global const float* gradients, float lr, unsigned int n_elements)
{
    int idx = get_global_id(0);
    if (idx < n_elements) {
        if (gradients != NULL) {
             weights_to_update[idx] -= lr * gradients[idx];
        }
    }
}

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


////////////////////////////
