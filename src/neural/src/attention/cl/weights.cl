
// update EV rows
__kernel void kernelUpdateEVrows(__global float* d_EV_rows, __global const float* d_vector_to_add,
    int num_rows_to_update, int num_cols)
{
    int row_idx = get_global_id(0); // Each work-item handles one row

    if (row_idx < num_rows_to_update) {
        for (int col_idx = 0; col_idx < num_cols; ++col_idx) {
            d_EV_rows[row_idx * num_cols + col_idx] += d_vector_to_add[col_idx];
        }
    }
}

__kernel void kernelUpdateWeights_EH_EV(__global float* mh_a, __global float* mv_a, __global float* mq_a, __global float* mk_a,
                                        __global float* eh, __global float* ev,
                                        __global const float* grad_mh, __global const float* grad_mv,
                                        __global const float* grad_mq, __global const float* grad_mk,
                                        __global const float* grad_eh, __global const float* grad_ev_scaled,
                                        float learning_rate, int update_eh, int update_ev,
                                        int mat_heights, int embedding_dim, int context_win)
{
    int idx = get_global_id(0);
    int ev_size = context_win * embedding_dim; // Define ev_size
    int matrix_size = mat_heights * embedding_dim;
    if (idx < matrix_size) {
        mh_a[idx] -= learning_rate * grad_mh[idx];
        mv_a[idx] -= learning_rate * grad_mv[idx];
        mq_a[idx] -= learning_rate * grad_mq[idx];
        mk_a[idx] -= learning_rate * grad_mk[idx];
    }
    if (update_eh != 0 && idx < embedding_dim) {
        eh[idx] -= learning_rate * grad_eh[idx];
    }
    if (update_ev != 0) {
        if (idx < ev_size) {
            int embed_idx = idx % embedding_dim;
            ev[idx] -= learning_rate * grad_ev_scaled[embed_idx];
        }
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
    // only update EH when updateEH is true, this for all heads of blocks,except first column
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

__kernel void kernelUpdateWeights_EV_V(__global float* mv_a, __global float* mq_a, __global float* mk_a, __global float* ev,
                                       __global const float* grad_mv, __global const float* grad_mq, // grad_mv, grad_mq are mat_heights x embedding_dim
                                       __global const float* grad_mk_correction, // grad_mk_correction is mat_heights x embedding_dim
                                       __global const float* grad_ev_full,
                                       float learning_rate,
                                       int update_ev, int mat_heights, int embedding_dim, int context_win)
{
    int idx = get_global_id(0);
    int matrix_size = mat_heights * embedding_dim;
    if (idx < matrix_size) {
        mv_a[idx] -= learning_rate * grad_mv[idx];
        mq_a[idx] -= learning_rate * grad_mq[idx];
        mk_a[idx] -= learning_rate * grad_mk_correction[idx];
    }
    // update for all blocks, except first block
    if (update_ev != 0) {
        int ev_size = context_win * embedding_dim;
        if (idx < ev_size) {
            ev[idx] -= learning_rate * grad_ev_full[idx];
        }
    }
}

// Provided reference, with correction for `total_gradient != NULL`
__kernel void kernelUpdateWeights_General(__global float* weights,
    __global const float* gradients, float learning_rate,
    float lambda_l1, float lambda_l2, float max_grad_clip_value,
    int total_elements)
{
    int idx = get_global_id(0);
    if (idx < total_elements) {
        // This kernel explicitly handles grad_ptr potentially being NULL, hence no need for apply_weight_grad_process_and_clip_elastic
        // to handle NULLs within its body, but the check is still needed here for gradients.
        float current_weight = weights[idx];
        float error_gradient = (gradients != NULL) ? gradients[idx] : 0.0f; // Handle NULL gradients

        float l1_reg_term = lambda_l1 * sign_f(current_weight); // L1 regularization term
        float l2_reg_term = 2.0f * lambda_l2 * current_weight;  // L2 regularization term (derivative of lambda*w^2), now consistent

        float total_gradient = error_gradient + l1_reg_term + l2_reg_term;
        if (isnan(total_gradient)) {
            total_gradient = 0.0f;
        } else if (isinf(total_gradient)) {
            total_gradient = copysign(FLT_MAX, total_gradient);
        }
        // Apply element-wise gradient clipping
        if (fabs(total_gradient) > max_grad_clip_value) {
            total_gradient = copysign(FLT_MAX, total_gradient);
        }

        weights[idx] -= learning_rate * total_gradient;
    }
}


__kernel void kernelUpdateSimple(__global float* weights_to_update, __global const float* gradients, float lr, unsigned int n_elements)
{
    int idx = get_global_id(0);
    if (idx < n_elements) {
        if (gradients == NULL) {
            return;
        }
        float tgradients = gradients[idx];
        // NOTE: Using a hardcoded clip value. It is recommended to pass max_grad_clip_value
        // to this kernel for consistency with other update kernels.
        const float clip_val = 1000.0f;

        if (isnan(tgradients)) {
            tgradients = 0.0f;
        }
        else if (isinf(tgradients)) {
            tgradients = copysign(clip_val, tgradients);
        }

        // Apply element-wise clipping for large gradients
        if (fabs(tgradients) > clip_val) tgradients = copysign(clip_val, tgradients);

        weights_to_update[idx] -= lr * tgradients;
    }
}