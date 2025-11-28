// Helper for sign function for float4
inline float4 signf4(float4 x) {
    return (float4)(sign_f(x.x), sign_f(x.y), sign_f(x.z), sign_f(x.w));
}

// Helper for isnan for float4
inline int4 isnan_float4(float4 x) {
    return (int4)(isnan(x.x), isnan(x.y), isnan(x.z), isnan(x.w));
}

// Helper for isinf for float4
inline int4 isinf_float4(float4 x) {
    return (int4)(isinf(x.x), isinf(x.y), isinf(x.z), isinf(x.w));
}

// Helper for fabs for float4
inline float4 fabs_float4(float4 x) {
    return (float4)(fabs(x.x), fabs(x.y), fabs(x.z), fabs(x.w));
}

// Helper for copysign for float4
inline float4 copysign_float4(float4 mag, float4 sgn) {
    return (float4)(copysign(mag.x, sgn.x), copysign(mag.y, sgn.y), copysign(mag.z, sgn.z), copysign(mag.w, sgn.w));
}

// Inline function to apply Elastic Net regularization and gradient clipping for single float
inline void apply_elastic_net_and_clip(volatile __global float* weight_ptr, __global const float* gradient_ptr,
                                       int index, float learning_rate, float lambda_l1, float lambda_l2,
                                       float max_grad_clip_value)
{
    float current_weight = weight_ptr[index];
    float error_gradient = (gradient_ptr != NULL) ? gradient_ptr[index] : 0.0f;

    float l1_reg_term = lambda_l1 * sign_f(current_weight);
    // taking 0.5 * current_weight^2 -> lambda * current_weight
    float l2_reg_term = lambda_l2 * current_weight;

    float total_gradient = error_gradient + l1_reg_term + l2_reg_term;

    if (isnan(total_gradient)) {
        total_gradient = 0.0f;
    }
    else if (isinf(total_gradient)) {
        total_gradient = copysign(MAXFLOAT, total_gradient);
    }
    // Apply element-wise gradient clipping
    // if (fabs(total_gradient) > max_grad_clip_value) { total_gradient = copysign(max_grad_clip_value, total_gradient); }

    weight_ptr[index] -= learning_rate * total_gradient;
}


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


__kernel void kernelUpdateWeightsHeadHVElastic(__global float* mh_a, __global float* mv_a, __global float* mq_a, __global float* mk_a,
                                        __global float* eh, __global float* ev,
                                        __global const float* grad_mh, __global const float* grad_mv,
                                        __global const float* grad_mq, __global const float* grad_mk,
                                        __global const float* grad_eh, __global const float* grad_ev_scaled,
                                        int update_eh, int update_ev, int mat_heights, int embedding_dim, int context_win,
                                        float learning_rate, float lambda_l1, float lambda_l2, float max_grad_clip_value)
{
    int idx = get_global_id(0);
    int ev_size = context_win * embedding_dim;
    int matrix_size = mat_heights * embedding_dim;

    if (idx < matrix_size) {
        if(mh_a != NULL) apply_elastic_net_and_clip(mh_a, grad_mh, idx, learning_rate, lambda_l1, lambda_l2, max_grad_clip_value);
        if(mv_a != NULL) apply_elastic_net_and_clip(mv_a, grad_mv, idx, learning_rate, lambda_l1, lambda_l2, max_grad_clip_value);
        if(mq_a != NULL) apply_elastic_net_and_clip(mq_a, grad_mq, idx, learning_rate, lambda_l1, lambda_l2, max_grad_clip_value);
        if(mk_a != NULL) apply_elastic_net_and_clip(mk_a, grad_mk, idx, learning_rate, lambda_l1, lambda_l2, max_grad_clip_value);
    }

    if (update_eh != 0 && idx < embedding_dim) {
        if(eh != NULL) apply_elastic_net_and_clip(eh, grad_eh, idx, learning_rate, lambda_l1, lambda_l2, max_grad_clip_value);
    }

    if (update_ev != 0) {
        if (idx < ev_size) {
            int embed_idx = idx % embedding_dim;
            if(ev != NULL) apply_elastic_net_and_clip(ev, grad_ev_scaled, idx, learning_rate, lambda_l1, lambda_l2, max_grad_clip_value);
        }
    }
}


__kernel void kernelUpdateWeightsHeadElastic(__global float* mh_a, __global float* mv_a, __global float* mq_a, __global float* mk_a,
                                            __global float* eh,
                                            __global const float* grad_mh, __global const float* grad_mv,
                                            __global const float* grad_mq, __global const float* grad_mk,
                                            __global const float* grad_eh,
                                            int update_eh, int mat_heights, int embedding_dim,
                                            float learning_rate, float lambda_l1, float lambda_l2, float max_grad_clip_value)
{
    int idx = get_global_id(0);
    int matrix_size = mat_heights * embedding_dim;

    if (idx < matrix_size) {
        if(mh_a != NULL) apply_elastic_net_and_clip(mh_a, grad_mh, idx, learning_rate, lambda_l1, lambda_l2, max_grad_clip_value);
        if(mv_a != NULL) apply_elastic_net_and_clip(mv_a, grad_mv, idx, learning_rate, lambda_l1, lambda_l2, max_grad_clip_value);
        if(mq_a != NULL) apply_elastic_net_and_clip(mq_a, grad_mq, idx, learning_rate, lambda_l1, lambda_l2, max_grad_clip_value);
        if(mk_a != NULL) apply_elastic_net_and_clip(mk_a, grad_mk, idx, learning_rate, lambda_l1, lambda_l2, max_grad_clip_value);
    }
    // only update EH when updateEH is true, this for all heads of blocks,except first column
    if (update_eh != 0 && idx < embedding_dim) {
        if(eh != NULL) apply_elastic_net_and_clip(eh, grad_eh, idx, learning_rate, lambda_l1, lambda_l2, max_grad_clip_value);
    }
}


__kernel void kernelUpdateSimple(__global float* weights_to_update, __global const float* gradients, float lr, unsigned int n_elements)
{
    int idx = get_global_id(0);
    if (idx < n_elements) {
        if (gradients == NULL) {
            return;
        }
        float current_weight = weights_to_update[idx];
        float tgradients = gradients[idx];
        // NOTE: Using a hardcoded clip value. It is recommended to pass max_grad_clip_value
        // to this kernel for consistency with other update kernels.
        const float clip_val = 1.0f;

        if (isnan(tgradients)) {
            tgradients = 0.0f;
        }
        else if (isinf(tgradients)) {
            tgradients = copysign(clip_val, tgradients);
        }

        // Apply element-wise clipping for large gradients
        if (fabs(tgradients) > clip_val) tgradients = copysign(clip_val, tgradients);

        // Apply weight decay: w = w * (1 - lr * weight_decay) - lr * gradient
        // weights_to_update[idx] = current_weight * (1.0f - lr * weight_decay) - lr * tgradients;

        weights_to_update[idx] -= lr * tgradients;
    }
}

__kernel void kernelUpdateSimple_Elastic(__global float* weights_to_update, __global const float* gradients, float lr, unsigned int n_elements,
                                 float lambda_l1, float lambda_l2, float max_grad_clip_value)
{
    int idx = get_global_id(0);
    if (idx < n_elements) {
        if (gradients == NULL) {
            // Apply regularization even if gradients are NULL
            float current_weight = weights_to_update[idx];
            float l1_reg_term = lambda_l1 * sign_f(current_weight);
            float l2_reg_term = lambda_l2 * current_weight;
            float total_gradient = l1_reg_term + l2_reg_term;

            if (isnan(total_gradient)) {
                total_gradient = 0.0f;
            } else if (isinf(total_gradient)) {
                total_gradient = copysign(MAXFLOAT, total_gradient);
            }
            if (fabs(total_gradient) > max_grad_clip_value) {
                total_gradient = copysign(max_grad_clip_value, total_gradient);
            }
            weights_to_update[idx] -= lr * total_gradient;
            return;
        }

        float current_weight = weights_to_update[idx];
        float error_gradient = gradients[idx];

        float l1_reg_term = lambda_l1 * sign_f(current_weight);
        float l2_reg_term = lambda_l2 * current_weight;

        float total_gradient = error_gradient + l1_reg_term + l2_reg_term;

        // NOTE: Using max_grad_clip_value passed to this kernel for consistency.
        // The original `clip_val` of 1000.0f is now replaced.
        const float clip_val = max_grad_clip_value;

        if (isnan(total_gradient)) {
            total_gradient = 0.0f;
        }
        else if (isinf(total_gradient)) {
            total_gradient = copysign(clip_val, total_gradient);
        }

        // Apply element-wise clipping for large gradients
        if (fabs(total_gradient) > clip_val) total_gradient = copysign(clip_val, total_gradient);

        // Apply weight decay: w = w * (1 - lr * weight_decay) - lr * gradient
        // weights_to_update[idx] = current_weight * (1.0f - lr * weight_decay) - lr * tgradients;

        weights_to_update[idx] -= lr * total_gradient;
    }
}

__kernel void kernelUpdateWeightsGeneral(__global float* weights, __global const float* gradients, float learning_rate,
    float lambda_l1, float lambda_l2, int total_elements)
{
    int idx = get_global_id(0);
    if (idx < total_elements) {
        // This kernel explicitly handles grad_ptr potentially being NULL, hence no need for apply_weight_grad_process_and_clip_elastic
        // to handle NULLs within its body, but the check is still needed here for gradients.
        float current_weight = weights[idx];
        float error_gradient = (gradients != NULL) ? gradients[idx] : 0.0f; // Handle NULL gradients

        float l1_reg_term = lambda_l1 * sign_f(current_weight); // L1 regularization term
        float l2_reg_term = lambda_l2 * current_weight;  // L2 regularization term (derivative of lambda*w^2), now consistent

        float total_gradient = error_gradient + l1_reg_term + l2_reg_term;
        if (isnan(total_gradient)) {
            total_gradient = 0.0f;
        } else if (isinf(total_gradient)) {
            total_gradient = copysign(MAXFLOAT, total_gradient);
        }

        // Apply weight decay: w = w * (1 - lr * weight_decay) - lr * gradient
        // weights[idx] = current_weight * (1.0f - learning_rate * weight_decay) - learning_rate * total_gradient;

        weights[idx] -= learning_rate * total_gradient;
    }
}

__kernel void kernelUpdateWeightsGeneral_f4(__global float* weights, __global const float* gradients, float learning_rate,
    float lambda_l1, float lambda_l2, int total_elements)
{
    int idx = get_global_id(0);
    int total_elements_f4 = total_elements / 4;

    if (idx < total_elements_f4) {
        volatile __global float4* weights_f4 = (__global float4*)weights;
        __global const float4* gradients_f4 = (__global const float4*)gradients;

        float4 current_weight = weights_f4[idx];
        float4 error_gradient = (gradients_f4 != NULL) ? gradients_f4[idx] : (float4)(0.0f);

        float4 l1_reg_term = lambda_l1 * sign(current_weight);
        float4 l2_reg_term = lambda_l2 * current_weight;

        float4 total_gradient = error_gradient + l1_reg_term + l2_reg_term;

        // Handle NaNs and Infs for each component
        total_gradient.x = isnan(total_gradient.x) ? 0.0f : (isinf(total_gradient.x) ? copysign(MAXFLOAT, total_gradient.x) : total_gradient.x);
        total_gradient.y = isnan(total_gradient.y) ? 0.0f : (isinf(total_gradient.y) ? copysign(MAXFLOAT, total_gradient.y) : total_gradient.y);
        total_gradient.z = isnan(total_gradient.z) ? 0.0f : (isinf(total_gradient.z) ? copysign(MAXFLOAT, total_gradient.z) : total_gradient.z);
        total_gradient.w = isnan(total_gradient.w) ? 0.0f : (isinf(total_gradient.w) ? copysign(MAXFLOAT, total_gradient.w) : total_gradient.w);

        weights_f4[idx] -= learning_rate * total_gradient;
    }
    // Handle remaining elements if total_elements is not a multiple of 4
    if (idx == total_elements_f4 && (total_elements % 4) != 0) {
        for (int i = 0; i < (total_elements % 4); ++i) {
            int current_idx = total_elements_f4 * 4 + i;
            float current_weight = weights[current_idx];
            float error_gradient = (gradients != NULL) ? gradients[current_idx] : 0.0f;
            float l1_reg_term = lambda_l1 * sign_f(current_weight);
            float l2_reg_term = lambda_l2 * current_weight;
            float total_gradient = error_gradient + l1_reg_term + l2_reg_term;
            if (isnan(total_gradient)) {
                total_gradient = 0.0f;
            } else if (isinf(total_gradient)) {
                total_gradient = copysign(MAXFLOAT, total_gradient);
            }

            // Apply weight decay: w = w * (1 - lr * weight_decay) - lr * gradient
            // weights[current_idx] = current_weight * (1.0f - learning_rate * weight_decay) - lr * total_gradient;

            weights[current_idx] -= learning_rate * total_gradient;
        }
    }

}


__kernel void updateEmbeddings(__global float* embeddings, __global const float* gradientsVector, float learning_rate,
    float lambda_l1, float lambda_l2, int embeddingRows, int embeddingDim)
{
    int global_id = get_global_id(0); // This will be the row index (embeddingRows)

    if (global_id < embeddingRows) {
        // Calculate the starting index for the current row in the embeddings matrix
        int row_start_idx = global_id * embeddingDim;

        // Process elements in chunks of 4 (float4)
        int num_elements_f4 = embeddingDim / 4;
        for (int i = 0; i < num_elements_f4; ++i) {
            int current_f4_idx = row_start_idx / 4 + i; // Index for float4 in the global memory

            volatile __global float4* embeddings_f4 = (__global float4*)embeddings;
            __global const float4* gradientsVector_f4 = (__global const float4*)gradientsVector;

            float4 current_weight = embeddings_f4[current_f4_idx];
            float4 error_gradient = (gradientsVector_f4 != NULL) ? gradientsVector_f4[i] : (float4)(0.0f); // gradientsVector is size embeddingDim

            float4 l1_reg_term = lambda_l1 * sign(current_weight);
            float4 l2_reg_term = lambda_l2 * current_weight;

            float4 total_gradient = error_gradient + l1_reg_term + l2_reg_term;

            // Handle NaNs and Infs for each component
            total_gradient.x = isnan(total_gradient.x) ? 0.0f : (isinf(total_gradient.x) ? copysign(MAXFLOAT, total_gradient.x) : total_gradient.x);
            total_gradient.y = isnan(total_gradient.y) ? 0.0f : (isinf(total_gradient.y) ? copysign(MAXFLOAT, total_gradient.y) : total_gradient.y);
            total_gradient.z = isnan(total_gradient.z) ? 0.0f : (isinf(total_gradient.z) ? copysign(MAXFLOAT, total_gradient.z) : total_gradient.z);
            total_gradient.w = isnan(total_gradient.w) ? 0.0f : (isinf(total_gradient.w) ? copysign(MAXFLOAT, total_gradient.w) : total_gradient.w);

            embeddings_f4[current_f4_idx] -= learning_rate * total_gradient;
        }

        // Handle remaining elements if embeddingDim is not a multiple of 4
        for (int i = num_elements_f4 * 4; i < embeddingDim; ++i) {
            int current_idx = row_start_idx + i;
            float current_weight = embeddings[current_idx];
            float error_gradient = (gradientsVector != NULL) ? gradientsVector[i] : 0.0f;
            float l1_reg_term = lambda_l1 * sign_f(current_weight);
            float l2_reg_term = lambda_l2 * current_weight;
            float total_gradient = error_gradient + l1_reg_term + l2_reg_term;
            if (isnan(total_gradient)) {
                total_gradient = 0.0f;
            } else if (isinf(total_gradient)) {
                total_gradient = copysign(MAXFLOAT, total_gradient);
            }
            embeddings[current_idx] -= learning_rate * total_gradient;
        }
    }
}
