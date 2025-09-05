#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "include/attention.hpp"
#include <vector>
#include <iostream>
#include <stdexcept>
#include <numeric>
#include <cmath>
#include <maths.hpp>
#include <string>

// --- CUDA Device-side Utility Functions ---

/**
 * @brief Processes, sanitizes, and clips a gradient value, then applies it to a parameter.
 *        Used for parameters that do NOT have regularization (e.g., EH, EV, or raw Adam input gradients).
 * @param param_ptr         Pointer to the parameter value to be updated.
 * @param raw_grad_val_ptr  Pointer to the raw gradient value. Can be NULL if gradient is optional.
 * @param lr                Learning rate.
 * @param clip_val          Maximum absolute value for gradient clipping.
 */
__device__ inline void apply_plain_grad_process_and_clip(float* param_ptr, const float* raw_grad_val_ptr, float lr, float clip_val) {
    if (raw_grad_val_ptr == nullptr) {
        return; // No gradient provided, nothing to do
    }
    float raw_grad = *raw_grad_val_ptr;

    // Sanitize gradient
    if (isnan(raw_grad)) raw_grad = 0.0001f;
    else if (isinf(raw_grad)) raw_grad = copysignf(FLT_MAX, raw_grad);

    // Apply element-wise clipping
    if (fabsf(raw_grad) > clip_val) raw_grad = copysignf(clip_val, raw_grad);

    // Apply update
    *param_ptr -= lr * raw_grad;
}

/**
 * @brief Processes, sanitizes, and clips a gradient value with L1 regularization,
 *        then applies it to a weight.
 * @param weight_ptr        Pointer to the weight value to be updated.
 * @param current_weight_val Current value of the weight.
 * @param raw_grad_val_ptr  Pointer to the raw gradient value. Can be NULL if gradient is optional.
 * @param l1_lambda         L1 regularization parameter.
 * @param lr                Learning rate.
 * @param clip_val          Maximum absolute value for gradient clipping.
 */
__device__ inline void apply_weight_grad_process_and_clip_l1(float* weight_ptr, float current_weight_val, const float* raw_grad_val_ptr, float l1_lambda, float lr, float clip_val) {
    if (raw_grad_val_ptr == nullptr) {
        return;
    }
    float raw_grad = *raw_grad_val_ptr;
    float l1_reg_term = l1_lambda * sign_f(current_weight_val);
    float total_grad = raw_grad + l1_reg_term;

    // Sanitize gradient
    if (isnan(total_grad)) total_grad = 0.0001f;
    else if (isinf(total_grad)) total_grad = copysignf(1000.0f, total_grad);

    // Apply element-wise clipping
    if (fabsf(total_grad) > clip_val) total_grad = copysignf(clip_val, total_grad);

    // Apply update
    *weight_ptr -= lr * total_grad;
}

/**
 * @brief Processes, sanitizes, and clips a gradient value with L2 regularization,
 *        then applies it to a weight.
 * @param weight_ptr        Pointer to the weight value to be updated.
 * @param current_weight_val Current value of the weight.
 * @param raw_grad_val_ptr  Pointer to the raw gradient value. Can be NULL if gradient is optional.
 * @param l2_lambda         L2 regularization parameter.
 * @param lr                Learning rate.
 * @param clip_val          Maximum absolute value for gradient clipping.
 */
__device__ inline void apply_weight_grad_process_and_clip_l2(float* weight_ptr, float current_weight_val, const float* raw_grad_val_ptr, float l2_lambda, float lr, float clip_val) {
    if (raw_grad_val_ptr == nullptr) {
        return;
    }
    float raw_grad = *raw_grad_val_ptr;
    float l2_reg_term = 2.0f * l2_lambda * current_weight_val;
    float total_grad = raw_grad + l2_reg_term;

    // Sanitize gradient
    if (isnan(total_grad)) total_grad = 0.0001f;
    else if (isinf(total_grad)) total_grad = copysignf(1000.0f, total_grad);

    // Apply element-wise clipping
    if (fabsf(total_grad) > clip_val) total_grad = copysignf(clip_val, total_grad);

    // Apply update
    *weight_ptr -= lr * total_grad;
}

/**
 * @brief Processes, sanitizes, and clips a gradient value with Elastic Net regularization,
 *        then applies it to a weight.
 * @param weight_ptr        Pointer to the weight value to be updated.
 * @param current_weight_val Current value of the weight.
 * @param raw_grad_val_ptr  Pointer to the raw gradient value. Can be NULL if gradient is optional.
 * @param l1_lambda         L1 regularization parameter.
 * @param l2_lambda         L2 regularization parameter.
 * @param lr                Learning rate.
 * @param clip_val          Maximum absolute value for gradient clipping.
 */
__device__ inline void apply_weight_grad_process_and_clip_elastic(float* weight_ptr, float current_weight_val, const float* raw_grad_val_ptr, float l1_lambda, float l2_lambda, float lr, float clip_val) {
    if (raw_grad_val_ptr == nullptr) {
        return;
    }
    float raw_grad = *raw_grad_val_ptr;
    float l1_reg_term = l1_lambda * sign_f(current_weight_val);
    float l2_reg_term = 2.0f * l2_lambda * current_weight_val;
    float total_grad = raw_grad + l1_reg_term + l2_reg_term;

    // Sanitize gradient
    if (isnan(total_grad)) total_grad = 0.0001f;
    else if (isinf(total_grad)) total_grad = copysignf(1000.0f, total_grad);

    // Apply element-wise clipping
    if (fabsf(total_grad) > clip_val) total_grad = copysignf(clip_val, total_grad);

    // Apply update
    *weight_ptr -= lr * total_grad;
}


// --- CUDA Kernel Implementations ---

__global__ void kernelUpdateSimple(float* weights_to_update, const float* gradients, float lr, unsigned int n_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        if (gradients != nullptr) {
            float tgradients = gradients[idx];
            if (isnan(tgradients)) {
                tgradients = 0.0001f; // Replace NaN with a small, non-zero value
            } else if (isinf(tgradients)) {
                // Replace infinity with a large but finite value, preserving the sign.
                tgradients = copysignf(1000.0f, tgradients);
            }
            weights_to_update[idx] -= lr * tgradients;
        }
    }
}

__global__ void updateEVRowsKernel(float* d_EV_rows, const float* d_vector_to_add,
    int num_rows_to_update, int num_cols)
{
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x; // Each thread handles one row

    if (row_idx < num_rows_to_update) {
        for (int col_idx = 0; col_idx < num_cols; ++col_idx) {
            d_EV_rows[row_idx * num_cols + col_idx] += d_vector_to_add[col_idx];
        }
    }
}

__global__ void kernelUpdateWeights_EH_EV(float* mh_a, float* mv_a, float* mq_a, float* mk_a,
                                        float* eh, float* ev,
                                        const float* grad_mh, const float* grad_mv,
                                        const float* grad_mq, const float* grad_mk,
                                        const float* grad_eh, const float* grad_ev_scaled,
                                        float learning_rate, int update_eh, int update_ev,
                                        float max_grad_clip_value,
                                        int mat_heights, int embedding_dim, int context_win)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ev_size = context_win * embedding_dim;
    int matrix_size = mat_heights * embedding_dim;

    if (idx < matrix_size) {
        // Here, these are raw gradients (no regularization)
        apply_plain_grad_process_and_clip(&mh_a[idx], &grad_mh[idx], learning_rate, max_grad_clip_value);
        apply_plain_grad_process_and_clip(&mv_a[idx], &grad_mv[idx], learning_rate, max_grad_clip_value);
        apply_plain_grad_process_and_clip(&mq_a[idx], &grad_mq[idx], learning_rate, max_grad_clip_value);
        apply_plain_grad_process_and_clip(&mk_a[idx], &grad_mk[idx], learning_rate, max_grad_clip_value);
    }
    if (update_eh != 0 && idx < embedding_dim) {
        apply_plain_grad_process_and_clip(&eh[idx], &grad_eh[idx], learning_rate, max_grad_clip_value);
    }
    if (update_ev != 0) {
        if (idx < ev_size) {
            int embed_idx = idx % embedding_dim;
            apply_plain_grad_process_and_clip(&ev[idx], &grad_ev_scaled[embed_idx], learning_rate, max_grad_clip_value);
        }
    }
}

__global__ void kernelUpdateWeights_1stHead_H(float* mh_a, float* mv_a, float* mq_a, float* mk_a,
                                            float* eh,
                                            const float* grad_mh, const float* grad_mv,
                                            const float* grad_mq, const float* grad_mk,
                                            const float* grad_eh,
                                            float learning_rate, int update_eh,
                                            float max_grad_clip_value,
                                            int mat_heights, int embedding_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int matrix_size = mat_heights * embedding_dim;

    if (idx < matrix_size) {
        apply_plain_grad_process_and_clip(&mh_a[idx], &grad_mh[idx], learning_rate, max_grad_clip_value);
        apply_plain_grad_process_and_clip(&mv_a[idx], &grad_mv[idx], learning_rate, max_grad_clip_value);
        apply_plain_grad_process_and_clip(&mq_a[idx], &grad_mq[idx], learning_rate, max_grad_clip_value);
        apply_plain_grad_process_and_clip(&mk_a[idx], &grad_mk[idx], learning_rate, max_grad_clip_value);
    }
    if (update_eh != 0 && idx < embedding_dim) {
        apply_plain_grad_process_and_clip(&eh[idx], &grad_eh[idx], learning_rate, max_grad_clip_value);
    }
}

__global__ void kernelUpdateWeights_1stHead_V(float* mv_a, float* mq_a, float* mk_a,
                                            const float* grad_mv, const float* grad_mq,
                                            const float* grad_mk_correction,
                                            float learning_rate,
                                            float max_grad_clip_value,
                                            int mat_heights, int embedding_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int matrix_size = mat_heights * embedding_dim;

    if (idx < matrix_size) {
        apply_plain_grad_process_and_clip(&mv_a[idx], &grad_mv[idx], learning_rate, max_grad_clip_value);
        apply_plain_grad_process_and_clip(&mq_a[idx], &grad_mq[idx], learning_rate, max_grad_clip_value);
        apply_plain_grad_process_and_clip(&mk_a[idx], &grad_mk_correction[idx], learning_rate, max_grad_clip_value);
    }
}

__global__ void kernelUpdateWeights_1stHead_HV(float* mh_a, float* mv_a, float* mq_a, float* mk_a,
                                             const float* grad_mh, const float* grad_mv,
                                             const float* grad_mq, const float* grad_mk,
                                             float learning_rate,
                                             float max_grad_clip_value,
                                             int mat_heights, int embedding_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int matrix_size = mat_heights * embedding_dim;

    if (idx < matrix_size) {
        apply_plain_grad_process_and_clip(&mh_a[idx], &grad_mh[idx], learning_rate, max_grad_clip_value);
        apply_plain_grad_process_and_clip(&mv_a[idx], &grad_mv[idx], learning_rate, max_grad_clip_value);
        apply_plain_grad_process_and_clip(&mq_a[idx], &grad_mq[idx], learning_rate, max_grad_clip_value);
        apply_plain_grad_process_and_clip(&mk_a[idx], &grad_mk[idx], learning_rate, max_grad_clip_value);
    }
}

__global__ void kernelUpdateWeights_EV_V(float* mv_a, float* mq_a, float* mk_a, float* ev,
                                       const float* grad_mv, const float* grad_mq,
                                       const float* grad_mk_correction,
                                       const float* grad_ev_full,
                                       float learning_rate,
                                       float max_grad_clip_value,
                                       int update_ev, int mat_heights, int embedding_dim, int context_win)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int matrix_size = mat_heights * embedding_dim;
    int ev_size = context_win * embedding_dim;

    if (idx < matrix_size) {
        apply_plain_grad_process_and_clip(&mv_a[idx], &grad_mv[idx], learning_rate, max_grad_clip_value);
        apply_plain_grad_process_and_clip(&mq_a[idx], &grad_mq[idx], learning_rate, max_grad_clip_value);
        apply_plain_grad_process_and_clip(&mk_a[idx], &grad_mk_correction[idx], learning_rate, max_grad_clip_value);
    }
    if (update_ev != 0) {
        if (idx < ev_size) {
            int embed_idx = idx % embedding_dim; // Original logic
            apply_plain_grad_process_and_clip(&ev[idx], &grad_ev_full[embed_idx], learning_rate, max_grad_clip_value);
        }
    }
}

// 1. L1-only variant for kernelUpdateWeights_EV_V
__global__ void kernelUpdateWeights_EV_V_L1(float* mv_a, float* mq_a, float* mk_a, float* ev,
                                       const float* grad_mv, const float* grad_mq,
                                       const float* grad_mk_correction,
                                       const float* grad_ev_full,
                                       float learning_rate,
                                       float lambda_l1,
                                       float max_grad_clip_value,
                                       int update_ev, int mat_heights, int embedding_dim, int context_win)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int matrix_size = mat_heights * embedding_dim;
    int ev_size = context_win * embedding_dim;

    if (idx < matrix_size) {
        if(grad_mv != nullptr) {
            apply_weight_grad_process_and_clip_l1(&mv_a[idx], mv_a[idx], &grad_mv[idx], lambda_l1, learning_rate, max_grad_clip_value);
        }
        if(grad_mq != nullptr) {
            apply_weight_grad_process_and_clip_l1(&mq_a[idx], mq_a[idx], &grad_mq[idx], lambda_l1, learning_rate, max_grad_clip_value);
        }
        if(grad_mk_correction != nullptr) {
            apply_weight_grad_process_and_clip_l1(&mk_a[idx], mk_a[idx], &grad_mk_correction[idx], lambda_l1, learning_rate, max_grad_clip_value);
        }
    }
    if (update_ev != 0) {
        if (idx < ev_size) {
            int embed_idx = idx % embedding_dim; // Original logic
            apply_plain_grad_process_and_clip(&ev[idx], &grad_ev_full[embed_idx], learning_rate, max_grad_clip_value);
        }
    }
}

// 2. L1-only variant for kernelUpdateWeights_EH_EV
__global__ void kernelUpdateWeights_EH_EV_L1(float* mh_a, float* mv_a, float* mq_a, float* mk_a,
                                        float* eh, float* ev,
                                        const float* grad_mh, const float* grad_mv,
                                        const float* grad_mq, const float* grad_mk,
                                        const float* grad_eh, const float* grad_ev_scaled,
                                        float learning_rate, int update_eh, int update_ev,
                                        float lambda_l1,
                                        float max_grad_clip_value,
                                        int mat_heights, int embedding_dim, int context_win)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ev_size = context_win * embedding_dim;
    int matrix_size = mat_heights * embedding_dim;

    if (idx < matrix_size) {
        apply_weight_grad_process_and_clip_l1(&mh_a[idx], mh_a[idx], &grad_mh[idx], lambda_l1, learning_rate, max_grad_clip_value);
        apply_weight_grad_process_and_clip_l1(&mv_a[idx], mv_a[idx], &grad_mv[idx], lambda_l1, learning_rate, max_grad_clip_value);
        apply_weight_grad_process_and_clip_l1(&mq_a[idx], mq_a[idx], &grad_mq[idx], lambda_l1, learning_rate, max_grad_clip_value);
        apply_weight_grad_process_and_clip_l1(&mk_a[idx], mk_a[idx], &grad_mk[idx], lambda_l1, learning_rate, max_grad_clip_value);
    }
    if (update_eh != 0 && idx < embedding_dim) {
        apply_plain_grad_process_and_clip(&eh[idx], &grad_eh[idx], learning_rate, max_grad_clip_value);
    }
    if (update_ev != 0) {
        if (idx < ev_size) {
            int embed_idx = idx % embedding_dim;
            apply_plain_grad_process_and_clip(&ev[idx], &grad_ev_scaled[embed_idx], learning_rate, max_grad_clip_value);
        }
    }
}

// 3. L1-only variant for kernelUpdateWeights_1stHead_H
__global__ void kernelUpdateWeights_1stHead_H_L1(float* mh_a, float* mv_a, float* mq_a, float* mk_a,
                                            float* eh,
                                            const float* grad_mh, const float* grad_mv,
                                            const float* grad_mq, const float* grad_mk,
                                            const float* grad_eh,
                                            float learning_rate, int update_eh,
                                            float lambda_l1,
                                            float max_grad_clip_value,
                                            int mat_heights, int embedding_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int matrix_size = mat_heights * embedding_dim;

    if (idx < matrix_size) {
        apply_weight_grad_process_and_clip_l1(&mh_a[idx], mh_a[idx], &grad_mh[idx], lambda_l1, learning_rate, max_grad_clip_value);
        apply_weight_grad_process_and_clip_l1(&mv_a[idx], mv_a[idx], &grad_mv[idx], lambda_l1, learning_rate, max_grad_clip_value);
        apply_weight_grad_process_and_clip_l1(&mq_a[idx], mq_a[idx], &grad_mq[idx], lambda_l1, learning_rate, max_grad_clip_value);
        apply_weight_grad_process_and_clip_l1(&mk_a[idx], mk_a[idx], &grad_mk[idx], lambda_l1, learning_rate, max_grad_clip_value);
    }
    if (update_eh != 0 && idx < embedding_dim) {
        apply_plain_grad_process_and_clip(&eh[idx], &grad_eh[idx], learning_rate, max_grad_clip_value);
    }
}

// 4. L1-only variant for kernelUpdateWeights_1stHead_V
__global__ void kernelUpdateWeights_1stHead_V_L1(float* mv_a, float* mq_a, float* mk_a,
                                            const float* grad_mv, const float* grad_mq,
                                            const float* grad_mk_correction,
                                            float learning_rate,
                                            float lambda_l1,
                                            float max_grad_clip_value,
                                            int mat_heights, int embedding_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int matrix_size = mat_heights * embedding_dim;

    if (idx < matrix_size) {
        apply_weight_grad_process_and_clip_l1(&mv_a[idx], mv_a[idx], &grad_mv[idx], lambda_l1, learning_rate, max_grad_clip_value);
        apply_weight_grad_process_and_clip_l1(&mq_a[idx], mq_a[idx], &grad_mq[idx], lambda_l1, learning_rate, max_grad_clip_value);
        apply_weight_grad_process_and_clip_l1(&mk_a[idx], mk_a[idx], &grad_mk_correction[idx], lambda_l1, learning_rate, max_grad_clip_value);
    }
}

// 5. L1-only variant for kernelUpdateWeights_1stHead_HV
__global__ void kernelUpdateWeights_1stHead_HV_L1(float* mh_a, float* mv_a, float* mq_a, float* mk_a,
                                             const float* grad_mh, const float* grad_mv,
                                             const float* grad_mq, const float* grad_mk,
                                             float learning_rate,
                                             float lambda_l1,
                                             float max_grad_clip_value,
                                             int mat_heights, int embedding_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int matrix_size = mat_heights * embedding_dim;

    if (idx < matrix_size) {
        apply_weight_grad_process_and_clip_l1(&mh_a[idx], mh_a[idx], &grad_mh[idx], lambda_l1, learning_rate, max_grad_clip_value);
        apply_weight_grad_process_and_clip_l1(&mv_a[idx], mv_a[idx], &grad_mv[idx], lambda_l1, learning_rate, max_grad_clip_value);
        apply_weight_grad_process_and_clip_l1(&mq_a[idx], mq_a[idx], &grad_mq[idx], lambda_l1, learning_rate, max_grad_clip_value);
        apply_weight_grad_process_and_clip_l1(&mk_a[idx], mk_a[idx], &grad_mk[idx], lambda_l1, learning_rate, max_grad_clip_value);
    }
}

// 1. L2-only variant for kernelUpdateWeights_EV_V
__global__ void kernelUpdateWeights_EV_V_L2(float* mv_a, float* mq_a, float* mk_a, float* ev,
                                       const float* grad_mv, const float* grad_mq,
                                       const float* grad_mk_correction,
                                       const float* grad_ev_full,
                                       float learning_rate,
                                       float lambda_l2,
                                       float max_grad_clip_value,
                                       int update_ev, int mat_heights, int embedding_dim, int context_win)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int matrix_size = mat_heights * embedding_dim;
    int ev_size = context_win * embedding_dim;

    if (idx < matrix_size) {
        if(grad_mv != nullptr) {
            apply_weight_grad_process_and_clip_l2(&mv_a[idx], mv_a[idx], &grad_mv[idx], lambda_l2, learning_rate, max_grad_clip_value);
        }
        if(grad_mq != nullptr) {
            apply_weight_grad_process_and_clip_l2(&mq_a[idx], mq_a[idx], &grad_mq[idx], lambda_l2, learning_rate, max_grad_clip_value);
        }
        if(grad_mk_correction != nullptr) {
            apply_weight_grad_process_and_clip_l2(&mk_a[idx], mk_a[idx], &grad_mk_correction[idx], lambda_l2, learning_rate, max_grad_clip_value);
        }
    }
    if (update_ev != 0) {
        if (idx < ev_size) {
            int embed_idx = idx % embedding_dim; // Original logic
            apply_plain_grad_process_and_clip(&ev[idx], &grad_ev_full[embed_idx], learning_rate, max_grad_clip_value);
        }
    }
}

// 2. L2-only variant for kernelUpdateWeights_EH_EV
__global__ void kernelUpdateWeights_EH_EV_L2(float* mh_a, float* mv_a, float* mq_a, float* mk_a,
                                        float* eh, float* ev,
                                        const float* grad_mh, const float* grad_mv,
                                        const float* grad_mq, const float* grad_mk,
                                        const float* grad_eh, const float* grad_ev_scaled,
                                        float learning_rate, int update_eh, int update_ev,
                                        float lambda_l2,
                                        float max_grad_clip_value,
                                        int mat_heights, int embedding_dim, int context_win)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ev_size = context_win * embedding_dim;
    int matrix_size = mat_heights * embedding_dim;

    if (idx < matrix_size) {
        apply_weight_grad_process_and_clip_l2(&mh_a[idx], mh_a[idx], &grad_mh[idx], lambda_l2, learning_rate, max_grad_clip_value);
        apply_weight_grad_process_and_clip_l2(&mv_a[idx], mv_a[idx], &grad_mv[idx], lambda_l2, learning_rate, max_grad_clip_value);
        apply_weight_grad_process_and_clip_l2(&mq_a[idx], mq_a[idx], &grad_mq[idx], lambda_l2, learning_rate, max_grad_clip_value);
        apply_weight_grad_process_and_clip_l2(&mk_a[idx], mk_a[idx], &grad_mk[idx], lambda_l2, learning_rate, max_grad_clip_value);
    }
    if (update_eh != 0 && idx < embedding_dim) {
        apply_plain_grad_process_and_clip(&eh[idx], &grad_eh[idx], learning_rate, max_grad_clip_value);
    }
    if (update_ev != 0) {
        if (idx < ev_size) {
            int embed_idx = idx % embedding_dim;
            apply_plain_grad_process_and_clip(&ev[idx], &grad_ev_scaled[embed_idx], learning_rate, max_grad_clip_value);
        }
    }
}

// 3. L2-only variant for kernelUpdateWeights_1stHead_H
__global__ void kernelUpdateWeights_1stHead_H_L2(float* mh_a, float* mv_a, float* mq_a, float* mk_a,
                                            float* eh,
                                            const float* grad_mh, const float* grad_mv,
                                            const float* grad_mq, const float* grad_mk,
                                            const float* grad_eh,
                                            float learning_rate, int update_eh,
                                            float lambda_l2,
                                            float max_grad_clip_value,
                                            int mat_heights, int embedding_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int matrix_size = mat_heights * embedding_dim;

    if (idx < matrix_size) {
        apply_weight_grad_process_and_clip_l2(&mh_a[idx], mh_a[idx], &grad_mh[idx], lambda_l2, learning_rate, max_grad_clip_value);
        apply_weight_grad_process_and_clip_l2(&mv_a[idx], mv_a[idx], &grad_mv[idx], lambda_l2, learning_rate, max_grad_clip_value);
        apply_weight_grad_process_and_clip_l2(&mq_a[idx], mq_a[idx], &grad_mq[idx], lambda_l2, learning_rate, max_grad_clip_value);
        apply_weight_grad_process_and_clip_l2(&mk_a[idx], mk_a[idx], &grad_mk[idx], lambda_l2, learning_rate, max_grad_clip_value);
    }
    if (update_eh != 0 && idx < embedding_dim) {
        apply_plain_grad_process_and_clip(&eh[idx], &grad_eh[idx], learning_rate, max_grad_clip_value);
    }
}

// 4. L2-only variant for kernelUpdateWeights_1stHead_V
__global__ void kernelUpdateWeights_1stHead_V_L2(float* mv_a, float* mq_a, float* mk_a,
                                            const float* grad_mv, const float* grad_mq,
                                            const float* grad_mk_correction,
                                            float learning_rate,
                                            float lambda_l2,
                                            float max_grad_clip_value,
                                            int mat_heights, int embedding_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int matrix_size = mat_heights * embedding_dim;

    if (idx < matrix_size) {
        apply_weight_grad_process_and_clip_l2(&mv_a[idx], mv_a[idx], &grad_mv[idx], lambda_l2, learning_rate, max_grad_clip_value);
        apply_weight_grad_process_and_clip_l2(&mq_a[idx], mq_a[idx], &grad_mq[idx], lambda_l2, learning_rate, max_grad_clip_value);
        apply_weight_grad_process_and_clip_l2(&mk_a[idx], mk_a[idx], &grad_mk_correction[idx], lambda_l2, learning_rate, max_grad_clip_value);
    }
}

// 5. L2-only variant for kernelUpdateWeights_1stHead_HV
__global__ void kernelUpdateWeights_1stHead_HV_L2(float* mh_a, float* mv_a, float* mq_a, float* mk_a,
                                             const float* grad_mh, const float* grad_mv,
                                             const float* grad_mq, const float* grad_mk,
                                             float learning_rate,
                                             float lambda_l2,
                                             float max_grad_clip_value,
                                             int mat_heights, int embedding_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int matrix_size = mat_heights * embedding_dim;

    if (idx < matrix_size) {
        apply_weight_grad_process_and_clip_l2(&mh_a[idx], mh_a[idx], &grad_mh[idx], lambda_l2, learning_rate, max_grad_clip_value);
        apply_weight_grad_process_and_clip_l2(&mv_a[idx], mv_a[idx], &grad_mv[idx], lambda_l2, learning_rate, max_grad_clip_value);
        apply_weight_grad_process_and_clip_l2(&mq_a[idx], mq_a[idx], &grad_mq[idx], lambda_l2, learning_rate, max_grad_clip_value);
        apply_weight_grad_process_and_clip_l2(&mk_a[idx], mk_a[idx], &grad_mk[idx], lambda_l2, learning_rate, max_grad_clip_value);
    }
}

// --- Elastic Net Regularization Kernels ---

__global__ void kernelUpdateWeights_EH_EV_ElasticNet(float* mh_a, float* mv_a, float* mq_a, float* mk_a,
                                        float* eh, float* ev,
                                        const float* grad_mh, const float* grad_mv,
                                        const float* grad_mq, const float* grad_mk,
                                        const float* grad_eh, const float* grad_ev_scaled,
                                        float learning_rate, int update_eh, int update_ev,
                                        float lambda_l1, float lambda_l2,
                                        float max_grad_clip_value,
                                        int mat_heights, int embedding_dim, int context_win)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ev_size = context_win * embedding_dim;
    int matrix_size = mat_heights * embedding_dim;

    if (idx < matrix_size) {
        apply_weight_grad_process_and_clip_elastic(&mh_a[idx], mh_a[idx], &grad_mh[idx], lambda_l1, lambda_l2, learning_rate, max_grad_clip_value);
        apply_weight_grad_process_and_clip_elastic(&mv_a[idx], mv_a[idx], &grad_mv[idx], lambda_l1, lambda_l2, learning_rate, max_grad_clip_value);
        apply_weight_grad_process_and_clip_elastic(&mq_a[idx], mq_a[idx], &grad_mq[idx], lambda_l1, lambda_l2, learning_rate, max_grad_clip_value);
        apply_weight_grad_process_and_clip_elastic(&mk_a[idx], mk_a[idx], &grad_mk[idx], lambda_l1, lambda_l2, learning_rate, max_grad_clip_value);
    }
    if (update_eh != 0 && idx < embedding_dim) {
        apply_plain_grad_process_and_clip(&eh[idx], &grad_eh[idx], learning_rate, max_grad_clip_value);
    }
    if (update_ev != 0) {
        if (idx < ev_size) {
            int embed_idx = idx % embedding_dim;
            apply_plain_grad_process_and_clip(&ev[idx], &grad_ev_scaled[embed_idx], learning_rate, max_grad_clip_value);
        }
    }
}

__global__ void kernelUpdateWeights_1stHead_H_ElasticNet(float* mh_a, float* mv_a, float* mq_a, float* mk_a,
                                            float* eh,
                                            const float* grad_mh, const float* grad_mv,
                                            const float* grad_mq, const float* grad_mk,
                                            const float* grad_eh,
                                            float learning_rate, int update_eh,
                                            float lambda_l1, float lambda_l2,
                                            float max_grad_clip_value,
                                            int mat_heights, int embedding_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int matrix_size = mat_heights * embedding_dim;

    if (idx < matrix_size) {
        apply_weight_grad_process_and_clip_elastic(&mh_a[idx], mh_a[idx], &grad_mh[idx], lambda_l1, lambda_l2, learning_rate, max_grad_clip_value);
        apply_weight_grad_process_and_clip_elastic(&mv_a[idx], mv_a[idx], &grad_mv[idx], lambda_l1, lambda_l2, learning_rate, max_grad_clip_value);
        apply_weight_grad_process_and_clip_elastic(&mq_a[idx], mq_a[idx], &grad_mq[idx], lambda_l1, lambda_l2, learning_rate, max_grad_clip_value);
        apply_weight_grad_process_and_clip_elastic(&mk_a[idx], mk_a[idx], &grad_mk[idx], lambda_l1, lambda_l2, learning_rate, max_grad_clip_value);
    }
    if (update_eh != 0 && idx < embedding_dim) {
        apply_plain_grad_process_and_clip(&eh[idx], &grad_eh[idx], learning_rate, max_grad_clip_value);
    }
}

__global__ void kernelUpdateWeights_1stHead_V_ElasticNet(float* mv_a, float* mq_a, float* mk_a,
                                            const float* grad_mv, const float* grad_mq,
                                            const float* grad_mk_correction,
                                            float learning_rate,
                                            float lambda_l1, float lambda_l2,
                                            float max_grad_clip_value,
                                            int mat_heights, int embedding_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int matrix_size = mat_heights * embedding_dim;

    if (idx < matrix_size) {
        apply_weight_grad_process_and_clip_elastic(&mv_a[idx], mv_a[idx], &grad_mv[idx], lambda_l1, lambda_l2, learning_rate, max_grad_clip_value);
        apply_weight_grad_process_and_clip_elastic(&mq_a[idx], mq_a[idx], &grad_mq[idx], lambda_l1, lambda_l2, learning_rate, max_grad_clip_value);
        apply_weight_grad_process_and_clip_elastic(&mk_a[idx], mk_a[idx], &grad_mk_correction[idx], lambda_l1, lambda_l2, learning_rate, max_grad_clip_value);
    }
}

__global__ void kernelUpdateWeights_EV_V_ElasticNet(float* mv_a, float* mq_a, float* mk_a, float* ev,
                                        const float* grad_mv, const float* grad_mq,
                                        const float* grad_mk_correction,
                                        const float* grad_ev_full,
                                        float learning_rate,
                                        float lambda_l1, float lambda_l2,
                                        float max_grad_clip_value,
                                        int update_ev, int mat_heights, int embedding_dim, int context_win)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int matrix_size = mat_heights * embedding_dim;
    int ev_size = context_win * embedding_dim;

    if (idx < matrix_size) {
        apply_weight_grad_process_and_clip_elastic(&mv_a[idx], mv_a[idx], &grad_mv[idx], lambda_l1, lambda_l2, learning_rate, max_grad_clip_value);
        apply_weight_grad_process_and_clip_elastic(&mq_a[idx], mq_a[idx], &grad_mq[idx], lambda_l1, lambda_l2, learning_rate, max_grad_clip_value);
        apply_weight_grad_process_and_clip_elastic(&mk_a[idx], mk_a[idx], &grad_mk_correction[idx], lambda_l1, lambda_l2, learning_rate, max_grad_clip_value);
    }
    if (update_ev != 0) {
        if (idx < ev_size) {
            int embed_idx = idx % embedding_dim;
            apply_plain_grad_process_and_clip(&ev[idx], &grad_ev_full[embed_idx], learning_rate, max_grad_clip_value);
        }
    }
}

__global__ void kernelUpdateWeights_1stHead_HV_ElasticNet(float* mh_a, float* mv_a, float* mq_a, float* mk_a,
                                            const float* grad_mh, const float* grad_mv,
                                            const float* grad_mq, const float* grad_mk,
                                            float learning_rate,
                                            float lambda_l1, float lambda_l2,
                                            float max_grad_clip_value,
                                            int mat_heights, int embedding_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int matrix_size = mat_heights * embedding_dim;

    if (idx < matrix_size) {
        apply_weight_grad_process_and_clip_elastic(&mh_a[idx], mh_a[idx], &grad_mh[idx], lambda_l1, lambda_l2, learning_rate, max_grad_clip_value);
        apply_weight_grad_process_and_clip_elastic(&mv_a[idx], mv_a[idx], &grad_mv[idx], lambda_l1, lambda_l2, learning_rate, max_grad_clip_value);
        apply_weight_grad_process_and_clip_elastic(&mq_a[idx], mq_a[idx], &grad_mq[idx], lambda_l1, lambda_l2, learning_rate, max_grad_clip_value);
        apply_weight_grad_process_and_clip_elastic(&mk_a[idx], mk_a[idx], &grad_mk[idx], lambda_l1, lambda_l2, learning_rate, max_grad_clip_value);
    }
}

__global__ void kernelUpdateWeights_General(
    float* weights,
    const float* gradients,
    float learning_rate,
    float lambda_l1,
    float lambda_l2,
    float max_grad_clip_value,
    int total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        float current_weight = weights[idx];
        float error_gradient = (gradients != nullptr) ? gradients[idx] : 0.0f; // Handle NULL gradients

        float l1_reg_term = lambda_l1 * sign_f(current_weight);
        float l2_reg_term = 2.0f * lambda_l2 * current_weight;

        float total_gradient = error_gradient + l1_reg_term + l2_reg_term;
        if (isnan(total_gradient)) {
            total_gradient = 0.0001f;
        } else if (isinf(total_gradient)) {
            total_gradient = copysignf(1000.0f, total_gradient);
        }
        
        if (fabsf(total_gradient) > max_grad_clip_value) {
            total_gradient = copysignf(max_grad_clip_value, total_gradient);
        }

        weights[idx] -= learning_rate * total_gradient;
    }
}

#endif