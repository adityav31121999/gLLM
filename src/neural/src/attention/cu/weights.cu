#ifdef USE_CUDA
#include "include/attention.hpp"
#include <vector>
#include <stdexcept>
#include <numeric>
#include <cmath>
#include <maths.hpp>
#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ inline void apply_elastic_net_and_clip(float* weight_ptr, const float* gradient_ptr, // Changed to volatile
                                       int index, float learning_rate, float lambda_l1, float lambda_l2,
                                       float max_grad_clip_value)
{
    float current_weight = weight_ptr[index];
    float error_gradient = (gradient_ptr != nullptr) ? gradient_ptr[index] : 0.0f;

    float l1_reg_term = lambda_l1 * ((__signbit(current_weight) == 0) ? 1.0f : -1.0f);
    float l2_reg_term = lambda_l2 * current_weight;

    float total_gradient = error_gradient + l1_reg_term + l2_reg_term;

    if (isnan(total_gradient)) {
        total_gradient = 0.0f;
    }
    else if (isinf(total_gradient)) {
        total_gradient = copysignf(FLT_MAX, total_gradient);
    }
    // Apply element-wise gradient clipping
    // if (fabsf(total_gradient) > max_grad_clip_value) { total_gradient = copysignf(max_grad_clip_value, total_gradient); }

    weight_ptr[index] -= learning_rate * total_gradient;
}

// update EV rows
__global__ void kernelUpdateEVrows(float* d_EV_rows, const float* d_vector_to_add,
    int num_rows_to_update, int num_cols)
{
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x; // Each work-item handles one row

    if (row_idx < num_rows_to_update) {
        for (int col_idx = 0; col_idx < num_cols; ++col_idx) {
            d_EV_rows[row_idx * num_cols + col_idx] += d_vector_to_add[col_idx];
        }
    }
}


__global__ void kernelUpdateWeightsHeadHVElastic(float* mh_a, float* mv_a, float* mq_a, float* mk_a,
                                        float* eh, float* ev,
                                        const float* grad_mh, const float* grad_mv,
                                        const float* grad_mq, const float* grad_mk,
                                        const float* grad_eh, const float* grad_ev_scaled,
                                        float learning_rate, int update_eh, int update_ev,
                                        float lambda_l1, float lambda_l2, float max_grad_clip_value,
                                        int mat_heights, int embedding_dim, int context_win)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ev_size = context_win * embedding_dim;
    int matrix_size = mat_heights * embedding_dim;

    if (idx < matrix_size) {
        if(mh_a != nullptr) apply_elastic_net_and_clip(mh_a, grad_mh, idx, learning_rate, lambda_l1, lambda_l2, max_grad_clip_value);
        if(mv_a != nullptr) apply_elastic_net_and_clip(mv_a, grad_mv, idx, learning_rate, lambda_l1, lambda_l2, max_grad_clip_value);
        if(mq_a != nullptr) apply_elastic_net_and_clip(mq_a, grad_mq, idx, learning_rate, lambda_l1, lambda_l2, max_grad_clip_value);
        if(mk_a != nullptr) apply_elastic_net_and_clip(mk_a, grad_mk, idx, learning_rate, lambda_l1, lambda_l2, max_grad_clip_value);
    }
    if (update_eh != 0 && idx < embedding_dim) {
        if(eh != nullptr) apply_elastic_net_and_clip(eh, grad_eh, idx, learning_rate, lambda_l1, lambda_l2, max_grad_clip_value);
    }
    if (update_ev != 0) {
        if (idx < ev_size) {
            if(ev != nullptr) apply_elastic_net_and_clip(ev, grad_ev_scaled, idx, learning_rate, lambda_l1, lambda_l2, max_grad_clip_value);
        }
    }
}

__global__ void kernelUpdateWeightsHeadElastic(float* mh_a, float* mv_a, float* mq_a, float* mk_a,
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
        if(mh_a != nullptr) apply_elastic_net_and_clip(mh_a, grad_mh, idx, learning_rate, lambda_l1, lambda_l2, max_grad_clip_value);
        if(mv_a != nullptr) apply_elastic_net_and_clip(mv_a, grad_mv, idx, learning_rate, lambda_l1, lambda_l2, max_grad_clip_value);
        if(mq_a != nullptr) apply_elastic_net_and_clip(mq_a, grad_mq, idx, learning_rate, lambda_l1, lambda_l2, max_grad_clip_value);
        if(mk_a != nullptr) apply_elastic_net_and_clip(mk_a, grad_mk, idx, learning_rate, lambda_l1, lambda_l2, max_grad_clip_value);
    }
    if (update_eh != 0 && idx < embedding_dim) {
        if(eh != nullptr) apply_elastic_net_and_clip(eh, grad_eh, idx, learning_rate, lambda_l1, lambda_l2, max_grad_clip_value);
    }
}

__global__ void kernelUpdateWeightsGeneral(float* weights,
                                const float* gradients,
                                float learning_rate,
                                float lambda_l1,
                                float lambda_l2,
                                int total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        float current_weight = weights[idx];
        float error_gradient = (gradients != nullptr) ? gradients[idx] : 0.0f; // Handle NULL gradients

        float l1_reg_term = lambda_l1 * ((__signbit(current_weight) == 0) ? 1.0f : -1.0f);
        float l2_reg_term = lambda_l2 * current_weight;

        float total_gradient = error_gradient + l1_reg_term + l2_reg_term;
        if (isnan(total_gradient)) {
            total_gradient = 0.0f;
        }
        else if (isinf(total_gradient)) {
            total_gradient = copysignf(FLT_MAX, total_gradient);
        }

        weights[idx] -= learning_rate * total_gradient;
    }
}

__global__ void kernelUpdateSimple(float* weights_to_update, const float* gradients, float lr, unsigned int n_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        if (gradients == nullptr) {
            return;
        }
        float tgradients = gradients[idx];
        const float clip_val = 1.0f;

        if (isnan(tgradients)) {
            tgradients = 0.0f;
        }
        else if (isinf(tgradients)) {
            tgradients = copysignf(clip_val, tgradients);
        }

        if (fabsf(tgradients) > clip_val) tgradients = copysignf(clip_val, tgradients);

        weights_to_update[idx] -= lr * tgradients;
    }
}

__global__ void kernelUpdateSimple_Elastic(float* weights_to_update, const float* gradients, float lr, unsigned int n_elements,
                                 float lambda_l1, float lambda_l2, float max_grad_clip_value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        if (gradients == nullptr) {
            // Apply regularization even if gradients are NULL
            float current_weight = weights_to_update[idx];
            float l1_reg_term = lambda_l1 * copysignf(1.0f, current_weight);
            float l2_reg_term = lambda_l2 * current_weight;
            float total_gradient = l1_reg_term + l2_reg_term;

            if (isnan(total_gradient)) {
                total_gradient = 0.0f;
            } else if (isinf(total_gradient)) {
                total_gradient = copysignf(FLT_MAX, total_gradient);
            }
            if (fabsf(total_gradient) > max_grad_clip_value) {
                total_gradient = copysignf(max_grad_clip_value, total_gradient);
            }
            weights_to_update[idx] -= lr * total_gradient;
            return;
        }

        float current_weight = weights_to_update[idx];
        float error_gradient = gradients[idx];

        float l1_reg_term = lambda_l1 * copysignf(1.0f, current_weight);
        float l2_reg_term = lambda_l2 * current_weight;

        float total_gradient = error_gradient + l1_reg_term + l2_reg_term;

        if (isnan(total_gradient)) total_gradient = 0.0f;
        else if (isinf(total_gradient)) total_gradient = copysignf(max_grad_clip_value, total_gradient);

        if (fabsf(total_gradient) > max_grad_clip_value) total_gradient = copysignf(max_grad_clip_value, total_gradient);

        weights_to_update[idx] -= lr * total_gradient;
    }
}

__global__ void kernelUpdateWeightsGeneral_f4(float* weights, const float* gradients, float learning_rate,
    float lambda_l1, float lambda_l2, int total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements_f4 = total_elements / 4;

    if (idx < total_elements_f4) {
        float4* weights_f4 = (float4*)weights;
        const float4* gradients_f4 = (const float4*)gradients;

        float4 current_weight = weights_f4[idx];
        float4 error_gradient = (gradients_f4 != nullptr) ? gradients_f4[idx] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        float4 l1_reg_term = make_float4(lambda_l1 * copysignf(1.0f, current_weight.x), 
                                         lambda_l1 * copysignf(1.0f, current_weight.y),
                                         lambda_l1 * copysignf(1.0f, current_weight.z),
                                         lambda_l1 * copysignf(1.0f, current_weight.w));
        float4 l2_reg_term = make_float4(lambda_l2 * current_weight.x,
                                         lambda_l2 * current_weight.y,
                                         lambda_l2 * current_weight.z,
                                         lambda_l2 * current_weight.w);

        float4 total_gradient = make_float4(error_gradient.x + l1_reg_term.x + l2_reg_term.x,
                                            error_gradient.y + l1_reg_term.y + l2_reg_term.y,
                                            error_gradient.z + l1_reg_term.z + l2_reg_term.z,
                                            error_gradient.w + l1_reg_term.w + l2_reg_term.w);

        // Handle NaNs and Infs for each component
        total_gradient.x = isnan(total_gradient.x) ? 0.0f : (isinf(total_gradient.x) ? copysignf(FLT_MAX, total_gradient.x) : total_gradient.x);
        total_gradient.y = isnan(total_gradient.y) ? 0.0f : (isinf(total_gradient.y) ? copysignf(FLT_MAX, total_gradient.y) : total_gradient.y);
        total_gradient.z = isnan(total_gradient.z) ? 0.0f : (isinf(total_gradient.z) ? copysignf(FLT_MAX, total_gradient.z) : total_gradient.z);
        total_gradient.w = isnan(total_gradient.w) ? 0.0f : (isinf(total_gradient.w) ? copysignf(FLT_MAX, total_gradient.w) : total_gradient.w);

        float4 update = make_float4(learning_rate * total_gradient.x,
                                    learning_rate * total_gradient.y,
                                    learning_rate * total_gradient.z,
                                    learning_rate * total_gradient.w);
        weights_f4[idx] = make_float4(current_weight.x - update.x, current_weight.y - update.y, current_weight.z - update.z, current_weight.w - update.w);
    }

    // Handle remaining elements if total_elements is not a multiple of 4
    if (idx == total_elements_f4 && (total_elements % 4) != 0) {
        for (int i = 0; i < (total_elements % 4); ++i) {
            int current_idx = total_elements_f4 * 4 + i;
            float current_weight = weights[current_idx];
            float error_gradient = (gradients != nullptr) ? gradients[current_idx] : 0.0f;
            float l1_reg_term = lambda_l1 * copysignf(1.0f, current_weight);
            float l2_reg_term = lambda_l2 * current_weight;
            float total_gradient = error_gradient + l1_reg_term + l2_reg_term;
            if (isnan(total_gradient)) {
                total_gradient = 0.0f;
            } else if (isinf(total_gradient)) {
                total_gradient = copysignf(FLT_MAX, total_gradient);
            }
            weights[current_idx] -= learning_rate * total_gradient;
        }
    }
}

__global__ void updateEmbeddings(float* embeddings, const float* gradientsVector, float learning_rate,
    float lambda_l1, float lambda_l2, int embeddingRows, int embeddingDim)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_id < embeddingRows) {
        int row_start_idx = global_id * embeddingDim;

        int num_elements_f4 = embeddingDim / 4;
        for (int i = 0; i < num_elements_f4; ++i) {
            int current_f4_idx = (row_start_idx / 4) + i;
            float4* embeddings_f4 = (float4*)embeddings;
            const float4* gradientsVector_f4 = (const float4*)gradientsVector;

            float4 current_weight = embeddings_f4[current_f4_idx];
            float4 error_gradient = (gradientsVector_f4 != nullptr) ? gradientsVector_f4[i] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

            float4 l1_reg_term = make_float4(lambda_l1 * copysignf(1.0f, current_weight.x),
                                             lambda_l1 * copysignf(1.0f, current_weight.y),
                                             lambda_l1 * copysignf(1.0f, current_weight.z),
                                             lambda_l1 * copysignf(1.0f, current_weight.w));
            float4 l2_reg_term = make_float4(lambda_l2 * current_weight.x,
                                             lambda_l2 * current_weight.y,
                                             lambda_l2 * current_weight.z,
                                             lambda_l2 * current_weight.w);

            float4 total_gradient = make_float4(error_gradient.x + l1_reg_term.x + l2_reg_term.x,
                                                error_gradient.y + l1_reg_term.y + l2_reg_term.y,
                                                error_gradient.z + l1_reg_term.z + l2_reg_term.z,
                                                error_gradient.w + l1_reg_term.w + l2_reg_term.w);

            total_gradient.x = isnan(total_gradient.x) ? 0.0f : (isinf(total_gradient.x) ? copysignf(FLT_MAX, total_gradient.x) : total_gradient.x);
            total_gradient.y = isnan(total_gradient.y) ? 0.0f : (isinf(total_gradient.y) ? copysignf(FLT_MAX, total_gradient.y) : total_gradient.y);
            total_gradient.z = isnan(total_gradient.z) ? 0.0f : (isinf(total_gradient.z) ? copysignf(FLT_MAX, total_gradient.z) : total_gradient.z);
            total_gradient.w = isnan(total_gradient.w) ? 0.0f : (isinf(total_gradient.w) ? copysignf(FLT_MAX, total_gradient.w) : total_gradient.w);

            embeddings_f4[current_f4_idx] = make_float4(current_weight.x - learning_rate * total_gradient.x,
                                                        current_weight.y - learning_rate * total_gradient.y,
                                                        current_weight.z - learning_rate * total_gradient.z,
                                                        current_weight.w - learning_rate * total_gradient.w);
        }

        for (int i = num_elements_f4 * 4; i < embeddingDim; ++i) {
            apply_elastic_net_and_clip(embeddings, gradientsVector, row_start_idx + i, learning_rate, lambda_l1, lambda_l2, FLT_MAX);
        }
    }
}

#endif