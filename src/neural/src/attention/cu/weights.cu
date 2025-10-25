#ifdef USE_CUDA
#include "include/attention.hpp"
#include <vector>
#include <iostream>
#include <stdexcept>
#include <numeric>
#include <cmath>
#include <maths.hpp>
#include <string>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

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

__device__ inline void apply_elastic_net_and_clip(float* weight_ptr, const float* gradient_ptr,
                                       int index, float learning_rate, float lambda_l1, float lambda_l2,
                                       float max_grad_clip_value)
{
    float current_weight = weight_ptr[index];
    float error_gradient = (gradient_ptr != nullptr) ? gradient_ptr[index] : 0.0f;

    float l1_reg_term = lambda_l1 * ((__signbit(current_weight) == 0) ? 1.0f : -1.0f);
    float l2_reg_term lambda_l2 * current_weight;

    float total_gradient = error_gradient + l1_reg_term + l2_reg_term;

    if (isnan(total_gradient)) {
        total_gradient = 0.0f;
    }
    else if (isinf(total_gradient)) {
        total_gradient = copysignf(FLT_MAX, total_gradient);
    }
    // Apply element-wise gradient clipping
    if (fabsf(total_gradient) > max_grad_clip_value) {
        total_gradient = copysignf(max_grad_clip_value, total_gradient);
    }

    weight_ptr[index] -= learning_rate * total_gradient;
}

__global__ void kernelUpdateWeightsHeadHVElastic(float* mh_a, float* mv_a, float* mq_a, float* mk_a,
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
        apply_elastic_net_and_clip(&mh_a[idx], &grad_mh[idx], idx, learning_rate, lambda_l1, lambda_l2, max_grad_clip_value);
        apply_elastic_net_and_clip(&mv_a[idx], &grad_mv[idx], idx, learning_rate, lambda_l1, lambda_l2, max_grad_clip_value);
        apply_elastic_net_and_clip(&mq_a[idx], &grad_mq[idx], idx, learning_rate, lambda_l1, lambda_l2, max_grad_clip_value);
        apply_elastic_net_and_clip(&mk_a[idx], &grad_mk[idx], idx, learning_rate, lambda_l1, lambda_l2, max_grad_clip_value);
    }
    if (update_eh != 0 && idx < embedding_dim) {
        apply_elastic_net_and_clip(&eh[idx], &grad_eh[idx], idx, learning_rate, lambda_l1, lambda_l2, max_grad_clip_value);
    }
    if (update_ev != 0) {
        if (idx < ev_size) {
            int embed_idx = idx % embedding_dim;
            apply_elastic_net_and_clip(&ev[idx], &grad_ev_scaled[embed_idx], idx, learning_rate, lambda_l1, lambda_l2, max_grad_clip_value);
        }
    }
}

__global__ void kernelUpdateWeightsHeadElasticNet(float* mh_a, float* mv_a, float* mq_a, float* mk_a,
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
        apply_elastic_net_and_clip(&mh_a[idx], &grad_mh[idx], idx, learning_rate, lambda_l1, lambda_l2, max_grad_clip_value);
        apply_elastic_net_and_clip(&mv_a[idx], &grad_mv[idx], idx, learning_rate, lambda_l1, lambda_l2, max_grad_clip_value);
        apply_elastic_net_and_clip(&mq_a[idx], &grad_mq[idx], idx, learning_rate, lambda_l1, lambda_l2, max_grad_clip_value);
        apply_elastic_net_and_clip(&mk_a[idx], &grad_mk[idx], idx, learning_rate, lambda_l1, lambda_l2, max_grad_clip_value);
    }
    if (update_eh != 0 && idx < embedding_dim) {
        apply_elastic_net_and_clip(&eh[idx], &grad_eh[idx], idx, learning_rate, lambda_l1, lambda_l2, max_grad_clip_value);
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
        apply_elastic_net_and_clip(&mv_a[idx], &grad_mv[idx], idx, learning_rate, lambda_l1, lambda_l2, max_grad_clip_value);
        apply_elastic_net_and_clip(&mq_a[idx], &grad_mq[idx], idx, learning_rate, lambda_l1, lambda_l2, max_grad_clip_value);
        apply_elastic_net_and_clip(&mk_a[idx], &grad_mk_correction[idx], idx, learning_rate, lambda_l1, lambda_l2, max_grad_clip_value);
    }
}

__global__ void kernelUpdateWeightsGeneral(
    float* weights,
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
        float l2_reg_term lambda_l2 * current_weight;

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

__global__ void updateEmbeddings(float* embeddings, const float* gradientsVector, float learning_rate,
    float lambda_l1, float lambda_l2, int embeddingRows, int embeddingDim)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x; // This will be the row index (embeddingRows)

    if (global_id < embeddingRows) {
        int row_start_idx = global_id * embeddingDim;

        for (int i = 0; i < embeddingDim; ++i) {
            int current_idx = row_start_idx + i;
            float current_weight = embeddings[current_idx];
            float error_gradient = (gradientsVector != nullptr) ? gradientsVector[i] : 0.0f;// Use ternary operator
            float l1_reg_term = lambda_l1 * ((__signbit(current_weight) == 0) ? 1.0f : -1.0f);
            float l2_reg_term lambda_l2 * current_weight;
            float total_gradient = error_gradient + l1_reg_term + l2_reg_term;
            if (isnan(total_gradient)) {
                total_gradient = 0.0f;
            } else if (isinf(total_gradient)) {
                total_gradient = copysignf(FLT_MAX, total_gradient);
            }
            embeddings[current_idx] -= learning_rate * total_gradient;
        }
    }
}

#endif