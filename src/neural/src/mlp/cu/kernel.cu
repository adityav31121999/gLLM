#ifdef USE_CU

#include "include/mlp.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <maths.hpp>


/**
 * @brief kernel for calculating L1 penalty
 * @param[in] weights 3D vector whose penalty to be calculated
 * @param[out] result L1 penalty
 * @param[in] size size of weights
 */
__global__ void l1PenaltyKernel(float* weights, float* result, int size) {
    __shared__ float temp[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread computes absolute value for its element
    temp[tid] = (i < size) ? fabsf(weights[i]) : 0.0f;
    
    __syncthreads();
    
    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            temp[tid] += temp[tid + stride];
        }
        __syncthreads();
    }
    
    // Write the result for this block to global memory
    if (tid == 0) {
        atomicAdd(result, temp[0]);
    }
}

/**
 * @brief kernel for calculating L2 penalty
 * @param[in] weights 3D vector whose penalty to be calculated
 * @param[out] result L2 penalty
 * @param[in] size size of weights
 */
__global__ void l2PenaltyKernel(float* weights, float* result, int size) {
    __shared__ float temp[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread computes square for its element
    temp[tid] = (i < size) ? weights[i] * weights[i] : 0.0f;
    
    __syncthreads();
    
    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            temp[tid] += temp[tid + stride];
        }
        __syncthreads();
    }
    
    // Write the result for this block to global memory
    if (tid == 0) {
        atomicAdd(result, temp[0]);
    }
}



/**
 * @brief kernel to calculate absolute difference
 * @param[in] output original output vector from a process
 * @param[in] targets expected output vector from same process
 * @param[out] result absolute[output[i] - target[i]]
 * @param[in] size size of each vector
 */
__global__ void absDiffKernel(float* outputs, float* targets, float* result, int size) {
    __shared__ float temp[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread computes absolute difference for its element
    temp[tid] = (i < size) ? fabsf(outputs[i] - targets[i]) : 0.0f;
    
    __syncthreads();
    
    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            temp[tid] += temp[tid + stride];
        }
        __syncthreads();
    }
    
    // Write the result for this block to global memory
    if (tid == 0) {
        atomicAdd(result, temp[0]);
    }
}

/**
 * @brief kernel to calculate squared difference
 * @param[in] output original output vector from a process
 * @param[in] targets expected output vector from same process
 * @param[out] result absolute(output[i]^2 - target[i]^2)
 * @param[in] size size of each vector
 */
__global__ void squaredDiffKernel(float* outputs, float* targets, float* result, int size) {
    __shared__ float temp[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread computes squared difference for its element
    if (i < size) {
        float diff = outputs[i] - targets[i];
        temp[tid] = diff * diff;
    } else {
        temp[tid] = 0.0f;
    }
    
    __syncthreads();
    
    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            temp[tid] += temp[tid + stride];
        }
        __syncthreads();
    }
    
    // Write the result for this block to global memory
    if (tid == 0) {
        atomicAdd(result, temp[0]);
    }
}

/**
 * @brief CUDA kernel for calculating the Mean Squared Error (MSE).
 * This kernel computes the squared difference between the expected and actual output for each neuron
 * and accumulates the sum using atomic operations.
 * @param expected Pointer to the expected output data on the device.
 * @param output Pointer to the output data on the device.
 * @param mse Pointer to the MSE value on the device (will be updated).
 * @param size The number of output neurons.
 */
__global__ void cuMSEKernel(float* expected, float* output, float* mse, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    atomicAdd(mse, powf(expected[idx] - output[idx], 2));
}

#endif