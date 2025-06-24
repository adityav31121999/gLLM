#ifdef USE_CUDA

// vector operations in cuda programs
#include "include/basic.hpp"
#include <cuda.h>
#include <cuda_runtime.h>

/**
 * 
 */
__global__ void someKernelWithReduction(const float* input, float* output_scalar, int N) {
    extern __shared__ float sdata[]; // Dynamically sized shared memory

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    if (i < N) {
        sdata[tid] = input[i];
    } else {
        sdata[tid] = 0; // Neutral element for sum (or FLT_MAX for min, -FLT_MAX for max)
    }
    // Handle cases where N is not a multiple of blockDim.x if loading more than one element per thread

    __syncthreads();

    // Parallel reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s]; // For sum. Use fminf/fmaxf for min/max
        }
        __syncthreads();
    }

    // Thread 0 writes the result for this block
    if (tid == 0) {
        // If multiple blocks, need atomicAdd or further reduction on host/another kernel
        atomicAdd(output_scalar, sdata[0]);
        // Or if only one block is expected to produce the final scalar:
        // *output_scalar = sdata[0];
    }
}


/**
 * @brief CUDA kernel for matrix multiplication C = A * B (Row-Major)
 * A (rowsA x colsA), B (colsA x colsB), C (rowsA x colsB)
 * Used here for vector-matrix product (1xh * hxd -> 1xd)
 * @param[in] A points to vector
 * @param[in] B points to matrix
 * @param[out] c points to output vector
 * @param[in] rowsA rows of A (= 1)
 * @param[in] colsA columns of A and rows of B
 * @param[in] colsB columns of B
 */
__global__ void matrixMultiplyKernel(const float* A, const float* B, float* C, int rowsA, int colsA, int colsB)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;        // for rows
    int col = blockIdx.x * blockDim.x + threadIdx.x;        // for columns

    if (row < rowsA && col < colsB) {
        float sum = 0.0f;
        for (int i = 0; i < colsA; ++i) {
            // A[row * colsA + i] * B[i * colsB + col]
            sum += A[row * colsA + i] * B[i * colsB + col];
        }
        C[row * colsB + col] = sum;
    }
}


/**
 * @brief CUDA kernel for element-wise vector addition C = A + B
 * @param[in] A points to vector
 * @param[in] B points to vector
 * @param[out] c points to output vector
 * @param[in] len size of vectors
 */
__global__ void vectorAddKernel(const float* A, const float* B, float* C, int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        C[idx] = A[idx] + B[idx];
    }
}

#endif