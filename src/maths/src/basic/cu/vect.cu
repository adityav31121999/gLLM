
// vector operations in cuda programs
#include "include/basic.hpp"
#include <cuda.h>
#include <cuda_runtime.h>

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
