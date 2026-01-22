#ifndef CUSUP_HPP
#define CUSUP_HPP 1
#ifdef USE_CU
#include <cuda_runtime.h>

/**
 * @brief CUDA device function to compute the dot product of two vectors.
 * @param[in] vec1 Device pointer to the first vector.
 * @param[in] vec2 Device pointer to the second vector.
 * @param[in] dim The dimension (number of elements) of the vectors.
 * @return The scalar dot product of vec1 and vec2.
 */
__device__ inline float cuComputeDot(const float* vec1, const float* vec2, int dim) {
    float dot_product = 0.0f;
    for (int k = 0; k < dim; ++k) {
        dot_product += vec1[k] * vec2[k];
    }
    return dot_product;
}

/**
 * @brief CUDA device function to compute the quadratic form vec1 * matrix * vec2^T.
 * @param[in] vec1 Device pointer to the first vector (treated as a row vector, size dim).
 * @param[in] vec2 Device pointer to the second vector (treated as a column vector, size dim).
 * @param[in] matrix Device pointer to the matrix (row-major, dim x dim).
 * @param[in] dim The dimension of the vectors and the square matrix.
 * @return The scalar result of vec1 * matrix * vec2^T.
 */
__device__ inline float cuComputeDotvmv(const float* vec1, const float* vec2, const float* matrix, int dim)
{
    float final_dot_product = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float inner_sum = 0.0f;
        const float* matrix_row_i = matrix + i * dim;
        inner_sum = cuComputeDot(vec1, matrix_row_i, dim); // Re-use cuComputeDot
        final_dot_product += inner_sum * vec2[i];
    }
    return final_dot_product;
}

__global__ void cuSigmoid(float x, float* result);
__global__ void cuSigmoid(float* x, float* out, int size);
__global__ void cuSigmoid(float* x, float* out, int rows, int cols);
__global__ void cuSoftmax(const float* __restrict__ x, float* __restrict__ out, float temp, int size);
__global__ void cuSoftmax(const float* __restrict__ x, float* __restrict__ out, float temp, int rows, int cols);
__global__ void cuReLU(float x, float* result);
__global__ void cuReLU(float* x, float* out, int size);
__global__ void cuLOTA(float* y, float* out, int size);
__global__ void cuLOTA(float* y, float* out, int rows, int cols);
__global__ void cuLOTA(float* y, float* out, int rows, int cols, int limit, bool attentionType);

__global__ void cuSigmoidder(float x, float* result);
__global__ void cuSigmoidder(float* x, float* out, int rows, int cols);
__global__ void cuSoftmaxder(float* x, float* out, float temp, int size);
__global__ void cuSoftmaxder(float* x, float* out, float temp, int rows, int cols);
__global__ void cuReLUder(float x, float* result);
__global__ void cuReLUder(float* x, float* out, int size);
__global__ void cuLOTAder(float* y, float* out, int size);
__global__ void cuLOTAder(float* y, float* out, int rows, int cols);
__global__ void cuLOTAder(float* y, float* out, int rows, int cols, int limit, bool attentionType);

__global__ void operator_add(const float* a, const float* b, float* result, int size);
__global__ void operator_sub(const float* a, const float* b, float* result, int size);
__global__ void operator_mul(const float* a, float scalar, float* result, int size);
__global__ void operator_mul_reverse(float scalar, const float* a, float* result, int size);
__global__ void operator_div(const float* a, float scalar, float* result, int size);
__global__ void operator_add_2d(const float* a, const float* b, float* result, int rows, int cols);
__global__ void operator_sub_2d(const float* a, const float* b, float* result, int rows, int cols);
__global__ void operator_mul_2d(const float* a, float scalar, float* result, int rows, int cols);
__global__ void operator_div_2d(const float* a, float scalar, float* result, int rows, int cols);

__global__ void gradientdesc(const float* a, const float* b, float* result, int size);
__global__ void vdotv2val(const float* a, const float* b, float* result, int size);
__global__ void vdotv2scal(const float* a, const float* b, float* result, int size);
__global__ void MSE(const float* a, const float* b, float* result, int size);
__global__ void sum(const float* a, float* result, int size);
__global__ void sum_2d(const float* a, float* result, int rows, int cols);
__global__ void product(const float* a, float* result, int size);
__global__ void product_2d(const float* a, float* result, int rows, int cols);

__global__ void matrixMultiplyKernel(const float* A, const float* B, float* C, int rowsA, int colsA, int colsB);
__global__ void vectorAddKernel(const float* A, const float* B, float* C, int len);

#endif
#endif