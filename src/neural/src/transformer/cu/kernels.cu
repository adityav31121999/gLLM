
#include "include/transformer.hpp"
#include <cuda.h>
#include <cuda_runtime.h>

// --- Kernel Implementation for EH Accumulation ---
// This kernel assumes d_eh_pointers is a device array where each element
// points to a valid device memory location containing an EH vector of size embedding_dim.
__global__ void accumulateEH(float** d_eh_pointers, float* d_otok, int num_layers, int embedding_dim)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // Parallelize over embedding_dim

    if (tid < embedding_dim && d_eh_pointers != nullptr) {
        float sum = 0.0f;
        for (int layer = 0; layer < num_layers; ++layer) {
            float* current_eh_ptr = d_eh_pointers[layer];
            if (current_eh_ptr != nullptr) {
                sum += current_eh_ptr[tid];
            }
        }
        d_otok[tid] = sum;
    }
}

/**
 * @brief compute product of vector and a matrix
 * @param vector  A pointer to a device memory location holding a vector of floats (the input vector).
 * @param matrix  A pointer to a device memory location holding a matrix of floats (the input matrix). 
 * The matrix is assumed to be stored in row-major order.
 * @param results A pointer to a device memory location where the results (dot products) will be stored. 
 * Each element in `results` will hold the dot product of the input `vector` with a corresponding row of 
 * the input `matrix`.
 * @param num_rows The number of rows in the input `matrix`. This also represents the number of dot 
 * products to be computed and the size of the `results` array.
 * @param vector_dim The dimension (length) of the input `vector` and the number of columns in the input `
 * matrix`.
 */
__global__ void computeAllDotsKernel(const float* vector, const float* matrix, float* results, int num_rows, int vector_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_rows) {
        const float* matrix_row = matrix + idx * vector_dim;
        float dot_product = 0.0f;
        // Compute dot product using loop unrolling or optimized libraries like cuBLAS::dot for better performance if needed
        for (int k = 0; k < vector_dim; ++k) {
            dot_product += vector[k] * matrix_row[k];
        }
        results[idx] = dot_product;
    }
}
