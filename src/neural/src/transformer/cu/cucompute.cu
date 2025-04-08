
// CUDA compute functions
#include "include/transformer.hpp"
#include "include/block.hpp"
#include "include/attention.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>


// CUDA intrinsics need to be conditionally included when compiling with a C++ compiler
#ifdef __CUDACC__
// CUDA compiler path - direct access to intrinsics
#else
// C++ compiler path - provide fallbacks for CUDA intrinsics if needed
#define __syncthreads() ((void)0)
#define atomicAdd(a, b) (*(a) += (b))
#endif

// Define the scaling factor used in attention calculations
#ifndef SCALING
#define SCALING 8.0f
#endif

// CUDA error checking macro
#define CUDA_CHECK(call) \
        do { \
            cudaError_t error = call; \
            if (error != cudaSuccess) { \
                fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                        __FILE__, __LINE__, cudaGetErrorString(error)); \
                exit(EXIT_FAILURE); \
            } \
        } while(0)

// CUDA kernel for computing inner product of two vectors
__global__ void innerProductKernel(float* a, float* b, float* result, int size) {
    __shared__ float temp[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    temp[tid] = (i < size) ? a[i] * b[i] : 0.0f;
    
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

// CUDA kernel for matrix-vector multiplication
__global__ void matVecMultKernel(float* matrix, float* vector, float* result, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        float sum = 0.0f;
        for (int i = 0; i < cols; i++) {
            sum += matrix[row * cols + i] * vector[i];
        }
        result[row] = sum;
    }
}


/**
 * @brief Dot product of T1 x M x T2' using CUDA
 * @param T1 token embedding
 * @param T2 token embedding
 * @param M matrix for attention head calculation (MQ x MK')
 * @param dot = T1 x M x T2'
 */
void cuComputeDot(std::vector<float>& T1, std::vector<float>& T2, std::vector<std::vector<float>>& M, float& dot) {
    int T_size = T1.size();
    int M_rows = M.size();
    
    // Flatten M for CUDA
    std::vector<float> M_flat;
    for (int i = 0; i < M_rows; i++) {
        M_flat.insert(M_flat.end(), M[i].begin(), M[i].end());
    }
    
    // Allocate device memory
    float *d_T1, *d_T2, *d_M, *d_temp, *d_dot;
    CUDA_CHECK(cudaMalloc(&d_T1, T_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_T2, T_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_M, M_rows * T_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_temp, T_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dot, sizeof(float)));
    
    // Initialize d_dot to 0
    CUDA_CHECK(cudaMemset(d_dot, 0, sizeof(float)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_T1, T1.data(), T_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_T2, T2.data(), T_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_M, M_flat.data(), M_rows * T_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Compute T1 x M
    int threadsPerBlock = 256;
    int blocksPerGrid = (M_rows + threadsPerBlock - 1) / threadsPerBlock;
    matVecMultKernel<<<blocksPerGrid, threadsPerBlock>>>(d_M, d_T1, d_temp, M_rows, T_size);
    
    // Compute (T1 x M) x T2'
    blocksPerGrid = (T_size + threadsPerBlock - 1) / threadsPerBlock;
    innerProductKernel<<<blocksPerGrid, threadsPerBlock>>>(d_temp, d_T2, d_dot, T_size);
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(&dot, d_dot, sizeof(float), cudaMemcpyDeviceToHost));
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_T1));
    CUDA_CHECK(cudaFree(d_T2));
    CUDA_CHECK(cudaFree(d_M));
    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaFree(d_dot));
}

/**
 * @brief Dot product of Ti x M x Tj' using CUDA
 * @param Ti ith token
 * @param M QK' cache
 * @param Tj jth token (count as transpose)
 * @param dot = Ti x M x Tj'
 */
void cuComputeDot(std::vector<float>& Ti, mat M, std::vector<float>& Tj, float& dot) {
    int Ti_size = Ti.size();
    int M_rows = M.row;
    
    // Flatten M.a for CUDA
    std::vector<float> M_flat;
    for (int i = 0; i < M_rows; i++) {
        M_flat.insert(M_flat.end(), M.a[i].begin(), M.a[i].end());
    }
    
    // Allocate device memory
    float *d_Ti, *d_Tj, *d_M, *d_temp, *d_dot;
    CUDA_CHECK(cudaMalloc(&d_Ti, Ti_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Tj, Ti_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_M, M_rows * Ti_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_temp, Ti_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dot, sizeof(float)));
    
    // Initialize d_dot to 0
    CUDA_CHECK(cudaMemset(d_dot, 0, sizeof(float)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_Ti, Ti.data(), Ti_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Tj, Tj.data(), Ti_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_M, M_flat.data(), M_rows * Ti_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Compute Ti x M
    int threadsPerBlock = 256;
    int blocksPerGrid = (M_rows + threadsPerBlock - 1) / threadsPerBlock;
    matVecMultKernel<<<blocksPerGrid, threadsPerBlock>>>(d_M, d_Ti, d_temp, M_rows, Ti_size);
    
    // Compute (Ti x M) x Tj'
    blocksPerGrid = (Ti_size + threadsPerBlock - 1) / threadsPerBlock;
    innerProductKernel<<<blocksPerGrid, threadsPerBlock>>>(d_temp, d_Tj, d_dot, Ti_size);
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(&dot, d_dot, sizeof(float), cudaMemcpyDeviceToHost));
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_Ti));
    CUDA_CHECK(cudaFree(d_Tj));
    CUDA_CHECK(cudaFree(d_M));
    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaFree(d_dot));
}

