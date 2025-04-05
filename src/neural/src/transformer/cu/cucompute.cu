
// CUDA compute functions
#include "include/transformer.hpp"
#include "include/block.hpp"
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
 * @brief Dot product of T x M x T' using CUDA
 * @param T token embedding
 * @param M matrix for attention head calculation (MQ x MK')
 * @param dot = T x M x T'
 */
void cuComputeDot(std::vector<float>& T, std::vector<std::vector<float>>& M, float& dot) {
    int T_size = T.size();
    int M_rows = M.size();
    
    // Flatten M for CUDA
    std::vector<float> M_flat;
    for (int i = 0; i < M_rows; i++) {
        M_flat.insert(M_flat.end(), M[i].begin(), M[i].end());
    }
    
    // Allocate device memory
    float *d_T, *d_M, *d_temp, *d_dot;
    CUDA_CHECK(cudaMalloc(&d_T, T_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_M, M_rows * T_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_temp, T_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dot, sizeof(float)));
    
    // Initialize d_dot to 0
    CUDA_CHECK(cudaMemset(d_dot, 0, sizeof(float)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_T, T.data(), T_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_M, M_flat.data(), M_rows * T_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Compute T x M
    int threadsPerBlock = 256;
    int blocksPerGrid = (M_rows + threadsPerBlock - 1) / threadsPerBlock;
    matVecMultKernel<<<blocksPerGrid, threadsPerBlock>>>(d_M, d_T, d_temp, M_rows, T_size);
    
    // Compute (T x M) x T'
    blocksPerGrid = (T_size + threadsPerBlock - 1) / threadsPerBlock;
    innerProductKernel<<<blocksPerGrid, threadsPerBlock>>>(d_temp, d_T, d_dot, T_size);
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(&dot, d_dot, sizeof(float), cudaMemcpyDeviceToHost));
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_T));
    CUDA_CHECK(cudaFree(d_M));
    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaFree(d_dot));
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

/**
 * @brief KdotQ via tokens (TxMxT') where M = MQ x MK' using CUDA
 * @param KdotQ dot product
 * @param tokenEmbed tokens
 * @param M QK' cache
 * @param currentTokenCount number of tokens in context
 * @param promptCount tokens in prompt 
 * @param attentionType attention type, 1 for self, 0 for cross
 */
void cuComputeKdotQ(std::vector<std::vector<float>>& KdotQ, std::vector<std::vector<float>>& tokenEmbed, mat& M, int& currentTokenCount, 
    int& promptCount, bool& attentionType)
{
    // original input
    if (currentTokenCount == 0) {
        // single word like 'Hey', 'Hi', 'Hello', 'Oye', etc.
        if(promptCount == 1) {
            cuComputeDot(tokenEmbed[0], M, tokenEmbed[0], KdotQ[0][0]);
            KdotQ[0][0] = KdotQ[0][0] / SCALING;
            currentTokenCount += 1;
        }
        // long prompt input like 'Hello there, Obi'van Kenobi here', etc.
        else {
            for(int i = 0; i < promptCount; i++) {
                for(int j = 0; j < (attentionType ? i : promptCount); j++) {
                    cuComputeDot(tokenEmbed[i], M, tokenEmbed[j], KdotQ[i][j]);
                    KdotQ[i][j] = KdotQ[i][j] / SCALING;
                }
            }
            currentTokenCount += promptCount;
        }
    }
    // other prompts
    else {
        // promptCount >= 1
        // for single word prompt like 'Why', 'What', 'Who', 'How', 'seriously': promptCount = 1
        // for long prompt like 'are you serious', 'is this really true fact', etc.
        int c = currentTokenCount;
        for(int i = 0; i < promptCount; i++) {
            // diagonal element
            cuComputeDot(tokenEmbed[c + i], M, tokenEmbed[c + i], KdotQ[c + i][c + i]);
            KdotQ[c + i][c + i] = KdotQ[c + i][c + i] / SCALING;
            // for rows and columns
            for(int j = 0; j < currentTokenCount; j++) {
                // for row
                cuComputeDot(tokenEmbed[c + i], M, tokenEmbed[j], KdotQ[c+i][j]);
                KdotQ[c+i][j] = KdotQ[c+i][j] / SCALING;
                // for column, FOR CROSS ATTENTION ONLY 
                if(attentionType == 0) {
                    cuComputeDot(tokenEmbed[j], M, tokenEmbed[c+i], KdotQ[j][c+i]);
                    KdotQ[j][c+i] = KdotQ[j][c+i] / SCALING;
                }
            }
            currentTokenCount += 1;
        }
    }
}

// CUDA kernel for inner product calculation
__global__ void innerProductBatchKernel(float* K, float* Q, float* results, int K_rows, int K_cols, int Q_rows) {
    int i = blockIdx.x;
    int j = threadIdx.x;
    
    if (i < K_rows && j < Q_rows) {
        float sum = 0.0f;
        for (int k = 0; k < K_cols; k++) {
            sum += K[i * K_cols + k] * Q[j * K_cols + k];
        }
        results[i * Q_rows + j] = sum / sqrtf(K_cols); // Apply scaling
    }
}

/**
 * @brief KdotQ via QxK (Q[i].K[j]) using CUDA
 * @param KdotQ dot product
 * @param K Keys
 * @param Q Queries
 * @param currentTokenCount number of tokens in context
 * @param promptCount tokens in prompt
 * @param attentionType attention type, 1 for self, 0 for cross
 */
void cuComputeKdotQ(std::vector<std::vector<float>>& KdotQ, std::vector<std::vector<float>>& K, std::vector<std::vector<float>>& Q, 
    int& currentTokenCount, int& promptCount, bool& attentionType)
{
        // Using native CUDA for inner products
    if (currentTokenCount == 0) {
        if(promptCount == 1) {
            // Single token case
            float result = 0.0f;
            int size = K[0].size();
            
            // Allocate device memory
            float *d_K, *d_Q, *d_result;
            CUDA_CHECK(cudaMalloc(&d_K, size * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_Q, size * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));
            
            // Initialize result to 0
            CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));
            
            // Copy data to device
            CUDA_CHECK(cudaMemcpy(d_K, K[0].data(), size * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_Q, Q[0].data(), size * sizeof(float), cudaMemcpyHostToDevice));
            
            // Launch kernel
            int threadsPerBlock = 256;
            int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
            innerProductKernel<<<blocksPerGrid, threadsPerBlock>>>(d_K, d_Q, d_result, size);
            
            // Copy result back to host
            CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
            
            // Free device memory
            CUDA_CHECK(cudaFree(d_K));
            CUDA_CHECK(cudaFree(d_Q));
            CUDA_CHECK(cudaFree(d_result));
            
            KdotQ[0][0] = result / SCALING;
            currentTokenCount += 1;
        }
        else {
            // Multiple tokens in prompt
            for(int i = 0; i < promptCount; i++) {
                for(int j = 0; j < (attentionType ? i : promptCount); j++) {
                    float result = 0.0f;
                    int size = K[i].size();
                    
                    // Allocate device memory
                    float *d_K, *d_Q, *d_result;
                    CUDA_CHECK(cudaMalloc(&d_K, size * sizeof(float)));
                    CUDA_CHECK(cudaMalloc(&d_Q, size * sizeof(float)));
                    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));
                    
                    // Initialize result to 0
                    CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));
                    
                    // Copy data to device
                    CUDA_CHECK(cudaMemcpy(d_K, K[i].data(), size * sizeof(float), cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(d_Q, Q[j].data(), size * sizeof(float), cudaMemcpyHostToDevice));
                    
                    // Launch kernel
                    int threadsPerBlock = 256;
                    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
                    innerProductKernel<<<blocksPerGrid, threadsPerBlock>>>(d_K, d_Q, d_result, size);
                    
                    // Copy result back to host
                    CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
                    
                    // Free device memory
                    CUDA_CHECK(cudaFree(d_K));
                    CUDA_CHECK(cudaFree(d_Q));
                    CUDA_CHECK(cudaFree(d_result));
                    
                    KdotQ[i][j] = result / SCALING;
                }
            }
            currentTokenCount += promptCount;
        }
    }
    else {
        // promptCount >= 1
        int c = currentTokenCount;
        for(int i = 0; i < promptCount; i++) {
            // Diagonal element
            float result = 0.0f;
            int size = K[c+i].size();
            
            // Allocate device memory
            float *d_K, *d_Q, *d_result;
            CUDA_CHECK(cudaMalloc(&d_K, size * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_Q, size * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));
            
            // Initialize result to 0
            CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));
            
            // Copy data to device
            CUDA_CHECK(cudaMemcpy(d_K, K[c+i].data(), size * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_Q, Q[c+i].data(), size * sizeof(float), cudaMemcpyHostToDevice));
            
            // Launch kernel
            int threadsPerBlock = 256;
            int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
            innerProductKernel<<<blocksPerGrid, threadsPerBlock>>>(d_K, d_Q, d_result, size);
            
            // Copy result back to host
            CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
            
            KdotQ[c+i][c+i] = result / SCALING;
            
            for(int j = 0; j < currentTokenCount; j++) {
                // For row
                CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));
                CUDA_CHECK(cudaMemcpy(d_Q, Q[j].data(), size * sizeof(float), cudaMemcpyHostToDevice));
                innerProductKernel<<<blocksPerGrid, threadsPerBlock>>>(d_K, d_Q, d_result, size);
                CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
                KdotQ[c+i][j] = result / SCALING;
                
                // For column, FOR CROSS ATTENTION ONLY
                if(attentionType == 0) {
                    CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));
                    CUDA_CHECK(cudaMemcpy(d_K, K[j].data(), size * sizeof(float), cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(d_Q, Q[c+i].data(), size * sizeof(float), cudaMemcpyHostToDevice));
                    innerProductKernel<<<blocksPerGrid, threadsPerBlock>>>(d_K, d_Q, d_result, size);
                    CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
                    KdotQ[j][c+i] = result / SCALING;
                }
            }
            
            // Free device memory
            CUDA_CHECK(cudaFree(d_K));
            CUDA_CHECK(cudaFree(d_Q));
            CUDA_CHECK(cudaFree(d_result));
            
            currentTokenCount += 1;
        }
    }
}

/**
 * @brief compute key vectors using CUDA
 * @param[in] t token embedding
 * @param[in] m matrix input
 * @param[out] k key vector = t x m
 */
void cuComputeKeys(std::vector<float>& t, std::vector<std::vector<float>>& m, std::vector<float>& k) {
    int t_size = t.size();
    int m_rows = m.size();
    
    // Flatten m for CUDA
    std::vector<float> m_flat;
    for (int i = 0; i < m_rows; i++) {
        m_flat.insert(m_flat.end(), m[i].begin(), m[i].end());
    }
    
    // Allocate device memory
    float *d_t, *d_m, *d_k;
    CUDA_CHECK(cudaMalloc(&d_t, t_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_m, m_rows * t_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k, m_rows * sizeof(float)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_t, t.data(), t_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_m, m_flat.data(), m_rows * t_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (m_rows + threadsPerBlock - 1) / threadsPerBlock;
    matVecMultKernel<<<blocksPerGrid, threadsPerBlock>>>(d_m, d_t, d_k, m_rows, t_size);
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(k.data(), d_k, m_rows * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_t));
    CUDA_CHECK(cudaFree(d_m));
    CUDA_CHECK(cudaFree(d_k));
}

/**
 * @brief compute query vectors using CUDA
 * @param[in] t token embedding
 * @param[in] m query matrix input
 * @param[out] q query vector = t x m
 */
void cuComputeQuerys(std::vector<float>& t, std::vector<std::vector<float>>& m, std::vector<float>& q) {
    int t_size = t.size();
    int m_rows = m.size();
    
    // Flatten m for CUDA
    std::vector<float> m_flat;
    for (int i = 0; i < m_rows; i++) {
        m_flat.insert(m_flat.end(), m[i].begin(), m[i].end());
    }
    
    // Allocate device memory
    float *d_t, *d_m, *d_q;
    CUDA_CHECK(cudaMalloc(&d_t, t_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_m, m_rows * t_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_q, m_rows * sizeof(float)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_t, t.data(), t_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_m, m_flat.data(), m_rows * t_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (m_rows + threadsPerBlock - 1) / threadsPerBlock;
    matVecMultKernel<<<blocksPerGrid, threadsPerBlock>>>(d_m, d_t, d_q, m_rows, t_size);
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(q.data(), d_q, m_rows * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_t));
    CUDA_CHECK(cudaFree(d_m));
    CUDA_CHECK(cudaFree(d_q));
}

/**
 * @brief compute KdotQs of each head in the block using CUDA
 * @param promptCount number of tokens in prompt
 * @param currentTokenCount current count of tokens in full context
 * @param blockCount current block index
 * @param isSelf attention type
 */
void transformer::cuComputeKdotQs(int& promptCount, int& currentTokenCount, int& blockCount, bool& isSelf) {
    // for first block
    if (blockCount == 1) {
        for (int i = 0; i < x; i++) {
            for (int j = 0; j < y; j++) {
                for (int k = 0; k < promptCount; k++) {
                    cuComputeKeys(input[k], t[0].b[i][j].MK.a, t[0].b[i][j].K[k]);
                    cuComputeQuerys(input[k], t[0].b[i][j].MQ.a, t[0].b[i][j].Q[k]);
                }
                cuComputeKdotQ(t[0].b[i][j].KdotQ, t[0].b[i][j].K, t[0].b[i][j].Q, currentTokenCount, promptCount, isSelf);
            }
        }
    }
    // for other blocks
    else {
        for (int i = 0; i < x; i++) {
            for (int j = 0; j < y; j++) {
                for (int k = 0; k < promptCount; k++) {
                    cuComputeKeys(input[k], t[blockCount-1].b[i][j].MK.a, t[blockCount-1].b[i][j].K[k]);
                    cuComputeQuerys(input[k], t[blockCount-1].b[i][j].MQ.a, t[blockCount-1].b[i][j].Q[k]);
                }
                cuComputeKdotQ(t[blockCount-1].b[i][j].KdotQ, t[blockCount-1].b[i][j].K, t[blockCount-1].b[i][j].Q, currentTokenCount, promptCount, isSelf);
            }
        }
    }
}