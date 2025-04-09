
// CUDA compute functions
#include "include/transformer.hpp"
#include "include/block.hpp"
#include "include/attention.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>
#include <cmath> // Include cmath for device-side sqrt

// Helper function to check for CUDA errors
#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t error = call;                                                        \
        if (error != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,             \
                    cudaGetErrorString(error));                                          \
            exit(EXIT_FAILURE);                                                          \
        }                                                                                \
    } while (0)



/**
 * @brief CUDA kernel for computing the dot product of T1 x M x T2'
 * @param[in] T1 Pointer to the first vector (device memory)
 * @param[in] T2 Pointer to the second vector (device memory)
 * @param[in] M Pointer to the matrix (device memory)
 * @param[out] dot Pointer to the result (device memory)
 * @param[in] size Size of the vectors (EMBEDDING)
 */
__global__ void cuComputeDotKernel(const float* T1, const float* T2, const float* M, float* dot, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) return;

    float temp = 0.0f;
    for (int i = 0; i < size; ++i) {
        temp += T1[i] * M[tid * size + i];
    }
    atomicAdd(dot, temp * T2[tid]);
}

/**
 * @brief CUDA kernel for computing the dot product of Ti x M x Tj'
 * @param[in] Ti Pointer to the first vector (device memory)
 * @param[in] M Pointer to the matrix (device memory)
 * @param[in] Tj Pointer to the second vector (device memory)
 * @param[out] dot Pointer to the result (device memory)
 * @param[in] size Size of the vectors (EMBEDDING)
 */
__global__ void cuComputeDotKernel2(const float* Ti, const float* M, const float* Tj, float* dot, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) return;

    float temp = 0.0f;
    for (int i = 0; i < size; ++i) {
        temp += Ti[i] * M[tid * size + i];
    }
    atomicAdd(dot, temp * Tj[tid]);
}


/**
 * @brief Computes the dot product of T1 x M x T2' using CUDA
 * @param[in] T1 First vector
 * @param[in] T2 Second vector
 * @param[in] M Matrix
 * @param[out] dot Result of the dot product
 */
void cuComputeDot(std::vector<float>& T1, std::vector<float>& T2, std::vector<std::vector<float>>& M, float& dot) {
    int size = T1.size();
    float* d_T1, * d_T2, * d_M, * d_dot;

    CUDA_CHECK(cudaMalloc((void**)&d_T1, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_T2, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_M, size * size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_dot, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_T1, T1.data(), size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_T2, T2.data(), size * sizeof(float), cudaMemcpyHostToDevice));
    
    std::vector<float> M_flat;
    for (const auto& row : M) {
        M_flat.insert(M_flat.end(), row.begin(), row.end());
    }
    CUDA_CHECK(cudaMemcpy(d_M, M_flat.data(), size * size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_dot, 0, sizeof(float)));

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    cuComputeDotKernel<<<blocksPerGrid, threadsPerBlock>>>(d_T1, d_T2, d_M, d_dot, size);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(&dot, d_dot, sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_T1));
    CUDA_CHECK(cudaFree(d_T2));
    CUDA_CHECK(cudaFree(d_M));
    CUDA_CHECK(cudaFree(d_dot));
}

/**
 * @brief Computes the dot product of Ti x M x Tj' using CUDA
 * @param[in] Ti First vector
 * @param[in] M Matrix
 * @param[in] Tj Second vector
 * @param[out] dot Result of the dot product
 */
void cuComputeDot(std::vector<float>& Ti, mat M, std::vector<float>& Tj, float& dot) {
    int size = Ti.size();
    float* d_Ti, * d_Tj, * d_M, * d_dot;

    CUDA_CHECK(cudaMalloc((void**)&d_Ti, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_Tj, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_M, size * size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_dot, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_Ti, Ti.data(), size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Tj, Tj.data(), size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_M, M.a[0].data(), size * size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_dot, 0, sizeof(float)));

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    cuComputeDotKernel2<<<blocksPerGrid, threadsPerBlock>>>(d_Ti, d_M, d_Tj, d_dot, size);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(&dot, d_dot, sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_Ti));
    CUDA_CHECK(cudaFree(d_Tj));
    CUDA_CHECK(cudaFree(d_M));
    CUDA_CHECK(cudaFree(d_dot));
}

//------------------------COMPUTE KdotQ------------------------//

/**
 * @brief CUDA kernel for computing KdotQ via QxK (Q[i].K[j]) for training purpose
 * @param[in] K Pointer to the Keys (device memory)
 * @param[in] Q Pointer to the Queries (device memory)
 * @param[out] KdotQ Pointer to the result (device memory)
 * @param[in] currentTokenCount Number of tokens in full context
 * @param[in] promptCount Number of tokens in prompt
 * @param[in] size Size of the vectors (EMBEDDING)
 * @param[in] attentionType Attention type, 1 for self, 0 for cross
 * @param[in] scalingFactor Scaling factor for attention
 */
__global__ void cuComputeKdotQKernel(const float* K, const float* Q, float* KdotQ, int currentTokenCount, int promptCount, int size, 
    bool attentionType, float scalingFactor) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= promptCount || col >= (attentionType ? row + 1 : promptCount)) return;

    float dot = 0.0f;
    for (int i = 0; i < size; ++i) {
        dot += Q[row * size + i] * K[col * size + i];
    }
    KdotQ[row * (attentionType ? promptCount : currentTokenCount) + col] = dot / scalingFactor;
}


/**
 * @brief CUDA kernel for computing KdotQ via tokens (TxMxT') where M = MQ x MK' for use cases for first block
 * @param[in] tokenEmbed Pointer to the token embeddings (device memory)
 * @param[in] M Pointer to the QK' cache (device memory)
 * @param[out] KdotQ Pointer to the result (device memory)
 * @param[in] currentTokenCount Number of tokens in full context
 * @param[in] promptCount Number of tokens in prompt
 * @param[in] size Size of the vectors
 * @param[in] attentionType Attention type, 1 for self, 0 for cross
 * @param[in] scalingFactor Scaling factor for attention
 */
__global__ void cuComputeKdotQKernel2(const float* tokenEmbed, const float* M, float* KdotQ, int currentTokenCount, int promptCount,
    int size, bool attentionType, float scalingFactor) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= promptCount || col >= (attentionType ? row + 1 : currentTokenCount)) return;
    
    float dot = 0.0f;
    float temp = 0.0f;
    for (int i = 0; i < size; ++i) {
        temp = 0.0f;
        for(int j = 0; j < size; j++){
            temp += tokenEmbed[row * size + j] * M[i * size + j];
        }
        dot += temp * tokenEmbed[col * size + i];
    }
    KdotQ[row * (attentionType ? promptCount : currentTokenCount) + col] = dot / scalingFactor;
}


/**
 * @brief CUDA kernel for computing KdotQ via tokens (TxMxEVp') where M = MQ x MK' for use cases (for 2nd to last blocks)
 * @param[in] tokForBlock Pointer to the token embeddings for the current block (device memory)
 * @param[in] EVp Pointer to the EVp (device memory)
 * @param[in] M Pointer to the QK' cache (device memory)
 * @param[out] KdotQ Pointer to the result (device memory)
 * @param[in] currentTokenCount Number of tokens in full context
 * @param[in] promptCount Number of tokens in prompt
 * @param[in] size Size of the vectors
 * @param[in] c Number of tokens in current context
 * @param[in] attentionType Attention type, 1 for self, 0 for cross
 * @param[in] scalingFactor Scaling factor for attention
 */
__global__ void cuComputeKdotQKernel3(const float* tokForBlock, const float* EVp, const float* M, float* KdotQ, int currentTokenCount, 
    int promptCount, int size, int c, bool attentionType, float scalingFactor) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= promptCount || col >= (attentionType ? row + 1 : c)) return;

    float dot = 0.0f;
    float temp = 0.0f;
    for (int i = 0; i < size; ++i) {
        temp = 0.0f;
        for(int j = 0; j < size; j++){
            temp += tokForBlock[row * size + j] * M[i * size + j];
        }
        dot += temp * EVp[col * size + i];
    }
    KdotQ[row * (attentionType ? promptCount : c) + col] = dot / scalingFactor;
}

//------------------------COMPUTE KdotQ------------------------//

/**
 * @brief Computes KdotQ via QxK (Q[i].K[j]) for training purpose using CUDA
 * @param[out] KdotQ Dot product
 * @param[in] K Keys
 * @param[in] Q Queries
 * @param[in] currentTokenCount Number of tokens in full context
 * @param[in] promptCount Tokens in prompt
 * @param[in] attentionType Attention type, 1 for self, 0 for cross
 */
void cuComputeKdotQ(std::vector<std::vector<float>>& KdotQ, std::vector<std::vector<float>>& K, std::vector<std::vector<float>>& Q,
    int& currentTokenCount, int& promptCount, bool& attentionType) {
    int size = K[0].size();
    int rows = promptCount;
    int cols = attentionType ? promptCount : currentTokenCount;
    float* d_K, * d_Q, * d_KdotQ;
    float scalingFactor = std::sqrt((float)EMBEDDING); // Calculate scaling factor on host

    CUDA_CHECK(cudaMalloc((void**)&d_K, rows * size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_Q, rows * size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_KdotQ, rows * cols * sizeof(float)));

    std::vector<float> K_flat, Q_flat;
    for (const auto& row : K) {
        K_flat.insert(K_flat.end(), row.begin(), row.end());
    }
    for (const auto& row : Q) {
        Q_flat.insert(Q_flat.end(), row.begin(), row.end());
    }

    CUDA_CHECK(cudaMemcpy(d_K, K_flat.data(), rows * size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Q, Q_flat.data(), rows * size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_KdotQ, 0, rows * cols * sizeof(float)));

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cuComputeKdotQKernel<<<blocksPerGrid, threadsPerBlock>>>(d_K, d_Q, d_KdotQ, currentTokenCount, promptCount, size, attentionType, scalingFactor);
    CUDA_CHECK(cudaGetLastError());

    std::vector<float> KdotQ_flat(rows * cols);
    CUDA_CHECK(cudaMemcpy(KdotQ_flat.data(), d_KdotQ, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < (attentionType ? i + 1 : cols); ++j) {
            KdotQ[i][j] = KdotQ_flat[i * cols + j];
        }
    }

    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_KdotQ));
}

/**
 * @brief Computes KdotQ via tokens (TxMxT') where M = MQ x MK' for use cases for first block using CUDA
 * @param[out] KdotQ Dot product
 * @param[in] tokenEmbed Tokens
 * @param[in] M QK' cache
 * @param[in] currentTokenCount Number of tokens in full context
 * @param[in] promptCount Tokens in prompt
 * @param[in] attentionType Attention type, 1 for self, 0 for cross
 */
void cuComputeKdotQ(std::vector<std::vector<float>>& KdotQ, std::vector<std::vector<float>>& tokenEmbed, mat M, int& currentTokenCount,
    int& promptCount, bool& attentionType) {
    int size = tokenEmbed[0].size();
    int rows = promptCount;
    int cols = attentionType ? promptCount : currentTokenCount;
    float* d_tokenEmbed, * d_M, * d_KdotQ;
    float scalingFactor = std::sqrt((float)EMBEDDING); // Calculate scaling factor on host

    CUDA_CHECK(cudaMalloc((void**)&d_tokenEmbed, rows * size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_M, size * size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_KdotQ, rows * cols * sizeof(float)));

    std::vector<float> tokenEmbed_flat;
    for (const auto& row : tokenEmbed) {
        tokenEmbed_flat.insert(tokenEmbed_flat.end(), row.begin(), row.end());
    }

    CUDA_CHECK(cudaMemcpy(d_tokenEmbed, tokenEmbed_flat.data(), rows * size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_M, M.a[0].data(), size * size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_KdotQ, 0, rows * cols * sizeof(float)));

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cuComputeKdotQKernel2<<<blocksPerGrid, threadsPerBlock>>>(d_tokenEmbed, d_M, d_KdotQ, currentTokenCount, promptCount, size, attentionType, scalingFactor);
    CUDA_CHECK(cudaGetLastError());

    std::vector<float> KdotQ_flat(rows * cols);
    CUDA_CHECK(cudaMemcpy(KdotQ_flat.data(), d_KdotQ, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < (attentionType ? i + 1 : cols); ++j) {
            KdotQ[i][j] = KdotQ_flat[i * cols + j];
        }
    }

    CUDA_CHECK(cudaFree(d_tokenEmbed));
    CUDA_CHECK(cudaFree(d_M));
    CUDA_CHECK(cudaFree(d_KdotQ));
}

/**
 * @brief Computes KdotQ via tokens (TxMxEVp') where M = MQ x MK' for use cases (for 2nd to last blocks) using CUDA
 * @param[out] KdotQ Dot product
 * @param[in] tokForBlock Token embeddings for the current block
 * @param[in] EVp EVp
 * @param[in] M QK' cache
 * @param[in] currentTokenCount Number of tokens in full context
 * @param[in] promptCount Tokens in prompt
 * @param[in] blockCount Current block number
 * @param[in] attentionType Attention type, 1 for self, 0 for cross
 */
void cuComputeKdotQ(std::vector<std::vector<float>>& KdotQ, std::vector<std::vector<float>>& tokForBlock, std::vector<std::vector<float>>& EVp,
    mat M, int& currentTokenCount, int& promptCount, int& blockCount, bool& attentionType) {
    int size = tokForBlock[0].size();
    int rows = promptCount;
    int c = currentTokenCount - (blockCount - 1)*CONTEXT_WIN;
    int cols = attentionType ? promptCount : c;
    float* d_tokForBlock, * d_EVp, * d_M, * d_KdotQ;
    float scalingFactor = std::sqrt((float)EMBEDDING); // Calculate scaling factor on host

    CUDA_CHECK(cudaMalloc((void**)&d_tokForBlock, rows * size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_EVp, c * size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_M, size * size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_KdotQ, rows * cols * sizeof(float)));

    std::vector<float> tokForBlock_flat, EVp_flat;
    for (const auto& row : tokForBlock) {
        tokForBlock_flat.insert(tokForBlock_flat.end(), row.begin(), row.end());
    }
    for (const auto& row : EVp) {
        EVp_flat.insert(EVp_flat.end(), row.begin(), row.end());
    }

    CUDA_CHECK(cudaMemcpy(d_tokForBlock, tokForBlock_flat.data(), rows * size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_EVp, EVp_flat.data(), c * size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_M, M.a[0].data(), size * size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_KdotQ, 0, rows * cols * sizeof(float)));

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cuComputeKdotQKernel3<<<blocksPerGrid, threadsPerBlock>>>(d_tokForBlock, d_EVp, d_M, d_KdotQ, currentTokenCount, promptCount, size, c, attentionType, scalingFactor);
    CUDA_CHECK(cudaGetLastError());

    std::vector<float> KdotQ_flat(rows * cols);
    CUDA_CHECK(cudaMemcpy(KdotQ_flat.data(), d_KdotQ, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < (attentionType ? i + 1 : cols); ++j) {
            KdotQ[i][j] = KdotQ_flat[i * cols + j];
        }
    }

    CUDA_CHECK(cudaFree(d_tokForBlock));
    CUDA_CHECK(cudaFree(d_EVp));
    CUDA_CHECK(cudaFree(d_M));
    CUDA_CHECK(cudaFree(d_KdotQ));
}
