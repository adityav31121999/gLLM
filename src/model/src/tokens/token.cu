#ifdef USE_CUDA

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <iostream>
#include <stdexcept>
#include "include/tokenise.hpp"

// Helper for CUDA Error Checking
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

/** 
 * --> f(i, j, seed) = (i * j + 1) * C * (seed^[j%d])
 * where: C = 0.01, x = seed, and  d is the embedding dimension.
 */
__global__ void embeddingFormulaBatchKernel(float* all_embeddings, const float* all_seeds, const int N, const int d) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N && j < d) {
        float seed = all_seeds[i];
        const float C = 0.01f;
        float result = static_cast<float>(i * j + 1) * C;
        int exponent = j;
        result *= powf(seed, static_cast<float>(exponent));
        all_embeddings[i * d + j] = result;
    }
}


/**
 * @brief Computes the inverse (v / ||v||^2) for a batch of vectors in parallel.
 * Each CUDA block is responsible for processing one vector (one row of the matrix).
 * @param output The output matrix (N x d), flattened.
 * @param input The input matrix (N x d), flattened.
 * @param N The number of vectors (rows).
 * @param d The dimension of each vector (columns).
 */
__global__ void batchedVectorInverseKernel(float* output, const float* input, int N, int d) {
    // Use dynamic shared memory, sized by the kernel launch.
    // This will hold the values for the reduction.
    extern __shared__ float s_data[];

    // Identify which row (vector) this block is working on.
    const int row_idx = blockIdx.y;

    // Identify the thread's index within the block and its global column index.
    const int tid_in_block = threadIdx.x;
    const int col_idx = blockIdx.x * blockDim.x + tid_in_block;

    // --- Step 1: Parallel Reduction to find the squared magnitude ---

    float my_val = 0.0f;
    // Load the thread's value from the input matrix if it's within bounds.
    if (col_idx < d) {
        my_val = input[row_idx * d + col_idx];
    }
    
    // Store the square of the value in shared memory for reduction.
    s_data[tid_in_block] = my_val * my_val;

    // Synchronize to make sure all threads have written their squared value to shared memory.
    __syncthreads();

    // Perform the reduction in shared memory.
    // Each thread adds its right-half neighbor's value to its own.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid_in_block < s) {
            s_data[tid_in_block] += s_data[tid_in_block + s];
        }
        __syncthreads(); // Synchronize at each step of the reduction tree.
    }

    // After the loop, the first element of shared memory (s_data[0]) holds the
    // final squared magnitude for the entire row.
    const float squared_magnitude = s_data[0];

    // --- Step 2: Element-wise division ---

    // Ensure the thread is within bounds and the magnitude is non-zero to avoid division by zero.
    if (col_idx < d && squared_magnitude > 1e-9f) {
        output[row_idx * d + col_idx] = my_val / squared_magnitude;
    } else if (col_idx < d) {
        // Handle zero-magnitude vector case (output is all zeros).
        output[row_idx * d + col_idx] = 0.0f;
    }
}

// =================================================================================
// HOST-SIDE WRAPPER FUNCTIONS
// =================================================================================

/**
 * @brief Host wrapper to generate embeddings on the GPU.
 * @param embedding [out] 2D vector to store the results. Will be resized.
 * @param seeds [in] 1D vector of seed values, one for each token.
 * @param d [in] The embedding dimension.
 * @param vocSize [in] The number of tokens/seeds (N).
 */
void tokeniser::cuEmbeddingFormula(std::vector<std::vector<float>>& embedding,
    const std::vector<float>& seeds, int& d, int& vocSize) 
{
    if (vocSize == 0 || d == 0) return;
    if (seeds.size() != vocSize) {
        throw std::runtime_error("Seed vector size must match vocSize.");
    }

    // 1. Resize output vector and create a flat buffer for GPU results
    embedding.assign(vocSize, std::vector<float>(d));
    std::vector<float> h_flat_output(vocSize * d);
    
    // 2. Allocate device memory
    float *d_seeds, *d_embeddings;
    CHECK_CUDA(cudaMalloc(&d_seeds, vocSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_embeddings, (size_t)vocSize * d * sizeof(float)));

    // 3. Copy input seeds to device
    CHECK_CUDA(cudaMemcpy(d_seeds, seeds.data(), vocSize * sizeof(float), cudaMemcpyHostToDevice));

    // 4. Configure and launch kernel
    dim3 block_dim(16, 16);
    dim3 grid_dim((d + block_dim.x - 1) / block_dim.x, (vocSize + block_dim.y - 1) / block_dim.y);
    embeddingFormulaBatchKernel<<<grid_dim, block_dim>>>(d_embeddings, d_seeds, vocSize, d);
    CHECK_CUDA(cudaGetLastError());

    // 5. Copy flat results back to host
    CHECK_CUDA(cudaMemcpy(h_flat_output.data(), d_embeddings, (size_t)vocSize * d * sizeof(float), cudaMemcpyDeviceToHost));
    
    // 6. "Un-flatten" the results into the 2D output vector
    for (int i = 0; i < vocSize; ++i) {
        for (int j = 0; j < d; ++j) {
            embedding[i][j] = h_flat_output[i * d + j];
        }
    }

    // 7. Free device memory
    CHECK_CUDA(cudaFree(d_seeds));
    CHECK_CUDA(cudaFree(d_embeddings));
}

/**
 * @brief Host wrapper to calculate batched vector inverses on the GPU.
 * @param deEmbedding [out] 2D vector to store the results. Will be resized.
 * @param embedding [in] 2D vector of input vectors.
 * @param d [in] The dimension of each vector.
 * @param vocSize [in] The number of vectors.
 */
void cuVectorInverse(std::vector<std::vector<float>>& deEmbedding,
    const std::vector<std::vector<float>>& embedding, int& d, int& vocSize)
{
    if (vocSize == 0 || d == 0) return;
    if (embedding.size() != vocSize || embedding[0].size() != d) {
        throw std::runtime_error("Input embedding dimensions do not match vocSize and d.");
    }

    // 1. Resize output and flatten the 2D input vector for the GPU
    deEmbedding.assign(vocSize, std::vector<float>(d));
    std::vector<float> h_flat_input(vocSize * d);
    std::vector<float> h_flat_output(vocSize * d);
    for (int i = 0; i < vocSize; ++i) {
        for (int j = 0; j < d; ++j) {
            h_flat_input[i * d + j] = embedding[i][j];
        }
    }

    // 2. Allocate device memory
    float *d_input, *d_output;
    size_t total_size = (size_t)vocSize * d * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_input, total_size));
    CHECK_CUDA(cudaMalloc(&d_output, total_size));

    // 3. Copy flattened input data to device
    CHECK_CUDA(cudaMemcpy(d_input, h_flat_input.data(), total_size, cudaMemcpyHostToDevice));

    // 4. Configure and launch kernel
    const int block_size = 256; // Must be power of 2 for this reduction
    dim3 grid_dim((d + block_size - 1) / block_size, vocSize, 1);
    dim3 block_dim(block_size, 1, 1);
    size_t shared_mem_size = block_dim.x * sizeof(float);
    batchedVectorInverseKernel<<<grid_dim, block_dim, shared_mem_size>>>(d_output, d_input, vocSize, d);
    CHECK_CUDA(cudaGetLastError());

    // 5. Copy flat results back to host
    CHECK_CUDA(cudaMemcpy(h_flat_output.data(), d_output, total_size, cudaMemcpyDeviceToHost));

    // 6. "Un-flatten" the results into the 2D output vector
    for (int i = 0; i < vocSize; ++i) {
        for (int j = 0; j < d; ++j) {
            deEmbedding[i][j] = h_flat_output[i * d + j];
        }
    }

    // 7. Free device memory
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
}

#endif