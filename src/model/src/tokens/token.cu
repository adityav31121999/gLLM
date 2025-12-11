#ifdef USE_CU
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <chrono>
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
 * embedding formula kernel
 * --> f(i, j, seed) = (i * j + 1) * C * (seed^[j%d])
 * where: C = 0.01, x = seed, and  d is the embedding dimension.
 */
__global__ void embeddingFormulaBatchKernel(float* embeddings_out, const int d_dim,
    const float r1, const float r2, const unsigned int initial_seed_offset) 
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int seed = initial_seed_offset + global_id + 1;

    float normalized_val = (float)(seed ^ (seed << 13) ^ (seed >> 17) ^ (seed << 5)) / (float)0xFFFFFFFFU;
    embeddings_out[global_id] = r1 + normalized_val * (r2 - r1);
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
void tokeniser::cuEmbeddingFormula(std::vector<std::vector<float>>& embedding, const std::vector<float>& seeds_ignored, int& d, int& vocSize, float r1, float r2) {
    if (vocSize == 0 || d == 0) return;

    // Resize embedding vector to hold the results
    embedding.assign(vocSize, std::vector<float>(d));

    size_t total_elements = (size_t)vocSize * d;
    if (total_elements == 0) return;

    std::vector<float> flat_embeddings(total_elements);

    // 2. Allocate device memory
    float* d_embeddings;
    CHECK_CUDA(cudaMalloc(&d_embeddings, sizeof(float) * total_elements));

    // Initialize kernel
    dim3 block_dim(256);
    dim3 grid_dim((total_elements + block_dim.x - 1) / block_dim.x);

    // Determine initial seed offset based on time
    unsigned int initial_seed_offset = static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count());

    // Launch kernel
    embeddingFormulaBatchKernel<<<grid_dim, block_dim>>>(d_embeddings, d, r1, r2, initial_seed_offset);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: Kernel launch failed with error %s\n", cudaGetErrorString(err));
        cudaFree(d_embeddings);
        return;
    }

    // Read results back
    CHECK_CUDA(cudaMemcpy(flat_embeddings.data(), d_embeddings, sizeof(float) * total_elements, cudaMemcpyDeviceToHost));

    for (int i = 0; i < vocSize; ++i) {
        for (int j = 0; j < d; ++j) {
            embedding[i][j] = flat_embeddings[i * d + j];
        }
    }
    CHECK_CUDA(cudaFree(d_embeddings));
}

/**
 * @brief Host wrapper to generate embeddings on the GPU.
 * @param embedding [out] 2D vector to store the results. Will be resized.
 * @param seeds [in] 1D vector of seed values, one for each token.
 * @param d [in] The embedding dimension.
 * @param vocSize [in] The number of tokens/seeds (N).
 */
void tokeniser::cudeEmbeddingFormula(std::vector<std::vector<float>>& embedding, const std::vector<float>& seeds_ignored, int& d, int& vocSize, float r1, float r2) {
    if (vocSize == 0 || d == 0) return;

    // Resize embedding vector to hold the results
    embedding.assign(vocSize, std::vector<float>(d));

    size_t total_elements = (size_t)vocSize * d;
    if (total_elements == 0) return;

    std::vector<float> flat_embeddings(total_elements);

    // 2. Allocate device memory
    float* d_embeddings;
    CHECK_CUDA(cudaMalloc(&d_embeddings, sizeof(float) * total_elements));

    // Initialize kernel
    dim3 block_dim(256);
    dim3 grid_dim((total_elements + block_dim.x - 1) / block_dim.x);

    // Determine initial seed offset based on time
    unsigned int initial_seed_offset = static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count());

    // Launch kernel
    embeddingFormulaBatchKernel<<<grid_dim, block_dim>>>(d_embeddings, d, r1, r2, initial_seed_offset);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: Kernel launch failed with error %s\n", cudaGetErrorString(err));
        cudaFree(d_embeddings);
        return;
    }

    // Read results back
    CHECK_CUDA(cudaMemcpy(flat_embeddings.data(), d_embeddings, sizeof(float) * total_elements, cudaMemcpyDeviceToHost));

    for (int i = 0; i < vocSize; ++i) {
        for (int j = 0; j < d; ++j) {
            embedding[i][j] = flat_embeddings[i * d + j];
        }
    }
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