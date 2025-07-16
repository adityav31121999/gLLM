#ifdef USE_CUDA

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <iostream>
#include <stdexcept>
#include "include/tokenise.hpp"

// Helper for CUDA Error Checking
#define CHECK_CUDA(call) do {   \
    cudaError_t err = call;     \
    if (err != cudaSuccess) {   \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__);  \
        exit(EXIT_FAILURE);     \
    }                           \
} while (0)

// Basic XorShift32 PRNG for CUDA
// Each thread uses its unique thread index to get a distinct seed
__device__ unsigned int xorshift32_cuda(unsigned int x) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

// Function to convert a normalized [0,1] float to a custom range [r1, r2]
__device__ float scale_random_cuda(unsigned int* seed_ptr, float r1, float r2) {
    *seed_ptr = xorshift32_cuda(*seed_ptr);
    float normalized_val = (float)(*seed_ptr) / (float)UINT_MAX;
    return r1 + normalized_val * (r2 - r1);
}

// CUDA Kernel to generate embeddings
__global__ void generate_embeddings_kernel(
    float* embeddings_out,
    int d_dim,
    float r1,
    float r2,
    unsigned int initial_seed_offset,
    int total_elements) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Global thread ID

    if (idx < total_elements) {
        // Use global_id + an offset for a unique seed per thread
        unsigned int seed = initial_seed_offset + idx + 1; // +1 to avoid seed 0

        // Generate random float in [r1, r2]
        embeddings_out[idx] = scale_random_cuda(&seed, r1, r2);
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
__global__ void batchedVectorInverseKernel(float* output, const float* input, int N, int d) 
{
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


// Host-side wrapper function
void tokeniser::cuEmbeddingFormula(std::vector<std::vector<float>>& embedding, const std::vector<float>& seeds_ignored, int& d_dim, int& vocSize_val) {
    // Resize embedding vector to hold the results
    embedding.resize(vocSize_val, std::vector<float>(d_dim));

    size_t total_elements = (size_t)vocSize_val * d_dim;
    if (total_elements == 0) return;

    float* d_embeddings = nullptr; // Device pointer for embeddings

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_embeddings, total_elements * sizeof(float)));

    // Configure kernel launch parameters
    int threads_per_block = 256;
    int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;

    // Use a time-based seed offset for better randomness across runs
    unsigned int initial_seed_offset = static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count());

    // Launch kernel
    generate_embeddings_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_embeddings,
        d_dim,
        r1,
        r2,
        initial_seed_offset,
        total_elements
    );
    CUDA_CHECK(cudaGetLastError()); // Check for errors during kernel launch

    // Copy results back to host
    std::vector<float> flat_embeddings(total_elements);
    CUDA_CHECK(cudaMemcpy(flat_embeddings.data(), d_embeddings, total_elements * sizeof(float), cudaMemcpyDeviceToHost));

    // Copy flat_embeddings to the 2D embedding vector
    for (int i = 0; i < vocSize_val; ++i) {
        for (int j = 0; j < d_dim; ++j) {
            embedding[i][j] = flat_embeddings[i * d_dim + j];
        }
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_embeddings));
}


/**
 * @brief Host wrapper to calculate batched vector inverses on the GPU.
 * @param deEmbedding [out] 2D vector to store the results. Will be resized.
 * @param embedding [in] 2D vector of input vectors.
 * @param d [in] The dimension of each vector.
 * @param vocSize [in] The number of vectors.
 */
void tokeniser::cuVectorInverse(std::vector<std::vector<float>>& deEmbedding,
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