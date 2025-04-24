
#include "include/transformer.hpp"
#include "include/block.hpp"
#include "include/attention.hpp"
#include "include/mlp.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <stdexcept>
#include <iostream>

// Helper macro for CUDA error checking (assuming it's defined elsewhere or define it here)
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        throw std::runtime_error(cudaGetErrorString(err)); \
    } \
} while (0)


// --- Kernel Implementation for EH Accumulation ---
// This kernel assumes d_eh_pointers is a device array where each element
// points to a valid device memory location containing an EH vector of size embedding_dim.
__global__ void accumulateEH(float** d_eh_pointers, float* d_otok, int num_layers, int embedding_dim) {
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
 * @brief CUDA forward propagation for transformers
 * @param blockCount current block index (0-based)
 * @param currentTokenCount current total number of tokens processed
 * @param promptCount number of tokens in the current prompt segment being processed
 */
void transformer::cuForward(int &blockCount, int &currentTokenCount, int &promptCount)
{
    // Device pointers
    float* d_otok = nullptr;
    float** d_eh_pointers_host = nullptr; // Host array to hold device pointers
    float** d_eh_pointers_device = nullptr; // Device array to hold device pointers

    try {
        // Step 1: Compute KdotQ matrices for the current block in parallel
        for (int col = 0; col < y; ++col) {
             cuParallelKdotQs(promptCount, currentTokenCount, blockCount, col, isSelf, inTraining);
        }


        // Allocate device memory for the output token embedding
        CUDA_CHECK(cudaMalloc(&d_otok, d * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_otok, 0, d * sizeof(float))); // Initialize output token to zero

        // --- Prepare pointers for EH accumulation ---
        // Allocate host memory for an array of device pointers
        d_eh_pointers_host = new float*[x]; // x = number of layers

        // Step 2 & 3: Perform forward propagation for the relevant block
        if (blockCount == 0) {
            // --- Block 0 Forward Propagation ---
            // Assumes t[0].cuForprop updates internal states (like EH) on the device
            t[0].cuForprop(d, currentTokenCount, l); // Pass embedding dim, token count, mlp layers
            CUDA_CHECK(cudaDeviceSynchronize()); // Ensure block 0 forward prop is complete

            // --- Accumulate EH from the last column of block 0 ---
            // Populate the host array with device pointers to EH vectors from the last column (y-1)
            for (int j = 0; j < x; ++j) {
                 if (y > 0) {
                    //
                 } else {
                     // Handle case with no columns if necessary
                     d_eh_pointers_host[j] = nullptr;
                 }
            }
            CUDA_CHECK(cudaMalloc(&d_eh_pointers_device, x * sizeof(float*)));
            CUDA_CHECK(cudaMemcpy(d_eh_pointers_device, d_eh_pointers_host, x * sizeof(float*), cudaMemcpyHostToDevice));

            int threadsPerBlock = 256;
            int blocksPerGrid = (d + threadsPerBlock - 1) / threadsPerBlock;
            accumulateEH<<<blocksPerGrid, threadsPerBlock>>>(d_eh_pointers_device, d_otok, x, d);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        else { // blockCount > 0
            // --- Block N Forward Propagation ---
            if (blockCount > 0) { // Ensure blockCount-2 is valid
                t[blockCount - 1].cuForprop(t[blockCount - 2].EV, d, currentTokenCount, blockCount, l, n);
                CUDA_CHECK(cudaDeviceSynchronize());
                for (int j = 0; j < x; ++j) {
                    if (y > 0) {
                        // d_eh_pointers_host[j] = t[blockCount-1].b[j][y-1].getDeviceEHPointer(); // Hypothetical
                    } 
                    else {
                        d_eh_pointers_host[j] = nullptr;
                    }
                }
                CUDA_CHECK(cudaMalloc(&d_eh_pointers_device, x * sizeof(float*)));
                CUDA_CHECK(cudaMemcpy(d_eh_pointers_device, d_eh_pointers_host, x * sizeof(float*), cudaMemcpyHostToDevice));

                int threadsPerBlock = 256;
                int blocksPerGrid = (d + threadsPerBlock - 1) / threadsPerBlock;
                accumulateEH<<<blocksPerGrid, threadsPerBlock>>>(d_eh_pointers_device, d_otok, x, d);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
            } 
            else {
                 // Should not happen if blockCount > 0, but handle defensively
                 std::cerr << "Warning: blockCount is " << blockCount << " in else branch of cuForward." << std::endl;
            }
        }

        // Step 4: Compute Output Token Index
        // Assumes 'embeddings' member is already transferred to device pointer 'd_embeddings'
        // Need to manage d_embeddings allocation and transfer elsewhere (e.g., constructor or setup method)
        float* d_embeddings = nullptr;
        // Allocate device memory for the embeddings
        CUDA_CHECK(cudaMalloc(&d_embeddings, vocabsize * d * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_embeddings, embeddings.data()->data(), vocabsize * d * sizeof(float), cudaMemcpyHostToDevice));
        if (d_embeddings != nullptr) {
            cuComputeOutput(d_otok, d_embeddings, vocabsize, indexForToken, d);
        }
        else {
            throw std::runtime_error("Device embeddings pointer (d_embeddings) is null in cuForward.");
        }

        // TODO: Add logic to potentially update tokenEmbed[currentTokenCount] on host/device
        // if needed for subsequent steps or training loops, based on indexForToken.
        // Example: Get embedding vector for indexForToken and store it.
        // std::vector<float> next_token_vec = embeddings[indexForToken];
        // Update tokenEmbed container (potentially requires H->D copy if tokenEmbed is on device)
        CUDA_CHECK(cudaFree(d_embeddings));

        // Cleanup
        CUDA_CHECK(cudaFree(d_otok));
        CUDA_CHECK(cudaFree(d_eh_pointers_device));
        delete[] d_eh_pointers_host;

    } 
    catch (const std::exception& e) {
        std::cerr << "Error in transformer::cuForward: " << e.what() << std::endl;
        cudaFree(d_otok);
        cudaFree(d_eh_pointers_device);
        delete[] d_eh_pointers_host;
        throw;
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
__global__ void computeAllDotsKernel(const float* vector, const float* matrix, float* results, int num_rows, int vector_dim) {
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


/**
 * @brief compute output as index for new token string
 * @param d_output A pointer to a device memory location holding the output vector, which represents the
 * processed token embedding.
 * @param d_embeddings A pointer to a device memory location holding the embeddings matrix, where each 
 * row represents a token's embedding vector.
 * @param voc_size The vocabulary size, which is the number of rows in the `d_embeddings` matrix and the 
 * number of tokens in the vocabulary.
 * @param index A reference to an integer variable where the resulting index (the index of the most likely 
 * next token) will be stored.
 * @param embedding_dim The dimension of the embedding vectors (the number of columns in the `d_embeddings`
 * matrix).
 */
void cuComputeOutput(float* d_output, float* d_embeddings, int voc_size, int& index, int embedding_dim)
{
    if (voc_size <= 0 || embedding_dim <= 0 || d_output == nullptr || d_embeddings == nullptr) {
        index = -1;
        return;
    }
    float* d_dot_products = nullptr;
    try {
        // 1. Allocate device memory for the dot product results
        CUDA_CHECK(cudaMalloc(&d_dot_products, voc_size * sizeof(float)));

        // 2. Launch the kernel to compute all dot products in parallel
        int threadsPerBlock = 256; // Or 512, tune as needed
        int blocksPerGrid = (voc_size + threadsPerBlock - 1) / threadsPerBlock;
        computeAllDotsKernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_output,
            d_embeddings,
            d_dot_products,
            voc_size,
            embedding_dim
        );
        CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors
        CUDA_CHECK(cudaDeviceSynchronize()); // Wait for kernel completion

        // 3. Copy the dot products back to the host
        std::vector<float> h_dot_products(voc_size);
        CUDA_CHECK(cudaMemcpy(h_dot_products.data(), d_dot_products, voc_size * sizeof(float), cudaMemcpyDeviceToHost));

        // 4. Find the index of the maximum dot product on the host
        float max_dot = -std::numeric_limits<float>::infinity(); // Use -infinity for correct comparison
        int max_idx = 0; // Default to index 0 if voc_size > 0

        for (int i = 0; i < voc_size; ++i) {
            if (h_dot_products[i] > max_dot) {
                max_dot = h_dot_products[i];
                max_idx = i;
            }
        }
        index = max_idx;

        // 5. Cleanup device memory
        CUDA_CHECK(cudaFree(d_dot_products));
    }
    catch (const std::exception& e) {
        // Cleanup on error
        cudaFree(d_dot_products); // Safe to call even if null
        std::cerr << "Error during cuComputeOutput: " << e.what() << std::endl;
        // Re-throw or handle error appropriately
        throw;
    }
}
