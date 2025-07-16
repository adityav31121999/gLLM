#ifdef USE_CUDA
#include "include/transformer.hpp"
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


/**
 * @brief CUDA forward propagation for transformers
 * @param blockCount current block index (1-based)
 * @param currentTokenCount current total number of tokens processed
 * @param promptCount number of tokens in the current prompt segment being processed
 */
void transformer::cuForward(int& blockCount, int& currentTokenCount, int& promptCount)
{
    // Device pointers
    float* d_otok = nullptr;
    float** d_eh_pointers_host = nullptr; // Host array to hold device pointers
    std::vector<float*> temp_device_eh_buffers; // To keep track of temporary EH buffers for cleanup
    float** d_eh_pointers_device = nullptr; // Device array to hold device pointers

    try {
        // Step 1: Compute KdotQ matrices for the current block in parallel
        for (int col = 0; col < y; col++) {
            cuParallelKdotQs(promptCount, currentTokenCount, blockCount, col, isSelf, inTraining);
        }

        // Allocate device memory for the output token embedding
        CUDA_CHECK(cudaMalloc(&d_otok, d * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_otok, 0, d * sizeof(float))); // Initialize output token to zero

        // --- Prepare pointers for EH accumulation ---
        // Allocate host memory for an array of device pointers
        d_eh_pointers_host = new float*[x]; // x = number of layers

        // Step 2 & 3: Perform forward propagation for the relevant block
        if (blockCount == 1) {
            // --- Block 0 Forward Propagation ---
            std::cout << "-> cuForward: Executing cuForprop for Block 1..." << std::endl;
            // Assumes t[0].cuForprop updates internal states (like EH) on the device
            t[0].cuForprop(d, currentTokenCount, l); // Pass embedding dim, token count, mlp layers
            CUDA_CHECK(cudaDeviceSynchronize()); // Ensure block 0 forward prop is complete

            // --- Accumulate EH from the last column of block 0 ---
            // Populate the host array with device pointers to EH vectors from the last column (y-1)
            for (int j = 0; j < x; ++j) {
                if (y > 0) {
                    attention& head_obj = t[0].b[j][y-1];
                    float* d_temp_eh_for_accumulation = nullptr;
                    CUDA_CHECK(cudaMalloc(&d_temp_eh_for_accumulation, d * sizeof(float)));
                    temp_device_eh_buffers.push_back(d_temp_eh_for_accumulation); // Store for later cleanup
                    CUDA_CHECK(cudaMemcpy(d_temp_eh_for_accumulation, head_obj.EH.data(), d * sizeof(float), cudaMemcpyHostToDevice));
                    d_eh_pointers_host[j] = d_temp_eh_for_accumulation;
                }
                else {
                    // Handle case with no columns if necessary
                    d_eh_pointers_host[j] = nullptr;
                }
            }
            if (x == 0 && y == 0) { // No heads to accumulate from
                // d_otok remains zero, which is fine.
                std::cout << "NOTHING TO WORK WITH" << std::endl;
            }

            CUDA_CHECK(cudaMalloc(&d_eh_pointers_device, x * sizeof(float*)));
            CUDA_CHECK(cudaMemcpy(d_eh_pointers_device, d_eh_pointers_host, x * sizeof(float*), cudaMemcpyHostToDevice));

            int threadsPerBlock = 256;
            int blocksPerGrid = (d + threadsPerBlock - 1) / threadsPerBlock;
            accumulateEH<<<blocksPerGrid, threadsPerBlock>>>(d_eh_pointers_device, d_otok, x, d);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            // std::cout << "cuForward: Device EH accumulation for Block 1 finished." << std::endl;
        }
        else { // blockCount > 1
            // --- Block ith Forward Propagation ---
            if (blockCount > 1) {
                // std::cout << "-> clForward: Executing clForprop for Block " << blockCount << "..." << std::endl;
                // Assumes t[blockCount].cuForprop updates internal states (like EH) 1..." << std::endl;
                t[blockCount].cuForprop(t[blockCount - 1].EV, d, currentTokenCount, blockCount, l, n);
                CUDA_CHECK(cudaDeviceSynchronize());
                std::cout << "-> cuForward: Executing cuForprop for Block " << blockCount << " using EV from Block " << blockCount - 1 << "..." << std::endl;
                // Accumulate EH from the last column of the CURRENT block (blockCount)
                for (int j = 0; j < x; ++j) {
                    if (y > 0) {
                        attention& head_obj = t[blockCount].b[j][y-1];
                        float* d_temp_eh_for_accumulation = nullptr;
                        CUDA_CHECK(cudaMalloc(&d_temp_eh_for_accumulation, d * sizeof(float)));
                        temp_device_eh_buffers.push_back(d_temp_eh_for_accumulation);
                        CUDA_CHECK(cudaMemcpy(d_temp_eh_for_accumulation, head_obj.EH.data(), d * sizeof(float), cudaMemcpyHostToDevice));
                        d_eh_pointers_host[j] = d_temp_eh_for_accumulation;
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
                // std::cout << "cuForward: Device EH accumulation for Block " << blockCount << " finished." << std::endl;
            } 
            else {
                // Should not happen if blockCount > 0, but handle defensively
                std::cerr << "Warning: blockCount is " << blockCount << " in else branch of cuForward." << std::endl;
            }
        }

        // Step 4: Compute Output Token Index
        // Assumes 'embeddings' member is already transferred to device pointer 'd_embeddings'
        // Need to manage d_embeddings allocation and transfer elsewhere (e.g., constructor or setup method)
        if (!embeddings.mapped_data || embeddings.row != vocabsize || embeddings.col != d) {
            throw std::runtime_error("Transformer embeddings mat is not properly initialized or dimensions mismatch.");
        }

        float* d_embeddings = nullptr;
        // std::cout << "cuForward: Computing output token index using cuComputeOutput..." << std::endl;
        // Allocate device memory for the embeddings
        size_t embeddings_bytes = static_cast<size_t>(embeddings.row) * embeddings.col * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_embeddings, embeddings_bytes));
        CUDA_CHECK(cudaMemcpy(d_embeddings, embeddings.mapped_data, embeddings_bytes, cudaMemcpyHostToDevice));

        if (d_embeddings != nullptr) {
            cuComputeOutput(d_otok, d_embeddings, vocabsize, indexForToken, d); // indexForToken is updated here

            // Check token validity and print
            if (indexForToken >= 0 && static_cast<size_t>(indexForToken) < tokens.size() && indexForToken < vocabsize) {
                std::cout << "cuForward: cuComputeOutput finished. Predicted index: " << indexForToken << " | Token is: " << tokens[indexForToken] << "" << std::endl;
            } 
            else {
                std::cout << "cuForward: cuComputeOutput finished. Predicted index: " << indexForToken << std::endl;
            }
            // Separate warning for vocabsize, similar to clForward
            if (indexForToken < 0 || indexForToken >= vocabsize) {
                std::cerr << "Warning: cuForward resulted in invalid indexForToken: " << indexForToken << " (vocabsize: " << vocabsize << ")" << std::endl;
            }
        }
        else {
            // This case should ideally be caught by earlier checks or d_embeddings would not be null
            throw std::runtime_error("Device embeddings pointer (d_embeddings) is null in cuForward before calling cuComputeOutput.");
        }

        CUDA_CHECK(cudaMemcpy(this->otok.data(), d_otok, static_cast<size_t>(d) * sizeof(float), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_embeddings));
        CUDA_CHECK(cudaFree(d_otok));
        CUDA_CHECK(cudaFree(d_eh_pointers_device));

        for (float* buffer : temp_device_eh_buffers) {
            CUDA_CHECK(cudaFree(buffer));
        }
        temp_device_eh_buffers.clear();
        delete[] d_eh_pointers_host;
        // std::cout << "cuForward: Finished forward propagation for block " << blockCount << "." << std::endl;
    } 
    catch (const std::exception& e) {
        std::cerr << "Error in transformer::cuForward: " << e.what() << std::endl;
        if(d_otok) cudaFree(d_otok);
        if(d_eh_pointers_device) cudaFree(d_eh_pointers_device);
        for (float* buffer : temp_device_eh_buffers) {
            if(buffer) cudaFree(buffer);
        }
        delete[] d_eh_pointers_host;
        throw;
    }
}


/**
 * @brief compute output as index for new token string
 * @param d_output A pointer to a device memory location holding the output vector, which represents the processed token embedding.
 * @param d_embeddings A pointer to a device memory location holding the embeddings matrix, where each row represents a token's embedding vector.
 * @param voc_size The vocabulary size, which is the number of rows in the `d_embeddings` matrix and the number of tokens in the vocabulary.
 * @param index A reference to an integer variable where the resulting index (the index of the most likely next token) will be stored.
 * @param embedding_dim The dimension of the embedding vectors (the number of columns in the `d_embeddings` matrix).
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

#endif