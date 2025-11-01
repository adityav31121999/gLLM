#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <maths.hpp>
#include "include/mlp.hpp"


// Helper macro for CUDA error checking
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        throw std::runtime_error(cudaGetErrorString(err)); \
    } \
} while (0)


/**
 * @brief CUDA implementation of backpropagation with L1 regularization
 * @param in Input size
 * @param layers Number of layers
 * @param learning Learning rate
 */
void mlp::cuBackwithL1(int in, int layers, float learning) {
    float lambda = 0.01f; // Regularization parameter
    
    // Perform standard backpropagation to compute gradients
    cuBackprop(in, layers, learning);
    
    // Device memory pointers
    float *d_weights, *d_deltas, *d_prev_activations;
    
    try {
        // Update weights with L1 regularization for all layers
        for (int l = 0; l < layers; l++) {
            // Determine previous layer activations (input for first layer)
            std::vector<float> prev_activations;
            if (l == 0) {
                prev_activations = input;
            }
            else {
                prev_activations = activations[l-1];
            }
            
            // Access weights for current layer
            // Assuming weights[l] is an 'in x in' matrix for this function's logic.
            const mat& current_weights_mat = weights[l];
            // Ensure mat dimensions are consistent with 'in' if this function is to be robust.
            // For now, we proceed assuming weights[l].row == in and weights[l].col == in.
            size_t weight_matrix_bytes = static_cast<size_t>(in) * in * sizeof(float);

            // Calculate deltas for current layer using the specific formula from gweights
            std::vector<float> layer_deltas(in);
            const mat& current_gweights_mat = gweights[l];
            // Ensure gweights mat dimensions are consistent with 'in'.
            for (int i = 0; i < in; i++) {
                // Accessing gweights[l].mapped_data[row * num_cols + col]
                // Here, row = i, col = 0, num_cols = in (as per function's logic for gweights[l][i][0])
                if (prev_activations[0] == 0.0f) { // Avoid division by zero
                    layer_deltas[i] = 0.0f; // Or handle as an error/special case
                } else {
                    layer_deltas[i] = current_gweights_mat.mapped_data[i * in + 0] / prev_activations[0];
                }
            }
            
            // Allocate and copy data to device
            CUDA_CHECK(cudaMalloc(&d_weights, weight_matrix_bytes));
            CUDA_CHECK(cudaMalloc(&d_deltas, in * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_prev_activations, in * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_weights, current_weights_mat.mapped_data, weight_matrix_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_deltas, layer_deltas.data(), in * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_prev_activations, prev_activations.data(), in * sizeof(float), cudaMemcpyHostToDevice));
            
            // Update weights with L1 regularization
            dim3 blockDim(16, 16);
            dim3 gridDim((in + blockDim.x - 1) / blockDim.x, (in + blockDim.y - 1) / blockDim.y);

            updateWeightsL1Kernel<<<gridDim, blockDim>>>(d_weights, d_deltas, d_prev_activations, learning, lambda, in, in);
            CUDA_CHECK(cudaGetLastError());

            // Copy updated weights back to host
            // The mat object weights[l] will receive the updated flat data.
            CUDA_CHECK(cudaMemcpy(weights[l].mapped_data, d_weights,
                                 weight_matrix_bytes, cudaMemcpyDeviceToHost));


            // Free temporary memory
            CUDA_CHECK(cudaFree(d_weights));
            CUDA_CHECK(cudaFree(d_deltas));
            CUDA_CHECK(cudaFree(d_prev_activations));
        }

        // Compute loss with L1 penalty
        float loss = cucomputeLossWithL1(output, expected, *this, lambda);
        std::cout << "Loss with L1 penalty: " << loss << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "CUDA Exception in L1 regularization: " << e.what() << std::endl;
        
        // Cleanup on error
        cudaFree(d_weights);
        cudaFree(d_deltas);
        cudaFree(d_prev_activations);

        throw;
    }
}
#endif