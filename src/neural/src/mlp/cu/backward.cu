
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
 * @brief CUDA implementation of the backward function
 * @param in Input size
 * @param layers Number of layers
 * @param learning Learning rate
 */
void mlp::cuBackward(int in, int layers, float learning) {
    // gweights is std::vector<mat> and should be initialized by the constructor.
    // The old resize logic is not applicable here. This function updates weights directly.
    // Device memory pointers
    float *d_outputs, *d_expected, *d_deltas, *d_weights, *d_activations, *d_prev_activations;
    std::vector<float*> d_layer_deltas(layers, nullptr);
    
    try {
        // Allocate memory for output layer deltas
        CUDA_CHECK(cudaMalloc(&d_outputs, in * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_expected, in * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_deltas, in * sizeof(float)));
        
        // Copy output and expected values to device
        CUDA_CHECK(cudaMemcpy(d_outputs, output.data(), in * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_expected, expected.data(), in * sizeof(float), cudaMemcpyHostToDevice));
        
        int threadsPerBlock = 256;
        int blocksPerGrid = (in + threadsPerBlock - 1) / threadsPerBlock;

        // Calculate output layer deltas (output - expected)
        kernelOutputDelta<<<blocksPerGrid, threadsPerBlock>>>(d_outputs, d_expected, d_deltas, in);
        CUDA_CHECK(cudaGetLastError());

        // Allocate memory for all layer deltas 
        std::vector<std::vector<float>> layer_deltas(layers, std::vector<float>(in, 0.0));

        // Copy output layer deltas back to host
        CUDA_CHECK(cudaMemcpy(layer_deltas[layers-1].data(), d_deltas, in * sizeof(float), cudaMemcpyDeviceToHost));

        // Allocate device memory for each layer's deltas
        for (int l = 0; l < layers; l++) {
            CUDA_CHECK(cudaMalloc(&d_layer_deltas[l], in * sizeof(float)));
        }

        // Copy output layer deltas to device
        CUDA_CHECK(cudaMemcpy(d_layer_deltas[layers-1], layer_deltas[layers-1].data(), 
                             in * sizeof(float), cudaMemcpyHostToDevice));

        // Backpropagate through hidden layers
        for (int l = layers - 2; l >= 0; l--) {
            // Assuming weights[l+1] is an 'in x in' matrix for this function's logic.
            // The mat object weights[l+1] contains the flat data.
            const mat& current_weight_matrix = weights[l+1];
            // Ensure mat dimensions are consistent with 'in' if this function is to be robust.
            // For now, we proceed assuming weights[l+1].row == in and weights[l+1].col == in.
            size_t weight_matrix_bytes = static_cast<size_t>(in) * in * sizeof(float);

            // Allocate and copy data to device
            CUDA_CHECK(cudaMalloc(&d_weights, weight_matrix_bytes));
            CUDA_CHECK(cudaMalloc(&d_activations, in * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_weights, current_weight_matrix.mapped_data, weight_matrix_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_activations, activations[l].data(), in * sizeof(float), cudaMemcpyHostToDevice));
            
            // Calculate hidden layer deltas
            hiddenDeltaKernel<<<blocksPerGrid, threadsPerBlock>>>(d_layer_deltas[l+1], d_weights, 
                                                               d_activations, d_layer_deltas[l], in, in);
            CUDA_CHECK(cudaGetLastError());
            
            // Copy deltas back to host
            CUDA_CHECK(cudaMemcpy(layer_deltas[l].data(), d_layer_deltas[l], 
                                 in * sizeof(float), cudaMemcpyDeviceToHost));
            
            // Free temporary memory
            CUDA_CHECK(cudaFree(d_weights));
            CUDA_CHECK(cudaFree(d_activations));
        }
        
        // Update weights for all layers
        for (int l = 0; l < layers; l++) {
            // Determine previous layer activations (input for first layer)
            std::vector<float> prev_activations;
            if (l == 0) {
                // for first layer
                prev_activations = input;
            }
            else {
                // for other layers
                prev_activations = activations[l-1];
            }
            
            // Assuming weights[l] is an 'in x in' matrix for this function's logic.
            const mat& weights_to_update_mat = weights[l];
            // Ensure mat dimensions are consistent with 'in'.
            size_t current_weight_matrix_bytes = static_cast<size_t>(in) * in * sizeof(float);

            // Allocate and copy data to device
            CUDA_CHECK(cudaMalloc(&d_weights, current_weight_matrix_bytes));
            CUDA_CHECK(cudaMalloc(&d_prev_activations, in * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_weights, weights_to_update_mat.mapped_data,
                                 current_weight_matrix_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_prev_activations, prev_activations.data(), in * sizeof(float), cudaMemcpyHostToDevice));

            // Update weights
            dim3 blockDim(16, 16);
            dim3 gridDim((in + blockDim.x - 1) / blockDim.x, (in + blockDim.y - 1) / blockDim.y);
            updateWeightsKernel<<<gridDim, blockDim>>>(d_layer_deltas[l], d_prev_activations, 
                                                     d_weights, learning, in, in);
            CUDA_CHECK(cudaGetLastError());

            // Copy updated weights back to host
            // The mat object weights[l] will receive the updated flat data.
            CUDA_CHECK(cudaMemcpy(weights[l].mapped_data, d_weights, current_weight_matrix_bytes, cudaMemcpyDeviceToHost));

            // Free temporary memory
            CUDA_CHECK(cudaFree(d_weights));
            CUDA_CHECK(cudaFree(d_prev_activations));
        }
        
        // Free all allocated memory
        CUDA_CHECK(cudaFree(d_outputs));
        CUDA_CHECK(cudaFree(d_expected));
        CUDA_CHECK(cudaFree(d_deltas));
        
        for (int l = 0; l < layers; l++) {
            CUDA_CHECK(cudaFree(d_layer_deltas[l]));
        }
    }
    catch (const std::exception& e) {
        std::cerr << "CUDA Exception in backpropagation: " << e.what() << std::endl;
        
        // Cleanup on error
        cudaFree(d_outputs);
        cudaFree(d_expected);
        cudaFree(d_deltas);
        
        for (int l = 0; l < layers; l++) {
            cudaFree(d_layer_deltas[l]);
        }
        
        throw;
    }
}
