
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
 * @brief CUDA implementation of backpropagation with L2 regularization
 * @param in Input size
 * @param layers Number of layers
 * @param learning Learning rate
 */
void mlp::cuBackwithL2(int in, int layers, float learning) {
    float lambda = 0.01f; // Regularization parameter
    
    // Perform standard backpropagation to compute gradients
    cuBackprop(in, layers, learning);
    
    // Device memory pointers
    float *d_weights, *d_deltas, *d_prev_activations;
    
    try {
        // Update weights with L2 regularization for all layers
        for (int l = 0; l < layers; l++) {
            // Determine previous layer activations (input for first layer)
            std::vector<float> prev_activations;
            if (l == 0) {
                prev_activations = input;
            }
            else {
                prev_activations = activations[l-1];
            }
            
            // Flatten weights and deltas for current layer
            std::vector<float> flat_weights;
            std::vector<float> layer_deltas(in);
            
            for (int i = 0; i < in; i++) {
                flat_weights.insert(flat_weights.end(), 
                                   weights[l][i].begin(), 
                                   weights[l][i].end());
                layer_deltas[i] = gweights[l][i][0] / prev_activations[0]; // Extract delta from gradient
            }
            
            // Allocate and copy data to device
            CUDA_CHECK(cudaMalloc(&d_weights, in * in * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_deltas, in * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_prev_activations, in * sizeof(float)));
            
            CUDA_CHECK(cudaMemcpy(d_weights, flat_weights.data(), 
                                 in * in * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_deltas, layer_deltas.data(), 
                                 in * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_prev_activations, prev_activations.data(), 
                                 in * sizeof(float), cudaMemcpyHostToDevice));
            
            // Update weights with L2 regularization
            dim3 blockDim(16, 16);
            dim3 gridDim((in + blockDim.x - 1) / blockDim.x, (in + blockDim.y - 1) / blockDim.y);
            
            updateWeightsL2Kernel<<<gridDim, blockDim>>>(d_weights, d_deltas, d_prev_activations, 
                                                      learning, lambda, in, in);
            CUDA_CHECK(cudaGetLastError());
            
            // Copy updated weights back to host
            std::vector<float> updated_flat_weights(in * in);
            CUDA_CHECK(cudaMemcpy(updated_flat_weights.data(), d_weights, 
                                 in * in * sizeof(float), cudaMemcpyDeviceToHost));
            
            // Reshape flat weights back to 2D
            for (int i = 0; i < in; i++) {
                for (int j = 0; j < in; j++) {
                    weights[l][i][j] = updated_flat_weights[i * in + j];
                }
            }
            
            // Free temporary memory
            CUDA_CHECK(cudaFree(d_weights));
            CUDA_CHECK(cudaFree(d_deltas));
            CUDA_CHECK(cudaFree(d_prev_activations));
        }
        
        // Compute loss with L2 penalty
        float loss = cucomputeLossWithL2(output, expected, *this, lambda);
        std::cout << "Loss with L2 penalty: " << loss << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "CUDA Exception in L2 regularization: " << e.what() << std::endl;
        
        // Cleanup on error
        cudaFree(d_weights);
        cudaFree(d_deltas);
        cudaFree(d_prev_activations);
        
        throw;
    }
}

