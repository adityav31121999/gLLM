
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
    // Ensure network vectors are properly initialized
    if (gweights.size() != layers) {
        gweights.resize(layers, std::vector<std::vector<float>>(in, std::vector<float>(in, 0.0)));
    }
    
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
            // Flatten weights for the next layer
            std::vector<float> flat_weights;
            for (int i = 0; i < in; i++) {
                flat_weights.insert(flat_weights.end(), weights[l+1][i].begin(), weights[l+1][i].end());
            }

            // Allocate and copy data to device
            CUDA_CHECK(cudaMalloc(&d_weights, in * in * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_activations, in * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_weights, flat_weights.data(), in * in * sizeof(float), cudaMemcpyHostToDevice));
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
            
            // Flatten weights for current layer
            std::vector<float> flat_weights;
            for (int i = 0; i < in; i++) {
                flat_weights.insert(flat_weights.end(), 
                                   weights[l][i].begin(), weights[l][i].end());
            }

            // Allocate and copy data to device
            CUDA_CHECK(cudaMalloc(&d_weights, in * in * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_prev_activations, in * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_weights, flat_weights.data(), 
                                 in * in * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_prev_activations, prev_activations.data(), 
                                 in * sizeof(float), cudaMemcpyHostToDevice));

            // Update weights
            dim3 blockDim(16, 16);
            dim3 gridDim((in + blockDim.x - 1) / blockDim.x, (in + blockDim.y - 1) / blockDim.y);
            updateWeightsKernel<<<gridDim, blockDim>>>(d_layer_deltas[l], d_prev_activations, 
                                                     d_weights, learning, in, in);
            CUDA_CHECK(cudaGetLastError());

            // Copy updated weights back to host
            std::vector<float> updated_flat_weights(in * in);
            CUDA_CHECK(cudaMemcpy(updated_flat_weights.data(), d_weights, in * in * sizeof(float), cudaMemcpyDeviceToHost));
            
            // Reshape flat weights back to 2D
            for (int i = 0; i < in; i++) {
                for (int j = 0; j < in; j++) {
                    weights[l][i][j] = updated_flat_weights[i * in + j];
                }
            }
            
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
