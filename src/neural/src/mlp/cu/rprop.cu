
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
 * @brief CUDA implementation of Rprop algorithm for MLP
 * @param dataset Input dataset
 * @param layers Number of layers
 * @param in Input size
 * @param learning Learning rate (not used in Rprop)
 * @param epochs Number of epochs
 */
void mlp::cuRprop(std::vector<std::vector<float>>& dataset, int layers, int in, float learning, int epochs) {
    const float etaPlus = 1.2f;     // Increase factor
    const float etaMinus = 0.5f;    // Decrease factor
    const float deltaMax = 50.0f;   // Maximum update value
    const float deltaMin = 1e-6f;   // Minimum update value
    
    // Initialize gradient and delta weight matrices
    std::vector<std::vector<std::vector<float>>> prev_gradients(layers, 
                                                             std::vector<std::vector<float>>(in, 
                                                                                           std::vector<float>(in, 0.0f)));
    std::vector<std::vector<std::vector<float>>> delta_weights(layers,
                                                           std::vector<std::vector<float>>(in, 
                                                                                         std::vector<float>(in, deltaMin)));
    
    // Device memory pointers, remove unused variables
    float *d_weights, *d_gradients, *d_prev_gradients, *d_delta_weights;
    
    for (unsigned int epoch = 0; epoch < epochs; ++epoch) {
        float totalError = 0.0f;
        
        for (const auto& data : dataset) {
            // Set input and perform forward and backward passes
            input = data;
            cuForward(in, layers);
            cuBackprop(in, layers, learning); // This computes gradients and stores them in gweights
            
            // Compute mean square error
            float error = 0.0f;
            for (unsigned int i = 0; i < in; ++i) {
                error += std::pow(expected[i] - output[i], 2);
            }
            error /= in;
            totalError += error;
            
            try {
                // Update weights using Rprop for each layer
                for (unsigned int l = 0; l < layers; ++l) {
                    // Flatten weights, gradients, previous gradients, and delta weights
                    std::vector<float> flat_weights;
                    std::vector<float> flat_gradients;
                    std::vector<float> flat_prev_gradients;
                    std::vector<float> flat_delta_weights;
                    
                    for (unsigned int i = 0; i < in; ++i) {
                        for (unsigned int j = 0; j < in; ++j) {
                            flat_weights.push_back(weights[l][i][j]);
                            flat_gradients.push_back(gweights[l][i][j]);
                            flat_prev_gradients.push_back(prev_gradients[l][i][j]);
                            flat_delta_weights.push_back(delta_weights[l][i][j]);
                        }
                    }
                    
                    // Allocate device memory
                    int size = in * in;
                    CUDA_CHECK(cudaMalloc(&d_weights, size * sizeof(float)));
                    CUDA_CHECK(cudaMalloc(&d_gradients, size * sizeof(float)));
                    CUDA_CHECK(cudaMalloc(&d_prev_gradients, size * sizeof(float)));
                    CUDA_CHECK(cudaMalloc(&d_delta_weights, size * sizeof(float)));
                    
                    // Copy data to device
                    CUDA_CHECK(cudaMemcpy(d_weights, flat_weights.data(), size * sizeof(float), cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(d_gradients, flat_gradients.data(), size * sizeof(float), cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(d_prev_gradients, flat_prev_gradients.data(), size * sizeof(float), cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(d_delta_weights, flat_delta_weights.data(), size * sizeof(float), cudaMemcpyHostToDevice));
                    
                    // Launch Rprop update kernel
                    int threadsPerBlock = 256;
                    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
                    
                    rpropUpdateKernel<<<blocksPerGrid, threadsPerBlock>>>(d_weights, d_gradients, d_prev_gradients,
                                                                         d_delta_weights, etaPlus, etaMinus,
                                                                         deltaMax, deltaMin, size);
                    CUDA_CHECK(cudaGetLastError());
                    
                    // Copy updated data back to host
                    std::vector<float> updated_weights(size);
                    std::vector<float> updated_prev_gradients(size);
                    std::vector<float> updated_delta_weights(size);
                    
                    CUDA_CHECK(cudaMemcpy(updated_weights.data(), d_weights, size * sizeof(float), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(updated_prev_gradients.data(), d_prev_gradients, size * sizeof(float), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(updated_delta_weights.data(), d_delta_weights, size * sizeof(float), cudaMemcpyDeviceToHost));
                    
                    // Reshape flat arrays back to 3D
                    int idx = 0;
                    for (unsigned int i = 0; i < in; ++i) {
                        for (unsigned int j = 0; j < in; ++j) {
                            weights[l][i][j] = updated_weights[idx];
                            prev_gradients[l][i][j] = updated_prev_gradients[idx];
                            delta_weights[l][i][j] = updated_delta_weights[idx];
                            idx++;
                        }
                    }
                    
                    // Free device memory
                    CUDA_CHECK(cudaFree(d_weights));
                    CUDA_CHECK(cudaFree(d_gradients));
                    CUDA_CHECK(cudaFree(d_prev_gradients));
                    CUDA_CHECK(cudaFree(d_delta_weights));
                }
            }
            catch (const std::exception& e) {
                std::cerr << "CUDA Exception in Rprop: " << e.what() << std::endl;
                
                // Cleanup on error
                cudaFree(d_weights);
                cudaFree(d_gradients);
                cudaFree(d_prev_gradients);
                cudaFree(d_delta_weights);
                
                throw;
            }
        }
        
        totalError /= dataset.size();
        std::cout << "Epoch " << epoch + 1 << "/" << epochs << " - Mean Squared Error: " << totalError << std::endl;
        
        if (totalError < 0.01) {
            status = true;
            break;
        }
    }
}
