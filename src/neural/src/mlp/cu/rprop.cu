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
 * @brief CUDA implementation of Rprop algorithm for MLP
 * @param dataset Input dataset
 * @param layers Number of layers
 * @param in Input size
 * @param learning Learning rate (not used in Rprop)
 * @param epochs Number of epochs
 */
void mlp::cuRprop(std::vector<std::vector<float>>& dataset, int layers_param, int in_param, float learning, int epochs_param) {
    const float etaPlus = 1.2f;     // Increase factor
    const float etaMinus = 0.5f;    // Decrease factor
    const float deltaMax = 50.0f;   // Maximum update value
    const float deltaMin = 1e-6f;   // Minimum update value

    // Initialize Rprop state matrices (previous gradients and delta weights)
    // These will have the same dimensions as weights
    std::vector<mat> prev_gradients_mats;
    std::vector<mat> delta_weights_mats;

    prev_gradients_mats.resize(num_layers - 1);
    delta_weights_mats.resize(num_layers - 1);

    for (unsigned int l = 0; l < num_layers - 1; ++l) {
        prev_gradients_mats[l] = mat(weights[l].row, weights[l].col); // Create temp mapped file
        std::fill_n(prev_gradients_mats[l].mapped_data,
                    static_cast<size_t>(prev_gradients_mats[l].row) * prev_gradients_mats[l].col,
                    0.0f);

        delta_weights_mats[l] = mat(weights[l].row, weights[l].col); // Create temp mapped file
        std::fill_n(delta_weights_mats[l].mapped_data,
                    static_cast<size_t>(delta_weights_mats[l].row) * delta_weights_mats[l].col,
                    deltaMin);
    }

    for (unsigned int epoch = 0; epoch < epochs_param; ++epoch) {
        float totalError = 0.0f;
        
        for (const auto& data : dataset) {
            // Set input and perform forward and backward passes
            input = data;
            cuForward(in_param, layers_param);
            cuBackprop(in_param, layers_param, learning); // This computes gradients and stores them in gweights

            // Ensure cuForward and cuBackprop use consistent parameters based on MLP's structure
            // cuForward(layer_sizes[0], num_layers);
            // cuBackprop(num_layers, layer_sizes[0], learning); // Assuming cuBackprop updates gweights

            // Compute mean square error
            float error = 0.0f;
            for (unsigned int i = 0; i < output.size(); ++i) { // Use actual output size
                error += std::pow(expected[i] - output[i], 2);
            }
            if (!output.empty()) {
                error /= output.size();
            }
            totalError += error;
            
            // Device memory pointers - scope to the layer loop
            float *d_weights_layer = nullptr, *d_gradients_layer = nullptr,
                  *d_prev_gradients_layer = nullptr, *d_delta_weights_layer = nullptr;

            try {
                // Update weights using Rprop for each layer
                for (unsigned int l = 0; l < (num_layers - 1); ++l) {
                    mat& current_weights_mat = weights[l];
                    mat& current_gradients_mat = gweights[l]; // Gradients computed by cuBackprop
                    mat& current_prev_grad_mat = prev_gradients_mats[l];
                    mat& current_delta_w_mat = delta_weights_mats[l];

                    size_t num_elements = static_cast<size_t>(current_weights_mat.row) * current_weights_mat.col;
                    if (num_elements == 0) continue; // Skip if layer has no weights

                    size_t data_bytes = num_elements * sizeof(float);

                    // Allocate device memory
                    CUDA_CHECK(cudaMalloc(&d_weights_layer, data_bytes));
                    CUDA_CHECK(cudaMalloc(&d_gradients_layer, data_bytes));
                    CUDA_CHECK(cudaMalloc(&d_prev_gradients_layer, data_bytes));
                    CUDA_CHECK(cudaMalloc(&d_delta_weights_layer, data_bytes));

                    // Copy data to device
                    CUDA_CHECK(cudaMemcpy(d_weights_layer, current_weights_mat.mapped_data, data_bytes, cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(d_gradients_layer, current_gradients_mat.mapped_data, data_bytes, cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(d_prev_gradients_layer, current_prev_grad_mat.mapped_data, data_bytes, cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(d_delta_weights_layer, current_delta_w_mat.mapped_data, data_bytes, cudaMemcpyHostToDevice));

                    // Launch Rprop update kernel
                    int threadsPerBlock = 256;
                    int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

                    rpropUpdateKernel<<<blocksPerGrid, threadsPerBlock>>>(d_weights_layer, d_gradients_layer, d_prev_gradients_layer,
                                                                         d_delta_weights_layer, etaPlus, etaMinus,
                                                                         deltaMax, deltaMin, num_elements);
                    CUDA_CHECK(cudaGetLastError());

                    // Copy updated data back to host
                    CUDA_CHECK(cudaMemcpy(current_weights_mat.mapped_data, d_weights_layer, data_bytes, cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(current_prev_grad_mat.mapped_data, d_prev_gradients_layer, data_bytes, cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(current_delta_w_mat.mapped_data, d_delta_weights_layer, data_bytes, cudaMemcpyDeviceToHost));

                    // Free device memory
                    CUDA_CHECK(cudaFree(d_weights_layer)); d_weights_layer = nullptr;
                    CUDA_CHECK(cudaFree(d_gradients_layer)); d_gradients_layer = nullptr;
                    CUDA_CHECK(cudaFree(d_prev_gradients_layer)); d_prev_gradients_layer = nullptr;
                    CUDA_CHECK(cudaFree(d_delta_weights_layer)); d_delta_weights_layer = nullptr;
                }
            }
            catch (const std::exception& e) {
                std::cerr << "CUDA Exception in Rprop: " << e.what() << std::endl;
                
                // Cleanup on error
                if(d_weights_layer) cudaFree(d_weights_layer);
                if(d_gradients_layer) cudaFree(d_gradients_layer);
                if(d_prev_gradients_layer) cudaFree(d_prev_gradients_layer);
                if(d_delta_weights_layer) cudaFree(d_delta_weights_layer);
                throw;
            }
        }
        
        totalError /= dataset.size();
        std::cout << "Epoch " << epoch + 1 << "/" << epochs_param << " - Mean Squared Error: " << totalError << std::endl;
        
        if (totalError < 0.01) {
            status = true;
            break;
        }
    }
}
#endif