
// forprop.cu: CUDA implementations for forward propagation in MLP
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <maths.hpp>  // Include for activation functions like cuSigmoid
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
 * @brief The forward propagation function for MLP in CUDA. This function performs the forward 
 * propagation and calculates the activations of each layer. Assumes network.hlayers and
 * network.activations are already sized correctly, and network.activations[0] is set.
 * @param initial_inputs The input data for the first layer.
 * @param network The network to be used for forward propagation. This will be modified to store layer outputs and activations.
 * @return std::vector<float> The activations of the last layer.
 */
std::vector<float> cuforwardPropagate(const std::vector<float>& initial_inputs, mlp& network) {
    if (network.input.size() != network.layer_sizes[0]) { // Use network.input
        throw std::invalid_argument("Initial input size does not match network's input layer size definition.");
    }

    // Determine max layer size for efficient buffer allocation
    unsigned int max_neurons_in_layer = 0;
    for (unsigned int size : network.layer_sizes) {
        if (size > max_neurons_in_layer) {
            max_neurons_in_layer = size;
        }
    }
    unsigned int max_weights_in_matrix = 0;
    for (const auto& w_mat : network.weights) {
        if (static_cast<unsigned int>(w_mat.row * w_mat.col) > max_weights_in_matrix) {
            max_weights_in_matrix = w_mat.row * w_mat.col;
        }
    }

    // Device memory pointers
    float *d_ping_activations = nullptr; // Input to current layer op
    float *d_pong_activations = nullptr; // Output of current layer op (pre-activation)
    float *d_weights_buffer = nullptr;   // Buffer for current weight matrix

    try {
        // Allocate device memory once
        size_t max_activation_bytes = max_neurons_in_layer * sizeof(float);
        size_t max_weights_bytes = max_weights_in_matrix * sizeof(float);

        CUDA_CHECK(cudaMalloc(&d_ping_activations, max_activation_bytes));
        CUDA_CHECK(cudaMalloc(&d_pong_activations, max_activation_bytes)); // Used for both pre and post activation sequentially
        if (max_weights_in_matrix > 0) {
            CUDA_CHECK(cudaMalloc(&d_weights_buffer, max_weights_bytes));
        }

        // Copy initial input (network.activations[0]) to d_ping_activations
        CUDA_CHECK(cudaMemcpy(d_ping_activations, network.activations[0].data(),
                              network.layer_sizes[0] * sizeof(float), cudaMemcpyHostToDevice));

        // Loop through each layer transition (from layer L to layer L+1)
        // network.weights[i] connects layer_sizes[i] to layer_sizes[i+1]
        for (unsigned int i = 0; i < network.num_layers - 1; ++i) {
            unsigned int current_layer_node_count = network.layer_sizes[i];
            unsigned int next_layer_node_count = network.layer_sizes[i+1];
            const mat& current_weight_matrix = network.weights[i]; // mat connecting layer i to i+1

            if (current_weight_matrix.row != static_cast<int>(next_layer_node_count) ||
                current_weight_matrix.col != static_cast<int>(current_layer_node_count)) {
                throw std::runtime_error("Weight matrix dimensions mismatch for layer " + std::to_string(i));
            }

            size_t weights_bytes = static_cast<size_t>(current_weight_matrix.row) * current_weight_matrix.col * sizeof(float);
            size_t next_layer_bytes = next_layer_node_count * sizeof(float);

            // Copy current weights to device buffer
            CUDA_CHECK(cudaMemcpy(d_weights_buffer, current_weight_matrix.mapped_data,
                                 weights_bytes, cudaMemcpyHostToDevice));

            // Initialize next layer outputs to zero (important if kernel doesn't fully overwrite)
            CUDA_CHECK(cudaMemset(d_pong_activations, 0, next_layer_bytes)); // pong will hold pre-activations first

            // Matrix-vector multiplication kernel
            int threadsPerBlock = 256;
            int blocksPerGrid = (next_layer_node_count + threadsPerBlock - 1) / threadsPerBlock;

            layerForwardKernel<<<blocksPerGrid, threadsPerBlock>>>(
                d_ping_activations, d_weights_buffer, d_pong_activations, // d_pong now holds pre-activation
                current_layer_node_count, next_layer_node_count);
            CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors

            // Copy pre-activation values to host (hlayers)
            // Corrected indexing for hlayers: hlayers[i] stores pre-activations for layer i+1 (output of weights[i])
            CUDA_CHECK(cudaMemcpy(network.hlayers[i].data(), d_pong_activations,
                                 next_layer_bytes, cudaMemcpyDeviceToHost));

            // Apply activation function (sigmoid)
            // The output of sigmoid will overwrite d_pong_activations (which held pre-activations)
            // Or, if cuSigmoid needs separate input/output, use another buffer or modify cuSigmoid.
            // Assuming cuSigmoid can operate in-place or d_pong_activations is now the input to sigmoid
            // and its output will be the new d_ping_activations for the next layer.
            // For clarity, let's assume cuSigmoid writes to a new buffer, then we swap.
            // If cuSigmoid(input, output, size):
            cuSigmoid<<<blocksPerGrid, threadsPerBlock>>>(d_pong_activations, d_ping_activations, next_layer_node_count); // d_ping now holds post-activation
            CUDA_CHECK(cudaGetLastError());

            // Copy post-activation values to host (activations)
            CUDA_CHECK(cudaMemcpy(network.activations[i+1].data(), d_ping_activations, // d_ping holds post-activation
                                 next_layer_bytes, cudaMemcpyDeviceToHost));
            // No need to swap d_ping and d_pong here, as d_ping is already the input for the next iteration's layerForwardKernel
        }

        // Free device memory once after the loop
        CUDA_CHECK(cudaFree(d_ping_activations));
        CUDA_CHECK(cudaFree(d_pong_activations));
        if (d_weights_buffer) CUDA_CHECK(cudaFree(d_weights_buffer));

        return network.activations.back(); // Activations of the last layer
    }
    catch (const std::exception& e) {
        std::cerr << "CUDA Exception in forward propagation: " << e.what() << std::endl;
        // Cleanup on error - safely free any allocated memory
        if (d_ping_activations) cudaFree(d_ping_activations);
        if (d_pong_activations) cudaFree(d_pong_activations);
        if (d_weights_buffer) cudaFree(d_weights_buffer);
        throw;
    }
}

/**
 * @brief CUDA implementation of forward propagation for MLP
 * @param in The input layer index
 * @param layers Number of layers to process
 */
void mlp::cuForward(int in, int layers) {
    // The 'in' and 'layers' parameters are not used by cuforwardPropagate,
    // which relies on num_layers and layer_sizes.
    // Ensure hlayers and activations are correctly sized
    if (activations.size() != num_layers) activations.resize(num_layers);

    for (unsigned int i = 0; i < num_layers; ++i) {
        if (hlayers[i].size() != layer_sizes[i]) {
            hlayers[i].resize(layer_sizes[i], 0.0f);
        }
        if (activations[i].size() != layer_sizes[i]) {
            activations[i].resize(layer_sizes[i], 0.0f);
        }
    }

    // hlayers should be num_layers - 1
    if (hlayers.size() != (num_layers > 0 ? num_layers - 1 : 0)) hlayers.resize(num_layers > 0 ? num_layers - 1 : 0);
    for (unsigned int i = 0; i < (num_layers > 0 ? num_layers - 1 : 0); ++i) {
        if (hlayers[i].size() != layer_sizes[i+1]) hlayers[i].resize(layer_sizes[i+1], 0.0f);
    }
    activations[0] = input; // Set initial activations from the network's input member

    try {
        output = cuforwardPropagate(input, *this); // Pass input
    }
    catch (const std::exception& e) {
        std::cerr << "Error in CUDA forward propagation: " << e.what() << std::endl;
        throw;
    }
}
