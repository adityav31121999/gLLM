
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
    if (initial_inputs.size() != network.layer_sizes[0]) {
        throw std::invalid_argument("Initial input size does not match network's input layer size definition.");
    }

    // Set input layer activations
    // network.activations[0] is expected to be set by mlp::cuForward using network.input
    if (network.activations[0].size() != network.layer_sizes[0] || network.activations[0].data() != initial_inputs.data()) {
        // This check ensures consistency if mlp::cuForward sets network.activations[0] = network.input
        // and then calls cuforwardPropagate(network.input, network)
        // For now, we assume network.activations[0] is correctly pre-set.
    }

    // Prepare output vector
    std::vector<float> final_output;

    // Device memory pointers
    float *d_current_layer_activations = nullptr;
    float *d_weights_current_to_next = nullptr;
    float *d_next_layer_pre_activation = nullptr;
    float *d_next_layer_post_activation = nullptr;

    try {
        // Loop through each layer transition (from layer L to layer L+1)
        // network.weights[i] connects layer_sizes[i] to layer_sizes[i+1]
        for (unsigned int i = 0; i < network.num_layers - 1; ++i) {
            unsigned int current_layer_node_count = network.layer_sizes[i];
            unsigned int next_layer_node_count = network.layer_sizes[i+1];

            const mat& current_weight_matrix = network.weights[i]; // mat connecting layer i to i+1
            // Weight matrix dimensions: rows = next_layer_node_count, cols = current_layer_node_count
            if (current_weight_matrix.row != static_cast<int>(next_layer_node_count) ||
                current_weight_matrix.col != static_cast<int>(current_layer_node_count)) {
                throw std::runtime_error("Weight matrix dimensions mismatch for layer " + std::to_string(i));
            }

            size_t current_activations_bytes = current_layer_node_count * sizeof(float);
            size_t weights_bytes = static_cast<size_t>(current_weight_matrix.row) * current_weight_matrix.col * sizeof(float);
            size_t next_layer_bytes = next_layer_node_count * sizeof(float);

            // Allocate device memory
            CUDA_CHECK(cudaMalloc(&d_current_layer_activations, current_activations_bytes));
            CUDA_CHECK(cudaMalloc(&d_weights_current_to_next, weights_bytes));
            CUDA_CHECK(cudaMalloc(&d_next_layer_pre_activation, next_layer_bytes));
            CUDA_CHECK(cudaMalloc(&d_next_layer_post_activation, next_layer_bytes)); // For post-activation

            // Copy data to device
            CUDA_CHECK(cudaMemcpy(d_current_layer_activations, network.activations[i].data(),
                                 current_activations_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_weights_current_to_next, current_weight_matrix.mapped_data,
                                 weights_bytes, cudaMemcpyHostToDevice));

            // Initialize next layer outputs to zero (important if kernel doesn't fully overwrite)
            CUDA_CHECK(cudaMemset(d_next_layer_pre_activation, 0, next_layer_bytes));
            CUDA_CHECK(cudaMemset(d_next_layer_post_activation, 0, next_layer_bytes));

            // Matrix-vector multiplication kernel
            int threadsPerBlock = 256;
            int blocksPerGrid = (next_layer_node_count + threadsPerBlock - 1) / threadsPerBlock;

            // Calculate layer outputs (pre-activation)
            // layerForwardKernel(inputs, weights, outputs, input_size, output_size)
            // input_size = current_layer_node_count (cols of weight matrix)
            // output_size = next_layer_node_count (rows of weight matrix)
            layerForwardKernel<<<blocksPerGrid, threadsPerBlock>>>(
                d_current_layer_activations, d_weights_current_to_next, d_next_layer_pre_activation,
                current_layer_node_count, next_layer_node_count);
            CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors

            // Copy pre-activation values to host (hlayers)
            CUDA_CHECK(cudaMemcpy(network.hlayers[i+1].data(), d_next_layer_pre_activation,
                                 next_layer_bytes, cudaMemcpyDeviceToHost));

            // Apply activation function (sigmoid)
            // Assuming cuSigmoid is available and takes (input_device, output_device, size)
            cuSigmoid<<<blocksPerGrid, threadsPerBlock>>>(d_next_layer_pre_activation, d_next_layer_post_activation, next_layer_node_count);
            CUDA_CHECK(cudaGetLastError());

            // Copy post-activation values to host (activations)
            CUDA_CHECK(cudaMemcpy(network.activations[i+1].data(), d_next_layer_post_activation,
                                 next_layer_bytes, cudaMemcpyDeviceToHost));

            // Free device memory
            CUDA_CHECK(cudaFree(d_current_layer_activations)); d_current_layer_activations = nullptr;
            CUDA_CHECK(cudaFree(d_weights_current_to_next)); d_weights_current_to_next = nullptr;
            CUDA_CHECK(cudaFree(d_next_layer_pre_activation)); d_next_layer_pre_activation = nullptr;
            CUDA_CHECK(cudaFree(d_next_layer_post_activation)); d_next_layer_post_activation = nullptr;
        }

        // Set up the final output vector
        final_output = network.activations.back(); // Activations of the last layer

        return final_output;
    }
    catch (const std::exception& e) {
        std::cerr << "CUDA Exception in forward propagation: " << e.what() << std::endl;
        // Cleanup on error - safely free any allocated memory
        if (d_current_layer_activations) cudaFree(d_current_layer_activations);
        if (d_weights_current_to_next) cudaFree(d_weights_current_to_next);
        if (d_next_layer_pre_activation) cudaFree(d_next_layer_pre_activation);
        if (d_next_layer_post_activation) cudaFree(d_next_layer_post_activation);
        throw;
    }
}

/**
 * @brief CUDA implementation of forward propagation for MLP
 * @param in The input layer index
 * @param layers Number of layers to process
 */
void mlp::cuForward(int in, int layers) {
    // The 'in' and 'layers' parameters should ideally match this->layer_sizes[0] and this->num_layers.
    // We will use this->num_layers and this->layer_sizes for internal consistency.
    if (static_cast<unsigned int>(layers) != this->num_layers || static_cast<unsigned int>(in) != this->layer_sizes[0]) {
        // Optionally, throw an error or log a warning if passed parameters don't match MLP's configured structure.
        // For now, we proceed assuming they are consistent or this->members are authoritative.
    }

    // Ensure hlayers and activations are correctly sized
    if (hlayers.size() != this->num_layers) hlayers.resize(this->num_layers);
    if (activations.size() != this->num_layers) activations.resize(this->num_layers);

    for (unsigned int i = 0; i < this->num_layers; ++i) {
        if (hlayers[i].size() != this->layer_sizes[i]) {
            hlayers[i].resize(this->layer_sizes[i], 0.0f);
        }
        if (activations[i].size() != this->layer_sizes[i]) {
            activations[i].resize(this->layer_sizes[i], 0.0f);
        }
    }

    this->activations[0] = this->input; // Set initial activations from the network's input member

    try {
        this->output = cuforwardPropagate(this->input, *this); // Pass this->input
    }
    catch (const std::exception& e) {
        std::cerr << "Error in CUDA forward propagation: " << e.what() << std::endl;
        throw;
    }
}
