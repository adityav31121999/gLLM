
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
 * @brief CUDA kernel for MLP forward propagation
 * @param inputs Input data
 * @param weights Weights for the current layer
 * @param biases Biases for the current layer
 * @param outputs Output data
 * @param input_size Size of the input layer
 * @param output_size Size of the output layer
 */
// CUDA kernel for matrix-vector multiplication
__global__ void layerForwardKernel(float* inputs, float* weights, float* outputs, 
    int input_size, int output_size)
{
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (neuron_idx < output_size) {
        // Initialize sum to 0 or bias value if biases are provided
        float sum = 0.0f;

        // Perform matrix-vector multiplication
        for (int i = 0; i < input_size; i++) {
            sum += inputs[i] * weights[neuron_idx * input_size + i];
        }
        outputs[neuron_idx] = sum;
    }
}


/**
 * @brief The forward propagation function for MLP in CUDA. This function performs the forward 
 * propagation and calculates the activations of each layer.
 * @param inputs The input data.
 * @param network The network to be used for forward propagation. This will be modified to store layer outputs and activations.
 * @return std::vector<float> The activations of the last layer.
 */
std::vector<float> cuforwardPropagate(const std::vector<float>& inputs, mlp& network) {
    if (inputs.size() != network.hlayers[0].size()) {
        throw std::invalid_argument("Input size does not match network input layer size");
    }
    
    // Ensure network vectors are properly initialized
    int layers = network.hlayers.size();
    int in = network.hlayers[0].size();
    
    if (network.activations.size() != layers) {
        network.activations.resize(layers, std::vector<float>(in, 0.0));
    }
    
    // Set input layer activations
    network.activations[0] = inputs;
    
    // Prepare output vector
    std::vector<float> final_output;
    
    // Device memory pointers
    float *d_inputs, *d_weights, *d_outputs, *d_layer_output;
    
    try {
        for (size_t layer = 1; layer < network.hlayers.size(); layer++) {
            int input_size = network.hlayers[layer-1].size();
            int output_size = network.hlayers[layer].size();
            
            // Ensure vectors are properly sized
            if (network.hlayers[layer].size() != output_size) {
                network.hlayers[layer].resize(output_size, 0.0);
            }
            if (network.activations[layer].size() != output_size) {
                network.activations[layer].resize(output_size, 0.0);
            }
            
            // Flatten weights for current layer
            std::vector<float> flat_weights;
            for (int i = 0; i < output_size; i++) {
                flat_weights.insert(flat_weights.end(), 
                                   network.weights[layer-1][i].begin(), 
                                   network.weights[layer-1][i].end());
            }
            
            // Allocate device memory
            CUDA_CHECK(cudaMalloc(&d_inputs, input_size * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_weights, input_size * output_size * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_outputs, output_size * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_layer_output, output_size * sizeof(float)));
            
            // Copy data to device
            CUDA_CHECK(cudaMemcpy(d_inputs, network.activations[layer-1].data(), 
                                 input_size * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_weights, flat_weights.data(), 
                                 input_size * output_size * sizeof(float), cudaMemcpyHostToDevice));
            
            // Initialize outputs to zero (since we're not using biases)
            CUDA_CHECK(cudaMemset(d_outputs, 0, output_size * sizeof(float)));
            CUDA_CHECK(cudaMemset(d_layer_output, 0, output_size * sizeof(float)));
            
            // Matrix-vector multiplication kernel
            int threadsPerBlock = 256;
            int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;
            
            // Calculate layer outputs (pre-activation)
            layerForwardKernel<<<blocksPerGrid, threadsPerBlock>>>(
                d_inputs, d_weights, d_layer_output, input_size, output_size);

            // Copy pre-activation values to host (hlayers)
            CUDA_CHECK(cudaMemcpy(network.hlayers[layer].data(), d_layer_output, 
                                 output_size * sizeof(float), cudaMemcpyDeviceToHost));
            
            // Apply activation function (sigmoid)
            cuSigmoid<<<blocksPerGrid, threadsPerBlock>>>(d_layer_output, d_outputs, output_size);

            // Copy post-activation values to host (activations)
            CUDA_CHECK(cudaMemcpy(network.activations[layer].data(), d_outputs, 
                                 output_size * sizeof(float), cudaMemcpyDeviceToHost));

            // Free device memory
            CUDA_CHECK(cudaFree(d_inputs));
            CUDA_CHECK(cudaFree(d_weights));
            CUDA_CHECK(cudaFree(d_outputs));
            CUDA_CHECK(cudaFree(d_layer_output));
        }
        
        // Set up the final output vector
        final_output = network.activations[layers-1];
        
        return final_output;
    }
    catch (const std::exception& e) {
        std::cerr << "CUDA Exception in forward propagation: " << e.what() << std::endl;
        // Cleanup on error - safely free any allocated memory
        // These calls are safe even if the pointers are null
        cudaFree(d_inputs);
        cudaFree(d_weights);
        cudaFree(d_outputs);
        cudaFree(d_layer_output);
        throw;
    }
}

/**
 * @brief CUDA implementation of forward propagation for MLP
 * @param in The input layer index
 * @param layers Number of layers to process
 */
void mlp::cuForward(int in, int layers) {
    // Ensure vectors are properly initialized
    if (hlayers.size() != layers || activations.size() != layers) {
        hlayers.resize(layers, std::vector<float>(in, 0.0));
        activations.resize(layers, std::vector<float>(in, 0.0));
    }
    
    // Forward propagation using CUDA
    try {
        // Call the CUDA implementation of forward propagation
        // This will update hlayers and activations directly
        output = cuforwardPropagate(input, *this);
    }
    catch (const std::exception& e) {
        std::cerr << "Error in CUDA forward propagation: " << e.what() << std::endl;
        throw;
    }
}
