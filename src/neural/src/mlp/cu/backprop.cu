
// backprop.cu: CUDA implementations for backward propagation in MLP
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
 * @brief CUDA implementation of backpropagation for MLP with gradients.
 *        Calculates deltas (error * derivative) and updates weights.
 * @param in Input/layer size (assuming all layers have the same size 'in')
 * @param layers Number of layers (excluding input layer)
 * @param learning Learning rate
 */
void mlp::cuBackprop(int in, int layers, float learning) {
    // Basic validation
    if (output.size() != in || expected.size() != in || activations.empty() || weights.empty()) {
        throw std::runtime_error("MLP members not properly initialized for cuBackprop.");
    }
    if (activations.size() != layers || weights.size() != layers) {
        throw std::runtime_error("Mismatch between layer count and activation/weight vector sizes.");
    }
    for(const auto& layer_weights : weights) {
        if (layer_weights.empty() || layer_weights[0].size() != in) {
            throw std::runtime_error("Weight dimensions are incorrect.");
        }
    }
    for(const auto& layer_activations : activations) {
        if (layer_activations.size() != in) {
            throw std::runtime_error("Activation dimensions are incorrect.");
        }
    }
    if (input.size() != in) {
        throw std::runtime_error("Input vector size mismatch.");
    }

    // --- Initialize Host gweights if necessary ---
    // This ensures the host vector has the correct structure before copying data back.
    if (gweights.size() != layers) {
        gweights.resize(layers);
    }
    for (int l = 0; l < layers; ++l) {
        if (gweights[l].size() != in) {
            gweights[l].resize(in);
        }
        for (int i = 0; i < in; ++i) {
            if (gweights[l][i].size() != in) {
                gweights[l][i].resize(in, 0.0f); // Initialize with 0.0f
            }
        }
    }


    // --- Device Memory Allocation ---
    float *d_input = nullptr;
    float *d_output = nullptr;
    float *d_expected = nullptr;
    std::vector<float*> d_activations(layers, nullptr);     // Device pointers for activations of each layer
    std::vector<float*> d_weights(layers, nullptr);         // Device pointers for weights of each layer
    std::vector<float*> d_gweights(layers, nullptr);        // Device pointers for gradients of each layer <-- ADDED
    std::vector<float*> d_layer_deltas(layers, nullptr);    // Device pointers for deltas of each layer

    try {
        // Allocate memory for input, output, expected
        CUDA_CHECK(cudaMalloc(&d_input, in * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, in * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_expected, in * sizeof(float)));

        // Allocate memory for activations, weights, gradients, and deltas for each layer
        size_t layer_size_bytes = in * sizeof(float);
        size_t weights_size_bytes = in * in * sizeof(float); // Assuming square weight matrices [in x in]

        for (int l = 0; l < layers; ++l) {
            CUDA_CHECK(cudaMalloc(&d_activations[l], layer_size_bytes));
            CUDA_CHECK(cudaMalloc(&d_weights[l], weights_size_bytes));
            CUDA_CHECK(cudaMalloc(&d_gweights[l], weights_size_bytes)); // <-- Allocate gradient memory
            CUDA_CHECK(cudaMalloc(&d_layer_deltas[l], layer_size_bytes));
        }

        // --- Data Transfer: Host -> Device ---
        CUDA_CHECK(cudaMemcpy(d_input, input.data(), layer_size_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_output, output.data(), layer_size_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_expected, expected.data(), layer_size_bytes, cudaMemcpyHostToDevice));

        for (int l = 0; l < layers; ++l) {
            // Copy activations
            CUDA_CHECK(cudaMemcpy(d_activations[l], activations[l].data(), layer_size_bytes, cudaMemcpyHostToDevice));

            // Flatten and copy weights for layer l
            std::vector<float> flat_weights;
            flat_weights.reserve(in * in);
            for (int i = 0; i < in; ++i) {
                 if (weights[l].size() != in || weights[l][i].size() != in) {
                    throw std::runtime_error("Weight matrix dimension error during flattening.");
                 }
                flat_weights.insert(flat_weights.end(), weights[l][i].begin(), weights[l][i].end());
            }
            CUDA_CHECK(cudaMemcpy(d_weights[l], flat_weights.data(), weights_size_bytes, cudaMemcpyHostToDevice));
            // No need to copy gweights H->D as they are calculated on the device
        }

        // --- Kernel Launch Configuration ---
        int threadsPerBlock1D = 256;
        int blocksPerGrid1D = (in + threadsPerBlock1D - 1) / threadsPerBlock1D;
        dim3 blockDim1D(threadsPerBlock1D);
        dim3 gridDim1D(blocksPerGrid1D);

        // For weight/gradient updates (matrix operations)
        // Use 2D grid/block structure matching updateWeightsKernel logic
        dim3 blockDim2D(16, 16); // e.g., 16x16 = 256 threads per block
        // Grid dimensions cover the entire weight matrix (current_layer_size x prev_layer_size)
        dim3 gridDim2D((in + blockDim2D.x - 1) / blockDim2D.x, // Grid blocks for columns (prev_layer_size)
                       (in + blockDim2D.y - 1) / blockDim2D.y); // Grid blocks for rows (current_layer_size)

        // --- Backpropagation Steps ---
 
        // 1. Calculate Output Layer Deltas (Layer layers-1)
        kernelOutputDelta<<<gridDim1D, blockDim1D>>>(d_output, d_expected, d_layer_deltas[layers - 1], in);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize()); // Sync after kernel launch for safety/debugging
 
        // 2. Calculate Hidden Layer Deltas (Propagate backwards from layers-2 down to 0)
        for (int l = layers - 2; l >= 0; --l) {
            hiddenDeltaKernel<<<gridDim1D, blockDim1D>>>(
                d_layer_deltas[l + 1],    // Deltas from the next layer (l+1)
                d_weights[l + 1],         // Weights connecting layer l to layer l+1
                d_activations[l],         // Activations of the current layer (l)
                d_layer_deltas[l],        // Deltas to compute for the current layer (l)
                in,                       // current_layer_size
                in                        // next_layer_size (assuming square matrices)
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize()); // Sync after kernel launch
        }

        // 3. Calculate Gradients and Update Weights (Iterate through layers 0 to layers-1)
        for (int l = 0; l < layers; ++l) {
            float* d_prev_activations = (l == 0) ? d_input : d_activations[l - 1];
 
            updateWeightsKernel<<<gridDim2D, blockDim2D>>>(
                d_layer_deltas[l],        // Deltas for the current layer (l)
                d_prev_activations,       // Activations from the previous layer (l-1 or input)
                d_weights[l],             // Weights connecting previous layer to current layer (l) - TO BE UPDATED
                d_gweights[l],            // Gradients for layer l - TO BE CALCULATED <-- Pass gradient buffer
                learning,                 // Learning rate
                in,                       // current_layer_size
                in                        // prev_layer_size (assuming square matrices)
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize()); // Sync after kernel launch
        }

        // --- Data Transfer: Device -> Host ---
 
        // Copy updated weights back to the host mlp object
        for (int l = 0; l < layers; ++l) {
            std::vector<float> updated_flat_weights(in * in);
            CUDA_CHECK(cudaMemcpy(updated_flat_weights.data(), d_weights[l], weights_size_bytes, cudaMemcpyDeviceToHost));
 
            // Reshape flat weights back into the 3D vector structure
            for (int i = 0; i < in; ++i) {
                 if (weights[l].size() != in || weights[l][i].size() != in) {
                    throw std::runtime_error("Weight matrix dimension error during host update.");
                 }
                for (int j = 0; j < in; ++j) {
                    weights[l][i][j] = updated_flat_weights[i * in + j];
                }
            }
        }

        // Copy calculated gradients back to the host mlp object <-- ADDED SECTION
        for (int l = 0; l < layers; ++l) {
            std::vector<float> calculated_flat_gradients(in * in);
            CUDA_CHECK(cudaMemcpy(calculated_flat_gradients.data(), d_gweights[l], weights_size_bytes, cudaMemcpyDeviceToHost));
 
            // Reshape flat gradients back into the 3D vector structure
            for (int i = 0; i < in; ++i) {
                 // Ensure gweights structure is correct (already done at the beginning)
                 if (gweights[l].size() != in || gweights[l][i].size() != in) {
                    throw std::runtime_error("gweights matrix dimension error during host update.");
                 }
                for (int j = 0; j < in; ++j) {
                    gweights[l][i][j] = calculated_flat_gradients[i * in + j]; // <-- Store gradient
                }
            }
        }


        // --- Cleanup Device Memory ---
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_expected);
        for (int l = 0; l < layers; ++l) {
            cudaFree(d_activations[l]);
            cudaFree(d_weights[l]);
            cudaFree(d_gweights[l]); // <-- Free gradient memory
            cudaFree(d_layer_deltas[l]);
        }
    }
    catch (const std::exception& e) {
        std::cerr << "CUDA Exception in cuBackprop: " << e.what() << std::endl;

        // --- Cleanup on Error ---
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_expected);
        for (int l = 0; l < layers; ++l) {
            cudaFree(d_activations[l]); // Use cudaFree on potentially allocated pointers
            cudaFree(d_weights[l]);
            cudaFree(d_gweights[l]); // <-- Free gradient memory on error
            cudaFree(d_layer_deltas[l]);
        }
        // Re-throw the exception to signal failure
        throw;
    }
}


/**
 * @brief CUDA implementation of backpropagation that updates input vector
 * @param in Input size
 * @param layers Number of layers
 * @param learning Learning rate
 */
void mlp::cuBackprop2in(int in, int layers, float learning) {
    // First perform standard backpropagation
    cuBackprop(in, layers, learning);
    
    // Device memory pointers
    float *d_input, *d_weights, *d_deltas;
    
    try {
        // Allocate memory for input update
        CUDA_CHECK(cudaMalloc(&d_input, in * sizeof(float)));
        
        // Copy input to device
        CUDA_CHECK(cudaMemcpy(d_input, input.data(), in * sizeof(float), cudaMemcpyHostToDevice));
        
        // Compute output layer error
        std::vector<float> output_error(in, 0.0f);
        for (int i = 0; i < in; i++) {
            output_error[i] = output[i] - expected[i];
        }
        
        // Compute layer errors for all layers
        std::vector<std::vector<float>> layer_errors(layers, std::vector<float>(in, 0.0f));
        
        // Compute error for the last hidden layer
        for (int i = 0; i < in; i++) {
            float error_sum = 0.0f;
            for (int j = 0; j < in; j++) {
                error_sum += weights[layers-1][j][i] * output_error[j];
            }
            layer_errors[layers-1][i] = error_sum * activations[layers-1][i] * (1.0f - activations[layers-1][i]);
        }
        
        // Propagate error backward through the network
        for (int l = layers - 2; l >= 0; l--) {
            for (int i = 0; i < in; i++) {
                float error_sum = 0.0f;
                for (int j = 0; j < in; j++) {
                    error_sum += weights[l+1][j][i] * layer_errors[l+1][j];
                }
                
                if (l > 0) {
                    layer_errors[l][i] = error_sum * activations[l][i] * (1.0f - activations[l][i]);
                }
                else {
                    // For the first layer, use input values for activation derivative
                    layer_errors[l][i] = error_sum * activations[l][i] * (1.0f - activations[l][i]);
                }
            }
        }
        
        // Get deltas for first layer
        std::vector<float> first_layer_deltas = layer_errors[0];
        
        // Flatten weights for first layer
        std::vector<float> flat_weights;
        for (int i = 0; i < in; i++) {
            flat_weights.insert(flat_weights.end(), 
                               weights[0][i].begin(), 
                               weights[0][i].end());
        }
        
        // Allocate and copy data to device
        CUDA_CHECK(cudaMalloc(&d_weights, in * in * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_deltas, in * sizeof(float)));
        
        CUDA_CHECK(cudaMemcpy(d_weights, flat_weights.data(), 
                             in * in * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_deltas, first_layer_deltas.data(), 
                             in * sizeof(float), cudaMemcpyHostToDevice));
        
        // Launch kernel to update input vector
        int threadsPerBlock = 256;
        int blocksPerGrid = (in + threadsPerBlock - 1) / threadsPerBlock;
        updateInputVectorKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_weights, d_deltas, learning, in);
        CUDA_CHECK(cudaGetLastError());
        
        // Copy updated input back to host
        CUDA_CHECK(cudaMemcpy(input.data(), d_input, in * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Free device memory
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_weights));
        CUDA_CHECK(cudaFree(d_deltas));
    }
    catch (const std::exception& e) {
        std::cerr << "CUDA Exception in backprop2in: " << e.what() << std::endl;
        
        // Cleanup on error
        cudaFree(d_input);
        cudaFree(d_weights);
        cudaFree(d_deltas);
        
        throw;
    }
}
