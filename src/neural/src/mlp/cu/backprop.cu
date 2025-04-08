
// backprop.cu: CUDA implementations for backward propagation in MLP
#include "include/mlp.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <maths.hpp>

// Helper macro for CUDA error checking
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        throw std::runtime_error(cudaGetErrorString(err)); \
    } \
} while (0)

/**
 * @brief CUDA kernel for calculating output layer deltas
 * @param outputs Output layer values
 * @param expected Expected output values
 * @param deltas Output deltas to be computed
 * @param size Size of the arrays
 */
__global__ void outputDeltaKernel(float* outputs, float* expected, float* deltas, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // Calculate error gradient for output layer: (output - expected) * sigmoid_derivative(output)
        float sigmoid_der = outputs[idx] * (1.0f - outputs[idx]);
        deltas[idx] = (outputs[idx] - expected[idx]) * sigmoid_der;
    }
}

/**
 * @brief CUDA kernel for calculating hidden layer deltas
 * @param next_layer_deltas Deltas from the next layer
 * @param weights Weights connecting to the next layer
 * @param activations Activations of the current layer
 * @param deltas Deltas to be computed for the current layer
 * @param current_layer_size Size of the current layer
 * @param next_layer_size Size of the next layer
 */
__global__ void hiddenDeltaKernel(float* next_layer_deltas, float* weights, float* activations, 
                                 float* deltas, int current_layer_size, int next_layer_size) {
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (neuron_idx < current_layer_size) {
        float error_sum = 0.0f;
        
        // Sum up the weighted deltas from the next layer
        for (int j = 0; j < next_layer_size; j++) {
            error_sum += next_layer_deltas[j] * weights[j * current_layer_size + neuron_idx];
        }
        
        // Calculate delta using the sigmoid derivative
        float sigmoid_der = activations[neuron_idx] * (1.0f - activations[neuron_idx]);
        deltas[neuron_idx] = error_sum * sigmoid_der;
    }
}

/**
 * @brief CUDA kernel for updating weights
 * @param deltas Deltas for the current layer
 * @param prev_activations Activations from the previous layer (or inputs for first layer)
 * @param weights Weights to be updated
 * @param learning_rate Learning rate
 * @param current_layer_size Size of the current layer
 * @param prev_layer_size Size of the previous layer
 */
__global__ void updateWeightsKernel(float* deltas, float* prev_activations, float* weights,
                                   float learning_rate, int current_layer_size, int prev_layer_size) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < current_layer_size && col < prev_layer_size) {
        int weight_idx = row * prev_layer_size + col;
        weights[weight_idx] -= learning_rate * deltas[row] * prev_activations[col];
    }
}

/**
 * @brief CUDA kernel for L1 regularization weight update
 * @param weights Weights to be updated
 * @param deltas Deltas for the current layer
 * @param prev_activations Activations from the previous layer
 * @param learning_rate Learning rate
 * @param lambda L1 regularization parameter
 * @param current_layer_size Size of the current layer
 * @param prev_layer_size Size of the previous layer
 */
__global__ void updateWeightsL1Kernel(float* weights, float* deltas, float* prev_activations,
                                     float learning_rate, float lambda, 
                                     int current_layer_size, int prev_layer_size) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < current_layer_size && col < prev_layer_size) {
        int weight_idx = row * prev_layer_size + col;
        float gradient = deltas[row] * prev_activations[col];
        
        // L1 regularization update
        if (weights[weight_idx] > 0) {
            weights[weight_idx] -= learning_rate * (lambda + gradient);
        } else {
            weights[weight_idx] -= learning_rate * (-lambda + gradient);
        }
    }
}

/**
 * @brief CUDA kernel for L2 regularization weight update
 * @param weights Weights to be updated
 * @param deltas Deltas for the current layer
 * @param prev_activations Activations from the previous layer
 * @param learning_rate Learning rate
 * @param lambda L2 regularization parameter
 * @param current_layer_size Size of the current layer
 * @param prev_layer_size Size of the previous layer
 */
__global__ void updateWeightsL2Kernel(float* weights, float* deltas, float* prev_activations,
                                     float learning_rate, float lambda, 
                                     int current_layer_size, int prev_layer_size) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < current_layer_size && col < prev_layer_size) {
        int weight_idx = row * prev_layer_size + col;
        float gradient = deltas[row] * prev_activations[col];
        
        // L2 regularization update
        weights[weight_idx] -= learning_rate * (lambda * weights[weight_idx] + gradient);
    }
}

/**
 * @brief CUDA implementation of backpropagation for MLP
 * @param in Input size
 * @param layers Number of layers
 * @param learning Learning rate
 */
void mlp::cuBackprop(int in, int layers, float learning) {
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
        
        // Calculate output layer deltas
        int threadsPerBlock = 256;
        int blocksPerGrid = (in + threadsPerBlock - 1) / threadsPerBlock;
        
        outputDeltaKernel<<<blocksPerGrid, threadsPerBlock>>>(d_outputs, d_expected, d_deltas, in);
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
                flat_weights.insert(flat_weights.end(), 
                                   weights[l+1][i].begin(), 
                                   weights[l+1][i].end());
            }
            
            // Allocate and copy data to device
            CUDA_CHECK(cudaMalloc(&d_weights, in * in * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_activations, in * sizeof(float)));
            
            CUDA_CHECK(cudaMemcpy(d_weights, flat_weights.data(), 
                                 in * in * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_activations, activations[l].data(), 
                                 in * sizeof(float), cudaMemcpyHostToDevice));
            
            // Calculate hidden layer deltas
            hiddenDeltaKernel<<<blocksPerGrid, threadsPerBlock>>>(d_layer_deltas[l+1], d_weights, 
                                                               d_activations, d_layer_deltas[l], 
                                                               in, in);
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
                prev_activations = input;
            } else {
                prev_activations = activations[l-1];
            }
            
            // Flatten weights for current layer
            std::vector<float> flat_weights;
            for (int i = 0; i < in; i++) {
                flat_weights.insert(flat_weights.end(), 
                                   weights[l][i].begin(), 
                                   weights[l][i].end());
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
            } else {
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
            
            // Update weights with L1 regularization
            dim3 blockDim(16, 16);
            dim3 gridDim((in + blockDim.x - 1) / blockDim.x, (in + blockDim.y - 1) / blockDim.y);
            
            updateWeightsL1Kernel<<<gridDim, blockDim>>>(d_weights, d_deltas, d_prev_activations, 
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
            } else {
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


/**
 * @brief CUDA kernel for updating input vector
 * @param input Input vector to be updated
 * @param weights Weights of the first layer
 * @param deltas Deltas of the first layer
 * @param learning_rate Learning rate
 * @param size Size of the arrays
 */
__global__ void updateInputVectorKernel(float* input, float* weights, float* deltas, float learning_rate, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float update = 0.0f;
        for (int i = 0; i < size; i++) {
            update += deltas[i] * weights[i * size + idx];
        }
        input[idx] -= learning_rate * update;
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
                } else {
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

/**
 * @brief CUDA kernel for Rprop weight update
 * @param weights Weights to be updated
 * @param gradients Current gradients
 * @param prev_gradients Previous gradients
 * @param delta_weights Step sizes for each weight
 * @param eta_plus Increase factor
 * @param eta_minus Decrease factor
 * @param delta_max Maximum step size
 * @param delta_min Minimum step size
 * @param size Size of the arrays
 */
__global__ void rpropUpdateKernel(float* weights, float* gradients, float* prev_gradients, 
                                float* delta_weights, float eta_plus, float eta_minus, 
                                float delta_max, float delta_min, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float grad = gradients[idx];
        float prev_grad = prev_gradients[idx];
        float delta = delta_weights[idx];
        
        // Apply Rprop update rule
        if (grad * prev_grad > 0) {
            // Same sign - increase step size
            delta = fminf(delta * eta_plus, delta_max);
            weights[idx] -= copysignf(delta, grad);
            prev_gradients[idx] = grad;
        } 
        else if (grad * prev_grad < 0) {
            // Sign changed - decrease step size
            delta = fmaxf(delta * eta_minus, delta_min);
            // Revert previous step
            weights[idx] += copysignf(delta, prev_grad);
            prev_gradients[idx] = 0; // Set to zero to avoid oscillation
        } 
        else {
            // First iteration or zero gradient
            weights[idx] -= copysignf(delta, grad);
            prev_gradients[idx] = grad;
        }
        
        // Store updated delta
        delta_weights[idx] = delta;
    }
}

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

/**
 * @brief CUDA implementation of the backward function
 * @param in Input size
 * @param layers Number of layers
 * @param learning Learning rate
 */
void mlp::cuBackward(int in, int layers, float learning) {
    // This is an alternative implementation of backpropagation
    // For simplicity, we'll just call the main backpropagation implementation
    cuBackprop(in, layers, learning);
}
