
#include "include/mlp.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <maths.hpp>


/**
 * @brief Calculates the gradient w.r.t. the input of an MLP layer.
 *        grad_input[i] = sum_j (deltas[j] * weights[j * input_size + i])
 * @param deltas Deltas of the current layer (output layer of the conceptual block)
 * @param weights Weights connecting the input to the current layer (Row-major: [current_layer_size x input_size])
 * @param grad_input Gradient w.r.t. the input (to be computed)
 * @param current_layer_size Size of the current layer (number of deltas/rows in weights)
 * @param input_size Size of the input layer (number of columns in weights)
 */
__global__ void kernelComputeGradMLPInput(const float* deltas, const float* weights, float* grad_input,
    int current_layer_size, int input_size)
{
    int input_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (input_idx < input_size) {
        float sum = 0.0f;
        for (int j = 0; j < current_layer_size; ++j) {
            // weights are row-major: weights[row * num_cols + col]
            // Here: row = j, num_cols = input_size, col = input_idx
            sum += deltas[j] * weights[j * input_size + input_idx];
        }
        grad_input[input_idx] = sum;
    }
}


/**
* @brief CUDA kernel for calculating output layer deltas (Sigmoid derivative assumed).
*        delta = (activation - expected) * activation * (1.0f - activation)
* @param activations Activations of the output layer
* @param expected Expected output values
* @param deltas Deltas to be computed for the output layer
* @param size Size of the output layer
*/
__global__ void kernelOutputDelta(const float* activations, const float* expected, float* deltas, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float activation = activations[idx];
        float error = activation - expected[idx];
        float sigmoid_der = activation * (1.0f - activation); // Assuming Sigmoid was used
        deltas[idx] = error * sigmoid_der;
    }
}

/**
* @brief CUDA kernel for calculating hidden layer deltas (Sigmoid derivative assumed)
* @param next_layer_deltas Deltas from the next layer
* @param weights Weights connecting to the next layer (Row-major: [next_layer_size x current_layer_size])
* @param activations Activations of the current layer
* @param deltas Deltas to be computed for the current layer
* @param current_layer_size Size of the current layer
* @param next_layer_size Size of the next layer
*/
__global__ void hiddenDeltaKernel(const float* next_layer_deltas, const float* weights, const float* activations,
float* deltas, int current_layer_size, int next_layer_size)
{
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx < current_layer_size) {
        float error_sum = 0.0f;
        // Sum up the weighted deltas from the next layer
        // weights are row-major: weights[row * num_cols + col]
        // Here: row=j (next layer neuron), num_cols=current_layer_size, col=neuron_idx (current layer neuron)
        for (int j = 0; j < next_layer_size; j++) {
            error_sum += next_layer_deltas[j] * weights[j * current_layer_size + neuron_idx];
        }
        // Calculate delta using the sigmoid derivative
        float sigmoid_der = activations[neuron_idx] * (1.0f - activations[neuron_idx]);
        deltas[neuron_idx] = error_sum * sigmoid_der;
    }
}


// Helper kernel to compute delta for the last MLP layer given the gradient w.r.t. its output
// Assumes Sigmoid activation for the output layer.
__global__ void kernelLastLayerDelta(const float* grad_output, const float* activations, float* deltas, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float activation = activations[idx];
        float sigmoid_der = activation * (1.0f - activation);
        // If ReLU: float relu_der = (activation > 0.0f) ? 1.0f : 0.0f; deltas[idx] = grad_output[idx] * relu_der;
        deltas[idx] = grad_output[idx] * sigmoid_der;
    }
}


/**
 * @brief CUDA kernel for calculating output layer deltas
 * @param outputs Output layer values
 * @param expected Expected output values
 * @param deltas Output deltas to be computed
 * @param size Size of the arrays
 */
__global__ void kernelOutputDelta(float* output, float* expected, float* delta, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;    // index of thread in global context
    if(idx < size) {
        float result = 1 / (1-expf(output[idx]));
        delta[idx] = (output[idx] - expected[idx]) * (result/(1.0f - result));
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
float* deltas, int current_layer_size, int next_layer_size)
{
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx < current_layer_size) {
        float error_sum = 0.0f;
        // Sum up the weighted deltas from the next layer
        // weights are row-major: weights[row * num_cols + col]
        // Here: row=j (next layer neuron), num_cols=current_layer_size, col=neuron_idx (current layer neuron)
        for (int j = 0; j < next_layer_size; j++) {
            error_sum += next_layer_deltas[j] * weights[j * current_layer_size + neuron_idx];
        }
        // Calculate delta using the sigmoid derivative
        // If ReLU was used: float deriv = (activations[neuron_idx] > 0.0f) ? 1.0f : 0.0f;
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
* @brief CUDA kernel for updating weights and calculating gradients
* @param deltas Deltas for the current layer
* @param prev_activations Activations from the previous layer (or inputs for first layer)
* @param weights Weights to be updated (Row-major: [current_layer_size x prev_layer_size])
* @param gradients gradients to be calculated (Row-major: [current_layer_size x prev_layer_size])
* @param learning_rate Learning rate
* @param current_layer_size Size of the current layer
* @param prev_layer_size Size of the previous layer
*/
__global__ void updateWeightsKernel(const float* deltas, const float* prev_activations, float* weights,
    float* gradients, float learning_rate, int current_layer_size, int prev_layer_size)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Index for previous layer neuron (input to weight)
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Index for current layer neuron (output of weight)

    if (row < current_layer_size && col < prev_layer_size) {
        int weight_idx = row * prev_layer_size + col; // row-major index
        float grad = deltas[row] * prev_activations[col];
        if (gradients != nullptr) { // Optionally store gradient
            gradients[weight_idx] = grad;
        }
        weights[weight_idx] -= learning_rate * grad; // Update weight
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
__global__ void updateWeightsL1Kernel(float* weights, float* deltas, float* prev_activations, float learning_rate, 
    float lambda, int current_layer_size, int prev_layer_size) 
{
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
    float learning_rate, float lambda, int current_layer_size, int prev_layer_size) {
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
 * @brief CUDA kernel for updating input vector
 * @param input Input vector to be updated
 * @param weights Weights of the first layer
 * @param deltas Deltas of the first layer
 * @param learning_rate Learning rate
 * @param size Size of the arrays
 */
__global__ void updateInputVectorKernel(float* input, float* weights, float* deltas, float learning_rate, int size) 
{
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
 * @brief kernel for calculating L1 penalty
 * @param[in] weights 3D vector whose penalty to be calculated
 * @param[out] result L1 penalty
 * @param[in] size size of weights
 */
__global__ void l1PenaltyKernel(float* weights, float* result, int size) {
    __shared__ float temp[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread computes absolute value for its element
    temp[tid] = (i < size) ? fabsf(weights[i]) : 0.0f;
    
    __syncthreads();
    
    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            temp[tid] += temp[tid + stride];
        }
        __syncthreads();
    }
    
    // Write the result for this block to global memory
    if (tid == 0) {
        atomicAdd(result, temp[0]);
    }
}

/**
 * @brief kernel for calculating L2 penalty
 * @param[in] weights 3D vector whose penalty to be calculated
 * @param[out] result L2 penalty
 * @param[in] size size of weights
 */
__global__ void l2PenaltyKernel(float* weights, float* result, int size) {
    __shared__ float temp[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread computes square for its element
    temp[tid] = (i < size) ? weights[i] * weights[i] : 0.0f;
    
    __syncthreads();
    
    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            temp[tid] += temp[tid + stride];
        }
        __syncthreads();
    }
    
    // Write the result for this block to global memory
    if (tid == 0) {
        atomicAdd(result, temp[0]);
    }
}

/**
 * @brief kernel to calculate absolute difference
 * @param[in] output original output vector from a process
 * @param[in] targets expected output vector from same process
 * @param[out] result absolute[output[i] - target[i]]
 * @param[in] size size of each vector
 */
__global__ void absDiffKernel(float* outputs, float* targets, float* result, int size) {
    __shared__ float temp[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread computes absolute difference for its element
    temp[tid] = (i < size) ? fabsf(outputs[i] - targets[i]) : 0.0f;
    
    __syncthreads();
    
    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            temp[tid] += temp[tid + stride];
        }
        __syncthreads();
    }
    
    // Write the result for this block to global memory
    if (tid == 0) {
        atomicAdd(result, temp[0]);
    }
}

/**
 * @brief kernel to calculate squared difference
 * @param[in] output original output vector from a process
 * @param[in] targets expected output vector from same process
 * @param[out] result absolute(output[i]^2 - target[i]^2)
 * @param[in] size size of each vector
 */
__global__ void squaredDiffKernel(float* outputs, float* targets, float* result, int size) {
    __shared__ float temp[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread computes squared difference for its element
    if (i < size) {
        float diff = outputs[i] - targets[i];
        temp[tid] = diff * diff;
    } else {
        temp[tid] = 0.0f;
    }
    
    __syncthreads();
    
    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            temp[tid] += temp[tid + stride];
        }
        __syncthreads();
    }
    
    // Write the result for this block to global memory
    if (tid == 0) {
        atomicAdd(result, temp[0]);
    }
}

/**
 * @brief CUDA kernel for calculating the Mean Squared Error (MSE).
 * This kernel computes the squared difference between the expected and actual output for each neuron
 * and accumulates the sum using atomic operations.
 * @param expected Pointer to the expected output data on the device.
 * @param output Pointer to the output data on the device.
 * @param mse Pointer to the MSE value on the device (will be updated).
 * @param size The number of output neurons.
 */
__global__ void cuMSEKernel(float* expected, float* output, float* mse, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    atomicAdd(mse, powf(expected[idx] - output[idx], 2));
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
    float delta_max, float delta_min, int size) 
{
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
