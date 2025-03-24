
#include "include/mlp.hpp"

/**
 * @brief Forward propagation of the MLP using CUDA
 *  
 */
__global__ void forwardcu(const float* input, float* output, float* weights, float* hlayers, float* activations, 
                int in, int layers, int input_size, int* layer_offsets, int* weight_offsets)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // First hidden layer calculation
    if (idx < in) {
        float sum = 0.0f;
        for (int i = 0; i < input_size; i++) {
            sum += input[i] * weights[weight_offsets[0] + idx * input_size + i];
        }
        hlayers[idx] = sum;
        activations[idx] = cuda_sigmoid(sum);
    }

    // Synchronize threads before calculating next layers
    __syncthreads();

    // Hidden layers calculation
    for (int l = 1; l < layers; l++) {
        if (idx < in) {
            float sum = 0.0f;
            for (int i = 0; i < in; i++) {
                sum += activations[layer_offsets[l-1] + i] * weights[weight_offsets[l] + idx * in + i];
            }
            hlayers[layer_offsets[l] + idx] = sum;
            activations[layer_offsets[l] + idx] = cuda_sigmoid(sum);
        }
        // Synchronize before next layer
        __syncthreads();
    }

    // Output layer calculation
    if (idx < in) {
        float sum = 0.0f;
        for (int i = 0; i < in; i++) {
            sum += activations[layer_offsets[layers-1] + i] * weights[weight_offsets[layers] + idx * in + i];
        }
        output[idx] = sum;
    }
}
