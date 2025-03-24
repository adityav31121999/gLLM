
#include "include/mlp.hpp"

/**
 * @brief OpenCL kernel for forward propagation of mlp
 */
__kernel void forwardcl(__global const float* input, __global float* output, __global const float* weights, 
                       __global float* hlayers, __global float* activations, int in, int layers, int input_size,
                       __global const int* layer_offsets, __global const int* weight_offsets) 
{
    int idx = get_global_id(0);
    
    // First hidden layer calculation
    if (idx < in) {
        float sum = 0.0f;
        for (int i = 0; i < input_size; i++) {
            sum += input[i] * weights[weight_offsets[0] + idx * input_size + i];
        }
        hlayers[idx] = sum;
        activations[idx] = sigmoid(sum);
    }
    
    barrier(CLK_GLOBAL_MEM_FENCE);
    
    // Hidden layers calculation
    for (int l = 1; l < layers; l++) {
        if (idx < in) {
            float sum = 0.0f;
            for (int i = 0; i < in; i++) {
                sum += activations[layer_offsets[l-1] + i] * weights[weight_offsets[l] + idx * in + i];
            }
            hlayers[layer_offsets[l] + idx] = sum;
            activations[layer_offsets[l] + idx] = sigmoid(sum);
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
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
