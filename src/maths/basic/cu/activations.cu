
// activations and its derivative functions
#include "include/basic.hpp"

/**
 * @brief CUDA kernel for the sigmoid activation function.
 * @param x The input value.
 * @return The output value.
 */
__device__ float cuda_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}


/**
 * @brief Cuda Kernel for Derivative of sigmoid activation function. Calculates the derivative of the sigmoid function.
 *      The derivative of sigmoid(x) is given by: sigmoid_derivative(x) = sigmoid(x) * (1 - sigmoid(x))
 * @param x Input value
 * @return The derivative of sigmoid(x)
 */
__device__ float cuda_sigmoidder(float x) {
    // Calculate the sigmoid of x
    float s = cuda_sigmoid(x);
    // Calculate the derivative of sigmoid(x)
    return s * (1 - s);
}


