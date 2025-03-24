
// activations and its derivatives in OpenCL
#include "include/activations.hpp"

/**
 * @brief OpenCL kernel for sigmoid function
 * @param x input value
 * @return sigmoid of x
 */
float cl_sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}


/**
 * @brief OpenCL Kernel for Derivative of sigmoid activation function. Calculates the derivative of the sigmoid function.
 *      The derivative of sigmoid(x) is given by: sigmoid_derivative(x) = sigmoid(x) * (1 - sigmoid(x))
 * @param x Input value
 * @return The derivative of sigmoid(x)
 */
float cl_sigmoidder(float x) {
    // Calculate the sigmoid of x
    float s = cl_sigmoid(x);
    // Calculate the derivative of sigmoid(x)
    return s * (1 - s);
}

