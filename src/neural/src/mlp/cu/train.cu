
#include "include/mlp.hpp"
#include <iostream>
#include <vector>
#include <maths.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Helper macro for CUDA error checking
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        throw std::runtime_error(cudaGetErrorString(err)); \
    } \
} while (0)


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
 * @brief Trains the MLP using CUDA for a single input-output pair.
 * This function performs forward and backward propagation on the GPU to train the MLP.
 * It calculates the MSE and continues training until the MSE is below a threshold.
 * @param mse Reference to the Mean Squared Error (MSE) value.
 * @param in Number of input neurons.
 * @param layers Number of layers in the MLP.
 * @param learning The learning rate.
 */
void mlp::cuTrain(float& mse, int in, int layers, float learning) {
    unsigned int e = 0;
    while (true) {
        cuForward(in, layers);

        // Calculate MSE on GPU
        float* d_mse;
        CUDA_CHECK(cudaMalloc((void**)&d_mse, sizeof(float)));
        CUDA_CHECK(cudaMemset(d_mse, 0, sizeof(float)));

        float* d_expected;
        CUDA_CHECK(cudaMalloc((void**)&d_expected, expected.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_expected, expected.data(), expected.size() * sizeof(float), cudaMemcpyHostToDevice));

        float* d_output;
        CUDA_CHECK(cudaMalloc((void**)&d_output, output.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_output, output.data(), output.size() * sizeof(float), cudaMemcpyHostToDevice));

        int size = output.size();
        int blockSize = 256;
        int gridSize = (size + blockSize - 1) / blockSize;
        cuMSEKernel<<<gridSize, blockSize>>>(d_expected, d_output, d_mse, size);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(&mse, d_mse, sizeof(float), cudaMemcpyDeviceToHost));
        mse /= output.size();

        CUDA_CHECK(cudaFree(d_mse));
        CUDA_CHECK(cudaFree(d_expected));
        CUDA_CHECK(cudaFree(d_output));

        if (mse < 1e-6)
            break;
        std::cout << "Rep. NO.:" << e << " Errors: " << mse << std::endl;
        cuBackward(in, layers, learning);
        e++;
    }
    cuForward(in, layers);
}


/**
 * @brief Trains the MLP using CUDA for a set of input-output pairs.
 * This function performs forward and backward propagation on the GPU for each input in the
 * provided set. It calculates the average MSE across all inputs and continues training
 * until the average MSE is below a threshold.
 * @param inputs A vector of input vectors.
 * @param mse Reference to the Mean Squared Error (MSE) value.
 * @param in Number of input neurons.
 * @param layers Number of layers in the MLP.
 * @param learning The learning rate.
 */
void mlp::cuTrain(std::vector<std::vector<float>>& inputs, float& mse, int in, int layers, float learning) {
    unsigned int e = 0;
    float total_mse = 0.0;
    while (true) {
        total_mse = 0.0;
        for (const auto& single_input : inputs) {
            // Set the current input
            input = single_input;
            // Perform forward propagation
            cuForward(in, layers);

            // Calculate MSE on GPU
            float* d_mse;
            CUDA_CHECK(cudaMalloc((void**)&d_mse, sizeof(float)));
            CUDA_CHECK(cudaMemset(d_mse, 0, sizeof(float)));

            float* d_expected;
            CUDA_CHECK(cudaMalloc((void**)&d_expected, expected.size() * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_expected, expected.data(), expected.size() * sizeof(float), cudaMemcpyHostToDevice));

            float* d_output;
            CUDA_CHECK(cudaMalloc((void**)&d_output, output.size() * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_output, output.data(), output.size() * sizeof(float), cudaMemcpyHostToDevice));

            int size = output.size();
            int blockSize = 256;
            int gridSize = (size + blockSize - 1) / blockSize;
            cuMSEKernel<<<gridSize, blockSize>>>(d_expected, d_output, d_mse, size);
            CUDA_CHECK(cudaDeviceSynchronize());

            float current_mse;
            CUDA_CHECK(cudaMemcpy(&current_mse, d_mse, sizeof(float), cudaMemcpyDeviceToHost));
            current_mse /= output.size();
            total_mse += current_mse;

            CUDA_CHECK(cudaFree(d_mse));
            CUDA_CHECK(cudaFree(d_expected));
            CUDA_CHECK(cudaFree(d_output));

            // Perform backward propagation
            cuBackward(in, layers, learning);
        }
        e++;
        // Calculate average MSE for the epoch
        total_mse /= inputs.size();
        std::cout << "Epoch " << e << " Average MSE: " << total_mse << std::endl;
        if (total_mse < 1e-7)
            break;
    }
    mse = total_mse;
}


/**
 * @brief Validates the MLP using CUDA. This function performs forward propagation on the 
 * GPU using validation data and calculates the MSE.
 * @param in Number of input neurons.
 * @param layers Number of layers in the MLP.
 */
void mlp::cuValidate(int in, int layers) {
    // Assuming validation data is available in some form
    std::vector<float> validation_input(in, 0.0);      // Replace with actual validation input
    std::vector<float> validation_expected(in, 0.0);  // Replace with actual expected output
    // Set the input and expected output for validation
    input = validation_input;
    expected = validation_expected;
    // Perform forward propagation
    cuForward(in, layers);
    // Calculate mean squared error
    float mse = 0.0;

    float* d_mse;
    CUDA_CHECK(cudaMalloc((void**)&d_mse, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_mse, 0, sizeof(float)));

    float* d_expected;
    CUDA_CHECK(cudaMalloc((void**)&d_expected, expected.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_expected, expected.data(), expected.size() * sizeof(float), cudaMemcpyHostToDevice));

    float* d_output;
    CUDA_CHECK(cudaMalloc((void**)&d_output, output.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_output, output.data(), output.size() * sizeof(float), cudaMemcpyHostToDevice));

    int size = output.size();
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    cuMSEKernel<<<gridSize, blockSize>>>(d_expected, d_output, d_mse, size);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&mse, d_mse, sizeof(float), cudaMemcpyDeviceToHost));
    mse /= output.size();

    CUDA_CHECK(cudaFree(d_mse));
    CUDA_CHECK(cudaFree(d_expected));
    CUDA_CHECK(cudaFree(d_output));

    std::cout << "Validation MSE: " << mse << std::endl;
}


/**
 * @brief Tests the MLP using CUDA. This function performs forward propagation on 
 * the GPU using test data and outputs the results.
 * @param in Number of input neurons.
 * @param layers Number of layers in the MLP.
 */
void mlp::cuTest(int in, int layers) {
    // Assuming test data is available in some form
    std::vector<float> test_input(in, 0.0);        // Replace with actual test input
    std::vector<float> test_expected(in, 0.0);    // Replace with actual expected output
    // Set the input and expected output for testing
    input = test_input;
    expected = test_expected;
    // Perform forward propagation
    cuForward(in, layers);
    // Output the results
    std::cout << "Expected " << "<-> Output" << std::endl;
    std::cout << "Test Results:" << std::endl;
    for (size_t i = 0; i < output.size(); ++i) {
        std::cout << expected[i] << " <-> " << output[i] << std::endl;
    }
}
