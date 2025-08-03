#ifdef USE_CUDA
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

void mlp::cuAdamUpdate(unsigned long long t_adam_param, float beta1, float beta2, float epsilon, float learning_rate) {
    // Initialize them to nullptr for safety.
    float *d_weights = nullptr;
    float *d_gradients = nullptr;
    float *d_m = nullptr;
    float *d_v = nullptr;
    cudaError_t cuda_err = cudaSuccess; // Initialize for first check

    // Moved blockSize and numBlocks declaration here to avoid goto issues within the loop
    // These will be calculated per layer.
    int blockSize = 0;
    int numBlocks = 0;


    for (size_t l = 0; l < num_layers - 1; ++l) {
        mat& current_weights = weights[l];
        mat& current_gradients = gweights[l]; // Gradients computed in backward pass
        //  mat& m = moments[l]; // First moment
        //  mat& v = velocity[l]; // Second moment

        int total_elements = current_weights.row * current_weights.col;
        if (total_elements == 0) { // Handle empty matrices gracefully
            continue; // Skip to next layer
        }
        size_t matrix_byte_size = total_elements * sizeof(float);

        // Reset device pointers to nullptr for each iteration of the loop
        // before re-allocating to ensure safe cleanup in case of error mid-loop.
        d_weights = nullptr;
        d_gradients = nullptr;
        d_m = nullptr;
        d_v = nullptr;

        // Allocate device memory and copy host data
        cuda_err = cudaMalloc(&d_weights, matrix_byte_size);
        if (cuda_err != cudaSuccess) {
            fprintf(stderr, "CUDA malloc for weights (layer %zu) failed: %s\n", l, cudaGetErrorString(cuda_err));
            goto cleanup_layer; // Use a specific goto for layer cleanup
        }
        cuda_err = cudaMemcpy(d_weights, current_weights.mapped_data, matrix_byte_size, cudaMemcpyHostToDevice);
        if (cuda_err != cudaSuccess) {
            fprintf(stderr, "CUDA memcpyH2D for weights (layer %zu) failed: %s\n", l, cudaGetErrorString(cuda_err));
            goto cleanup_layer;
        }

        cuda_err = cudaMalloc(&d_gradients, matrix_byte_size);
        if (cuda_err != cudaSuccess) {
            fprintf(stderr, "CUDA malloc for gradients (layer %zu) failed: %s\n", l, cudaGetErrorString(cuda_err));
            goto cleanup_layer;
        }
        cuda_err = cudaMemcpy(d_gradients, current_gradients.mapped_data, matrix_byte_size, cudaMemcpyHostToDevice);
        if (cuda_err != cudaSuccess) {
            fprintf(stderr, "CUDA memcpyH2D for gradients (layer %zu) failed: %s\n", l, cudaGetErrorString(cuda_err));
            goto cleanup_layer;
        }

        cuda_err = cudaMalloc(&d_m, matrix_byte_size);
        if (cuda_err != cudaSuccess) {
            fprintf(stderr, "CUDA malloc for moments (layer %zu) failed: %s\n", l, cudaGetErrorString(cuda_err));
            goto cleanup_layer;
        }

        cuda_err = cudaMalloc(&d_v, matrix_byte_size);
        if (cuda_err != cudaSuccess) {
            fprintf(stderr, "CUDA malloc for velocity (layer %zu) failed: %s\n", l, cudaGetErrorString(cuda_err));
            goto cleanup_layer;
        }

        // Determine grid and block dimensions (now declared at the top of the loop's scope)
        blockSize = 256; // Typical block size, adjust for performance
        numBlocks = (total_elements + blockSize - 1) / blockSize;

        // Launch kernel (CORRECTED KERNEL NAME)
        adam_optimizer_kernel_cuda<<<numBlocks, blockSize>>>(d_weights, d_gradients, d_m, d_v,
                                                             learning_rate, beta1, beta2, epsilon,
                                                             t_adam_param, // Pass the global time step directly
                                                             total_elements);

        // Synchronize and check for errors after kernel launch
        cuda_err = cudaGetLastError();
        if (cuda_err != cudaSuccess) {
            fprintf(stderr, "CUDA kernel launch (layer %zu) failed: %s\n", l, cudaGetErrorString(cuda_err));
            goto cleanup_layer;
        }
        cuda_err = cudaDeviceSynchronize(); // Synchronize to ensure kernel completion before copying back
        if (cuda_err != cudaSuccess) {
            fprintf(stderr, "CUDA device synchronize (layer %zu) failed: %s\n", l, cudaGetErrorString(cuda_err));
            goto cleanup_layer;
        }

        // Copy results back to host memory (mat objects)
        cuda_err = cudaMemcpy(current_weights.mapped_data, d_weights, matrix_byte_size, cudaMemcpyDeviceToHost);
        if (cuda_err != cudaSuccess) {
            fprintf(stderr, "CUDA memcpyD2H for weights (layer %zu) failed: %s\n", l, cudaGetErrorString(cuda_err));
            goto cleanup_layer;
        }

    cleanup_layer: // Label for cleaning up resources for the *current* layer
        // Free device memory
        if (d_weights) cudaFree(d_weights);
        if (d_gradients) cudaFree(d_gradients);
        if (d_m) cudaFree(d_m);
        if (d_v) cudaFree(d_v);
        if (cuda_err != cudaSuccess) {
            std::cerr << "Error in layer " << l << ", stopping cuAdamUpdate for this MLP." << std::endl;
            break; // Exit the for loop for layers
        }
    } // End of for loop for layers
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

#endif