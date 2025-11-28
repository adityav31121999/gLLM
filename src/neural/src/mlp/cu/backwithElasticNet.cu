#ifdef USE_CU
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
 * @brief CUDA implementation of backpropagation with Elastic Net (L1+L2) regularization
 * @param in Input size (unused, dimensions taken from layer_sizes)
 * @param layers Number of layers (unused, dimensions taken from num_layers)
 * @param learning Learning rate
 */
void mlp::cuBackwithELasticNet(int in, int layers, float learning) {
    // Regularization parameters. Consider making these members of the mlp class.
    float lambda_l1 = this->lambda_l1;
    float lambda_l2 = this->lambda_l2;

    // Device memory pointers
    float *d_weights = nullptr, *d_deltas = nullptr, *d_prev_activations = nullptr, *d_gweights = nullptr;

    try {
        // This function assumes that a backpropagation pass has already computed the error gradients
        // and stored them in the `gweights` matrices.

        for (int l = num_layers - 2; l >= 0; --l) {
            const unsigned int current_layer_size = layer_sizes[l + 1];
            const unsigned int prev_layer_size = layer_sizes[l];

            // Determine previous layer activations (input for first layer)
            std::vector<float> prev_activations;
            if (l == 0) {
                prev_activations = input;
            } else {
                prev_activations = activations[l - 1];
            }

            const mat& current_weights_mat = weights[l];
            mat& current_gweights_mat = gweights[l];

            size_t weight_matrix_bytes = static_cast<size_t>(current_layer_size) * prev_layer_size * sizeof(float);
            size_t prev_activations_bytes = static_cast<size_t>(prev_layer_size) * sizeof(float);

            // Allocate and copy data to device
            CUDA_CHECK(cudaMalloc(&d_weights, weight_matrix_bytes));
            CUDA_CHECK(cudaMalloc(&d_gweights, weight_matrix_bytes));
            CUDA_CHECK(cudaMalloc(&d_prev_activations, prev_activations_bytes));

            CUDA_CHECK(cudaMemcpy(d_weights, current_weights_mat.mapped_data, weight_matrix_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_gweights, current_gweights_mat.mapped_data, weight_matrix_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_prev_activations, prev_activations.data(), prev_activations_bytes, cudaMemcpyHostToDevice));

            // Configure kernel launch
            dim3 blockDim(16, 16);
            dim3 gridDim((prev_layer_size + blockDim.x - 1) / blockDim.x, (current_layer_size + blockDim.y - 1) / blockDim.y);

            kernelUpdateElasticNet<<<gridDim, blockDim>>>(nullptr, d_prev_activations, d_weights, d_gweights, learning, lambda_l1, lambda_l2, current_layer_size, prev_layer_size);
            CUDA_CHECK(cudaGetLastError());

            // Copy updated weights and gradients back to host
            CUDA_CHECK(cudaMemcpy(weights[l].mapped_data, d_weights, weight_matrix_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(gweights[l].mapped_data, d_gweights, weight_matrix_bytes, cudaMemcpyDeviceToHost));

            // Free temporary memory
            CUDA_CHECK(cudaFree(d_weights));
            CUDA_CHECK(cudaFree(d_gweights));
            CUDA_CHECK(cudaFree(d_prev_activations));
            d_weights = d_gweights = d_prev_activations = nullptr;
        }

        float loss = computeLossWithElasticNet(output, expected, *this, lambda_l1, lambda_l2);
        std::cout << "Loss with Elastic Net penalty: " << loss << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "CUDA Exception in Elastic Net regularization: " << e.what() << std::endl;
        cudaFree(d_weights); cudaFree(d_deltas); cudaFree(d_prev_activations); cudaFree(d_gweights);
        throw;
    }
}
#endif