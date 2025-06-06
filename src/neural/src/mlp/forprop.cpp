
// forprop.cpp: forward propagation functions for mlp
#include "include/mlp.hpp"
#include <numeric>
#include <maths.hpp>

#ifdef USE_CPU

/**
 * @brief The forward propagation function. This function performs the
 * forward propagation and calculates the activations of each layer.
 */
void mlp::forward(int in, int layers) {
    if (hlayers.size() != layers || activations.size() != layers) {
        hlayers.resize(layers, std::vector<float>(in, 0.0));
        activations.resize(layers, std::vector<float>(in, 0.0));
    }
    const unsigned int num_input_neurons = input.size();

    for (unsigned int j = 0; j < layer_sizes[1]; ++j) {
        if (!weights[0].mapped_data) {
            throw std::runtime_error("Weights[0] not mapped.");
        }

        float sum = 0.0f;
        for (unsigned int k = 0; k < num_input_neurons; ++k) {
            size_t weight_index = static_cast<size_t>(k) * weights[0].col + j;
            if (weight_index >= weights[0].mapped_size / sizeof(float)) {
                throw std::out_of_range("Weight index out of range.");
            }
            sum += input[k] * weights[0].mapped_data[weight_index];
        }

        hlayers[0][j] = sum;
        activations[0][j] = sigmoid(sum);
    }

    for (unsigned int i = 1; i < layers; i++) {
        if (!weights[i].mapped_data) {
            throw std::runtime_error("Weights[" + std::to_string(i) + "] not mapped.");
        }

        for (unsigned int j = 0; j < layer_sizes[i + 1]; ++j) {
            float sum = 0.0f;
            for (unsigned int k = 0; k < layer_sizes[i]; ++k) {
                size_t weight_index = static_cast<size_t>(k) * weights[i].col + j;

                if (weight_index >= weights[i].mapped_size / sizeof(float)) {
                    throw std::out_of_range("Weight index out of range.");
                }
                sum += activations[i - 1][k] * weights[i].mapped_data[weight_index];
            }
            hlayers[i][j] = sum;
            activations[i][j] = sigmoid(sum);
        }
    }

    output.resize(layer_sizes.back(), 0.0f);
    if (!weights[layers].mapped_data) {
        throw std::runtime_error("Weights[" + std::to_string(layers) + "] not mapped.");
    }

    for (unsigned int i = 0; i < layer_sizes.back(); ++i) {
        float sum = 0.0f;
        for (unsigned int k = 0; k < layer_sizes[layers]; ++k) {
            size_t weight_index = static_cast<size_t>(k) * weights[layers].col + i;

            if (weight_index >= weights[layers].mapped_size / sizeof(float)) {
                throw std::out_of_range("Weight index out of range.");
            }
            sum += activations[layers - 1][k] * weights[layers].mapped_data[weight_index];
        }
        output[i] = sigmoid(sum);
    }
}
#endif