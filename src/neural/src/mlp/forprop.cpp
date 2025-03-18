
// forprop.cpp: forward propagation functions for mlp
#include "include/mlp.hpp"
#include <numeric>
#include <maths.hpp>

/**
 * @brief The forward propagation function. This function performs the
 * forward propagation and calculates the activations of each layer.
 */
void mlp::forward(int in, int layers) {
    // Ensure vectors are properly initialized
    if (hlayers.size() != layers || activations.size() != layers) {
        hlayers.resize(layers, std::vector<double>(in, 0.0));
        activations.resize(layers, std::vector<double>(in, 0.0));
    }

    // Calculate activation of the first hidden layer (input to hidden)
    for (unsigned int j = 0; j < in; j++) {
        double sum = std::inner_product(input.begin(), input.end(), weights[0][j].begin(), 0.0);
        hlayers[0][j] = sum;
        activations[0][j] = sigmoid(sum); // Apply activation function
    }

    // Calculate activations for the remaining hidden layers
    for (unsigned int i = 1; i < layers; i++) {
        for (unsigned int j = 0; j < in; j++) {
            double sum = std::inner_product(activations[i-1].begin(), activations[i-1].end(), weights[i][j].begin(), 0.0);
            hlayers[i][j] = sum;
            activations[i][j] = sigmoid(sum); // Apply activation function
        }
    }

    // Calculate output layer values (hidden to output)
    output.resize(in, 0.0);
    for (unsigned int i = 0; i < in; i++) {
        output[i] = std::inner_product(activations[layers-1].begin(), activations[layers-1].end(), 
                                      weights[layers][i].begin(), 0.0);
        // No activation function applied to the output layer
    }
}
