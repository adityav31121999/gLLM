// forprop.cpp: forward propagation functions for mlp
#include "include/mlp.hpp"
#include <numeric>

/**
 * @brief The forward propagation function. This function performs the
 * forward propagation and calculates the activations of each layer.
 */
void mlp::forward() {
    // Ensure vectors are properly initialized
    // assert(hlayers.size() == layers);
    // assert(activations.size() == layers);

    // Calculate activation of the first hidden layer
    for (int i = 0; i < neurons; i++) {
        double sum = std::inner_product(input.begin(), input.end(), iweights[i].begin(), 0.0);
        hlayers[0][i] = sum;
        activations[0][i] = sigmoid(sum); // Apply activation function
    }

    // Calculate activations of the remaining hidden layers
    for (int i = 1; i < layers - 1; i++) {
        for (int j = 0; j < neurons; j++) {
            double sum = std::inner_product(activations[i - 1].begin(), activations[i - 1].end(), weights[i - 1][j].begin(), 0.0);
            hlayers[i][j] = sum;
            activations[i][j] = sigmoid(sum); // Apply activation function
        }
    }

    // Calculate output layer activations
    for (int i = 0; i < out; i++) {
        double sum = std::inner_product(activations[layers - 2].begin(), activations[layers - 2].end(), oweights[i].begin(), 0.0);
        output[i] = sum;
        // output[i] = sigmoid(sum); // Apply activation function to output layer
    }
}

