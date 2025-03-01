
#include "include/mlp.hpp"
#include <random>

/**
 * @brief Function to initialize the weights of the multi-layer perceptron.
 * This function initializes the weights of the mlp using a normal distribution
 * with a mean of 0.0 and a standard deviation of 1.0.
 */
void mlp::initializeWeights() {
    // random number generator
    std::random_device rd;      // device
    std::mt19937 gen(rd());     // generator
    std::normal_distribution<double> dis(0.0, 1.0);     // min and max of distribution

    // initialize hidden to hidden weights
    for(int i = 0; i < layers; i++) {
        for(int j = 0; j < neurons; j++) {
            for(int k = 0; k < neurons; k++) {
                weights[i][j][k] = (i+j + dis(gen)) / (k+1);
            }
        }
    }
}
