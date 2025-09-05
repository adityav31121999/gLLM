
// train.cpp: Training, Validation and Testing Functions for MLP
#include "include/mlp.hpp"
#include <iostream>
#include <vector>
#include <maths.hpp>

#ifdef USE_CPU

/**
 * @brief Training fucntion for MLP (error threshold: 10^-6)
 * @param mse Mean Squared Error
 * @param in Number of inputs
 * @param layers Number of layers
 * @param learning Learning rate
 */
void mlp::train(float& mse, int in, int layers, float learning) {
    unsigned int e = 0;
    while (1) {
        forward(in, layers);
        mse = MSE(expected, output);
        if(mse < 1e-6)
            break;
        std::cout << "Rep. NO.:" << e << " Errors: " << mse << std::endl;
        backward(in, layers, learning);
        e++;
    }
    forward(in, layers);
}

/**
 * @brief Training function using multiple inputs for MLP 
 * (error threshold: 10^-6)
 * @param inputs 2D vector of Multiple Inputs
 * @param mse Mean Squared Error
 * @param in Number of inputs
 * @param layers Number of layers
 * @param learning Learning rate
 */
void mlp::train(std::vector<std::vector<float>>& inputs, float& mse, int in, int layers, float learning) {
    unsigned int e = 0;
    float total_mse = 0.0;
    while (1) {
        for (const auto& single_input : inputs) {
            input = single_input;
            forward(in, layers);
            float current_mse = 0.0;
            for (size_t i = 0; i < output.size(); ++i) {
                current_mse += std::pow(expected[i] - output[i], 2);
            }
            current_mse /= output.size();
            total_mse += current_mse;
            backward(in, layers, learning);
        }
        e++;
        total_mse /= inputs.size();
        std::cout << "Epoch " << e << " Average MSE: " << total_mse << std::endl;
        if(total_mse > 1e-7) 
            break;
    }
    mse = total_mse;
}

/**
 * @brief Validation function for MLP
 * @param in Number of inputs
 * @param layers Number of layers
 */
void mlp::validate(int in, int layers) {
    std::vector<float> validation_input(in, 0.0);
    std::vector<float> validation_expected(in, 0.0);
    input = validation_input;
    expected = validation_expected;
    forward(in, layers);
    float mse = 0.0;
    for (size_t i = 0; i < output.size(); ++i) {
        mse += std::pow(expected[i] - output[i], 2);
    }
    mse /= output.size();
    std::cout << "Validation MSE: " << mse << std::endl;
}

/**
 * @brief Testing function for MLP
 * @param in Number of inputs
 * @param layers Number of layers
 */
void mlp::test(int in, int layers) {
    std::vector<float> test_input(in, 0.0);
    std::vector<float> test_expected(in, 0.0);
    input = test_input;
    expected = test_expected;
    forward(in, layers);
    std::cout << "Expected " << "<-> Output" << std::endl;
    std::cout << "Test Results:" << std::endl;
    for (size_t i = 0; i < output.size(); ++i) {
        std::cout << expected[i] << " <-> " << output[i] << std::endl;
    }
}

#endif
