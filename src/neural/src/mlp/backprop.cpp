
// backprop.cpp: backward propagation functions for mlp
#include "include/mlp.hpp"
#include <iostream>
#include <maths.hpp>

/**
 * @brief Backpropagation till input vector
 */
void mlp::backprop2in(int in, int layers, float learning) {
    // Compute output layer error
    std::vector<float> output_error(in, 0.0);
    for (unsigned int i = 0; i < in; ++i) {
        output_error[i] = output[i] - expected[i];
    }

    // Initialize gradient weights if not already done
    if (gweights.size() != layers) {
        gweights.resize(layers, std::vector<std::vector<float>>(in, std::vector<float>(in, 0.0)));
    }

    // Compute gradients for the output layer
    for (unsigned int i = 0; i < in; ++i) {
        for (unsigned int j = 0; j < in; ++j) {
            gweights[layers - 1][i][j] = output_error[i] * activations[layers - 1][j];
        }
    }

    // Compute hidden layer errors and gradients
    std::vector<std::vector<float>> layer_errors(layers, std::vector<float>(in, 0.0));

    // Compute error for the last hidden layer
    for (unsigned int i = 0; i < in; ++i) {
        float error_sum = 0.0;
        for (unsigned int j = 0; j < in; ++j) {
            error_sum += weights[layers - 1][j][i] * output_error[j];
        }
        layer_errors[layers - 1][i] = error_sum * activations[layers - 1][i] * (1 - activations[layers - 1][i]);
    }

    // Propagate error backward through the network
    for (int l = layers - 2; l >= 0; --l) {
        for (unsigned int i = 0; i < in; ++i) {
            float error_sum = 0.0;
            for (unsigned int j = 0; j < in; ++j) {
                error_sum += weights[l + 1][j][i] * layer_errors[l + 1][j];
            }
            layer_errors[l][i] = error_sum * activations[l][i] * (1 - activations[l][i]);
        }
    }

    // Compute gradients for all layers and update weights
    for (unsigned int l = 0; l < layers; ++l) {
        for (unsigned int i = 0; i < in; ++i) {
            for (unsigned int j = 0; j < in; ++j) {
                if (l == 0) {
                    gweights[l][i][j] = layer_errors[l][i] * input[j];
                    // Update input vector using gradient
                    input[j] -= learning * layer_errors[l][i] * weights[l][i][j];
                } else {
                    gweights[l][i][j] = layer_errors[l][i] * activations[l - 1][j];
                }
                // Update weights
                weights[l][i][j] -= learning * gweights[l][i][j];
            }
        }
    }
}


/**
 * @brief The backward propagation function. This function performs the
 * backward propagation and calculates the error.
 */
void mlp::backward(int in, int layers, float learning) {
    // Initialize vectors
    std::vector<float> error(in);
    std::vector<std::vector<float>> delta_weights(layers, std::vector<float>(in, 0.0));
    std::vector<std::vector<float>> layer_deltas(layers, std::vector<float>(in, 0.0));
    
    // Calculate output layer error
    for(unsigned int i = 0; i < in; i++) {
        error[i] = output[i] - expected[i];
    }

    // Calculate output layer deltas (gradient of error with respect to output)
    for (unsigned int i = 0; i < in; i++) {
        layer_deltas[layers - 1][i] = error[i] * sigmoidder(output[i]);
    }

    // Backpropagate the error through hidden layers
    for (int l = layers - 2; l >= 0; l--) {
        for (unsigned int i = 0; i < in; i++) {
            float error_sum = 0.0;
            for (unsigned int j = 0; j < in; j++) {
                error_sum += layer_deltas[l + 1][j] * weights[l + 1][j][i];
            }
            layer_deltas[l][i] = error_sum * sigmoidder(hlayers[l][i]);
        }
    }

    // Update weights for all layers
    for (unsigned int l = 0; l < layers; l++) {
        for (unsigned int i = 0; i < in; i++) {
            for (unsigned int j = 0; j < (l == 0 ? in : in); j++) {
                if (l == 0) {
                    weights[l][i][j] += learning * layer_deltas[l][i] * input[j];
                }
                else {
                    weights[l][i][j] += learning * layer_deltas[l][i] * hlayers[l - 1][j];
                }
            }
        }
    }

    // Update output layer weights
    for (unsigned int i = 0; i < in; i++) {
        for (unsigned int j = 0; j < in; j++) {
            weights[layers - 1][i][j] += learning * error[i] * hlayers[layers - 1][j];
        }
    }
}

/**
 * @brief Backward propagation with gradients
 */
void mlp::backprop(int in, int layers, float learning) {
    // Compute output layer error
    std::vector<float> output_error(in, 0.0);
    for (unsigned int i = 0; i < in; ++i) {
        output_error[i] = output[i] - expected[i];
    }

    // Initialize gradient weights if not already done
    if (gweights.size() != layers) {
        gweights.resize(layers, std::vector<std::vector<float>>(in, std::vector<float>(in, 0.0)));
    }

    // Compute gradients for the output layer
    for (unsigned int i = 0; i < in; ++i) {
        for (unsigned int j = 0; j < in; ++j) {
            gweights[layers - 1][i][j] = output_error[i] * activations[layers - 1][j];
        }
    }

    // Compute hidden layer errors and gradients
    std::vector<std::vector<float>> layer_errors(layers, std::vector<float>(in, 0.0));

    // Compute error for the last hidden layer
    for (unsigned int i = 0; i < in; ++i) {
        float error_sum = 0.0;
        for (unsigned int j = 0; j < in; ++j) {
            error_sum += weights[layers - 1][j][i] * output_error[j];
        }
        layer_errors[layers - 1][i] = error_sum * activations[layers - 1][i] * (1 - activations[layers - 1][i]);
    }

    // Propagate error backward through the network
    for (int l = layers - 2; l >= 0; --l) {
        for (unsigned int i = 0; i < in; ++i) {
            float error_sum = 0.0;
            for (unsigned int j = 0; j < in; ++j) {
                error_sum += weights[l + 1][j][i] * layer_errors[l + 1][j];
            }
            layer_errors[l][i] = error_sum * activations[l][i] * (1 - activations[l][i]);
        }
    }

    // Compute gradients for all layers and update weights
    for (unsigned int l = 0; l < layers; ++l) {
        for (unsigned int i = 0; i < (l == layers - 1 ? in : in); ++i) {
            for (unsigned int j = 0; j < (l == 0 ? in : in); ++j) {
                if (l == 0) {
                    gweights[l][i][j] = layer_errors[l][i] * input[j];
                } else {
                    gweights[l][i][j] = layer_errors[l][i] * activations[l - 1][j];
                }
                // Update weights
                weights[l][i][j] -= learning * gweights[l][i][j];
            }
        }
    }
}


/**
 * @brief Compute backpropagation with L1 regularization
 */
void mlp::backwithL1(int in, int layers, float learning) {
    float lambda = 0.01; // Regularization parameter

    // Perform standard backpropagation to compute gradients
    backprop(in, layers, learning);

    // Compute L1 penalty
    float l1_penalty = 0.0;
    for (unsigned int l = 0; l < layers; ++l) {
        for (unsigned int i = 0; i < in; ++i) {
            for (unsigned int j = 0; j < in; ++j) {
                l1_penalty += std::abs(weights[l][i][j]);
            }
        }
    }

    // Update weights with L1 regularization
    for (unsigned int l = 0; l < layers; ++l) {
        for (unsigned int i = 0; i < in; ++i) {
            for (unsigned int j = 0; j < in; ++j) {
                float gradient = gweights[l][i][j];
                if (weights[l][i][j] > 0) {
                    weights[l][i][j] -= learning * (lambda + gradient);
                } else {
                    weights[l][i][j] -= learning * (-lambda + gradient);
                }
            }
        }
    }
    
    std::cout << "Into Backprop" << std::endl;
    // Compute loss with L1 penalty
    float loss = computeLossWithL1(output, expected, *this, lambda);
    std::cout << "Loss with L1 penalty: " << loss << std::endl;
}

/**
 * @brief Compute backpropagation with L2 regularization
 */
void mlp::backwithL2(int in, int layers, float learning) {
    float lambda = 0.01; // Regularization parameter

    // Perform standard backpropagation to compute gradients
    backprop(in, layers, learning);

    // Compute L2 penalty
    float l2_penalty = 0.0;
    for (unsigned int l = 0; l < layers; ++l) {
        for (unsigned int i = 0; i < in; ++i) {
            for (unsigned int j = 0; j < in; ++j) {
                l2_penalty += weights[l][i][j] * weights[l][i][j];
            }
        }
    }

    // Update weights with L2 regularization
    for (unsigned int l = 0; l < layers; ++l) {
        for (unsigned int i = 0; i < in; ++i) {
            for (unsigned int j = 0; j < in; ++j) {
                float gradient = gweights[l][i][j];
                weights[l][i][j] -= learning * (lambda * weights[l][i][j] + gradient);
            }
        }
    }

    // Compute loss with L2 penalty
    float loss = computeLossWithL2(output, expected, *this, lambda);
    std::cout << "Loss with L2 penalty: " << loss << std::endl;
}

/**
 * @brief Rprop algorithm for MLP
 * @param dataset Input dataset
 * @note This version only updates the weights and doesn't update the bias
 */
void mlp::rprop(std::vector<std::vector<float>>& dataset, int layers, int in, float learning, int epochs) {
    const float etaPlus = 1.2;     // Increase factor
    const float etaMinus = 0.5;    // Decrease factor
    const float deltaMax = 50.0;   // Maximum update value
    const float deltaMin = 1e-6;   // Minimum update value

    // Initialize gradient and delta weight matrices
    std::vector<std::vector<std::vector<float>>> prev_gradients(layers, std::vector<std::vector<float>>(in, std::vector<float>(in, 0.0)));
    std::vector<std::vector<std::vector<float>>> delta_weights(layers, std::vector<std::vector<float>>(in, std::vector<float>(in, deltaMin)));

    for (unsigned int epoch = 0; epoch < epochs; ++epoch) {
        float totalError = 0.0;
        
        for (const auto& data : dataset) {
            // Set input and perform forward and backward passes
            input = data;
            forward(in, layers);
            backprop(in, layers, learning); // This computes gradients and stores them in gweights
            
            // Compute mean square error
            float error = 0.0;
            for (unsigned int i = 0; i < in; ++i) {
                error += std::pow(expected[i] - output[i], 2);
            }
            error /= in;
            totalError += error;

            // Update weights using Rprop
            for (unsigned int l = 0; l < layers; ++l) {
                for (unsigned int i = 0; i < in; ++i) {
                    for (unsigned int j = 0; j < in; ++j) {
                        float grad = gweights[l][i][j];
                        
                        // Apply Rprop update rule
                        if (grad * prev_gradients[l][i][j] > 0) {
                            // Same sign - increase step size
                            delta_weights[l][i][j] = std::min(delta_weights[l][i][j] * etaPlus, deltaMax);
                            weights[l][i][j] -= std::copysign(delta_weights[l][i][j], grad);
                            prev_gradients[l][i][j] = grad;
                        } 
                        else if (grad * prev_gradients[l][i][j] < 0) {
                            // Sign changed - decrease step size
                            delta_weights[l][i][j] = std::max(delta_weights[l][i][j] * etaMinus, deltaMin);
                            // Revert previous step
                            weights[l][i][j] += std::copysign(delta_weights[l][i][j], prev_gradients[l][i][j]);
                            prev_gradients[l][i][j] = 0; // Set to zero to avoid oscillation
                        } 
                        else {
                            // First iteration or zero gradient
                            weights[l][i][j] -= std::copysign(delta_weights[l][i][j], grad);
                            prev_gradients[l][i][j] = grad;
                        }
                    }
                }
            }
        }

        totalError /= dataset.size();
        std::cout << "Epoch " << epoch + 1 << "/" << epochs << " - Mean Squared Error: " << totalError << std::endl;
        
        if (totalError < 0.01) {
            status = true;
            break;
        }
    }
}
