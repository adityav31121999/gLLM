
// backprop.cpp: backward propagation functions for mlp
#include "include/mlp.hpp"
#include <numeric>
#include <iostream>

/**
 * @brief The backward propagation function. This function performs the
 * backward propagation and calculates the error.
 */
void mlp::backward() {
    // Initialize vectors
    std::vector<double> error(out);
    std::vector<std::vector<double>> ow(out, std::vector<double>(neurons, 0.0));
    std::vector<double> dh1(neurons, 0.0);  // Initialize to 0
    
    // Calculate output layer error
    for(int i = 0; i < out; i++) {
        error[i] = expected[i] - output[i];
    }

    // Calculate gradients for output weights
    for(int i = 0; i < out; i++) {
        for(int j = 0; j < neurons; j++) {
            ow[i][j] = error[i] * activations[layers-2][j];
        }
    }

    // Calculate hidden layer deltas
    for(int i = 0; i < neurons; i++) {
        double sum = 0.0;
        for(int j = 0; j < out; j++) {
            sum += error[j] * oweights[j][i];
        }
        dh1[i] = sum * sigmoidder(hlayers[layers-2][i]);  // Corrected index
    }

    // Update output weights
    for(int i = 0; i < out; i++) {
        for(int j = 0; j < neurons; j++) {
            oweights[i][j] += learning * ow[i][j];
        }
    }

    // Handle hidden layers
    std::vector<std::vector<std::vector<double>>> dweights(layers - 1,
        std::vector<std::vector<double>>(neurons, std::vector<double>(neurons, 0.0)));
    std::vector<double> layer_error = dh1;

    for(int i = layers-2; i >= 0; i--) {  // Corrected loop condition
        std::vector<double> next_error(neurons, 0.0);
        
        // Calculate weight updates
        for(int j = 0; j < neurons; j++) {
            for(int k = 0; k < neurons; k++) {
                dweights[i][j][k] = layer_error[k] * activations[i][j];
            }
        }

        // Calculate error for next layer back
        for(int j = 0; j < neurons; j++) {
            double sum = 0.0;
            for(int k = 0; k < neurons; k++) {
                sum += layer_error[k] * weights[i][k][j];
            }
            next_error[j] = sum * sigmoidder(hlayers[i][j]);
        }
        
        layer_error = next_error;
    }

    // Update hidden layer weights
    for(int i = layers-2; i >= 0; i--) {  // Corrected loop condition
        for(int j = 0; j < neurons; j++) {
            for(int k = 0; k < neurons; k++) {
                weights[i][j][k] += learning * dweights[i][j][k];
            }
        }
    }

    // Update input weights
    for(int i = 0; i < neurons; i++) {
        for(int j = 0; j < in; j++) {
            iweights[i][j] += learning * layer_error[i];
        }
    }

    // Reset activations and hlayers
    for(int i = 0; i < layers; i++) {
        std::fill(activations[i].begin(), activations[i].end(), 0.0);
        std::fill(hlayers[i].begin(), hlayers[i].end(), 0.0);
    }
    std::fill(output.begin(), output.end(), 0.0);
}


/**
 * @brief Backpropagation with gradients
 */
void mlp::backprop() {
    // Compute output layer error
    std::vector<double> output_error(out, 0.0);
    for (unsigned int i = 0; i < out; ++i) {
        output_error[i] = output[i] - expected[i];
    }
    // Compute output layer gradients
    for (unsigned int i = 0; i < out; ++i) {
        for (unsigned int j = 0; j < neurons; ++j) {
            goweights[i][j] = output_error[i] * activations[layers - 1][j];
        }
    }

    // Compute hidden layer gradients
    std::vector<double> hidden_error(neurons, 0.0);
    for (unsigned int i = 0; i < neurons; ++i) {
        double error_sum = 0.0;
        for (unsigned int j = 0; j < out; ++j) {
            error_sum += oweights[j][i] * output_error[j];
        }
        hidden_error[i] = error_sum * activations[layers - 1][i] * (1 - activations[layers - 1][i]);
    }
    
    // Compute gradients for the hidden layer
    for (unsigned int i = 0; i < neurons; ++i) {
        for (unsigned int j = 0; j < in; ++j) {
            giweights[i][j] = hidden_error[i] * input[j];
        }
    }
    
    // Propagate error backward through the network
    for (int l = layers - 2; l >= 0; --l) {
        std::vector<double> layer_error(neurons, 0.0);
        for (unsigned int i = 0; i < neurons; ++i) {
            double error_sum = 0.0;
            for (unsigned int j = 0; j < (neurons); ++j) {
                error_sum += weights[l][j][i] * gweights[l][j][i];
            }
            layer_error[i] = error_sum * activations[l][i] * (1 - activations[l][i]);
        }
        
        // Compute gradients for the current layer
        for (unsigned int i = 0; i < neurons; ++i) {
            for (unsigned int j = 0; j < neurons; ++j) {
                gweights[l][i][j] = layer_error[i] * activations[l][j];
            }
        }
    }
}


void mlp::backwithL1() {
    double lambda = 0.01; // Regularization parameter

    // Perform standard backpropagation to compute gradients
    backprop();

    // Compute L1 penalty
    double l1_penalty = 0.0;
    for (unsigned int l = 0; l < layers-1; ++l) {
        for (unsigned int i = 0; i < neurons; ++i) {
            for (unsigned int j = 0; j < neurons; ++j) {
                l1_penalty += std::abs(weights[l][i][j]);
            }
        }
    }
    for (unsigned int i = 0; i < neurons; ++i) {
        for (unsigned int j = 0; j < in; ++j) {
            l1_penalty += std::abs(iweights[i][j]);
        }
    }
    for (unsigned int i = 0; i < out; ++i) {
        for (unsigned int j = 0; j < neurons; ++j) {
            l1_penalty += std::abs(oweights[i][j]);
        }
    }

    // Update weights with L1 regularization
    for (unsigned int l = 0; l < layers-1; ++l) {
        for (unsigned int i = 0; i < neurons; ++i) {
            for (unsigned int j = 0; j < neurons; ++j) {
                double gradient = gweights[l][i][j];
                if (weights[l][i][j] > 0) {
                    weights[l][i][j] -= learning * (lambda + gradient);
                } else {
                    weights[l][i][j] -= learning * (-lambda + gradient);
                }
            }
        }
    }
    for (unsigned int i = 0; i < neurons; ++i) {
        for (unsigned int j = 0; j < in; ++j) {
            double gradient = giweights[i][j];
            if (iweights[i][j] > 0) {
                iweights[i][j] -= learning * (lambda + gradient);
            } else {
                iweights[i][j] -= learning * (-lambda + gradient);
            }
        }
    }
    for (unsigned int i = 0; i < out; ++i) {
        for (unsigned int j = 0; j < neurons; ++j) {
            double gradient = goweights[i][j];
            if (oweights[i][j] > 0) {
                oweights[i][j] -= learning * (lambda + gradient);
            } else {
                oweights[i][j] -= learning * (-lambda + gradient);
            }
        }
    }
    std::cout << "Into Backprop" << std::endl;
    // Compute loss with L1 penalty
    double loss = computeLossWithL1(output, expected, *this, lambda);
    std::cout << "Loss with L1 penalty: " << loss << std::endl;
}

void mlp::backwithL2() {
    double lambda = 0.01; // Regularization parameter

    // Perform standard backpropagation to compute gradients
    backprop();

    // Compute L2 penalty
    double l2_penalty = 0.0;
    for (unsigned int l = 0; l < layers-1; ++l) {
        for (unsigned int i = 0; i < neurons; ++i) {
            for (unsigned int j = 0; j < neurons; ++j) {
                l2_penalty += weights[l][i][j] * weights[l][i][j];
            }
        }
    }
    for (unsigned int i = 0; i < neurons; ++i) {
        for (unsigned int j = 0; j < in; ++j) {
            l2_penalty += iweights[i][j] * iweights[i][j];
        }
    }
    for (unsigned int i = 0; i < out; ++i) {
        for (unsigned int j = 0; j < neurons; ++j) {
            l2_penalty += oweights[i][j] * oweights[i][j];
        }
    }

    // Update weights with L2 regularization
    for (unsigned int l = 0; l < layers-1; ++l) {
        for (unsigned int i = 0; i < neurons; ++i) {
            for (unsigned int j = 0; j < neurons; ++j) {
                double gradient = gweights[l][i][j];
                weights[l][i][j] -= learning * (lambda * weights[l][i][j] + gradient);
            }
        }
    }
    for (unsigned int i = 0; i < neurons; ++i) {
        for (unsigned int j = 0; j < in; ++j) {
            double gradient = giweights[i][j];
            iweights[i][j] -= learning * (lambda * iweights[i][j] + gradient);
        }
    }
    for (unsigned int i = 0; i < out; ++i) {
        for (unsigned int j = 0; j < neurons; ++j) {
            double gradient = goweights[i][j];
            oweights[i][j] -= learning * (lambda * oweights[i][j] + gradient);
        }
    }

    // Compute loss with L2 penalty
    double loss = computeLossWithL2(output, expected, *this, lambda);
    std::cout << "Loss with L2 penalty: " << loss << std::endl;
}


/**
 * @brief Rprop algorithm for MLP
 * @param dataset Input dataset
 * @note This version only updates the weights and doesn't update the bias
 */
void mlp::rprop(std::vector<std::vector<double>> dataset) {
    const double etaPlus = 1.2;     // Increase factor
    const double etaMinus = 0.5;    // Decrease factor
    const double deltaMax = 50.0;   // Maximum update value
    const double deltaMin = 1e-6;   // Minimum update value

    std::vector<double> outputError(out, 0.0);
    std::vector<double> hiddenError(neurons, 0.0);
    std::vector<std::vector<double>> gradients(out, std::vector<double>(neurons, 0.0));
    std::vector<std::vector<double>> deltaWeights(out, std::vector<double>(neurons, deltaMin));

    for (unsigned int epoch = 0; epoch < epochs; ++epoch) {
        double totalError = 0.0;
        for (const auto& data : dataset) {
            input = data;
            forward();
            backward();

            // Compute mean square error
            double error = 0.0;
            for (unsigned int i = 0; i < out; ++i) {
                outputError[i] = (expected[i] - output[i]);
                error += std::pow(expected[i] - output[i], 2);
            }
            error /= out;
            totalError += error;

            // Update weights using Rprop
            for (unsigned int i = 0; i < out; ++i) {
                for (unsigned int j = 0; j < neurons; ++j) {
                    double grad = outputError[i] * hlayers[0][j];
                    if (grad * gradients[i][j] > 0) {
                        deltaWeights[i][j] = std::min(deltaWeights[i][j] * etaPlus, deltaMax);
                        oweights[i][j] += (grad > 0 ? -deltaWeights[i][j] : deltaWeights[i][j]);
                        gradients[i][j] = grad;
                    } else if (grad * gradients[i][j] < 0) {
                        deltaWeights[i][j] = std::max(deltaWeights[i][j] * etaMinus, deltaMin);
                        oweights[i][j] += (grad > 0 ? -deltaWeights[i][j] : deltaWeights[i][j]);
                        gradients[i][j] = 0;
                    } else {
                        oweights[i][j] += (grad > 0 ? -deltaWeights[i][j] : deltaWeights[i][j]);
                        gradients[i][j] = grad;
                    }
                }
            }
            for (unsigned int i = 0; i < neurons; ++i) {
                for (unsigned int j = 0; j < in; ++j) {
                    double grad = hiddenError[i] * input[j];
                    if (grad * gradients[i][j] > 0) {
                        deltaWeights[i][j] = std::min(deltaWeights[i][j] * etaPlus, deltaMax);
                        iweights[i][j] += (grad > 0 ? -deltaWeights[i][j] : deltaWeights[i][j]);
                        gradients[i][j] = grad;
                    } else if (grad * gradients[i][j] < 0) {
                        deltaWeights[i][j] = std::max(deltaWeights[i][j] * etaMinus, deltaMin);
                        iweights[i][j] += (grad > 0 ? -deltaWeights[i][j] : deltaWeights[i][j]);
                        gradients[i][j] = 0;
                    } else {
                        iweights[i][j] += (grad > 0 ? -deltaWeights[i][j] : deltaWeights[i][j]);
                        gradients[i][j] = grad;
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
