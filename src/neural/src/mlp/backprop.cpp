#ifdef USE_CPU

// backprop.cpp: backward propagation functions for mlp
#include "include/mlp.hpp"
#include <iostream>
#include <vector>
#include <maths.hpp>


/**
 * @brief The backward propagation function. This function performs the
 * backward propagation and calculates the error.
 * @param in dimension of input, output and size of weight layers
 * @param layers_param number of weight layers (Note: This parameter is now unused, logic relies on class members)
 * @param learning learning rate for mlp
 */
void mlp::backward(int in_param, int layers_param, float learning) {
    // Parameters in_param and layers_param are ignored in favor of class members
    // num_layers and layer_sizes for robustness and consistency.

    if (num_layers < 2) return;

    // Initialize deltas for each layer
    std::vector<std::vector<float>> layer_deltas(num_layers);
    for(size_t i = 0; i < num_layers; ++i) {
        layer_deltas[i].resize(layer_sizes[i], 0.0f);
    }

    // Calculate deltas for the output layer
    // Delta = (output - expected) * f'(output_activation)
    size_t output_layer_idx = num_layers - 1;
    for (unsigned int i = 0; i < layer_sizes[output_layer_idx]; ++i) {
        // Assuming output[i] is the activated value from the output layer
        // and sigmoidder(activated_value) computes activated_value * (1 - activated_value)
        layer_deltas[output_layer_idx][i] = (this->output[i] - this->expected[i]) * sigmoidder(this->activations[output_layer_idx][i]);
    }

    // Calculate deltas for hidden layers (from right to left)
    for (int l = num_layers - 2; l >= 1; --l) { // Iterate from the second to last layer down to the first hidden layer
        size_t current_layer_size = layer_sizes[l];
        size_t next_layer_size = layer_sizes[l+1];
        const mat& weights_to_next_layer = weights[l]; // weights[l] connects layer l to l+1
        const std::vector<float>& current_hidden_layer_activations = activations[l];

        for (unsigned int i = 0; i < current_layer_size; ++i) { // For each neuron 'i' in current hidden layer 'l'
            float error_sum = 0.0;
            for (unsigned int k = 0; k < next_layer_size; ++k) { // For each neuron 'k' in next layer 'l+1'
                error_sum += layer_deltas[l + 1][k] * weights_to_next_layer(k, i);
            }
            layer_deltas[l][i] = error_sum * sigmoidder(current_hidden_layer_activations[i]);
        }
    }

    // Update weights for all layers
    for (unsigned int l = 0; l < num_layers - 1; ++l) {
        size_t to_layer_size = layer_sizes[l+1];   // Number of neurons in the layer weights[l] projects TO (layer l+1)
        size_t from_layer_size = layer_sizes[l]; // Number of neurons in the layer weights[l] projects FROM (layer l)
        mat& current_weights = weights[l];
        const std::vector<float>& from_layer_activations = activations[l]; // Activations of layer 'l'

        for (unsigned int i = 0; i < to_layer_size; ++i) { // Neuron 'i' in layer 'l+1'
            for (unsigned int j = 0; j < from_layer_size; ++j) { // Neuron 'j' in layer 'l'
                // Weight W_ij connects neuron j (from_layer) to neuron i (to_layer)
                current_weights(i, j) -= learning * layer_deltas[l + 1][i] * from_layer_activations[j];
            }
        }
    }
}


/**
 * @brief Backward propagation with gradients, uses gradients to update weights.
 * @param in dimension of input and output vectors, size of each square layer
 * @param layers number of activation layers
 * @param learning learning rate for mlp
 */
void mlp::backprop(int in, int layers, float learning) {
    if (num_layers < 2) 
        return; // Need at least input and output layer

    // --- Initialize deltas for each layer ---
    std::vector<std::vector<float>> layer_deltas(num_layers);
    for(size_t i = 0; i < num_layers; ++i) {
        layer_deltas[i].resize(layer_sizes[i], 0.0f);
    }

    // --- Calculate deltas for the output layer ---
    // Delta = (output - expected) * f'(output_activation)
    // Assuming f'(x) = x * (1-x) for sigmoid, where x is the activated output.
    size_t output_layer_idx = num_layers - 1;
    size_t output_size = layer_sizes[output_layer_idx];
    for (unsigned int i = 0; i < output_size; ++i) {
        // 'this->output' contains the final activations of the output layer
        // and 'this->expected' contains the target values.
        // The derivative of sigmoid (if output is sigmoid activated) is output * (1 - output)
        layer_deltas[output_layer_idx][i] = (this->output[i] - this->expected[i]) * this->activations[output_layer_idx][i] * (1.0f - this->activations[output_layer_idx][i]);
    }

    // --- Calculate deltas for hidden layers (from right to left) ---
    for (int l = num_layers - 2; l >= 1; --l) {
        // 'l' is the index of the current hidden layer
        size_t current_layer_size = layer_sizes[l];
        size_t next_layer_size = layer_sizes[l+1];
        const mat& weights_to_next_layer = weights[l]; // weights[l] connects layer l to l+1
        const std::vector<float>& current_hidden_layer_activations = activations[l]; // Activations of the current hidden layer 'l'

        for (unsigned int i = 0; i < current_layer_size; ++i) {
            float error_sum = 0.0;
            // Sum of (delta_of_neuron_k_in_next_layer * weight_from_neuron_i_to_neuron_k)
            for (unsigned int k = 0; k < next_layer_size; ++k) { // Neuron 'k' in next layer 'l+1'
                error_sum += layer_deltas[l + 1][k] * weights_to_next_layer(k, i);
            }
            // Delta = error_sum * f'(activation_of_neuron_i_in_current_layer)
            layer_deltas[l][i] = error_sum * (current_hidden_layer_activations[i] * (1.0f - current_hidden_layer_activations[i]));
        }
    }

    // --- Update weights and calculate gradients for gweights ---
    for (unsigned int l = 0; l < num_layers - 1; ++l) {
        // 'l' is the index of the "from" layer for weights[l].
        // weights[l] connects layer 'l' to layer 'l+1'.
        size_t current_layer_size = layer_sizes[l+1]; 
        size_t prev_layer_size = layer_sizes[l];
        mat& current_weights = weights[l];
        mat& current_gweights = gweights[l];

        // Activations of the layer that is input to weights[l]
        const std::vector<float>& prev_layer_output_activations = activations[l];

        for (unsigned int i = 0; i < current_layer_size; ++i) {
            for (unsigned int j = 0; j < prev_layer_size; ++j) {
                // Gradient of loss w.r.t weight(i,j) = delta_of_neuron_i_in_layer(l+1) * activation_of_neuron_j_in_layer(l)
                current_gweights(i, j) = layer_deltas[l + 1][i] * prev_layer_output_activations[j];
                current_weights(i, j) -= learning * current_gweights(i, j);
            }
        }
    }
}


/**
 * @brief Compute backpropagation with L1 regularization
 */
void mlp::backwithL1(int in, int layers, float learning) {
    float lambda = 0.01;
    backprop(in, layers, learning);
    float l1_penalty = 0.0;
    for (unsigned int l = 0; l < num_layers - 1; ++l) {
        mat& current_weights = weights[l];
        size_t rows = current_weights.row;
        size_t cols = current_weights.col;
        for (unsigned int i = 0; i < rows; ++i) {
            for (unsigned int j = 0; j < cols; ++j) {
                float w_val = current_weights(i, j);
                float sign_w = (w_val > 0.0f) ? 1.0f : ((w_val < 0.0f) ? -1.0f : 0.0f);
                current_weights(i, j) -= learning * lambda * sign_w;
            }
        }
    }

    float loss = computeLossWithL1(output, expected, *this, lambda);
    std::cout << "Loss with L1 penalty: " << loss << std::endl;
}


/**
 * @brief Compute backpropagation with L2 regularization
 */
void mlp::backwithL2(int in, int layers, float learning) {
    float lambda = 0.01; // Regularization parameter

    backprop(in, layers, learning);
    float l2_penalty = 0.0;
    for (unsigned int l = 0; l < num_layers - 1; ++l) {
        mat& current_weights = weights[l];
        size_t rows = current_weights.row;
        size_t cols = current_weights.col;
        for (unsigned int i = 0; i < rows; ++i) {
            for (unsigned int j = 0; j < cols; ++j) {
                current_weights(i, j) -= learning * lambda * current_weights(i, j);
            }
        }
    }

    float loss = computeLossWithL2(output, expected, *this, lambda);
    std::cout << "Loss with L2 penalty: " << loss << std::endl;
}


/**
 * @brief Compute backpropagation with Elastic Net regularization (combines L1 and L2).
 *        Updates weights directly.
 *        Should be used with standard SGD `train` method if regularization is desired without Adam.
 */
void mlp::backwithElasticNet(int in, int layers, float learning) {
    backprop(in, layers, learning); // Calculate gradients

    for (unsigned int l = 0; l < num_layers - 1; ++l) {
        mat& current_weights = weights[l];
        mat& current_gweights = gweights[l];
        size_t rows = current_weights.row;
        size_t cols = current_weights.col;

        for (unsigned int i = 0; i < rows; ++i) {
            for (unsigned int j = 0; j < cols; ++j) {
                float w_val = current_weights(i, j);
                float sign_w = (w_val > 0.0f) ? 1.0f : ((w_val < 0.0f) ? -1.0f : 0.0f);

                // Combined gradient with regularization: grad_total = grad_from_backprop + lambda_l1 * sign(W) + lambda_l2 * W
                float regularized_gradient = current_gweights(i, j) + (lambda_l1 * sign_w) + (lambda_l2 * current_weights(i, j));

                // Update rule: W = W - learning_rate * grad_total
                current_weights(i, j) -= learning * regularized_gradient;
            }
        }
    }
}


/**
 * @brief Rprop algorithm for MLP
 * @param dataset Input dataset
 * @note This version only updates the weights and doesn't update the bias
 */
void mlp::rprop(std::vector<std::vector<float>>& dataset, int layers, int in, float learning, int epochs) {
    if (num_layers < 2) return;

    const float etaPlus = 1.2;
    const float etaMinus = 0.5;
    const float deltaMax = 50.0;
    const float deltaMin = 1e-6;

    std::vector<mat> prev_gradients;
    std::vector<mat> update_values;
    prev_gradients.reserve(num_layers - 1);
    update_values.reserve(num_layers - 1);

    for(size_t l=0; l < num_layers - 1; ++l) {
        size_t rows = layer_sizes[l+1];
        size_t cols = layer_sizes[l];
        prev_gradients.emplace_back(static_cast<int>(rows), static_cast<int>(cols));
        update_values.emplace_back(static_cast<int>(rows), static_cast<int>(cols));

        float* update_ptr = update_values[l].mapped_data;
        float* prev_grad_ptr = prev_gradients[l].mapped_data;
        size_t num_elements = rows * cols;
        if (update_ptr) {
            std::fill_n(update_ptr, num_elements, deltaMin);
        }
        if (prev_grad_ptr) {
             std::fill_n(prev_grad_ptr, num_elements, 0.0f);
        }
    }

    for (unsigned int epoch = 0; epoch < epochs; ++epoch) {
        float totalError = 0.0;

        for (const auto& data : dataset) {
            input = data;
            forward(0, 0);
            backprop(0, 0, 0.0f);

            // Compute mean square error
            float error = 0.0;
            size_t output_size = layer_sizes.back();
            for (unsigned int i = 0; i < output_size; ++i) {
                error += std::pow(expected[i] - output[i], 2);
            }
            error /= output_size;
            totalError += error;

            // Update weights using Rprop
            for (unsigned int l = 0; l < num_layers - 1; ++l) {
                mat& current_weights = weights[l];
                const mat& current_gweights = gweights[l];
                mat& current_prev_grads = prev_gradients[l];
                mat& current_update_vals = update_values[l];
                size_t rows = current_weights.row;
                size_t cols = current_weights.col;

                for (unsigned int i = 0; i < rows; ++i) {
                    for (unsigned int j = 0; j < cols; ++j) {
                        float grad = current_gweights(i, j);
                        float prev_grad = current_prev_grads(i, j);
                        float& update_val = current_update_vals(i, j);

                        // Apply Rprop update rule
                        float sign_change = grad * prev_grad;

                        if (sign_change > 0) {
                            update_val = std::min(update_val * etaPlus, deltaMax);
                            float delta_w = -std::copysign(update_val, grad);
                            current_weights(i, j) += delta_w;
                            current_prev_grads(i, j) = grad;
                        }
                        else if (sign_change < 0) {
                            update_val = std::max(update_val * etaMinus, deltaMin);
                            current_prev_grads(i, j) = 0; // Set previous gradient to zero
                        }
                        else {
                            float delta_w = -std::copysign(update_val, grad);
                            current_weights(i, j) += delta_w;
                            current_prev_grads(i, j) = grad;
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


/**
 * @brief Backpropagation till input vector
 */
void mlp::backprop2in(int in, int layers, float learning) {
    if (num_layers < 2) return;

    std::vector<std::vector<float>> layer_deltas(num_layers);
    for(size_t i = 0; i < num_layers; ++i) {
        layer_deltas[i].resize(layer_sizes[i], 0.0f);
    }

    size_t output_layer_idx = num_layers - 1;
    size_t output_size = layer_sizes[output_layer_idx];
    for (unsigned int i = 0; i < output_size; ++i) {
        layer_deltas[output_layer_idx][i] = (output[i] - expected[i]) * (output[i] * (1.0f - output[i]));
    }

    for (int l = num_layers - 2; l >= 1; --l) {
        size_t current_layer_size = layer_sizes[l];
        size_t next_layer_size = layer_sizes[l+1];
        const mat& next_weights = weights[l];
        const std::vector<float>& current_activations = activations[l-1];

        for (unsigned int i = 0; i < current_layer_size; ++i) {
            float error_sum = 0.0;
            for (unsigned int k = 0; k < next_layer_size; ++k) {
                error_sum += layer_deltas[l + 1][k] * next_weights(k, i);
            }
            layer_deltas[l][i] = error_sum * (current_activations[i] * (1.0f - current_activations[i]));
        }
    }

    for (unsigned int l = 0; l < num_layers - 1; ++l) {
        size_t current_layer_size = layer_sizes[l+1];
        size_t prev_layer_size = layer_sizes[l];
        mat& current_weights = weights[l]; 
        mat& current_gweights = gweights[l];
        const std::vector<float>& prev_activations = activations[l]; // Use activations[l] as it includes input at index 0

        for (unsigned int i = 0; i < current_layer_size; ++i) {
            for (unsigned int j = 0; j < prev_layer_size; ++j) {
                current_gweights(i, j) = layer_deltas[l + 1][i] * prev_activations[j];
                current_weights(i, j) -= learning * current_gweights(i, j);
            }
        }
    }

    const mat& first_weights = weights[0]; 
    size_t input_size = layer_sizes[0];
    size_t first_hidden_size = layer_sizes[1];

    for (unsigned int j = 0; j < input_size; ++j) {
        float input_grad_sum = 0.0;
        for (unsigned int i = 0; i < first_hidden_size; ++i) {
            input_grad_sum += layer_deltas[1][i] * first_weights(i, j);
        }
        input[j] -= learning * input_grad_sum;
    }
}

#endif
