
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
    if (num_layers < 2) {
        if (num_layers == 1 && !input.empty()) {
            output = input; // Or apply activation if input layer has one
        }
        return;
    }

    // Ensure input is copied to activations[0]
    if (activations[0].size() != layer_sizes[0]) {
        activations[0].resize(layer_sizes[0]);
    }
    if (input.size() == layer_sizes[0]) {
        activations[0] = input;
    } 
    else {
        throw std::runtime_error("MLP forward: Input vector size mismatch with input layer size.");
    }

    // Iterate through weight matrices: weights[0] to weights[num_layers - 2]
    // `l_idx` is the index for the weight matrix, hlayer, and the *previous* activation layer.
    for (unsigned int l_idx = 0; l_idx < num_layers - 1; ++l_idx) {
        const mat& current_weights = weights[l_idx]; // Connects layer l_idx to l_idx+1
                                                       // Dimensions: (layer_sizes[l_idx+1], layer_sizes[l_idx])
        const std::vector<float>& prev_layer_activations = activations[l_idx];
        std::vector<float>& current_hlayer_values = hlayers[l_idx]; // For layer l_idx+1
        std::vector<float>& current_output_activations = activations[l_idx+1]; // For layer l_idx+1

        if (!current_weights.mapped_data) {
            throw std::runtime_error("Weights[" + std::to_string(l_idx) + "] not mapped.");
        }

        // Ensure hlayers and activations for the current output layer are correctly sized
        if (current_hlayer_values.size() != layer_sizes[l_idx+1]) {
            current_hlayer_values.resize(layer_sizes[l_idx+1]);
        }
        if (current_output_activations.size() != layer_sizes[l_idx+1]) {
            current_output_activations.resize(layer_sizes[l_idx+1]);
        }

        // For each neuron 'j' in the current output layer (layer l_idx+1)
        for (unsigned int j = 0; j < layer_sizes[l_idx+1]; ++j) {
            float sum = 0.0f;
            // For each neuron 'k' in the previous layer (layer l_idx)
            for (unsigned int k = 0; k < layer_sizes[l_idx]; ++k) {
                // Weight from neuron k (prev layer) to neuron j (current output layer)
                // is weightsl_idx
                // Access: current_weights.mapped_data[j * current_weights.col + k]
                // current_weights.col is layer_sizes[l_idx]
                size_t weight_index = static_cast<size_t>(j) * current_weights.col + k;
                if (weight_index >= current_weights.mapped_size / sizeof(float)) {
                    throw std::out_of_range("Weight index out of range for weights[" + std::to_string(l_idx) + "]. Accessing index " + std::to_string(weight_index) + " with size " + std::to_string(current_weights.mapped_size / sizeof(float)));
                }
                sum += prev_layer_activations[k] * current_weights.mapped_data[weight_index];
            }
            current_hlayer_values[j] = sum;
            current_output_activations[j] = sigmoid(sum);
        }
    }

    // The final output is in activations[num_layers - 1]
    if (output.size() != layer_sizes.back()) {
        output.resize(layer_sizes.back());
    }
    output = activations[num_layers - 1];
}

#endif
