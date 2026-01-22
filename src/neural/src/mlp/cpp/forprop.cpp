#ifdef USE_CPU
#include "include/mlp.hpp"
#include <numeric>
#include <maths.hpp>

// --- MLP (1D Vector Version) ---
void mlp::forward() {
    if (num_layers < 2) return;

    // 1. Set input layer activations
    if (input.size() != layer_sizes[0]) {
        throw std::runtime_error("MLP Forward: Input size mismatch.");
    }
    activations[0] = input;

    // 2. Propagate through layers
    for (unsigned int l = 0; l < num_layers - 1; ++l) {
        const mat& W = weights[l]; // Shape: [NextSize x CurrSize]
        const std::vector<float>& prev_act = activations[l];
        std::vector<float>& curr_h = hlayers[l];
        std::vector<float>& curr_act = activations[l+1];

        // Ensure resizing
        if (curr_h.size() != layer_sizes[l+1]) curr_h.resize(layer_sizes[l+1]);
        if (curr_act.size() != layer_sizes[l+1]) curr_act.resize(layer_sizes[l+1]);

        // Compute: H = W * prev_act
        for (unsigned int i = 0; i < layer_sizes[l+1]; ++i) {
            float sum = 0.0f;
            for (unsigned int j = 0; j < layer_sizes[l]; ++j) {
                sum += W(i, j) * prev_act[j];
            }
            curr_h[i] = sum;
            curr_act[i] = sigmoid(sum);
        }
    }

    // 3. Set final output
    output = activations.back();
}

/**
 * @brief Forward propagation for 2D MLP.
 * Processes an [inHeight x inWidth] matrix through the network.
 */
void mlp2d::forward() {
    if (num_layers < 2 || input.empty()) return;

    // input is std::vector<std::vector<float>> [inHeight x inWidth]
    size_t rows = input.size();

    // 1. Initialize/Copy input to first activation layer
    activations[0] = input;

    // 2. Propagate through layers
    for (unsigned int l = 0; l < num_layers - 1; ++l) {
        // Current activations A: [rows x layer_sizes[l]]
        // Current weights W: [layer_sizes[l+1] x layer_sizes[l]]
        
        mat A_mat(activations[l]); 
        mat W_T = weights[l].transpose(); // W_T is [layer_sizes[l] x layer_sizes[l+1]]
        
        // H = A * W^T results in [rows x layer_sizes[l+1]]
        mat H_mat = A_mat * W_T;

        // Store pre-activation values in hlayers
        hlayers[l] = H_mat.make2dVector();

        // Apply activation for the next layer
        size_t next_layer_width = layer_sizes[l+1];
        if (activations[l+1].size() != rows) activations[l+1].resize(rows);

        for (size_t r = 0; r < rows; ++r) {
            if (activations[l+1][r].size() != next_layer_width) 
                activations[l+1][r].resize(next_layer_width);
                
            for (size_t c = 0; c < next_layer_width; ++c) {
                activations[l+1][r][c] = sigmoid(hlayers[l][r][c]);
            }
        }
    }

    // 3. Final output matrix [inHeight x outWidth]
    output = activations.back();
}

#endif