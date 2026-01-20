#ifdef USE_CPU
#include "include/mlp.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

// ---------------------- MLP ----------------------

/**
 * @brief Standard backpropagation: calculates gradients and updates weights.
 */
void mlp::backprop(int in, int layers, float learning) {
    if (num_layers < 2) return;

    // 1. Initialize deltas
    std::vector<std::vector<float>> layer_deltas(num_layers);
    for(size_t i = 0; i < num_layers; ++i) {
        layer_deltas[i].resize(layer_sizes[i], 0.0f);
    }

    // 2. Output Layer Delta: (actual - expected) * f'(h)
    size_t L = num_layers - 1;
    for (unsigned int i = 0; i < layer_sizes[L]; ++i) {
        // activations[L] is the output layer
        layer_deltas[L][i] = (activations[L][i] - expected[i]) * activations[L][i] * (1.0f - activations[L][i]);
    }

    // 3. Hidden Layer Deltas (Backwards from L-1 to 1)
    for (int l = L - 1; l >= 1; --l) {
        for (unsigned int i = 0; i < layer_sizes[l]; ++i) {
            float error_sum = 0.0f;
            // Weights[l] connects layer l to l+1
            for (unsigned int k = 0; k < layer_sizes[l+1]; ++k) {
                error_sum += layer_deltas[l + 1][k] * weights[l](k, i);
            }
            layer_deltas[l][i] = error_sum * activations[l][i] * (1.0f - activations[l][i]);
        }
    }

    // 4. Update weights and store gradients in gweights
    for (unsigned int l = 0; l < num_layers - 1; ++l) {
        mat& W = weights[l];
        mat& dW = gweights[l];
        const std::vector<float>& act_prev = activations[l];

        for (unsigned int i = 0; i < layer_sizes[l+1]; ++i) {
            for (unsigned int j = 0; j < layer_sizes[l]; ++j) {
                float gradient = layer_deltas[l + 1][i] * act_prev[j];
                dW(i, j) = gradient;
                W(i, j) -= learning * gradient;
            }
        }
    }
}

/**
 * @brief Simple backward wrapper (legacy logic)
 */
void mlp::backward(int in_param, int layers_param, float learning) {
    backprop(0, 0, learning);
}

/**
 * @brief Backpropagation with L1 regularization
 */
void mlp::backwithL1(int in, int layers, float learning) {
    backprop(in, layers, learning); // Standard update first
    
    float lambda = 0.01f;
    for (unsigned int l = 0; l < num_layers - 1; ++l) {
        mat& W = weights[l];
        for (int i = 0; i < W.row; ++i) {
            for (int j = 0; j < W.col; ++j) {
                float sign_w = (W(i, j) > 0.0f) ? 1.0f : ((W(i, j) < 0.0f) ? -1.0f : 0.0f);
                W(i, j) -= learning * lambda * sign_w;
            }
        }
    }
}

/**
 * @brief Backpropagation with L2 regularization
 */
void mlp::backwithL2(int in, int layers, float learning) {
    backprop(in, layers, learning); // Standard update first
    
    float lambda = 0.01f;
    for (unsigned int l = 0; l < num_layers - 1; ++l) {
        mat& W = weights[l];
        for (int i = 0; i < W.row; ++i) {
            for (int j = 0; j < W.col; ++j) {
                W(i, j) -= learning * lambda * W(i, j);
            }
        }
    }
}

/**
 * @brief Elastic Net regularization (L1 + L2)
 */
void mlp::backwithElastic(int in, int layers, float learning) {
    backprop(in, layers, learning); // Standard update first

    for (unsigned int l = 0; l < num_layers - 1; ++l) {
        mat& W = weights[l];
        for (int i = 0; i < W.row; ++i) {
            for (int j = 0; j < W.col; ++j) {
                float w_val = W(i, j);
                float sign_w = (w_val > 0.0f) ? 1.0f : ((w_val < 0.0f) ? -1.0f : 0.0f);
                
                // Penalty gradient = lambda_l1 * sign(w) + lambda_l2 * w
                float penalty = (lambda_l1 * sign_w) + (lambda_l2 * w_val);
                W(i, j) -= learning * penalty;
            }
        }
    }
}

/**
 * @brief Backpropagation that propagates error back to the input vector
 */
void mlp::backprop2in(int in, int layers, float learning) {
    if (num_layers < 2) return;

    // 1. Calculate deltas for all layers (same as backprop)
    std::vector<std::vector<float>> layer_deltas(num_layers);
    for(size_t i = 0; i < num_layers; ++i) layer_deltas[i].resize(layer_sizes[i]);

    size_t L = num_layers - 1;
    for (unsigned int i = 0; i < layer_sizes[L]; ++i) {
        layer_deltas[L][i] = (activations[L][i] - expected[i]) * activations[L][i] * (1.0f - activations[L][i]);
    }

    for (int l = L - 1; l >= 1; --l) {
        for (unsigned int i = 0; i < layer_sizes[l]; ++i) {
            float sum = 0.0f;
            for (unsigned int k = 0; k < layer_sizes[l+1]; ++k) {
                sum += layer_deltas[l + 1][k] * weights[l](k, i);
            }
            layer_deltas[l][i] = sum * activations[l][i] * (1.0f - activations[l][i]);
        }
    }

    // 2. Update Weights
    for (unsigned int l = 0; l < num_layers - 1; ++l) {
        for (unsigned int i = 0; i < layer_sizes[l+1]; ++i) {
            for (unsigned int j = 0; j < layer_sizes[l]; ++j) {
                weights[l](i, j) -= learning * layer_deltas[l + 1][i] * activations[l][j];
            }
        }
    }

    // 3. Update Input Vector (Layer 0)
    // Grad_input = Weights[0]^T * Delta[1]
    for (unsigned int j = 0; j < layer_sizes[0]; ++j) {
        float input_grad = 0.0f;
        for (unsigned int i = 0; i < layer_sizes[1]; ++i) {
            input_grad += layer_deltas[1][i] * weights[0](i, j);
        }
        input[j] -= learning * input_grad;
    }
}

/**
 * @brief Rprop implementation
 */
void mlp::rprop(std::vector<std::vector<float>>& dataset, int layers, int in, float learning, int epochs) {
    if (num_layers < 2) return;

    const float etaPlus = 1.2f, etaMinus = 0.5f;
    const float deltaMax = 50.0f, deltaMin = 1e-6f;

    std::vector<mat> prev_grads, update_values;
    for(size_t l=0; l < num_layers - 1; ++l) {
        prev_grads.emplace_back(layer_sizes[l+1], layer_sizes[l]);
        update_values.emplace_back(layer_sizes[l+1], layer_sizes[l]);
        // Initialize update values to deltaMin
        std::fill_n(update_values[l].mapped_data, update_values[l].row * update_values[l].col, 0.1f);
        std::fill_n(prev_grads[l].mapped_data, prev_grads[l].row * prev_grads[l].col, 0.0f);
    }

    for (unsigned int epoch = 0; epoch < epochs; ++epoch) {
        float totalError = 0.0;
        for (const auto& data : dataset) {
            input = data;
            forward(0, 0);
            
            // Standard gradient calculation via backprop logic (don't update weights yet)
            // For Rprop we actually need the gradient of the TOTAL error over the batch, 
            // but this implementation updates per-sample (Stochastic Rprop).
            backprop(0, 0, 0.0f); 

            for (unsigned int l = 0; l < num_layers - 1; ++l) {
                mat& W = weights[l];
                mat& dW = gweights[l];
                mat& pG = prev_grads[l];
                mat& uV = update_values[l];

                for (int i = 0; i < W.row; ++i) {
                    for (int j = 0; j < W.col; ++j) {
                        float change = dW(i, j) * pG(i, j);
                        if (change > 0) {
                            uV(i, j) = std::min(uV(i, j) * etaPlus, deltaMax);
                            W(i, j) -= std::copysign(uV(i, j), dW(i, j));
                            pG(i, j) = dW(i, j);
                        } else if (change < 0) {
                            uV(i, j) = std::max(uV(i, j) * etaMinus, deltaMin);
                            pG(i, j) = 0; 
                        } else {
                            W(i, j) -= std::copysign(uV(i, j), dW(i, j));
                            pG(i, j) = dW(i, j);
                        }
                    }
                }
            }
        }
    }
}

// --------------------- MLP2D ---------------------

/**
 * @brief Standard Backpropagation for MLP2D.
 * Computes gradients by averaging the error across the 'height' (rows) of the 2D input.
 */
void mlp2d::backprop(int layers, int in, float learning) {
    if (num_layers < 2 || input.empty()) return;

    size_t height = input.size();
    size_t L = num_layers - 1;

    // 1. Calculate Deltas for the Output Layer [Height x layer_sizes[L]]
    mat D_out(height, layer_sizes[L]);
    for (size_t r = 0; r < height; ++r) {
        for (unsigned int c = 0; c < layer_sizes[L]; ++c) {
            float act = activations[L][r][c];
            // derivative of sigmoid: act * (1 - act)
            // Error = (activation - expected)
            D_out(r, c) = (act - expected[r][c]) * act * (1.0f - act);
        }
    }

    std::vector<mat> layer_deltas(num_layers);
    layer_deltas[L] = std::move(D_out);

    // 2. Backpropagate Deltas through Hidden Layers (Right to Left)
    for (int l = L - 1; l >= 1; --l) {
        // Delta_l = (Delta_{l+1} * Weights_l) * f'(Activations_l)
        // [height x next] * [next x curr] -> [height x curr]
        mat error_prop = layer_deltas[l+1] * weights[l];
        
        layer_deltas[l] = mat(height, layer_sizes[l]);
        for (size_t r = 0; r < height; ++r) {
            for (unsigned int c = 0; c < layer_sizes[l]; ++c) {
                float act = activations[l][r][c];
                layer_deltas[l](r, c) = error_prop(r, c) * act * (1.0f - act);
            }
        }
    }

    // 3. Update Weights and Calculate Gradients
    float inv_height = 1.0f / static_cast<float>(height);
    for (unsigned int l = 0; l < num_layers - 1; ++l) {
        // Gradient = Delta_{l+1}^T * Activations_l
        // [Next x height] * [height x Curr] = [Next x Curr]
        mat A_l(activations[l]);
        mat grad_total = layer_deltas[l+1].transpose() * A_l;

        mat& W = weights[l];
        mat& dW = gweights[l];

        for (int i = 0; i < W.row; ++i) {
            for (int j = 0; j < W.col; ++j) {
                // Average gradient over the height dimension
                float avg_grad = grad_total(i, j) * inv_height;
                dW(i, j) = avg_grad;
                W(i, j) -= learning * avg_grad;
            }
        }
    }
}

/**
 * @brief Backpropagation with L1 regularization for MLP2D.
 */
void mlp2d::backwithL1(int layers, int in, float learning) {
    backprop(layers, in, learning); // Update weights with gradients first

    float lambda = 0.01f;
    for (unsigned int l = 0; l < num_layers - 1; ++l) {
        mat& W = weights[l];
        for (int i = 0; i < W.row; ++i) {
            for (int j = 0; j < W.col; ++j) {
                float sign_w = (W(i, j) > 0.0f) ? 1.0f : ((W(i, j) < 0.0f) ? -1.0f : 0.0f);
                W(i, j) -= learning * lambda * sign_w;
            }
        }
    }
}

/**
 * @brief Backpropagation with L2 regularization for MLP2D.
 */
void mlp2d::backwithL2(int layers, int in, float learning) {
    backprop(layers, in, learning); // Update weights with gradients first

    float lambda = 0.01f;
    for (unsigned int l = 0; l < num_layers - 1; ++l) {
        mat& W = weights[l];
        for (int i = 0; i < W.row; ++i) {
            for (int j = 0; j < W.col; ++j) {
                W(i, j) -= learning * lambda * W(i, j);
            }
        }
    }
}

/**
 * @brief Backpropagation with Elastic Net regularization for MLP2D.
 */
void mlp2d::backwithElastic(int in, int layers, float learning) {
    backprop(layers, in, learning);

    for (unsigned int l = 0; l < num_layers - 1; ++l) {
        mat& W = weights[l];
        for (int i = 0; i < W.row; ++i) {
            for (int j = 0; j < W.col; ++j) {
                float w_val = W(i, j);
                float sign_w = (w_val > 0.0f) ? 1.0f : ((w_val < 0.0f) ? -1.0f : 0.0f);
                float penalty = (lambda_l1 * sign_w) + (lambda_l2 * w_val);
                W(i, j) -= learning * penalty;
            }
        }
    }
}

/**
 * @brief Updates the input matrix by propagating error all the way back.
 */
void mlp2d::backprop2in(int layers, int in, float learning) {
    if (num_layers < 2 || input.empty()) return;
    size_t height = input.size();

    // 1. Calculate all layer deltas (logic same as backprop)
    std::vector<mat> layer_deltas(num_layers);
    size_t L = num_layers - 1;
    
    layer_deltas[L] = mat(height, layer_sizes[L]);
    for (size_t r = 0; r < height; ++r) {
        for (unsigned int c = 0; c < layer_sizes[L]; ++c) {
            float act = activations[L][r][c];
            layer_deltas[L](r, c) = (act - expected[r][c]) * act * (1.0f - act);
        }
    }

    for (int l = L - 1; l >= 1; --l) {
        mat error_prop = layer_deltas[l+1] * weights[l];
        layer_deltas[l] = mat(height, layer_sizes[l]);
        for (size_t r = 0; r < height; ++r) {
            for (unsigned int c = 0; c < layer_sizes[l]; ++c) {
                float act = activations[l][r][c];
                layer_deltas[l](r, c) = error_prop(r, c) * act * (1.0f - act);
            }
        }
    }

    // 2. Update Weights
    float inv_height = 1.0f / static_cast<float>(height);
    for (unsigned int l = 0; l < num_layers - 1; ++l) {
        mat A_l(activations[l]);
        mat grad = layer_deltas[l+1].transpose() * A_l;
        for (int i = 0; i < weights[l].row; ++i) {
            for (int j = 0; j < weights[l].col; ++j) {
                weights[l](i, j) -= learning * (grad(i, j) * inv_height);
            }
        }
    }

    // 3. Update the input matrix [Height x InWidth]
    // Grad_input = Delta_1 * Weights_0
    // [Height x Hidden1] * [Hidden1 x InWidth] = [Height x InWidth]
    mat grad_input = layer_deltas[1] * weights[0];
    for (size_t r = 0; r < height; ++r) {
        for (unsigned int c = 0; c < layer_sizes[0]; ++c) {
            input[r][c] -= learning * grad_input(r, c);
        }
    }
}

/**
 * @brief Rprop algorithm for MLP2D.
 * Updates weights based on the sign of the average gradient across the 2D height.
 */
void mlp2d::rprop(std::vector<std::vector<std::vector<float>>>& dataset, int layers, int in, float learning, int epochs) {
    if (num_layers < 2) return;

    const float etaPlus = 1.2f;
    const float etaMinus = 0.5f;
    const float deltaMax = 50.0f;
    const float deltaMin = 1e-6f;

    // Persistent matrices to store state across epochs/samples
    std::vector<mat> prev_gradients;
    std::vector<mat> update_values;

    for(size_t l=0; l < num_layers - 1; ++l) {
        prev_gradients.emplace_back(layer_sizes[l+1], layer_sizes[l]);
        update_values.emplace_back(layer_sizes[l+1], layer_sizes[l]);

        // Initialize update values (deltas) to a small positive constant
        std::fill_n(update_values[l].mapped_data, (size_t)update_values[l].row * update_values[l].col, 0.1f);
        std::fill_n(prev_gradients[l].mapped_data, (size_t)prev_gradients[l].row * prev_gradients[l].col, 0.0f);
    }

    for (unsigned int epoch = 0; epoch < epochs; ++epoch) {
        float totalMse = 0.0f;

        for (const auto& data_2d : dataset) {
            // data_2d is [Height x Width]
            input = data_2d;
            forward(0, 0);

            // 1. Compute Gradients via backprop logic. 
            // Note: We pass learning=0.0 because Rprop handles its own weight updates.
            // backprop() will populate this->gweights with the average gradient across the height.
            backprop(0, 0, 0.0f); 

            // 2. Compute MSE for reporting
            size_t height = input.size();
            size_t out_dim = layer_sizes.back();
            float sample_error = 0.0f;
            for (size_t r = 0; r < height; ++r) {
                for (size_t c = 0; c < out_dim; ++c) {
                    sample_error += std::pow(expected[r][c] - output[r][c], 2);
                }
            }
            totalMse += (sample_error / (height * out_dim));

            // 3. Apply Rprop Update Rule to each weight layer
            for (unsigned int l = 0; l < num_layers - 1; ++l) {
                mat& W = weights[l];
                mat& G = gweights[l]; // Average gradient from backprop()
                mat& pG = prev_gradients[l];
                mat& delta = update_values[l];

                for (int i = 0; i < W.row; ++i) {
                    for (int j = 0; j < W.col; ++j) {
                        float sign_change = G(i, j) * pG(i, j);

                        if (sign_change > 0) {
                            // Gradients have the same sign: speed up
                            delta(i, j) = std::min(delta(i, j) * etaPlus, deltaMax);
                            W(i, j) -= std::copysign(delta(i, j), G(i, j));
                            pG(i, j) = G(i, j);
                        } 
                        else if (sign_change < 0) {
                            // Gradients have opposite signs: slow down and skip update
                            delta(i, j) = std::max(delta(i, j) * etaMinus, deltaMin);
                            pG(i, j) = 0; // Force sign_change = 0 in next iteration
                        } 
                        else {
                            // One gradient is zero or sign just reset
                            W(i, j) -= std::copysign(delta(i, j), G(i, j));
                            pG(i, j) = G(i, j);
                        }
                    }
                }
            }
        }

        std::cout << "Epoch " << epoch + 1 << " MSE: " << (totalMse / dataset.size()) << std::endl;
    }
}

#endif