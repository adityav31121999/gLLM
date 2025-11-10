
// loss.cpp: calculate losses and penalties required for mlp
#include "include/mlp.hpp" // Includes maths.hpp -> mat.hpp
#include <cmath>
#include <numeric> // For std::accumulate (optional, but can be cleaner)
#include <cstddef> // For size_t

/**
 * @brief Calculates the L1 penalty for all weights in the network.
 * The L1 penalty is the sum of the absolute value of all the weights in the network.
 * @param weights A vector of memory-mapped matrices representing the network weights.
 * @return The L1 penalty for the network.
 */
float getL1Penalty(const std::vector<mat>& weights) {
    double penalty = 0.0;
    for (const auto& matrix : weights) {
        if (matrix.mapped_data) {
            size_t num_elements = static_cast<size_t>(matrix.row) * matrix.col;
            size_t elements_in_map = matrix.mapped_size / sizeof(float);
            size_t count = (std::min)(num_elements, elements_in_map);

            for (size_t k = 0; k < count; ++k) {
                penalty += std::abs(static_cast<double>(matrix.mapped_data[k]));
            }
        }
    }
    if (penalty > (std::numeric_limits<float>::max)()) {
        throw std::overflow_error("L1 penalty calculation resulted in overflow.");
    }
    return static_cast<float>(penalty);
}

/**
 * @brief Calculates the L2 penalty for all weights in the network.
 * The L2 penalty is the sum of the squares of all the weights in the network.
 * @param weights 3d matrix
 * @return The L2 penalty for the network.
 */
float getL2Penalty(const std::vector<mat>& weights) {
    double penalty = 0.0;
    for (const auto& matrix : weights) {
        if (matrix.mapped_data) {
            size_t num_elements = static_cast<size_t>(matrix.row) * matrix.col;
            size_t elements_in_map = matrix.mapped_size / sizeof(float);
            size_t count = (std::min)(num_elements, elements_in_map);

            for (size_t k = 0; k < count; ++k) {
                double w = static_cast<double>(matrix.mapped_data[k]);
                penalty += w * w;
            }
        }
    }
    if (penalty > (std::numeric_limits<float>::max)()) {
        throw std::overflow_error("L2 penalty calculation resulted in overflow.");
    }
    return static_cast<float>(penalty);
}


/**
 * @brief Computes the loss with L1 regularization. The loss is the sum of 
 * the absolute difference between the predicted output and the target output.
 * The L1 regularization term is added to the loss.
 * @param outputs The predicted output of the network.
 * @param targets The target output of the network.
 * @param network The network to compute the loss for.
 * @param lambda The regularization parameter.
 * @return The loss with L1 regularization.
 */
float computeLossWithL1(std::vector<float>& outputs, std::vector<float>& targets, mlp& network, float lambda) {
    if (outputs.size() != targets.size()) {
        throw std::invalid_argument("Output and target vector sizes must match for loss calculation.");
    }
    double loss = 0.0;
    for (size_t i = 0; i < outputs.size(); ++i) {
        loss += std::abs(static_cast<double>(outputs[i]) - static_cast<double>(targets[i]));
    }
    double total_loss = loss + 0.5 * static_cast<double>(lambda) * static_cast<double>(getL1Penalty(network.weights));
    return static_cast<float>(total_loss);
}


/**
 * @brief Computes the loss with L2 regularization. The loss is the sum of the 
 * squared difference between the predicted output and the target output.
 * The L2 regularization term is added to the loss.
 * @param outputs The predicted output of the network.
 * @param targets The target output of the network.
 * @param network The network to compute the loss for.
 * @param lambda The regularization parameter.
 * @return The loss with L2 regularization.
 */
float computeLossWithL2(std::vector<float>& outputs, std::vector<float>& targets, mlp& network, float lambda) {
    if (outputs.size() != targets.size()) {
        throw std::invalid_argument("Output and target vector sizes must match for loss calculation.");
    }
    double loss = 0.0;
    for (size_t i = 0; i < outputs.size(); ++i) {
        double diff = static_cast<double>(outputs[i]) - static_cast<double>(targets[i]);
        loss += diff * diff;
    }
    double total_loss = 0.5 * loss + 0.5 * static_cast<double>(lambda) * static_cast<double>(getL2Penalty(network.weights));
    return static_cast<float>(total_loss); // Cast back at the end
}

/**
 * @brief Computes the loss with Elastic Net regularization.
 * The loss is the Mean Squared Error (MSE) plus the combined L1 and L2 regularization terms.
 * Elastic Net Loss = MSE + lambda_l1 * L1_Penalty + lambda_l2 * L2_Penalty
 * @param outputs The predicted output of the network.
 * @param targets The target output of the network.
 * @param network The network to compute the loss for.
 * @param lambda_l1 The L1 regularization parameter.
 * @param lambda_l2 The L2 regularization parameter.
 * @return The loss with Elastic Net regularization.
 */
float computeLossWithElasticNet(std::vector<float>& outputs, std::vector<float>& targets, mlp& network, float lambda_l1, float lambda_l2) {
    if (outputs.size() != targets.size()) {
        throw std::invalid_argument("Output and target vector sizes must match for loss calculation.");
    }

    // 1. Calculate the base loss (Mean Squared Error)
    double mse_loss = 0.0;
    for (size_t i = 0; i < outputs.size(); ++i) {
        double diff = static_cast<double>(outputs[i]) - static_cast<double>(targets[i]);
        mse_loss += diff * diff;
    }
    mse_loss /= outputs.size(); // Divide by number of outputs to get the mean

    // 2. Calculate L1 and L2 penalties
    double l1_penalty = static_cast<double>(getL1Penalty(network.weights));
    double l2_penalty = static_cast<double>(getL2Penalty(network.weights));

    // 3. Combine base loss with weighted regularization terms
    // Standard Elastic Net combines MSE with lambda_l1 * |W| + lambda_l2 * W^2
    double total_loss = mse_loss +
                        static_cast<double>(lambda_l1) * l1_penalty +
                        static_cast<double>(lambda_l2) * l2_penalty;

    return static_cast<float>(total_loss);
}


/**
 * @brief Computes the loss with dropout generalization. The loss is the sum of 
 * the squared difference between the predicted output and the target output.
 * The dropout generalization term is added to the loss.
 * @param outputs The predicted output of the network.
 * @param targets The target output of the network.
 * @param network The network to compute the loss for.
 * @param p The dropout probability.
 * @return The loss with dropout generalization.
 */
float dropoutGeneralisation(std::vector<float>& outputs, std::vector<float>& targets, mlp& network, float p) {
    if (outputs.size() != targets.size()) {
        throw std::invalid_argument("Output and target vector sizes must match for loss calculation.");
    }
    if (p < 0.0f || p >= 1.0f) {
        throw std::invalid_argument("Dropout probability p must be in the range [0, 1).");
    }
    double loss = 0.0;
    for (size_t i = 0; i < outputs.size(); ++i) {
        double diff = static_cast<double>(outputs[i]) - static_cast<double>(targets[i]);
        loss += diff * diff;
    }
    return static_cast<float>(loss / (1.0 - static_cast<double>(p)));
}
