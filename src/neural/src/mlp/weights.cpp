
#include "include/mlp.hpp"
#include <random>

/**
 * @brief Function to initialize the weights of the multi-layer perceptron.
 * This function initializes the weights of the mlp using a normal distribution
 * with a mean of 0.0 and a standard deviation of 1.0.
 */
void mlp::initializeWeights() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0, 1.0);

    for (size_t i = 0; i < weights.size(); ++i) {
        mat& current_weight_matrix = weights[i];
        for (int row_idx = 0; row_idx < current_weight_matrix.row; ++row_idx) {
            for (int col_idx = 0; col_idx < current_weight_matrix.col; ++col_idx) {
                current_weight_matrix(row_idx, col_idx) = dis(gen) * sqrt(2.0f / current_weight_matrix.col);
            }
        }
    }
}


/**
 * @brief flatten the 3D vector into 1D vector
 * @param weights 3D vector
 * @return 1D vector
 */
std::vector<float> flattenWeights(const std::vector<std::vector<std::vector<float>>>& weights) {
    std::vector<float> flat_weights;
    size_t total_weights = 0;
    for (const auto& layer : weights) {
        for (const auto& neuron : layer) {
            total_weights += neuron.size();
        }
    }
    flat_weights.reserve(total_weights);
    for (const auto& layer : weights) {
        for (const auto& neuron : layer) {
            flat_weights.insert(flat_weights.end(), neuron.begin(), neuron.end());
        }
    }
    return flat_weights;
}


/**
 * @brief Flattens a collection of 'mat' objects (matrices) into a single 1D vector.
 * Each matrix in the input vector is appended sequentially to the output vector.
 * @param weights_collection A vector of 'mat' objects.
 * @return A 1D std::vector<float> containing all elements from all matrices.
 */
std::vector<float> flattenWeights(const std::vector<mat>& weights_collection) {
    std::vector<float> flat_weights;
    size_t total_elements = 0;

    for (const mat& matrix : weights_collection) {
        if (matrix.mapped_data && matrix.row > 0 && matrix.col > 0) {
            total_elements += static_cast<size_t>(matrix.row) * matrix.col;
        }
    }

    if (total_elements == 0) {
        return {};
    }
    flat_weights.reserve(total_elements);

    for (const mat& matrix : weights_collection) {
        if (matrix.mapped_data && matrix.row > 0 && matrix.col > 0) {
            size_t num_elements_in_matrix = static_cast<size_t>(matrix.row) * matrix.col;
            flat_weights.insert(flat_weights.end(),
                                matrix.mapped_data,
                                matrix.mapped_data + num_elements_in_matrix);
        }
    }
    return flat_weights;
}