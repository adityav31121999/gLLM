#include "include/cppsup.hpp"
#include <stdexcept>
#include <string>
#include <cmath>

/**
 * @brief Calculate the error between two vectors
 * This function takes two vectors as an input and returns a new vector
 * where each element is the difference between the corresponding elements
 * of the input vectors divided by 100.
 * @param x first vector
 * @param y second vector
 * @return a vector of errors
 */
std::vector<float> error(std::vector<float>& x, std::vector<float>& y) {
    if(x.size() != y.size()) {
        throw std::runtime_error("Vectors must be of the same size");
    }
    // Create a vector to hold the errors
    std::vector<float> b(x.size());
    // Use std::transform to calculate the error
    std::transform(x.begin(), x.end(), y.begin(), b.begin(), \
                // lambda to calculate the error
                [](const auto& i, const auto& j) { return (i - j); });
    // Return the vector of errors
    return b;
}

/**
 * @brief Calculate the Mean Squared Error (MSE) between two vectors. This function 
 * takes two vectors as input and returns the mean squared error between them.
 * The mean squared error is calculated as the average of the squared differences 
 * between the corresponding elements of the two vectors.
 * @param a The first vector.
 * @param b The second vector.
 * @return The mean squared error between the two vectors.
 */
float MAE(std::vector<float>& a, std::vector<float>& b) {
    if(a.size() != b.size())
        throw std::runtime_error("Same Size Vectors are ALLOWED only");
    float sum = 0.0;
    for(int i = 0; i < a.size(); i++) {
        // Calculate the squared difference between the elements of the two vectors
        sum += std::abs(a[i] - b[i]);
    }
    return sum / a.size();
}

/**
 * @brief Calculate the Mean Squared Error (MSE) between two vectors. This function 
 * takes two vectors as input and returns the mean squared error between them.
 * The mean squared error is calculated as the average of the squared differences 
 * between the corresponding elements of the two vectors.
 * @param a The first vector.
 * @param b The second vector.
 * @return The mean squared error between the two vectors.
 */
float MSE(std::vector<float>& a, std::vector<float>& b) {
    if(a.size() != b.size())
        throw std::runtime_error("Same Size Vectors are ALLOWED only");
    float sum = 0.0;
    for(int i = 0; i < a.size(); i++) {
        // Calculate the squared difference between the elements of the two vectors
        sum += pow(a[i] - b[i], 2);
    }
    return sum / a.size();
}

/**
 * @brief Calculates the Cross-Entropy loss between two probability distributions.
 * @details Computes the Cross-Entropy loss for classification tasks, measuring the difference
 * between the true probability distribution (y_true, typically one-hot encoded) and the
 * predicted probability distribution (y_pred). The formula is:
 * L = -log(y_pred[i]) for i = expected index (after softmax)
 * A small epsilon is added to y_pred to prevent taking the logarithm of zero.
 * @param[in] y_true The true values (one-hot encoded).
 * @param[in] y_pred The predicted values (logits, softmax will be applied internally).
 * @return The Cross-Entropy Loss (positive).
 * @throws std::runtime_error if vector sizes do not match or y_true is not a valid one-hot vector.
 */
float crossEntropy(std::vector<float>& y_true, std::vector<float>& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::runtime_error("crossEntropy: Vector sizes do not match: " + 
                                 std::to_string(y_true.size()) + " vs " + 
                                 std::to_string(y_pred.size()));
    }

    std::vector<float> y_pred_softmax = softmax(y_pred);
    float loss = 0.0f;
    float epsilon = 1e-15f; // Prevent log(0)
    bool is_one_hot = false;
    int one_hot_index = -1;

    // Validate y_true is one-hot (one element is 1, others are 0)
    for (size_t i = 0; i < y_true.size(); ++i) {
        if (y_true[i] == 1.0f) {
            if (is_one_hot) {
                throw std::runtime_error("crossEntropy: y_true is not a valid one-hot vector (multiple 1s)");
            }
            is_one_hot = true;
            one_hot_index = i;
        } else if (y_true[i] != 0.0f) {
            throw std::runtime_error("crossEntropy: y_true is not a valid one-hot vector (non-zero, non-one value)");
        }
    }
    if (!is_one_hot) {
        throw std::runtime_error("crossEntropy: y_true is not a valid one-hot vector (no 1 found)");
    }

    // Compute loss for the one-hot index
    loss = -std::log(y_pred_softmax[one_hot_index] + epsilon);
    return loss;
}

/**
 * @brief Calculates the Binary Cross-Entropy loss between two probability distributions.
 * @details This function computes the Binary Cross-Entropy loss, which is commonly used as a loss function
 * in binary classification tasks. It measures the difference between the predicted probability distribution
 * (y_pred, after sigmoid) and the true probability distribution (y_true, should be in [0,1]).
 * The formula for Binary Cross-Entropy is: 
 *  L = -1 * sum(y_true[i] * log(y_pred[i]) + (1 - y_true[i]) * log(1 - y_pred[i])) / size()
 * @param[in] a The true values (raw).
 * @param[in] b The predicted values (raw).
 * @return The Binary Cross-Entropy loss
 * @throws std::runtime_error if the vector sizes do not match.
 */
float binaryCrossEntropy(std::vector<float>& a, std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("binaryCrossEntropy: Vector sizes do not match: " + std::to_string(a.size()) + " vs " + std::to_string(b.size()));
    }
    std::vector<float> y_true = sigmoid(a);
    std::vector<float> y_pred = sigmoid(b);
    float loss = 0.0f;
    float epsilon = 1e-10f;
    for (size_t i = 0; i < y_true.size(); ++i) {
        float p = std::max<float>(y_pred[i], epsilon);
        float p_inv = std::max<float>(1.0f - y_pred[i], epsilon);
        loss += (y_true[i] * std::log(p)) + ((1.0f - y_true[i]) * std::log(p_inv));
    }
    return -(loss / static_cast<float>(y_true.size()));
}

/**
 * @brief Calculates Categorical Cross-Entropy loss between true and predicted probability distributions.
 * @param[in] y_true One-hot encoded true labels (vector of vectors, each inner vector has one 1.0 and rest 0.0).
 * @param[in] y_pred Predicted probabilities (logits, softmax applied internally).
 * @return The Categorical Cross-Entropy loss.
 * @throws std::runtime_error if vector sizes mismatch or y_true is invalid.
 */
float categoricalCrossEntropy(std::vector<std::vector<float>>& y_true, std::vector<std::vector<float>>& y_pred) {
    if (y_true.size() != y_pred.size() || y_true.empty()) {
        throw std::runtime_error("categoricalCrossEntropy: Vector sizes mismatch or empty");
    }
    size_t num_classes = y_true[0].size();
    std::vector<std::vector<float>> y_pred_softmax = softmax(y_pred, 1.0f); // Apply softmax to logits
    float loss = 0.0f;
    float epsilon = 1e-15f;
    for (size_t i = 0; i < y_true.size(); ++i) {
        if (y_true[i].size() != num_classes) {
            throw std::runtime_error("y_true[" + std::to_string(i) + "] size mismatch");
        }
        for (size_t j = 0; j < num_classes; ++j) {
            if (y_true[i][j] != 0.0f && y_true[i][j] != 1.0f) {
                throw std::runtime_error("y_true[" + std::to_string(i) + "][" + std::to_string(j) + "] not 0 or 1");
            }
            float p = std::max<float>(y_pred_softmax[i][j], epsilon);
            loss += y_true[i][j] * std::log(p);
        }
    }
    return -loss / static_cast<float>(y_true.size());
}
