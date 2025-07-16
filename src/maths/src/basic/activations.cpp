// activation functions
#include "include/basic.hpp"
#include <vector>       // Ensure vector is included
#include <cmath>        // For std::exp, std::max, std::pow, std::abs
#include <numeric>      // For std::accumulate, std::inner_product
#include <algorithm>    // For std::transform, std::max_element, std::min_element, std::sort
#include <limits>       // For std::numeric_limits
#include <stdexcept>    // For std::invalid_argument (used in expectation)

//----------------SIGMOID----------------//

/**
 * @brief Sigmoid activation function. Applies the sigmoid function to the input value.
 * @param x Input value
 * @return Sigmoid of x
 */
float sigmoid(const float& x) {
    // The sigmoid function is defined as 1 / (1 + exp(-x)).
    // Use std::exp for clarity and standard library usage
    return (1.0f / (1.0f + std::exp(-x))); // Use float literals
}

/**
 * @brief Derivative of sigmoid activation function. Calculates the derivative of the sigmoid function.
 *      The derivative of sigmoid(x) is given by: sigmoid_derivative(x) = sigmoid(x) * (1 - sigmoid(x))
 * @param x Input value
 * @return The derivative of sigmoid(x)
 */
float sigmoidder(const float& x) {
    // Calculate the sigmoid of x
    float s = sigmoid(x);
    // Calculate the derivative of sigmoid(x)
    return s * (1.0f - s); // Use float literal
}

/**
 * @brief Applies the sigmoid function to each element of a vector.
 * @param x A const reference to the input vector.
 * @return A new vector containing the element-wise sigmoid results.
 */
std::vector<float> sigmoid(const std::vector<float>& x) {
    std::vector<float> y = x; // Create a copy to store results
    // CORRECTED LAMBDA PARAMETER: const float& i
    std::transform(x.begin(), x.end(), y.begin(), [](const float& i){ return sigmoid(i); });
    return y;
}

/**
 * @brief Derivative of sigmoid activation function for a vector. This function calculates
 *      the derivative of the sigmoid function for each element of the input vector.
 * @param x Input vector (const reference)
 * @return A new vector where each element is the derivative of the corresponding
 *      element in the input vector.
 */
std::vector<float> sigmoidder(const std::vector<float>& x) {
    // Create a copy of the input vector to store results
    std::vector<float> y = x;
    // CORRECTED LAMBDA PARAMETER: const float& i
    std::transform(x.begin(), x.end(), y.begin(), [](const float& i){ return sigmoidder(i); });
    return y;
}

/**
 * @brief Apply sigmoid function to each element of a matrix.
 * @param x A const reference to the input matrix.
 * @return A new matrix containing the element-wise sigmoid results.
 */
std::vector<std::vector<float>> sigmoid(const std::vector<std::vector<float>>& x) {
    if (x.empty()) return {}; // Handle empty matrix
    size_t rows = x.size();
    size_t cols = x[0].size(); // Assume non-ragged
    std::vector<std::vector<float>> result(rows, std::vector<float>(cols));
    for (size_t i = 0; i < rows; ++i) {
        // Check for ragged matrix (optional but good practice)
        // if (x[i].size() != cols) { /* Handle error */ }
        for(size_t j = 0; j < cols; ++j) {
            result[i][j] = sigmoid(x[i][j]);
        }
    }
    return result;
}

/**
 * @brief Derivative of sigmoid activation function for a matrix. This function calculates
 *      the derivative of the sigmoid function for each element of the input matrix.
 * @param x Input matrix (const reference)
 * @return A new matrix where each element is the derivative of the corresponding
 *      element in the input matrix.
 */
std::vector<std::vector<float>> sigmoidder(const std::vector<std::vector<float>>& x) {
     if (x.empty()) return {}; // Handle empty matrix
    size_t rows = x.size();
    size_t cols = x[0].size(); // Assume non-ragged
    std::vector<std::vector<float>> result(rows, std::vector<float>(cols));
    for (size_t i = 0; i < rows; ++i) {
         // if (x[i].size() != cols) { /* Handle error */ }
        for(size_t j = 0; j < cols; ++j) {
            result[i][j] = sigmoidder(x[i][j]);
        }
    }
    return result;
}

//----------------SOFTMAX----------------//

/**
 * @brief Softmax activation function for a vector. Applies the softmax function.
 *          Numerically stable version using max_val subtraction.
 * @param x The input vector (const reference).
 * @param temp The temperature parameter (passed by value, as it's not modified).
 * @return Vector of softmax probabilities.
 */
std::vector<float> softmax(const std::vector<float>& x, float temp) { // Pass temp by value
    if (x.empty()) return {};

    if (temp <= 0.0f) {
        // Handle invalid temperature, maybe throw or return uniform distribution
        // For now, return uniform as a fallback
         return std::vector<float>(x.size(), 1.0f / static_cast<float>(x.size()));
    }

    float inv_temp = 1.0f / temp;
    // Find max element for numerical stability
    float max_val = *std::max_element(x.begin(), x.end());

    std::vector<float> exps(x.size());
    float sum = 0.0f;
    for (size_t i = 0; i < x.size(); ++i) {
        // Subtract max_val before exponentiating
        exps[i] = std::exp((x[i] - max_val) * inv_temp);
        sum += exps[i];
    }

    // Handle case where sum is zero (e.g., all inputs were -inf after scaling)
    if (sum == 0.0f || !std::isfinite(sum)) {
        // Return uniform distribution if sum is zero or non-finite
        return std::vector<float>(x.size(), 1.0f / static_cast<float>(x.size()));
    }

    // Normalize
    for (float& val : exps) {
         val /= sum;
    }
    return exps;
}


/**
 * @brief Derivative of softmax activation function for a vector.
 *          Calculates the diagonal elements of the Jacobian: s_i * (1 - s_i).
 * @param x Input vector (const reference).
 * @param temp Temperature parameter (passed by value).
 * @return The vector of derivatives s_i * (1 - s_i).
 */
std::vector<float> softmaxder(const std::vector<float>& x, float temp) { // Pass temp by value
    // Calculate softmax probabilities first
    std::vector<float> s = softmax(x, temp); // Use the corrected softmax
    if (s.empty()) return {};

    std::vector<float> grad(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        // Calculate the diagonal element of the Jacobian
        grad[i] = s[i] * (1.0f - s[i]);
    }
    return grad;
}


/**
 * @brief Softmax activation function for a matrix (applies row-wise).
 * @param x The 2D input vector (matrix, const reference).
 * @param temp The temperature parameter (passed by value).
 * @return Matrix with softmax applied row-wise.
 */
std::vector<std::vector<float>> softmax(const std::vector<std::vector<float>>& x, float temp) { // Pass temp by value
    std::vector<std::vector<float>> result;
    if (x.empty()) return result;

    result.reserve(x.size());
    for (const auto& row : x) {
        // Apply the vector softmax to each row
        result.push_back(softmax(row, temp));
    }
    return result;
}


/**
 * @brief Derivative of softmax activation function for a matrix (applies row-wise).
 * @param x Input 2D vector (matrix, const reference).
 * @param temp Temperature parameter (passed by value).
 * @return Matrix with softmax derivative applied row-wise.
 */
std::vector<std::vector<float>> softmaxder(const std::vector<std::vector<float>>& x, float temp) { // Pass temp by value
    std::vector<std::vector<float>> result;
     if (x.empty()) return result;

    result.reserve(x.size());
    for (const auto& row : x) {
        // Apply the vector softmax derivative to each row
        result.push_back(softmaxder(row, temp));
    }
    return result;
}

//----------------ReLU----------------//

/**
 * @brief ReLU activation function. Calculates the ReLU of a single input value.
 * @param x Input value (const reference).
 * @return The ReLU of x, which is the maximum of 0 and x.
 */
float ReLU(const float& x) {
    // The ReLU function is defined as max(0, x)
    return (std::max)(0.0f, x); // Use float literal 0.0f, parenthesize to avoid macro
}

/**
 * @brief Calculates the derivative of the ReLU (Rectified Linear Unit)
 *      activation function.
 * @param x Input value (const reference).
 * @return 1.0f if x > 0, 0.0f otherwise.
 */
float ReLUder(const float& x) {
    // The derivative of the ReLU function is 1 if x > 0, and 0 otherwise.
    return (x > 0.0f) ? 1.0f : 0.0f; // Use float literals
}

/**
 * @brief ReLU activation function for a vector. Applies the ReLU function element-wise.
 * @param x Input vector (const reference).
 * @return A new vector where each element is the ReLU of the corresponding element.
 */
std::vector<float> ReLU(const std::vector<float>& x) {
    // Create a copy to store results
    std::vector<float> y = x;
    // CORRECTED LAMBDA PARAMETER: const float& i
    std::transform(x.begin(), x.end(), y.begin(), [](const float& i){ return ReLU(i); });
    return y;
}

/**
 * @brief Derivative of ReLU activation function for a vector. Applies the ReLU derivative element-wise.
 * @param x Input vector (const reference).
 * @return A new vector where each element is the ReLU derivative of the corresponding element.
 */
std::vector<float> ReLUder(const std::vector<float>& x) {
    // Create a copy to store results
    std::vector<float> y = x;
    // CORRECTED LAMBDA PARAMETER: const float& i
    std::transform(x.begin(), x.end(), y.begin(), [](const float& i){ return ReLUder(i); });
    return y;
}

/**
 * @brief ReLU of 2D vector (matrix). Applies ReLU element-wise.
 * @param x input matrix (const reference).
 * @param t Number of rows/cols to process (assuming square submatrix, passed by value).
 * @return A new matrix containing element-wise ReLU results up to t x t.
 */
// Note: The 't' parameter seems unusual here. If you want to apply to the whole matrix, remove 't'.
// If 't' defines a submatrix size, the allocation should match 't'.
// Assuming 't' means process the first 't' rows and 't' columns.
std::vector<std::vector<float>> ReLU(const std::vector<std::vector<float>>& x, int t_val) {
    if (x.empty() || t_val <= 0) return {};
    size_t num_rows = (std::min)((size_t)t_val, x.size()); // Parenthesize to avoid macro
    std::vector<std::vector<float>> result;
    result.reserve(num_rows);

    for (size_t i = 0; i < num_rows; ++i) {
        size_t num_cols = std::min<size_t>((size_t)t_val, x[i].size()); // Parenthesize to avoid macro
        std::vector<float> row_result(num_cols);
        for(size_t j = 0; j < num_cols; ++j) {
            row_result[j] = ReLU(x[i][j]);
        }
        // If you need the result matrix to always be t x t, pad with zeros if necessary
        // row_result.resize(t, 0.0f); // Uncomment if padding is needed
        result.push_back(row_result);
    }
     // If you need the result matrix to always be t x t, pad rows if necessary
     // while (result.size() < t) { result.push_back(std::vector<float>(t, 0.0f)); } // Uncomment if padding is needed
    return result;
}

/**
 * @brief Calculate the derivative of the ReLU activation function for each element in a matrix.
 * @param x input matrix (const reference).
 * @param t Number of rows/cols to process (passed by value).
 * @return A new matrix containing element-wise ReLU derivative results up to t x t.
 */
std::vector<std::vector<float>> ReLUder(const std::vector<std::vector<float>>& x, int t_val) {
    if (x.empty() || t_val <= 0) return {};
    size_t num_rows = std::min<int>((size_t)t_val, x.size()); // Parenthesize to avoid macro
    std::vector<std::vector<float>> result;
    result.reserve(num_rows);

    for (size_t i = 0; i < num_rows; ++i) {
        size_t num_cols = (std::min)((size_t)t_val, x[i].size()); // Parenthesize to avoid macro
        std::vector<float> row_result(num_cols);
        for(size_t j = 0; j < num_cols; ++j) {
            row_result[j] = ReLUder(x[i][j]);
        }
        // Optional padding similar to ReLU matrix function if needed
        // row_result.resize(t, 0.0f);
        result.push_back(row_result);
    }
    // Optional padding similar to ReLU matrix function if needed
    // while (result.size() < t) { result.push_back(std::vector<float>(t, 0.0f)); }
    return result;
}


//----------------Least of them all (LOTA)----------------//

/**
 * @brief Applies the LOTA activation function to a vector.
 *        LOTA(x_i) = (x_i + abs(min(x))) / sum(x_j + abs(min(x)))
 * @param y Input vector (const reference).
 * @return A new vector containing the LOTA results.
 */
std::vector<float> LOTA(const std::vector<float>& y) {
    if (y.empty()) {
        return {}; // Return empty for empty input
    }
    if (y.size() == 1) {
        return {1.0f}; // Single element always results in probability 1
    }

    // Find the minimum value in the input vector
    float min_val = *std::min_element(y.begin(), y.end());
    float abs_min_val = std::abs(min_val);

    // Create a temporary vector for transformed values (x_i + abs(min_val))
    std::vector<float> transformed_x = y; // Copy constructor
    float sum = 0.0f;
    for(float& val : transformed_x) {
        val += abs_min_val;
        sum += val;
    }

    // Normalize the transformed vector
    if (sum > 0.0f) { // Avoid division by zero
        for(float& val : transformed_x) {
            val /= sum;
        }
    } else if (!transformed_x.empty()) {
        // Handle sum == 0 case (e.g., all elements were -abs_min_val) -> uniform distribution
        float uniform_prob = 1.0f / static_cast<float>(transformed_x.size());
        std::fill(transformed_x.begin(), transformed_x.end(), uniform_prob);
    }
    // If sum is non-positive and vector is empty, it returns empty anyway

    return transformed_x; // Return the normalized vector
}


/**
 * @brief Calculates the derivative of the LOTA activation function for a vector.
 *        LOTA'(x_i) = (sum - transformed_x_i) / sum^2
 *        where transformed_x_i = x_i + abs(min(x)) and sum = sum(transformed_x_j)
 * @param y Input vector (const reference).
 * @return A new vector containing the LOTA derivative results.
 */
std::vector<float> LOTAder(const std::vector<float>& y) {
     if (y.empty()) {
        return {};
    }
     if (y.size() == 1) {
         // Derivative for a single element LOTA(x)=1 is 0
         return {0.0f};
     }

    // Find the minimum value
    float min_val = *std::min_element(y.begin(), y.end());
    float abs_min_val = std::abs(min_val);

    // Calculate transformed values and their sum
    std::vector<float> transformed_x = y; // Copy
    float sum = 0.0f;
    for(float& val : transformed_x) {
        val += abs_min_val;
        sum += val;
    }

    // Calculate the derivative
    std::vector<float> derivative_x(y.size());
    float sum_sq = sum * sum; // Calculate sum squared once

    if (sum > 0.0f) { // Avoid division by zero
        for (size_t i = 0; i < y.size(); ++i) {
            // Derivative: (sum - transformed_element) / sum^2
            derivative_x[i] = (sum - transformed_x[i]) / sum_sq;
        }
    } else {
        // Handle sum == 0 case (derivative is likely 0 or undefined)
        std::fill(derivative_x.begin(), derivative_x.end(), 0.0f);
    }

    return derivative_x;
}


/**
 * @brief Applies the LOTA (least of them all) activation function to a 2D vector (matrix),
 *        considering only relevant elements defined by 't' and 'attentionType'.
 * @param y Input 2D vector (const reference).
 * @param t Dimension limit (passed by value).
 * @param attentionType If true, process only the lower triangle (incl. diagonal); otherwise, process up to t x t square (passed by value).
 * @return A new 2D vector with LOTA applied to relevant elements, others potentially zeroed.
 */
std::vector<std::vector<float>> LOTA(const std::vector<std::vector<float>>& y, int t, bool attentionType) { // Pass t and attentionType by value
    if (y.empty() || y[0].empty() || t <= 0) return {{}}; // Handle edge cases

    std::vector<std::vector<float>> x = y; // Operate on a copy

    // Handle 1x1 case explicitly if t=1
    if (t == 1 && !x.empty() && !x[0].empty()) {
        x[0][0] = 1.0f;
        // Zero out other elements if needed based on desired output shape
        // for (size_t j = 1; j < x[0].size(); ++j) x[0][j] = 0.0f;
        // for (size_t i = 1; i < x.size(); ++i) std::fill(x[i].begin(), x[i].end(), 0.0f);
        return x;
    }

    // Find the minimum value in the relevant region
    float min_val = (std::numeric_limits<float>::max)(); // Parenthesize to avoid macro
    bool found_value = false;
    size_t max_rows = (std::min)((size_t)t, x.size()); // Parenthesize to avoid macro
    for (size_t i = 0; i < max_rows; ++i) {
        size_t limit_j = attentionType ? (i + 1) : (size_t)t;
        limit_j = (std::min)(limit_j, x[i].size()); // Boundary check cols, parenthesize
        for (size_t j = 0; j < limit_j; ++j) {
            min_val = (std::min)(min_val, x[i][j]); // Parenthesize to avoid macro
            found_value = true;
        }
    }
    if (!found_value) 
        min_val = 0.0f; // Handle case where relevant region is effectively empty

    float abs_min_val = std::abs(min_val);

    // Transform relevant elements: element + abs(min_val) and calculate sum
    float sum = 0.0f;
    int relevant_count = 0;
    for (size_t i = 0; i < max_rows; ++i) {
        size_t limit_j = attentionType ? (i + 1) : (size_t)t;
        limit_j = (std::min)(limit_j, x[i].size()); // Boundary check cols, parenthesize
        for (size_t j = 0; j < limit_j; ++j) {
            x[i][j] += abs_min_val;
            sum += x[i][j];
            relevant_count++;
        }
        // Zero out non-relevant elements in the row if attentionType is true
        if (attentionType) {
            for (size_t j = limit_j; j < x[i].size(); ++j) {
                x[i][j] = 0.0f;
            }
        }
    }
    // Normalize relevant elements by the global sum
    if (sum > 0.0f) {
        for (size_t i = 0; i < max_rows; ++i) {
            size_t limit_j = attentionType ? (i + 1) : (size_t)t;
            limit_j = (std::min)(limit_j, x[i].size()); // Boundary check cols, parenthesize
            for (size_t j = 0; j < limit_j; ++j) {
                x[i][j] /= sum;
            }
        }
    } 
    else if (relevant_count > 0) {
        // Handle sum=0 case -> uniform distribution over relevant elements
        float uniform_val = 1.0f / static_cast<float>(relevant_count);
        for (size_t i = 0; i < max_rows; ++i) {
            size_t limit_j = attentionType ? (i + 1) : (size_t)t;
            limit_j = (std::min)(limit_j, x[i].size()); // Boundary check cols, parenthesize
            for (size_t j = 0; j < limit_j; ++j) {
                x[i][j] = uniform_val;
            }
        }
    }
    // Non-relevant elements remain 0 (if zeroed out) or their original value
    return x; // Return the modified 2D vector
}


/**
 * @brief Derivative of the LOTA activation function for a 2D vector (matrix),
 *        considering only relevant elements defined by 't' and 'attentionType'.
 * @param y Input 2D vector (const reference).
 * @param t Dimension limit (passed by value).
 * @param attentionType If true, process only the lower triangle; otherwise, up to t x t square (passed by value).
 * @return A new 2D vector with LOTA derivative applied to relevant elements.
 */
std::vector<std::vector<float>> LOTAder(const std::vector<std::vector<float>>& y, int t, bool attentionType) { // Pass t and attentionType by value
    if (y.empty() || y[0].empty() || t <= 0) return {{}};

    std::vector<std::vector<float>> result = y; // Create a copy to store derivatives

    // Handle 1x1 case explicitly if t=1
    if (t == 1 && !result.empty() && !result[0].empty()) {
        result[0][0] = 0.0f; // Derivative of LOTA(x)=1 is 0
        return result;
    }

    // Find the minimum value in the relevant region
    float min_val = (std::numeric_limits<float>::max)(); // Parenthesize to avoid macro
    bool found_value = false;
    size_t max_rows = (std::min)((size_t)t, y.size()); // Use y for reading min, parenthesize
    for (size_t i = 0; i < max_rows; ++i) {
        size_t limit_j = attentionType ? (i + 1) : (size_t)t;
        limit_j = (std::min)(limit_j, y[i].size()); // Boundary check cols, parenthesize
        for (size_t j = 0; j < limit_j; ++j) {
            min_val = (std::min)(min_val, y[i][j]); // Parenthesize to avoid macro
            found_value = true;
        }
    }
    if (!found_value) min_val = 0.0f;

    float abs_min_val = std::abs(min_val);
    // Calculate the sum of (element + abs(min_val)) in the relevant region
    // Also store transformed values temporarily
    float sum = 0.0f;
    std::vector<std::vector<float>> transformed_x = y; // Temp storage
    for (size_t i = 0; i < max_rows; ++i) {
        size_t limit_j = attentionType ? (i + 1) : (size_t)t;
        limit_j = (std::min)(limit_j, y[i].size()); // Boundary check cols, parenthesize
        for (size_t j = 0; j < limit_j; ++j) {
            transformed_x[i][j] = y[i][j] + abs_min_val; // Use original y here
            sum += transformed_x[i][j];
        }
    }

    // Calculate the derivative for each element in the relevant region
    float sum_sq = sum * sum;
    if (sum > 0.0f) { // Avoid division by zero
        for (size_t i = 0; i < max_rows; ++i) {
            size_t limit_j = attentionType ? (i + 1) : (size_t)t;
            limit_j = (std::min)(limit_j, y[i].size()); // Boundary check cols, parenthesize
            for (size_t j = 0; j < limit_j; ++j) {
                // Derivative: (sum - transformed_element) / sum^2
                result[i][j] = (sum - transformed_x[i][j]) / sum_sq;
            }
            // Zero out non-relevant elements in the row if attentionType is true
            if (attentionType) {
                for (size_t j = limit_j; j < result[i].size(); ++j) { // Use result here
                    result[i][j] = 0.0f; // Parenthesize to avoid macro
                }
            }
        }
    } 
    else {
        // Handle sum=0 case (derivative is likely 0 or undefined)
        for (size_t i = 0; i < max_rows; ++i) {
            size_t limit_j = attentionType ? (i + 1) : (size_t)t;
            limit_j = (std::min)(limit_j, y[i].size()); // Boundary check cols, parenthesize
            for (size_t j = 0; j < limit_j; ++j) {
                result[i][j] = 0.0f; // Set derivative to 0
            }
            // Zero out non-relevant elements if attentionType is true
            if (attentionType) {
                for (size_t j = limit_j; j < result[i].size(); ++j) {
                    result[i][j] = 0.0f;
                }
            }
        }
    }

    return result; // Return the derivative matrix
}
