
// activation functions
#include "include/basic.hpp"

//----------------SIGMOID----------------//

/**
 * @brief Sigmoid activation function. Applies the sigmoid function to the input value.
 * @param x Input value
 * @return Sigmoid of x
 */
float sigmoid(const float& x) {
    // The sigmoid function is defined as 1 / (1 + exp(-x)).
    return (1 / (1 + std::exp(-x)));
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
    return s * (1 - s);
}

/**
 * @brief Applies the sigmoid function to each element of a vector.
 * @param x A reference to the input vector. The function modifies this
 * vector in-place.
 * @return Sigmoid of vector: element-wise
 */
std::vector<float> sigmoid(const std::vector<float>& x) {
    std::vector<float> y(x);
    std::transform(x.begin(), x.end(), y.begin(), [](float& i){ return sigmoid(i); });
    return y;
}

/**
 * @brief Derivative of sigmoid activation function. This function calculates 
 *      the derivative of the sigmoid function for each element of the input vector.
 * @param x Input vector
 * @return A vector where each element is the derivative of the corresponding 
 *      element in the input vector.
 */
std::vector<float> sigmoidder(const std::vector<float>& x) {
    // Create a copy of the input vector
    std::vector<float> y(x);
    // Apply the derivative of sigmoid function to each element of the input vector
    std::transform(x.begin(), x.end(), y.begin(), [](float& i){ return sigmoidder(i); });
    return y;
}

/**
 * @brief Apply sigmoid function to each element of a matrix.
 * @param x A reference to the input matrix. 
 * @return Sigmoid of matrix: element-wise
 */
std::vector<std::vector<float>> sigmoid(const std::vector<std::vector<float>>& x) {
    std::vector<std::vector<float>> result(x.size(), std::vector<float>(x[0].size()));
    for (size_t i = 0; i < x.size(); ++i) {
        for(size_t j = 0; j < x[0].size(); ++j) {
            result[i][j] = sigmoid(x[i][j]);
        }
    }
    return result;
}

/**
 * @brief Derivative of sigmoid activation function. This function calculates 
 *      the derivative of the sigmoid function for each element of the input matrix.
 * @param x Input matrix
 * @return A matrix where each element is the derivative of the corresponding 
 *      element in the input matrix.
 */
std::vector<std::vector<float>> sigmoidder(const std::vector<std::vector<float>>& x) {
    std::vector<std::vector<float>> result(x.size(), std::vector<float>(x[0].size()));
    for (size_t i = 0; i < x.size(); ++i) {
        for(size_t j = 0; j < x[0].size(); ++j) {
            result[i][j] = sigmoidder(x[i][j]);
        }
    }
    return result;
}

//----------------SOFTMAX----------------//

/**
 * @brief Softmax activation function. Applies the softmax function to each element of a vector.
 *          The softmax function is used to map a vector of real numbers to a probability distribution.
 *          It is often used in the output layer of a neural network to output probabilities.
 *          The softmax function is defined as follows: softmax(x) = exp(x) / sum(exp(x))
 * @param x The input vector
 * @param temp The temperature parameter for the softmax function. A high temperature results in a more 
 *          uniform distribution.
 * @return Vector of softmax values
 */
std::vector<float> softmax(const std::vector<float>& x, float& temp) {
    if (x.empty()) return {};
    
    float inv_temp = 1.0f / temp;
    float max_val = *std::max_element(x.begin(), x.end());
    
    std::vector<float> exps(x.size());
    float sum = 0.0f;
    for (size_t i = 0; i < x.size(); ++i) {
        exps[i] = std::exp((x[i] - max_val) * inv_temp);
        sum += exps[i];
    }

    if (sum == 0.0f) return std::vector<float>(x.size(), 1.0f / x.size());

    for (float& val : exps) val /= sum;
    return exps;
}



/**
 * @brief Derivative of softmax activation function This function calculates the derivative of 
 *          the softmax function for a vector of input values. The derivative of the softmax 
 *          function is given by: softmax_derivative(x) = softmax(x) * (1 - softmax(x)). The 
 *          derivative of softmax(x) is a vector of the same size as the input vector, where each 
 *          element is the derivative of the softmax function with respect to the corresponding 
 *          input value.
 * @param x Input vector
 * @param temp Temperature parameter for the softmax function. A high temperature results in a more 
 *          uniform distribution.
 * @return The derivative of softmax(x) for each input value
 */
std::vector<float> softmaxder(const std::vector<float>& x, float& temp) {
    auto s = softmax(x, temp);
    std::vector<float> grad(s.size());

    for (size_t i = 0; i < s.size(); ++i)
        grad[i] = s[i] * (1.0f - s[i]);

    return grad;

}


/**
 * @brief Softmax activation function. Applies the softmax function to each element of a 2D vector.
 *          The softmax function is used to map a vector of real numbers to a probability distribution.
 *          It is often used in the output layer of a neural network to output probabilities.
 *          The softmax function is defined as follows: softmax(x) = exp(x) / sum(exp(x))
 * @param x The 2D input vector
 * @param temp The temperature parameter for the softmax function. A high temperature results in a more 
 *          uniform distribution.
 * @return Vector of softmax values
 */
std::vector<std::vector<float>> softmax(const std::vector<std::vector<float>>& x, float temp) {
    std::vector<std::vector<float>> result;
    result.reserve(x.size());

    for (const auto& row : x)
        result.push_back(softmax(row, temp));

    return result;
}


/**
 * @brief Derivative of softmax activation function. Calculates the derivative of the softmax function for each element of a 2D vector.
 *          The derivative of the softmax function is given by: softmax_derivative(x) = softmax(x) * (1 - softmax(x)). The
 *          derivative of softmax(x) is a vector of the same size as the input vector, where each
 *          element is the derivative of the softmax function with respect to the corresponding
 *          input value.
 * @param x Input 2D vector
 * @param temp Temperature parameter for the softmax function. A high temperature results in a more
 *          uniform distribution.
 * @return The derivative of softmax(x) for each input value
 */
std::vector<std::vector<float>> softmaxder(const std::vector<std::vector<float>>& x, float temp) {
    std::vector<std::vector<float>> result;
    result.reserve(x.size());

    for (const auto& row : x)
        result.push_back(softmaxder(row, temp));

    return result;
}

//----------------ReLU----------------//

/**
 * @brief ReLU activation function. Calculates the ReLU of a single input value.
 * @param x Input value
 * @return The ReLU of x, which is the maximum of 0 and x
 */
float ReLU(const float& x) {
    // The ReLU function is defined as max(0, x)
    return std::max(float(0), x); // Return the maximum of 0 and x
}

/**
 * @brief Calculates the derivative of the ReLU (Rectified Linear Unit)
 *      activation function. The derivative of the ReLU function is 0 if the input value is less than 0,
 *      and 1 otherwise.
 * @param x Input value
 * @return 0 if x < 0, 1 otherwise
 */
float ReLUder(const float& x) {
    // The derivative of the ReLU function is 0 if the input value is less than 0, and 1 otherwise.
    return (x > 0) ? 1 : 0; // Return 1 if x > 0, 0 otherwise
}

/**
 * @brief ReLU activation function. Applies the ReLU function to each element of a vector.
 * @param x Input vector
 * @return A vector where each element is the ReLU of the corresponding element in the input vector.
 */
std::vector<float> ReLU(const std::vector<float>& x) {
    // Create a copy of the input vector
    std::vector<float> y(x);
    // Apply the ReLU function to each element of the input vector
    std::transform(x.begin(), x.end(), y.begin(), [](float& i){ return ReLU(i); });
    return y;
}

/**
 * @brief Derivative of ReLU activation function. Applies the ReLU function to each element of a vector.
 * @param x Input vector
 * @return A vector where each element is the ReLU of the corresponding element in the input vector.
 */
std::vector<float> ReLUder(const std::vector<float>& x) {
    // Create a copy of the input vector
    std::vector<float> y(x);
    // Apply the ReLU function to each element of the input vector
    std::transform(x.begin(), x.end(), y.begin(), [](float& i){ return ReLUder(i); });
    return y;
}

/**
 * @brief ReLU of 2D vector
 * @param x input matrix
 * @param t allowable terms
 */
std::vector<std::vector<float>> ReLU(const std::vector<std::vector<float>>& x, int& t) {
    std::vector<std::vector<float>> result(t, std::vector<float>(t, 0.0f));
    for (int i = 0; i < t; i++) {
        for(int j = 0; j < t; j++) {
            result[i][j] = ReLU(x[i][j]);
        }
    }
    return result;
}

/**
 * @brief Calculate the derivative of the ReLU activation function for each element in a matrix.
 *      The derivative of the ReLU function is 0 if the input value is less than 0, and 1 otherwise.
 *      This function applies the derivative of the ReLU function to each element in the input matrix.
 * @param x input matrix
 */
std::vector<std::vector<float>> ReLUder(const std::vector<std::vector<float>>& x, int& t) {
    std::vector<std::vector<float>> result(t, std::vector<float>(t, 0.0f));
    for (int i = 0; i < t; i++) {
        for(int j = 0; j < t; j++) {
            result[i][j] = ReLUder(x[i][j]);
        }
    }
    return result;
}

//----------------Least of them all----------------//

/**
 * @brief Applies the LOTA (Least Of Them All) activation function to a 2D vector.
 *        The LOTA function is defined as:
 *        f(x) = x - min(x) for each element, and
 *        f(x) = f(x) / sum(f(x)) for normalization
 * @param y Input 2D vector
 * @return A 2D vector where each vector is the result of the LOTA function applied to the corresponding vector in the input.
 */
std::vector<float> LOTA(const std::vector<float>& y) {
    if(y.empty()) {
        return {0};
    }

    if(y.size() == 1)
        return {1};
    // Create a copy of the input vector
    std::vector<float> x(y);
    // Find the minimum value in the input vector
    float min_val = 0.0;
    min_val = *std::min_element(x.begin(), x.end());
    min_val = std::abs(min_val);
    // Subtract the minimum value from each element in the vector
    std::transform(x.begin(), x.end(), x.begin(), [&min_val](float& i){ return (i + min_val); });
    // Calculate the sum of the elements in the vector
    float sum = std::accumulate(x.begin(), x.end(), 0.0);
    // Normalize the vector by dividing each element by the sum
    std::transform(x.begin(), x.end(), x.begin(), [&sum](float& i){ return i / sum; });
    return x;
}


/**
 * @brief Calculates the derivative of the LOTA (Least Of Them All) activation function for a vector.
 *        This function calculates the derivative of the LOTA function for each element in a vector.
 *        The LOTA derivative is defined as:
 *        f'(x) = (sum - x) / sum^2 for normalization
 * @param y Input vector
 * @return A vector where each element is the derivative of the LOTA function applied to the corresponding element in the input vector.
 */
std::vector<float> LOTAder(const std::vector<float>& y) {
    // Create a copy of the input vector
    std::vector<float> v(y);
    // Find the minimum value in the entire vector
    float min_val =  *std::min_element(v.begin(), v.end());
    min_val = std::abs(min_val);
    // Subtract the minimum value from each element in the vector
    std::transform(v.begin(), v.end(), v.begin(), [&min_val](float& i){ return (i + min_val); });
    // Calculate the sum of the elements in the vector
    float sum = std::accumulate(v.begin(), v.end(), 0.0);
    // Normalize the vector by dividing each element by the sum
    std::transform(v.begin(), v.end(), v.begin(), [&sum](float& i){ return ((sum - i) / static_cast<float>(std::pow(sum, 2))); });
    return v;
}


/**
 * @brief Applies the LOTA (Least Of Them All) activation function to a 2D vector.
 *        The LOTA function is defined as:
 *        f(x) = x - min(x) for each element, and
 *        f(x) = f(x) / sum(f(x)) for normalization
 * @param y Input 2D vector
 * @param t allowable terms
 * @return A 2D vector where each vector is the result of the LOTA function applied to the corresponding vector in the input.
 */
std::vector<std::vector<float>> LOTA(const std::vector<std::vector<float>>& y, int& t, bool& attentionType) {
    // Create a copy of the input 2D vector
    std::vector<std::vector<float>> x = y; // Operate on copy
    if (y.empty() || y[0].empty()) return {{}};

    if(y.size() == 1 && y[0].size() == 1 && t == 1) {
        x[0][0] = 1.0f; // Ensure float
        return x;
    }

    // Find the minimum value in the relevant region
    float min_val = std::numeric_limits<float>::max(); // Initialize with max float
    bool found_value = false;
    for (int i = 0; i < t; ++i) {
        int limit_j = attentionType ? (i + 1) : t; // Corrected limit for attentionType=1
        limit_j = std::min(limit_j, (int)x[i].size()); // Boundary check
        for (int j = 0; j < limit_j; ++j) {
             if (i < x.size()) { // Boundary check
                min_val = std::min(min_val, x[i][j]);
                found_value = true;
             }
        }
    }
     if (!found_value) min_val = 0.0f; // Handle empty relevant region

    float abs_min_val = std::abs(min_val);

    // Transform relevant elements: element + abs(min_val)
    float sum = 0.0f;
    int relevant_count = 0;
    for (int i = 0; i < t; ++i) {
        int limit_j = attentionType ? (i + 1) : t; // Corrected limit
        limit_j = std::min(limit_j, (int)x[i].size()); // Boundary check
         if (i < x.size()) { // Boundary check
            for (int j = 0; j < limit_j; ++j) {
                x[i][j] = x[i][j] + abs_min_val;
                sum += x[i][j];
                relevant_count++;
            }
            // Zero out non-relevant elements if attentionType=1
             if (attentionType) {
                 for (int j = limit_j; j < x[i].size(); ++j) {
                     x[i][j] = 0.0f;
                 }
             }
        }
    }

    // Normalize relevant elements by the global sum
    if (sum > 0.0f) {
        for (int i = 0; i < t; ++i) {
            int limit_j = attentionType ? (i + 1) : t; // Corrected limit
            limit_j = std::min(limit_j, (int)x[i].size()); // Boundary check
             if (i < x.size()) { // Boundary check
                for (int j = 0; j < limit_j; ++j) {
                    x[i][j] /= sum;
                }
            }
        }
    } 
    else if (relevant_count > 0) {
        // Handle sum=0 case (e.g., all relevant elements were -abs_min_val)
        float uniform_val = 1.0f / relevant_count;
         for (int i = 0; i < t; ++i) {
            int limit_j = attentionType ? (i + 1) : t; // Corrected limit
            limit_j = std::min(limit_j, (int)x[i].size()); // Boundary check
             if (i < x.size()) { // Boundary check
                for (int j = 0; j < limit_j; ++j) {
                    x[i][j] = uniform_val; // Assign uniform probability
                }
            }
        }
    }
    // Non-relevant elements remain 0 (or their original value if not attentionType=1 and j>=t)

    return x; // Return the normalized 2D vector
}


/**
 * @brief Derivative of the LOTA (Least Of Them All) activation function for a 2D vector.
 *        This function calculates the derivative of the LOTA function for each element
 *        in a 2D vector. The LOTA derivative is defined as:
 *        f'(x) = (sum - x) / sum^2 for normalization
 * @param y Input 2D vector
 * @param t allowable terms
 * @return A 2D vector where each element is the derivative of the LOTA function applied 
 *         to the corresponding element in the input vector.
 */
std::vector<std::vector<float>> LOTAder(const std::vector<std::vector<float>>& y, int& t, bool& attentionType) {
    if (y.empty() || y[0].empty()) return {{}};
    std::vector<std::vector<float>> x = y; // Create a copy

    // Find the minimum value *in the relevant region*
    float min_val = std::numeric_limits<float>::max();
    bool found_value = false;
    for (int i = 0; i < t; ++i) {
        int limit_j = attentionType ? (i + 1) : t; // Corrected limit
        limit_j = std::min(limit_j, (int)x[i].size()); // Boundary check
         if (i < x.size()) { // Boundary check
            for (int j = 0; j < limit_j; ++j) {
                min_val = std::min(min_val, x[i][j]);
                found_value = true;
            }
        }
    }
    if (!found_value) min_val = 0.0f;

    float abs_min_val = std::abs(min_val);

    // Calculate the sum of (element + abs(min_val)) *in the relevant region*
    float sum = 0.0f;
    int relevant_count = 0;
    std::vector<std::vector<float>> transformed_x = x; // Store transformed values temporarily
    for (int i = 0; i < t; ++i) {
        int limit_j = attentionType ? (i + 1) : t; // Corrected limit
        limit_j = std::min(limit_j, (int)x[i].size()); // Boundary check
         if (i < x.size()) { // Boundary check
            for (int j = 0; j < limit_j; ++j) {
                transformed_x[i][j] = x[i][j] + abs_min_val;
                sum += transformed_x[i][j];
                relevant_count++;
            }
        }
    }

    // Calculate the derivative for each element *in the relevant region*
    float sum_sq = sum * sum; // Use float for pow(sum, 2)
    if (sum > 0.0f) { // Avoid division by zero in derivative
        for (int i = 0; i < t; ++i) {
            int limit_j = attentionType ? (i + 1) : t; // Corrected limit
            limit_j = std::min(limit_j, (int)x[i].size()); // Boundary check
             if (i < x.size()) { // Boundary check
                for (int j = 0; j < limit_j; ++j) {
                    // Derivative: (sum - transformed_element) / sum^2
                    x[i][j] = (sum - transformed_x[i][j]) / sum_sq;
                }
                // Zero out non-relevant elements if attentionType=1
                 if (attentionType) {
                     for (int j = limit_j; j < x[i].size(); ++j) {
                         x[i][j] = 0.0f;
                     }
                 }
            }
        }
    } 
    else {
         // Handle sum=0 case for derivative (derivative is likely 0 or undefined)
         for (int i = 0; i < t; ++i) {
            int limit_j = attentionType ? (i + 1) : t; // Corrected limit
            limit_j = std::min(limit_j, (int)x[i].size()); // Boundary check
             if (i < x.size()) { // Boundary check
                for (int j = 0; j < limit_j; ++j) {
                    x[i][j] = 0.0f; // Set derivative to 0
                }
                 // Zero out non-relevant elements if attentionType=1
                 if (attentionType) {
                     for (int j = limit_j; j < x[i].size(); ++j) {
                         x[i][j] = 0.0f;
                     }
                 }
            }
        }
    }
    // Non-relevant elements remain 0 (or their original value if not attentionType=1 and j>=t)

    return x; // Return the derived 2D vector
}
