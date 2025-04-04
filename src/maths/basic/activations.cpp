
// activation functions
#include "include/basic.hpp"

//----------------SIGMOID----------------//

/**
 * @brief Sigmoid activation function. Applies the sigmoid function to the input value.
 * @param x Input value
 * @return Sigmoid of x
 */
float sigmoid(float& x) {
    // The sigmoid function is defined as 1 / (1 + exp(-x)).
    return (1 / (1 + std::exp(-x)));
}

/**
 * @brief Derivative of sigmoid activation function. Calculates the derivative of the sigmoid function.
 *      The derivative of sigmoid(x) is given by: sigmoid_derivative(x) = sigmoid(x) * (1 - sigmoid(x))
 * @param x Input value
 * @return The derivative of sigmoid(x)
 */
float sigmoidder(float& x) {
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
std::vector<float> sigmoid(std::vector<float>& x) {
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
std::vector<float> sigmoidder(std::vector<float> x) {
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
std::vector<std::vector<float>> sigmoid(std::vector<std::vector<float>>& x) {
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
std::vector<std::vector<float>> sigmoidder(std::vector<std::vector<float>>& x) {
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
std::vector<float> softmax(std::vector<float> x, float temp = 1.0) {
    // Create a copy of the input vector
    std::vector<float> y(x);
    // Calculate the sum of the exponentials of the vector elements
    float sum = 0;
    // Calculate the sum of the exponentials of the vector elements
    std::transform(y.begin(), y.end(), y.begin(), [temp](float& val) { return exp(val / temp); });
    sum = std::accumulate(y.begin(), y.end(), 0.0);
    // Normalize the exponentials by dividing each by the sum
    std::transform(y.begin(), y.end(), y.begin(), [sum](float val) {
        return val / sum;
    });
    // Return the normalized vector
    return y;
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
std::vector<float> softmaxder(std::vector<float> x, float temp = 1.0) {
    // Create a copy of the input vector
    std::vector<float> y(x);
    // Calculate the sum of exponential of each element of the input vector
    float sum = 0;
    std::for_each(y.begin(), y.end(), [&sum, temp](float& val) { sum += exp(val/temp); });
    // Calculate the softmax of each element of the input vector
    std::for_each(y.begin(), y.end(), [&sum](float& val) { val = exp(val) / sum; });
    // Calculate the derivative of softmax(x) for each input value
    std::vector<float> result(y.size(), 0.0);
    for (size_t i = 0; i < y.size(); ++i) {
        // Calculate the derivative of softmax(x) using the formula: softmax_derivative(x) = softmax(x) * (1 - softmax(x))
        result[i] = y[i] * (1 - y[i]);
        // Subtract the softmax of each other element from the derivative of softmax(x)
        for (size_t j = 0; j < y.size(); ++j) {
            if (i == j) {
                continue;
            }
            result[i] -= y[j];
        }
    }
    // Return the vector of derivatives
    return result;
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
std::vector<std::vector<float>> softmax(std::vector<std::vector<float>> x, float temp = 1.0) {
    // Create a copy of the input vector
    std::vector<std::vector<float>> y(x);
    // Calculate the sum of the exponentials of the vector elements
    float sum = 0.0;
    for (auto& v : x) {
        std::transform(v.begin(), v.end(), v.begin(), [&temp](float& i){ return exp(i/temp); });
        sum += std::accumulate(v.begin(), v.end(), 0.0);
    }
    // Normalize each element by dividing it by the total sum
    for (auto& v: x) {
        std::transform(v.begin(), v.end(), v.begin(), [&sum](float& i){ return i / sum; });
    }
    return y;
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
std::vector<std::vector<float>> softmaxder(std::vector<std::vector<float>> x, float temp = 1.0) {
    // Create a copy of the input vector
    std::vector<std::vector<float>> y(x);
    // Calculate the sum of the exponentials of the vector elements
    float sum = 0.0;
    for (auto& v : x) {
        std::transform(v.begin(), v.end(), v.begin(), [&temp](float& i){ return exp(i/temp); });
        sum += std::accumulate(v.begin(), v.end(), 0.0);
    }
    // Normalize each element by dividing it by the total sum
    for (auto& v: x) {
        std::transform(v.begin(), v.end(), v.begin(), [&sum](float& i){ return i / sum; });
    }
    std::vector<std::vector<float>> result(y.size(), std::vector<float>(y[0].size()));       // Initialize the result vector
    // Calculate the derivative of softmax(x) for each input value
    for (size_t i = 0; i < y.size(); ++i) {
        for (size_t j = 0; j < y[0].size(); ++j) {
            result[i][j] = y[i][j] * (1 - y[i][j]);
            // Subtract the softmax of each other element from the derivative of softmax(x)
            std::transform(y[i].begin(), y[i].end(), result[i].begin(), [i, &y](float val){ 
                float sum = 0.0;
                for (size_t k = 0; k < y[0].size(); ++k) {
                    if (k == i) {
                        continue;
                    }
                    sum += y[i][k];
                }
                return val * (1 - val) - sum;
            });
        }
    }
    return y;
}

//----------------ReLU----------------//

/**
 * @brief ReLU activation function. Calculates the ReLU of a single input value.
 * @param x Input value
 * @return The ReLU of x, which is the maximum of 0 and x
 */
float ReLU(float& x) {
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
float ReLUder(float& x) {
    // The derivative of the ReLU function is 0 if the input value is less than 0, and 1 otherwise.
    return (x > 0) ? 1 : 0; // Return 1 if x > 0, 0 otherwise
}

/**
 * @brief ReLU activation function. Applies the ReLU function to each element of a vector.
 * @param x Input vector
 * @return A vector where each element is the ReLU of the corresponding element in the input vector.
 */
std::vector<float> ReLU(std::vector<float> x) {
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
std::vector<float> ReLUder(std::vector<float> x) {
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
std::vector<std::vector<float>> ReLU(std::vector<std::vector<float>>& x, int& t) {
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
std::vector<std::vector<float>> ReLUder(std::vector<std::vector<float>>& x, int& t) {
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
std::vector<float> LOTA(std::vector<float>& y) {
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
std::vector<float> LOTAder(std::vector<float>& y) {
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
std::vector<std::vector<float>> LOTA(std::vector<std::vector<float>>& y, int& t, bool& attentionType) {
    // Create a copy of the input 2D vector
    std::vector<std::vector<float>> x(y);
    // Create a copy of the input 2D vector
    if(y.size() == 1 && y[0].size() == 1) {
        x[0][0] = 1;
        return x;
    }
    // set all the values of x which are above diagonal and right side of diagonal to 0
    if(attentionType == 1) {
        for (int i = 0; i < t; i++) {
            for(int j = i+1; j < t; j++) {
                x[i][j] = 0;
            }
        }
    }
    // Find the minimum value in the entire 2D vector
    float min_val = 0.0;
    for (int i = 0; i < t; i++) {
        for(int j = 0; j < (attentionType ? i : t); j++) {
            if (x[i][j] < min_val)
                min_val = x[i][j];
        }
    }
    min_val = std::abs(min_val);
    // Subtract the minimum value from each element in the 2D vector
    for (int i = 0; i < t; i++) {
        std::transform(x[i].begin(), x[i].begin() + (attentionType ? i : t), x[i].begin(), [&min_val](float& j){ 
            return (j + min_val); 
        });
    }
    float sum = 0.0;        // Variable to store the sum of all elements
    // Calculate the sum of all elements in the 2D vector
    for (int i = 0; i < t; i++) {
        sum += std::accumulate(x[i].begin(), x[i].begin() + (attentionType ? i : t), 0.0);
    }
    // Normalize each element by dividing it by the total sum
    for (int i = 0; i < t; i++) {
        std::transform(x[i].begin(), x[i].begin() + (attentionType ? i : t), x[i].begin(), [&sum](float& j){ 
            return j / sum;
        });
    }
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
std::vector<std::vector<float>> LOTAder(std::vector<std::vector<float>>& y, int& t, bool& attentionType) {
    // Create a copy of the input 2D vector
    std::vector<std::vector<float>> x(y);
    // Find the minimum value in the entire 2D vector
    float min_val = 0.0; 
    for (int i = 0; i < t; i++) {
        for(int j = 0; j < t; j++) {
            if (x[i][j] < min_val)
                min_val = x[i][j];
        }
    }
    min_val = std::abs(min_val);
    // Subtract the minimum value from each element in the 2D vector
    for (int i = 0; i < t; i++) {
        std::transform(x[i].begin(), x[i].begin() + t, x[i].begin(), [&min_val](float& i){ return (i + min_val); });
    }
    float sum = 0.0; // Variable to store the sum of all elements
    // Calculate the sum of all elements in the 2D vector
    for (int i = 0; i < t; i++) {
        sum += std::accumulate(x[i].begin(), x[i].begin() + t, 0.0);
    }
    // Calculate the derivative of the LOTA function for each element
    for (int i = 0; i < t; i++) {
        std::transform(x[i].begin(), x[i].begin() + t, x[i].begin(), [&sum](float& i) {
            return ((sum - i) / static_cast<float>(std::pow(sum, 2)));
        });
    }
    return x; // Return the derived 2D vector
}
