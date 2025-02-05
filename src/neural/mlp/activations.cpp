
// activations functions
#include "include/activations.hpp"
#include <cmath>
#include <numeric>
#include <vector>
#include <algorithm>
#include <stdexcept>

//----------------ERRORS----------------//

/**
 * @brief Computes the Mean Squared Error (MSE) loss between the predicted and target vectors.
 * @param predicted The predicted output of the network.
 * @param target The target output of the network.
 * @return The mean squared error between the predicted and target vectors.
 */
double MSE(std::vector<double> predicted, std::vector<double> target) {
    if (predicted.size() != target.size()) {
        throw std::invalid_argument("Vectors must be of the same length");
    }
    double sum = 0.0;
    // Calculate the difference between the predicted and target vectors
    for (size_t i = 0; i < predicted.size(); ++i) {
        double diff = predicted[i] - target[i];
        // Square the difference and add it to the sum
        sum += diff * diff;
    }
    // Calculate the mean of the squared differences
    return sum / predicted.size();
}

/**
 * @brief Computes the Root Mean Squared Error (RMSE) loss between the predicted and target vectors.
 * @param predicted The predicted output of the network.
 * @param target The target output of the network.
 * @return The root mean squared error between the predicted and target vectors.
 */
double rMSE(std::vector<double> predicted, std::vector<double> target) {
    if (predicted.size() != target.size()) {
        throw std::invalid_argument("Vectors must be of the same length");
    }
    double sum = 0.0;
    // Calculate the difference between the predicted and target vectors
    for (size_t i = 0; i < predicted.size(); ++i) {
        double diff = predicted[i] - target[i];
        // Square the difference and add it to the sum
        sum += diff * diff;
    }
    // Calculate the mean of the squared differences
    sum = sum / predicted.size();
    // Calculate the root mean squared error
    return std::sqrt(sum);
}

//----------------SIGMOID----------------//

/**
 * @brief Sigmoid activation function. Applies the sigmoid function to the input value.
 * @param x Input value
 * @return Sigmoid of x
 */
double sigmoid(double x) {
    // The sigmoid function is defined as 1 / (1 + exp(-x)).
    return (1 / (1 + std::exp(-x)));
}

/**
 * @brief Derivative of sigmoid activation function. Calculates the derivative of the sigmoid function.
 *      The derivative of sigmoid(x) is given by: sigmoid_derivative(x) = sigmoid(x) * (1 - sigmoid(x))
 * @param x Input value
 * @return The derivative of sigmoid(x)
 */
double sigmoidder(double x) {
    // Calculate the sigmoid of x
    double s = sigmoid(x);
    // Calculate the derivative of sigmoid(x)
    return s * (1 - s);
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
std::vector<double> softmax(std::vector<double> x, double temp = 1.0) {
    // Create a copy of the input vector
    std::vector<double> y(x);
    // Calculate the sum of the exponentials of the vector elements
    double sum = 0;
    // Calculate the sum of the exponentials of the vector elements
    std::transform(y.begin(), y.end(), y.begin(), [temp](double& val) { return exp(val / temp); });
    sum = std::accumulate(y.begin(), y.end(), 0.0);
    // Normalize the exponentials by dividing each by the sum
    std::transform(y.begin(), y.end(), y.begin(), [sum](double val) {
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
std::vector<double> softmaxder(std::vector<double> x, double temp = 1.0) {
    // Create a copy of the input vector
    std::vector<double> y(x);
    // Calculate the sum of exponential of each element of the input vector
    double sum = 0;
    std::for_each(y.begin(), y.end(), [&sum, temp](double& val) { sum += exp(val/temp); });
    // Calculate the softmax of each element of the input vector
    std::for_each(y.begin(), y.end(), [&sum](double& val) { val = exp(val) / sum; });
    // Calculate the derivative of softmax(x) for each input value
    std::vector<double> result(y.size(), 0.0);
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
std::vector<std::vector<double>> softmax(std::vector<std::vector<double>> x, double temp = 1.0) {
    // Create a copy of the input vector
    std::vector<std::vector<double>> y(x);
    // Calculate the sum of the exponentials of the vector elements
    double sum = 0.0;
    for (auto& v : x) {
        std::transform(v.begin(), v.end(), v.begin(), [&temp](double& i){ return exp(i/temp); });
        sum += std::accumulate(v.begin(), v.end(), 0.0);
    }
    // Normalize each element by dividing it by the total sum
    for (auto& v: x) {
        std::transform(v.begin(), v.end(), v.begin(), [&sum](double& i){ return i / sum; });
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
std::vector<std::vector<double>> softmaxder(std::vector<std::vector<double>> x, double temp = 1.0) {
    // Create a copy of the input vector
    std::vector<std::vector<double>> y(x);
    // Calculate the sum of the exponentials of the vector elements
    double sum = 0.0;
    for (auto& v : x) {
        std::transform(v.begin(), v.end(), v.begin(), [&temp](double& i){ return exp(i/temp); });
        sum += std::accumulate(v.begin(), v.end(), 0.0);
    }
    // Normalize each element by dividing it by the total sum
    for (auto& v: x) {
        std::transform(v.begin(), v.end(), v.begin(), [&sum](double& i){ return i / sum; });
    }
    std::vector<std::vector<double>> result(y.size(), std::vector<double>(y[0].size()));       // Initialize the result vector
    // Calculate the derivative of softmax(x) for each input value
    for (size_t i = 0; i < y.size(); ++i) {
        for (size_t j = 0; j < y[0].size(); ++j) {
            result[i][j] = y[i][j] * (1 - y[i][j]);
            // Subtract the softmax of each other element from the derivative of softmax(x)
            std::transform(y[i].begin(), y[i].end(), result[i].begin(), [i, &y](double val){ 
                double sum = 0.0;
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
double ReLU(double x) {
    // The ReLU function is defined as max(0, x)
    return std::max(double(0), x); // Return the maximum of 0 and x
}

/**
 * @brief Calculates the derivative of the ReLU (Rectified Linear Unit)
 *      activation function. The derivative of the ReLU function is 0 if the input value is less than 0,
 *      and 1 otherwise.
 * @param x Input value
 * @return 0 if x < 0, 1 otherwise
 */
double ReLUder(double x) {
    // The derivative of the ReLU function is 0 if the input value is less than 0, and 1 otherwise.
    return (x > 0) ? 1 : 0; // Return 1 if x > 0, 0 otherwise
}

//----------------SeLU----------------//

/**
 * @brief Applies the SeLU (Scale-Exponential Linear Unit) activation function to the input.
 *      The SeLU function is defined as:
 *      f(x) = x if x > 0
 *      f(x) = 0.1 * x otherwise
 * @param x input value
 * @return the value of the SeLU function applied to the input
 */
double SeLU(double x) {
    // Apply the SeLU function to the input
    return (x > 0) ? x : 0.1 * x;
}

/**
 * @brief The derivative of the SeLU (Scale-Exponential Linear Unit) activation function.
 *        The SeLU derivative is defined as:
 *        f'(x) = 1 if x > 0
 *        f'(x) = 0.1 otherwise
 * @param x input value
 * @return the value of the SeLU derivative function applied to the input
 */
double SeLUder(double x) {
    // Apply the SeLU derivative function to the input
    return (x > 0) ? 1 : 0.1;
}
