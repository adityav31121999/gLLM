#include "cppsup.hpp"

/**
 * @brief Randomly initialize weights and biases using random numbers
 * @param weights Reference to the 2D weights vector
 */
void randomweights(std::vector<std::vector<float>> weights) {
    // Randomly intialize weights
    for(size_t i = 0; i < weights.size(); i++) {
        std::transform(weights[i].begin(), weights[i].end(), weights[i].begin(), [i](float x) {
            return static_cast<float>(pow(-1, x)*i + (rand() % 10));
        });
    }
}

/**
 * @brief Initialize weights randomly, incrementing by 1 and shuffling the weights
 */
void jumbledwbs(std::vector<std::vector<float>> weights) {
    // Randomly initialize weights
    for (size_t i = 0; i < weights.size(); ++i) {
        // Fill the weights vector with random values and increment by 1
        std::iota(weights[i].begin(), weights[i].end(), static_cast<float>((i - rand()) % 10));
        // Randomize the order of weights values
        std::shuffle(weights[i].begin(), weights[i].end(), std::mt19937{std::random_device{}()});
    }
    // Final shuffle
    std::shuffle(weights.begin(), weights.end(), std::mt19937{std::random_device{}()});
}

/**
 * @brief Randomly initialize weights using ij-based calculation
 *        The weights are initialized with the result of (i+1)*(j+1) * rand() % (i+j)
 * @param[in,out] weights Reference to the weights vector
 */
void ijbasedwbs(std::vector<std::vector<float>> weights) {
    // Randomly initialize weights
    for(int i = 0; i < weights.size(); i++) {
        std::transform(weights[i].begin(), weights[i].end(), weights[i].begin(), 
                       [i](float x) mutable { 
                        return static_cast<float>(static_cast<int>(((i+1)*(x+1)) * rand()) % static_cast<int>(i + x)); 
                       });
    }
}

/**
 * @brief Creates a matrix with random values between 0 and 1.
 * This function creates a matrix with the specified number of rows and columns, and fills it with random values between 0 and 1.
 * @param row The number of rows in the matrix.
 * @param col The number of columns in the matrix.
 * @return A matrix with the specified size and random values.
 */
void Random(std::vector<std::vector<float>> weights) {
    std::mt19937 gen(std::random_device{}()); // Create a random number generator
    std::normal_distribution<float> dist(-10.0, 10.0); // Create a distribution that generates random numbers between 0 and 1
    for (int i = 0; i < weights.size(); ++i) {
        for (int j = 0; j < weights[i].size(); ++j) {
            weights[i][j] = dist(gen); // Generate a random number and assign it to the current position in the matrix
        }
    }
}
