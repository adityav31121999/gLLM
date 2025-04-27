
#include "include/mlp.hpp"
#include <random>

/**
 * @brief Function to initialize the weights of the multi-layer perceptron.
 * This function initializes the weights of the mlp using a normal distribution
 * with a mean of 0.0 and a standard deviation of 1.0.
 */
void mlp::initializeWeights(int in, int layers) {
    // random number generator
    std::random_device rd;      // device
    std::mt19937 gen(rd());     // generator
    std::normal_distribution<float> dis(0.0, 1.0);     // min and max of distribution

    // initialize hidden to hidden weights
    for(int i = 0; i < layers; i++) {
        for(int j = 0; j < in; j++) {
            for(int k = 0; k < in; k++) {
                weights[i][j][k] = (i+j + dis(gen)) / (k+1);
            }
        }
    }
}

/**
 * @brief Flattens a 2D vector into a 1D vector (row-major).
 * @param vec2d The input 2D vector.
 * @return A 1D vector containing the flattened data. Returns empty if input is empty.
 */
std::vector<float> flatten(const std::vector<std::vector<float>>& vec2d) {
    if (vec2d.empty() || vec2d[0].empty()) {
        return {};
    }
    size_t rows = vec2d.size();
    size_t cols = vec2d[0].size();
    std::vector<float> flat_vec;
    flat_vec.reserve(rows * cols); // Reserve space for efficiency
    for (size_t i = 0; i < rows; ++i) {
        // Check consistency only if cols > 0 to avoid issues with empty inner vectors
        if (cols > 0 && vec2d[i].size() != cols) {
             fprintf(stderr, "Warning: Inconsistent column count in flatten (Row %zu has %zu, expected %zu). Data might be truncated or padded later.\n", i, vec2d[i].size(), cols);
             // Decide how to handle: Use first row's count? Throw? Pad/Truncate?
             // Current CUDA code inserts anyway, let's mimic that but keep the warning.
        }
        // Ensure we don't read out of bounds if a row is unexpectedly short
        // flat_vec.insert(flat_vec.end(), vec2d[i].begin(), vec2d[i].begin() + std::min(vec2d[i].size(), cols));
        // The original code just inserts whatever the row has. Let's stick to that for direct porting.
         flat_vec.insert(flat_vec.end(), vec2d[i].begin(), vec2d[i].end());
    }
     // Post-check: If sizes were inconsistent, the total size might not be rows*cols.
     if (flat_vec.size() != rows * cols) {
         fprintf(stderr, "Warning: Flattened vector size (%zu) does not match expected (%zu * %zu) due to inconsistent rows.\n", flat_vec.size(), rows, cols);
     }
    return flat_vec;
}

/**
 * @brief Flattens a mat object into a 1D vector (row-major).
 * @param matrix The input mat object. Assumes mat has member `a` which is std::vector<std::vector<float>>.
 * @return A 1D vector containing the flattened data.
 */
std::vector<float> flatten(const mat& matrix) {
    // Assuming mat class has a public member 'a' of type std::vector<std::vector<float>>
    return flatten(matrix.a);
}

/**
 * @brief Unflattens a 1D vector back into a 2D vector (row-major).
 * @param flat_vec The input 1D vector.
 * @param[out] vec2d The output 2D vector. Will be resized and populated.
 * @param rows The number of rows expected in the output 2D vector.
 * @param cols The number of columns expected in the output 2D vector.
 */
void unflatten(const std::vector<float>& flat_vec, std::vector<std::vector<float>>& vec2d, size_t rows, size_t cols) {
    if (rows == 0 || cols == 0) {
        vec2d.clear();
        // If rows > 0 but cols == 0, maybe create rows empty vectors?
        if (rows > 0) {
            vec2d.resize(rows);
        }
        return; // Nothing to do for zero dimensions in flat_vec expected
    }
    if (flat_vec.size() != rows * cols) {
        fprintf(stderr, "Error: Cannot unflatten vector, size mismatch (%zu != %zu * %zu).\n", flat_vec.size(), rows, cols);
        // Set vec2d to a default state or throw?
        vec2d.assign(rows, std::vector<float>(cols, 0.0f)); // Example: fill with zeros
        // Or: throw std::runtime_error("Unflatten size mismatch");
        return;
    }
    vec2d.resize(rows);
    auto it = flat_vec.begin();
    for (size_t i = 0; i < rows; ++i) {
        // Assign elements for the current row using iterators
        vec2d[i].assign(it, it + cols);
        it += cols; // Move iterator to the start of the next row
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
 * @brief make transpose of a flatten matrix
 * @param[in] input matrix
 * @param[out] output_flat flattened transpose of input
 * @param[in] rows number of rows
 * @param[in] cols number of columns
 */
void transposeFlattenMatrix(const std::vector<std::vector<float>>& input, std::vector<float>& output_flat, int rows, int cols) {
    if (input.empty()) { // Allow empty input (e.g., if d or h is 0)
        output_flat.clear();
        return;
    }
     if (input[0].empty() && cols != 0) { // Rows exist but are empty, cols expected
        throw std::runtime_error("Transpose input has empty rows but non-zero columns expected.");
    }
     if (input[0].empty() && cols == 0) { // Empty rows and zero cols expected is valid
         output_flat.clear();
         return;
     }
    if (static_cast<int>(input.size()) != rows || static_cast<int>(input[0].size()) != cols) {
        throw std::runtime_error("Transpose dimension mismatch.");
    }
    output_flat.resize(static_cast<size_t>(cols) * rows); // Transposed dimensions
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // output[j][i] = input[i][j] -> output_flat[j * rows + i]
            output_flat[static_cast<size_t>(j) * rows + i] = input[i][j];
        }
    }
}
