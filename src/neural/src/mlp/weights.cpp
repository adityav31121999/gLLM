
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
    flat_vec.reserve(rows * cols);
    for (size_t i = 0; i < rows; ++i) {
        if (cols > 0 && vec2d[i].size() != cols) {
             fprintf(stderr, "Warning: Inconsistent column count in flatten (Row %zu has %zu, expected %zu). Data might be truncated or padded later.\n", i, vec2d[i].size(), cols);
        }
         flat_vec.insert(flat_vec.end(), vec2d[i].begin(), vec2d[i].end());
    }
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
    if (!matrix.mapped_data || matrix.row <= 0 || matrix.col <= 0) {
        return {};
    }
    size_t num_elements = static_cast<size_t>(matrix.row) * matrix.col;
    std::vector<float> flat_vec(num_elements);
    memcpy(flat_vec.data(), matrix.mapped_data, num_elements * sizeof(float));
    return flat_vec;
}


/**
 * @brief Flattens a specified range of rows from a 2D vector into a 1D vector.
 * @param vec2d The input 2D vector (vector of vectors).
 * @param start_row The starting row index (inclusive).
 * @param num_rows The number of rows to flatten.
 * @return A 1D vector containing the flattened elements.
 * @throws std::out_of_range if start_row or num_rows are invalid.
 */
inline std::vector<float> flatten_range(const std::vector<std::vector<float>>& vec2d, size_t start_row, size_t num_rows) {
    if (vec2d.empty() || num_rows == 0) {
        return {};
    }
    if (start_row >= vec2d.size() || start_row + num_rows > vec2d.size()) {
        throw std::out_of_range("flatten_range: Invalid start_row or num_rows exceeds vector bounds.");
    }
    size_t cols = vec2d[start_row].size();
    std::vector<float> flat_vec;
    flat_vec.reserve(num_rows * cols);
    for (size_t i = 0; i < num_rows; ++i) {
        flat_vec.insert(flat_vec.end(), vec2d[start_row + i].begin(), vec2d[start_row + i].end());
    }
    return flat_vec;
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
        if (rows > 0) {
            vec2d.resize(rows);
        }
        return;
    }
    if (flat_vec.size() != rows * cols) {
        fprintf(stderr, "Error: Cannot unflatten vector, size mismatch (%zu != %zu * %zu).\n", flat_vec.size(), rows, cols);
        vec2d.assign(rows, std::vector<float>(cols, 0.0f)); // Example: fill with zeros
        return;
    }
    vec2d.resize(rows);
    auto it = flat_vec.begin();
    for (size_t i = 0; i < rows; ++i) {
        vec2d[i].assign(it, it + cols);
        it += cols;
    }
}


/**
 * @brief Unflattens a 1D vector back into a 2D vector (row-major).
 * @param flat_vec The input 1D vector.
 * @param[out] vec2d The output matrix. Will be resized and populated.
 * @param rows The number of rows expected in the output 2D vector.
 * @param cols The number of columns expected in the output 2D vector.
 */
void unflatten(const std::vector<float>& flat_vec, mat& vec2d, size_t rows, size_t cols) {
    if (rows == 0 || cols == 0) {
        if (!flat_vec.empty()) {
            fprintf(stderr, "Error: Cannot unflatten non-empty vector into zero-dimension mat.\n");
        }
        return;
    }
    if (flat_vec.size() != rows * cols) {
        fprintf(stderr, "Error: Cannot unflatten vector, size mismatch (%zu != %zu * %zu).\n", flat_vec.size(), rows, cols);
        return;
    }
    if (!vec2d.mapped_data || static_cast<size_t>(vec2d.row) != rows || static_cast<size_t>(vec2d.col) != cols) {
        fprintf(stderr, "Error: Output mat is not correctly sized or not mapped for unflatten operation.\n");
        return;
    }
    memcpy(vec2d.mapped_data, flat_vec.data(), flat_vec.size() * sizeof(float));
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


/**
 * @brief make transpose of a flatten matrix
 * @param[in] input matrix
 * @param[out] output_flat flattened transpose of input
 * @param[in] rows number of rows
 * @param[in] cols number of columns
 */
void transposeFlattenMatrix(const std::vector<std::vector<float>>& input, std::vector<float>& output_flat, int rows, int cols) {
    if (input.empty()) {
        output_flat.clear();
        return;
    }
     if (input[0].empty() && cols != 0) {
        throw std::runtime_error("Transpose input has empty rows but non-zero columns expected.");
    }
     if (input[0].empty() && cols == 0) {
         output_flat.clear();
         return;
     }
    if (static_cast<int>(input.size()) != rows || static_cast<int>(input[0].size()) != cols) {
        throw std::runtime_error("Transpose dimension mismatch.");
    }
    output_flat.resize(static_cast<size_t>(cols) * rows);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            output_flat[static_cast<size_t>(j) * rows + i] = input[i][j];
        }
    }
}

/**
 * @brief make transpose of a flatten matrix (from mat object)
 * @param[in] input matrix object
 * @param[out] output_flat flattened transpose of input
 */
void transposeFlattenMatrix(const mat& input, std::vector<float>& output_flat) {
    if (!input.mapped_data || input.row <= 0 || input.col <= 0) {
        output_flat.clear();
        return;
    }
    int rows = input.row;
    int cols = input.col;

    output_flat.resize(static_cast<size_t>(cols) * rows);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            output_flat[static_cast<size_t>(j) * rows + i] = input(i, j);
        }
    }
}
