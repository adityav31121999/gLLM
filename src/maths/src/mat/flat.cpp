
#include <vector>
#include "include/mat.hpp"


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

/**
 * @brief Flattens a 2D vector into a 1D vector (row-major), validating against expected dimensions.
 * @param vec2d The input 2D vector.
 * @param[out] output_flat The output 1D vector. Will be cleared and populated.
 * @param expected_rows The expected number of rows in the input 2D vector.
 * @param expected_cols The expected number of columns in each row of the input 2D vector.
 * @throws std::runtime_error if the input vector's dimensions do not match expected_rows or if rows have inconsistent column counts.
 */
void flatten2DVector(const std::vector<std::vector<float>>& vec2d, std::vector<float>& output_flat, size_t expected_rows, size_t expected_cols) {
    output_flat.clear();
    if (expected_rows == 0) { // If expecting zero rows, output is empty.
        if (!vec2d.empty() && !vec2d[0].empty()) {
             fprintf(stderr, "Warning: flatten2DVector called with expected_rows=0 but vec2d is not empty.\n");
        }
        return;
    }
    if (vec2d.size() != expected_rows) {
        throw std::runtime_error("flatten2DVector: Input vector row count (" + std::to_string(vec2d.size()) + ") does not match expected_rows (" + std::to_string(expected_rows) + ").");
    }

    output_flat.reserve(expected_rows * expected_cols);

    for (size_t i = 0; i < expected_rows; ++i) {
        if (vec2d[i].size() != expected_cols) {
            throw std::runtime_error("flatten2DVector: Row " + std::to_string(i) + " has " + std::to_string(vec2d[i].size()) + " columns, but expected_cols is " + std::to_string(expected_cols) + ".");
        }
        output_flat.insert(output_flat.end(), vec2d[i].begin(), vec2d[i].end());
    }
}



// Helper function for transposing a mat object's data into a flat vector (row-major)
// Takes an R x C matrix m and produces output_flat representing a C x R matrix.
// (Copied from attention/cu/forward.cu for self-containment)
void transposeMatToFlatVector(const mat& m, std::vector<float>& output_flat) {
    if (!m.mapped_data) {
        output_flat.clear();
        if (m.row != 0 || m.col != 0) { // Invalid state: dimensions but no data
            throw std::runtime_error("Mat has non-zero dimensions but null mapped_data in transposeMatToFlatVector.");
        }
        return; // Valid empty mat
    }
    if (m.row == 0 || m.col == 0) { // Valid empty mat
        output_flat.clear();
        return;
    }
    int R = m.row; // Original rows
    int C = m.col; // Original columns
    output_flat.resize(static_cast<size_t>(R) * C); // Will store data for a C x R matrix

    for (int j = 0; j < C; ++j) {        // Iterate original columns (these become rows in the transposed version)
        for (int i_orig = 0; i_orig < R; ++i_orig) {    // Iterate original rows (these become columns in the transposed version)
            output_flat[static_cast<size_t>(j) * R + i_orig] = m(i_orig, j); // Access m(original_row, original_col)
        }
    }
}


/**
 * @brief Flattens a 2D vector into a 1D vector (row-major).
 * @param vec2d The input 2D vector.
 * @return A 1D vector containing the flattened data. Returns empty if input is empty.
 */
std::vector<float> flatten(const std::vector<std::vector<float>>& vec2d) {
    if (vec2d.empty()) {
        return {};
    }
    size_t rows = vec2d.size();
    size_t cols = vec2d[0].size();
    std::vector<float> flat_vec;
    flat_vec.reserve(rows * cols);
    for (size_t i = 0; i < rows; ++i) {
        if (vec2d[i].size() != cols) {
            throw std::runtime_error("flatten: Inconsistent column count. Row " + std::to_string(i) + " has " + std::to_string(vec2d[i].size()) + " columns, but expected " + std::to_string(cols) + ".");
        }
        flat_vec.insert(flat_vec.end(), vec2d[i].begin(), vec2d[i].end());
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
