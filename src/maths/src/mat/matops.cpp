#include "include/mat.hpp"
#include <algorithm>
#include <numeric>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <vector>
#include "basic.hpp"


// Access operator (non-const)
float& mat::operator()(int i, int j) {
    if(!mapped_data) {
        throw std::runtime_error("Matrix data is not mapped.");
    }
    float* current_data_ptr = is_shared_segment ? data_segment_start : mapped_data;
    if (!current_data_ptr || i < 0 || i >= row || j < 0 || j >= col) {
        throw std::out_of_range("Matrix index out of bounds: " + std::to_string(i) + ", " + std::to_string(j));
    }
    // Check potential overflow before calculating index
    size_t index = static_cast<size_t>(i) * col + j;
    if (index >= (static_cast<size_t>(row) * col) ) { // Check against logical elements
        throw std::out_of_range("Calculated index exceeds mapped data bounds.");
    }
    return current_data_ptr[index];
}

// Access operator (const)
const float& mat::operator()(int i, int j) const {
    if(!mapped_data) {
        throw std::runtime_error("Matrix data is not mapped.");
    }
    const float* current_data_ptr = is_shared_segment ? data_segment_start : mapped_data;
    if (!current_data_ptr || i < 0 || i >= row || j < 0 || j >= col) {
        throw std::out_of_range("Matrix index out of bounds: " + std::to_string(i) + ", " + std::to_string(j));
    }
    size_t index = static_cast<size_t>(i) * col + j;
        if (index >= (static_cast<size_t>(row) * col) ) { // Check against logical elements
            throw std::out_of_range("Calculated index exceeds mapped data bounds.");
        }
    return current_data_ptr[index];
}

// return row of matrix from mapped file using 0-based row index
std::vector<float> mat::operator()(int i) const {
    const float* current_data_ptr = is_shared_segment ? data_segment_start : mapped_data;
    if (!current_data_ptr || i < 0 || i >= row) {
        throw std::out_of_range("Row index out of bounds or matrix not mapped.");
    }
    std::vector<float> row_data(col);
    // Check potential overflow before calculating offset
    size_t offset = static_cast<size_t>(i) * col;
    if (offset + col > mapped_size / sizeof(float)) {
        throw std::out_of_range("Row data exceeds mapped file capacity.");
    }
    std::copy(current_data_ptr + offset, current_data_ptr + offset + col, row_data.begin());
    return row_data;
}

/**
 * @brief Retrieves a specific row from the matrix.
 * @param m The matrix.
 * @param row_idx The 0-based index of the row to retrieve.
 * @return A std::vector<float> containing the elements of the specified row.
 * @throws std::out_of_range if row_idx is invalid or if access is out of mapped bounds.
 * @throws std::runtime_error if matrix data is not mapped when it's expected.
 */
std::vector<float> getRow(const mat& m, int row_idx) {
    if (row_idx < 0 || row_idx >= m.row) {
        throw std::out_of_range("getRow: Row index " + std::to_string(row_idx) + " is out of bounds for matrix with " + std::to_string(m.row) + " rows.");
    }

    if (m.col == 0) { // Matrix has columns of zero width
        return {}; // Return an empty vector
    }

    if (!m.mapped_data) {
        // This should ideally only be true if row or col is 0.
        // If row > 0 and col > 0, mapped_data should be valid.
        throw std::runtime_error("getRow: Matrix data is not mapped for a non-empty row.");
    }

    std::vector<float> row_vector(m.col);
    size_t offset = static_cast<size_t>(row_idx) * m.col;

    // Check if the read will go out of bounds of the mapped data
    if (m.mapped_size > 0 && (offset + static_cast<size_t>(m.col)) * sizeof(float) > m.mapped_size) {
        throw std::out_of_range("getRow: Calculated row data exceeds mapped memory bounds.");
    }
    
    const float* start_ptr = m.mapped_data + offset;
    std::copy(start_ptr, start_ptr + m.col, row_vector.begin());
    
    return row_vector;
}

/**
 * @brief Retrieves a specific column from the matrix.
 * @param m The matrix.
 * @param col_idx The 0-based index of the column to retrieve.
 * @return A std::vector<float> containing the elements of the specified column.
 * @throws std::out_of_range if col_idx is invalid.
 * @throws std::runtime_error if matrix data is not mapped.
 */
std::vector<float> getCol(const mat& m, int col_idx) {
    if (col_idx < 0 || col_idx >= m.col) {
        throw std::out_of_range("getCol: Column index " + std::to_string(col_idx) + " is out of bounds for matrix with " + std::to_string(m.col) + " columns.");
    }
    if (m.row == 0) {
        return {}; // No rows, so column is empty.
    }
    if (!m.mapped_data) {
        throw std::runtime_error("getCol: Matrix data is not mapped for a non-empty column.");
    }

    std::vector<float> col_vector(m.row);
    for (int i = 0; i < m.row; ++i) {
        // Access element (i, col_idx)
        col_vector[i] = m(i, col_idx);
    }
    return col_vector;
}

/**
 * @brief Assigns new values to a specific row in the matrix.
 * @param m The matrix to modify.
 * @param row_idx The 0-based index of the row to set.
 * @param data A std::vector<float> containing the new values for the row.
 * @throws std::out_of_range if row_idx is invalid or if access is out of mapped bounds.
 * @throws std::invalid_argument if data size does not match matrix column count.
 * @throws std::runtime_error if matrix data is not mapped when it's expected.
 */
void setRow(mat& m, int row_idx, const std::vector<float>& data) {
    if (row_idx < 0 || row_idx >= m.row) {
        throw std::out_of_range("setRow: Row index " + std::to_string(row_idx) + " is out of bounds for matrix with " + std::to_string(m.row) + " rows.");
    }
    if (static_cast<int>(data.size()) != m.col) {
        throw std::invalid_argument("setRow: Input data size (" + std::to_string(data.size()) + ") does not match matrix column count (" + std::to_string(m.col) + ").");
    }

    if (m.col == 0) { // Matrix has columns of zero width; data must be empty.
        return; // Nothing to set
    }

    if (!m.mapped_data) {
        // This should ideally only be true if row or col is 0.
        // If row > 0 and col > 0, mapped_data should be valid.
        throw std::runtime_error("setRow: Matrix data is not mapped for a non-empty row.");
    }

    size_t offset = static_cast<size_t>(row_idx) * m.col;
    // Check if the write will go out of bounds of the mapped data
    if (m.mapped_size > 0 && (offset + static_cast<size_t>(m.col)) * sizeof(float) > m.mapped_size) {
        throw std::out_of_range("setRow: Calculated row data exceeds mapped memory bounds for writing.");
    }

    float* start_ptr = m.mapped_data + offset;
    std::copy(data.begin(), data.end(), start_ptr);
}

/**
 * @brief Assigns new values to a specific column in the matrix.
 * @param m The matrix to modify.
 * @param col_idx The 0-based index of the column to set.
 * @param data A std::vector<float> containing the new values for the column.
 * @throws std::out_of_range if col_idx is invalid.
 * @throws std::invalid_argument if data size does not match matrix row count.
 * @throws std::runtime_error if matrix data is not mapped.
 */
void setCol(mat& m, int col_idx, const std::vector<float>& data) {
    if (col_idx < 0 || col_idx >= m.col) {
        throw std::out_of_range("setCol: Column index " + std::to_string(col_idx) + " is out of bounds for matrix with " + std::to_string(m.col) + " columns.");
    }
    if (static_cast<int>(data.size()) != m.row) {
        throw std::invalid_argument("setCol: Input data size (" + std::to_string(data.size()) + ") does not match matrix row count (" + std::to_string(m.row) + ").");
    }
    if (m.row == 0) {
        return; // Nothing to set.
    }
    if (!m.mapped_data) {
        throw std::runtime_error("setCol: Matrix data is not mapped for a non-empty column.");
    }

    for (int i = 0; i < m.row; ++i) {
        // Access element (i, col_idx) to set it
        m(i, col_idx) = data[i];
    }
}

/**
 * @brief provide values to row from vector
 * @param vec value vector be placed in row
 * @param i 0-based index of row in matrix
 */
void mat::addRow(const std::vector<float> & vec, int i)
{
    if (i < 0 || i >= row) {
        throw std::out_of_range("Row index out of bounds.");
    }
    if (static_cast<int>(vec.size()) != col) {
        throw std::invalid_argument("Data size does not match the number of columns.");
    }
    size_t offset = static_cast<size_t>(i) * col;
    if (offset + col > mapped_size / sizeof(float)) {
        throw std::out_of_range("Row data exceeds mapped file capacity.");
    }
    std::copy(vec.begin(), vec.end(), mapped_data + offset);
}

/**
 * @brief provide values to a column from a vector
 * @param vec value vector be placed in column
 * @param j 0-based index of column in matrix
 */
void mat::addCol(const std::vector<float>& vec, int j) {
    if (j < 0 || j >= col) {
        throw std::out_of_range("Column index out of bounds.");
    }
    if (static_cast<int>(vec.size()) != row) {
        throw std::invalid_argument("Data size does not match the number of rows.");
    }
    if (!mapped_data) {
        throw std::runtime_error("addCol: Matrix data is not mapped.");
    }

    for (int i = 0; i < row; ++i) {
        // Access element (i, j) to set it
        (*this)(i, j) = vec[i];
    }
}

/**
 * @brief Fills a vector with the data from a specific column.
 * @param out_vec The vector to be filled with column data.
 * @param j The 0-based index of the column to retrieve.
 */
void mat::getCol(std::vector<float>& out_vec, int j) const {
    if (j < 0 || j >= col) {
        throw std::out_of_range("getCol: Column index out of bounds.");
    }
    if (static_cast<int>(out_vec.size()) != row) {
        out_vec.resize(row);
    }
    if (!mapped_data && !is_shared_segment) {
        throw std::runtime_error("getCol: Matrix data is not mapped.");
    }

    for (int i = 0; i < row; ++i) {
        out_vec[i] = (*this)(i, j);
    }
}

/**
 * @brief Fills a vector with the data from a specific row.
 * @param out_vec The vector to be filled with row data.
 * @param i The 0-based index of the row to retrieve.
 */
void mat::getRow(std::vector<float>& out_vec, int i) const {
    if (i < 0 || i >= row) {
        throw std::out_of_range("getRow: Row index out of bounds.");
    }
    if (static_cast<int>(out_vec.size()) != col) {
        out_vec.resize(col);
    }
    std::copy_n(mapped_data + static_cast<size_t>(i) * col, col, out_vec.begin());
}
/**
 * @brief make 2d vector out of mapped file
 */
std::vector<std::vector<float>> mat::make2dVector()
{
    std::vector<std::vector<float>> result(row, std::vector<float>(col, 0.0f));
    // Copy data from this matrix to the result vector
    for (int i = 0; i < row; ++i) {
        // Copy the i-th row from the mapped data to the result vector
        std::copy_n(mapped_data + static_cast<size_t>(i) * col, col, result[i].begin());
    }

    return result;
}

/**
 * @brief make 2d vector out of mapped file
 * @param other matrix with values in mapped file
 * @param row rows of matrix
 * @param col columns of matrix
 */
std::vector<std::vector<float>> mat::make2dVector(const mat &other, int row, int col)
{
    // Validate input dimensions against 'other' matrix dimensions
    if (row <= 0 || col <= 0) {
        // Return an empty vector if requested dimensions are invalid
        return {};
    }
    if (!other.mapped_data) {
        throw std::runtime_error("Source matrix 'other' is not mapped or initialized.");
    }

    // Determine the actual number of rows/cols to copy (minimum of requested and available)
    int rows_to_copy = std::min<int>(row, other.row);
    int cols_to_copy = std::min<int>(col, other.col);

    // Create the result vector with the specified dimensions (row x col)
    // Initialize with zeros or default values.
    std::vector<std::vector<float>> result(row, std::vector<float>(col, 0.0f));

    // Copy data from 'other' matrix to the result vector
    for (int i = 0; i < rows_to_copy; ++i) {
        // Access 'other' safely using its operator() and copy to result
        std::copy_n(other.mapped_data + static_cast<size_t>(i) * other.col, cols_to_copy, result[i].begin());
    }

    // Return the result vector by value (cannot safely return a reference '&' as requested)
    return result;
}

std::vector<float> mat::flatten()
{
    std::vector<float> flattened(row * col, 0.0f);
    std::copy_n(mapped_data, row * col, flattened.begin());
    return flattened;
}

std::vector<float> mat::flatten(const mat& other)
{
    std::vector<float> flattened(other.row * other.col, 0.0f);
    std::copy_n(other.mapped_data, other.row * other.col, flattened.begin());
    return flattened;
}

/**
 * @brief In-place transpose of a matrix.
 * @details This function transposes the matrix in-place, meaning that it modifies the current matrix.
 * The transpose of a matrix is a matrix whose row and column indices are swapped, i.e.
 * the element at row i and column j is swapped with the element at row j and column i.
 * @throws throws error when the matrix is not square.
 */
void mat::transpose_inplace()
{
    if(!ifsquare()) {
        throw std::runtime_error("In-place transpose requires a square matrix!");
    }
    else {
        for(int i = 0; i < row; ++i) {
            for(int j = i + 1; j < col; ++j) {
                std::swap((*this)(i, j), (*this)(j, i));
            }
        }
    }
}


/**
 * @brief Creates a matrix with random values between 0 and 1.
 * This function creates a matrix with the specified number of rows and columns, and fills it with random values between 0 and 1.
 * @param row The number of rows in the matrix.
 * @param col The number of columns in the matrix.
 * @return A matrix with the specified size and random values.
 */
mat mat::Random(int row, int col) { // Definition matches static declaration
    mat result(row, col); // Creates temp file and maps
    std::mt19937 gen(std::random_device{}()); // Create a random number generator
    // Original code used normal dist, but comment said 0 to 1. Let's use uniform.
    std::uniform_real_distribution<float> dist(-3.0, 3.0); // Random numbers between 0 and 1
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            result(i, j) = dist(gen); // Use operator() to write to mapped memory
        }
    }
    return result;
}

/**
* @brief Calculate the weighted dot product of a matrix and a vector.
* @param a The matrix
* @param b The vector
* @return The dot product of the matrix and the vector
* @throws std::runtime_error if the row vectors of the matrix have different sizes than the vector
*/
std::vector<float> dot(const mat& a, const std::vector<float>& b) {
    if(a.col != static_cast<int>(b.size())) {
        throw std::runtime_error("Matrix columns and vector size must match for dot product.");
    }

    std::vector<float> c(a.row); // Result vector size is number of matrix rows

    for(int i = 0; i < a.row; i++) {
        float sum = 0.0f;
        for (int j = 0; j < a.col; ++j) { // a.col == b.size()
            sum += a(i, j) * b[j];
        }
        c[i] = sum;
    }
    return c;
}

/**
* @brief Calculate the weighted dot product of a vector and a matrix.
* @param a The vector
* @param b The matrix
* @return The dot product of the vector and the matrix
* @throws std::runtime_error if the row vectors of the matrix have different sizes than the vector
*/
std::vector<float> dot(const std::vector<float>& a, const mat& b) {
    // This calculates a^T * b if a is treated as a row vector.
    // Or a * b if a is treated as row vector and result is row vector.
    // Let's assume standard vector * matrix (vector is row vector) -> row vector
    if (static_cast<int>(a.size()) != b.row) {
        throw std::runtime_error("Vector size and matrix rows must match for dot product.");
    }
    std::vector<float> c(b.col, 0.0f); // Result vector size is number of matrix columns
    for (int j = 0; j < b.col; ++j) {
        for (int i = 0; i < b.row; ++i) { // b.row == a.size()
            c[j] += a[i] * b(i, j);
        }
    }
    return c;
}

/**
 * @brief Multiplies a matrix by a vector using hadamard matrix multiplication.
 * @param[in] a 2D vector of floats representing a matrix
 * @param[in] b 1D vector of floats representing a vector
 * @return a 1D vector of floats representing the result of the matrix multiplication
 * @throws throws error if the sizes of the matrices are not compatible
 */
// Standalone matmul seems redundant with operator* and dot product functions.
// If needed, it should take `const mat&` as input.
std::vector<float> matmul(const mat& a, const std::vector<float>& b) {
    if(a.col != static_cast<int>(b.size())) {
        throw std::runtime_error("Matrix columns and vector size must match for matmul.");
    }
    std::vector<float> c(a.row, 0.0f);
    for(int i = 0; i < a.row; i++) {
        for(int j = 0; j < a.col; j++) {
            c[i] += a(i, j) * b[j];
        }
    }
    return c;
}

/**
 * @brief Compute the trace of the matrix.
 * @details The trace of a matrix is the sum of the elements on the main diagonal.
 * @return The trace of the matrix.
 * @throws throws error if the matrix is not square.
 */
float mat::trace() const {
    // Check if the matrix is square
    if(!ifsquare()) { // Assumes ifsquare() is updated
        throw std::invalid_argument("Matrix is not square");
    }
    float tr = 0;  // Initialize the trace to 0
    // Iterate over the main diagonal and add the elements to the trace
    for(int i = 0; i < row; i++) { // row == col here
        tr += (*this)(i, i); // Use operator()
    }
    return tr;
}

/**
 * @brief Transpose a matrix.
 * @details This function takes a 2D vector (matrix) as input and returns its transpose.
 * The transpose of a matrix is obtained by swapping its rows and columns.
 * @param b 2D vector of floats representing the matrix to be transposed.
 * @return A 2D vector of floats representing the transposed matrix.
 */
// Standalone transpose function. Better to use a member function mat::transpose()
// which returns a new transposed `mat` object.
mat mat::transpose() const {
    mat result(col, row); // Create result matrix with swapped dimensions
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            result(j, i) = (*this)(i, j);
        }
    }
    return result;
}

/**
 * @brief Compute the covariance matrix of a dataset.
 * @details This function computes the covariance matrix for a matrix of data points.
 * Each column of the input matrix represents a variable, and each row represents a data point.
 * @param a A 2D vector of floats representing the dataset.
 * @return A 2D vector of floats representing the covariance matrix.
 */
mat covariance(const mat& a) {
    // Assumes rows are observations, columns are variables
    int n_obs = a.row;
    int n_vars = a.col;
    if (n_obs <= 1) {
        throw std::runtime_error("Covariance requires at least 2 observations.");
    }

    // Compute mean of each column
    std::vector<float> means(n_vars, 0.0f);
    for (int j = 0; j < n_vars; ++j) {
        for (int i = 0; i < n_obs; ++i) {
            means[j] += a(i, j);
        }
        means[j] /= n_obs;
    }

    // Create mean-centered matrix (copy original data first)
    mat centered_a = a; // Uses copy constructor
    for (int i = 0; i < n_obs; ++i) {
        for (int j = 0; j < n_vars; ++j) {
            centered_a(i, j) -= means[j];
        }
    }

    mat centered_a_T = centered_a.transpose(); // Use the new transpose member
    // compute covariance matrix
    mat cov = (centered_a_T * centered_a) * (1.0f / (n_obs - 1));
    return cov;
}

//----------------LOTA for mat----------------//

/**
 * @brief Applies the LOTA activation function to a memory-mapped matrix (mat),
 *        considering only relevant elements defined by 't' and 'attentionType'.
 * @param y Input mat object (const reference).
 * @param t Dimension limit (passed by value).
 * @param attentionType If true, process only the lower triangle (incl. diagonal); otherwise, process up to t x t square (passed by value).
 * @return A new mat object containing LOTA results for relevant elements, others potentially zeroed.
 */
mat LOTA(const mat& y, int t, bool attentionType) { // Pass t and attentionType by value
    if (y.row <= 0 || y.col <= 0 || t <= 0) {
        // Return an empty or appropriately sized zero matrix? Returning empty for now.
        return mat();
    }

    // Create a result matrix (temporary file backing)
    // Output size should ideally match input, or be t x t? Let's match input for now.
    mat result(y.row, y.col); // Creates temp file backing
    std::fill_n(result.mapped_data, result.row * result.col, 0.0f); // Initialize result to 0

    // Handle 1x1 case explicitly if t=1
    if (t == 1 && y.row > 0 && y.col > 0) {
        result(0, 0) = 1.0f;
        return result; // Already zeroed out others
    }

    // Find the minimum value in the relevant region of the input matrix y
    float min_val = (std::numeric_limits<float>::max)(); // Parenthesize to avoid macro
    bool found_value = false;
    size_t max_rows = (std::min)((size_t)t, (size_t)y.row); // Parenthesize to avoid macro
    for (size_t i = 0; i < max_rows; ++i) {
        size_t limit_j = attentionType ? (i + 1) : (size_t)t;
        limit_j = (std::min)(limit_j, (size_t)y.col); // Boundary check cols, parenthesize
        for (size_t j = 0; j < limit_j; ++j) {
            min_val = (std::min)(min_val, y(i, j)); // Use mat access, parenthesize
            found_value = true;
        }
    }
    if (!found_value) min_val = 0.0f; // Handle case where relevant region is effectively empty

    float abs_min_val = std::abs(min_val);

    // Transform relevant elements: element + abs(min_val), calculate sum, and store in result
    float sum = 0.0f;
    int relevant_count = 0;
    for (size_t i = 0; i < max_rows; ++i) {
        size_t limit_j = attentionType ? (i + 1) : (size_t)t;
        limit_j = (std::min)(limit_j, (size_t)y.col); // Boundary check cols, parenthesize
        for (size_t j = 0; j < limit_j; ++j) {
            float transformed_val = y(i, j) + abs_min_val;
            result(i, j) = transformed_val; // Store transformed value in result mat
            sum += transformed_val;
            relevant_count++;
        }
        // Non-relevant elements in result remain 0 (already initialized)
    }

    // Normalize relevant elements in the result matrix by the global sum
    if (sum > 0.0f) {
        for (size_t i = 0; i < max_rows; ++i) {
            size_t limit_j = attentionType ? (i + 1) : (size_t)t;
            limit_j = (std::min)(limit_j, (size_t)result.col); // Use result.col here, parenthesize
            for (size_t j = 0; j < limit_j; ++j) {
                result(i, j) /= sum; // Normalize the value already stored
            }
        }
    } else if (relevant_count > 0) {
        // Handle sum=0 case -> uniform distribution over relevant elements
        float uniform_val = 1.0f / static_cast<float>(relevant_count);
        for (size_t i = 0; i < max_rows; ++i) {
            size_t limit_j = attentionType ? (i + 1) : (size_t)t;
            limit_j = (std::min)(limit_j, (size_t)result.col); // Use result.col here, parenthesize
            for (size_t j = 0; j < limit_j; ++j) {
                result(i, j) = uniform_val; // Set uniform value
            }
        }
    }
    // Non-relevant elements remain 0

    return result; // Return the new mat object
}

//----------------LOTA for mat----------------//

/**
 * @brief Derivative of the LOTA activation function for a memory-mapped matrix (mat),
 *        considering only relevant elements defined by 't' and 'attentionType'.
 * @param y Input mat object (const reference).
 * @param t Dimension limit (passed by value).
 * @param attentionType If true, process only the lower triangle; otherwise, up to t x t square (passed by value).
 * @return A new mat object containing LOTA derivative results for relevant elements.
 */
mat LOTAder(const mat& y, int t, bool attentionType) { // Pass t and attentionType by value
    if (y.row <= 0 || y.col <= 0 || t <= 0) {
        return mat(); // Return empty mat for invalid input
    }

    // Create a result matrix (temporary file backing)
    mat result(y.row, y.col);
    std::fill_n(result.mapped_data, result.row * result.col, 0.0f); // Initialize result to 0

    // Handle 1x1 case explicitly if t=1
    if (t == 1 && y.row > 0 && y.col > 0) {
        result(0, 0) = 0.0f; // Derivative of LOTA(x)=1 is 0
        return result;
    }

    // Find the minimum value in the relevant region of the input matrix y
    float min_val = (std::numeric_limits<float>::max)(); // Parenthesize to avoid macro
    bool found_value = false;
    size_t max_rows = (std::min)((size_t)t, (size_t)y.row); // Parenthesize to avoid macro
    for (size_t i = 0; i < max_rows; ++i) {
        size_t limit_j = attentionType ? (i + 1) : (size_t)t;
        limit_j = (std::min)(limit_j, (size_t)y.col); // Boundary check cols, parenthesize
        for (size_t j = 0; j < limit_j; ++j) {
            min_val = (std::min)(min_val, y(i, j)); // Use mat access, parenthesize
            found_value = true;
        }
    }
    if (!found_value) min_val = 0.0f;

    float abs_min_val = std::abs(min_val);

    // Calculate the sum of (element + abs(min_val)) in the relevant region
    // Also store transformed values temporarily if needed (or recalculate)
    float sum = 0.0f;
    for (size_t i = 0; i < max_rows; ++i) {
        size_t limit_j = attentionType ? (i + 1) : (size_t)t;
        limit_j = (std::min)(limit_j, (size_t)y.col); // Boundary check cols, parenthesize
        for (size_t j = 0; j < limit_j; ++j) {
            sum += (y(i, j) + abs_min_val); // Use original y here
        }
    }

    // Calculate the derivative for each element in the relevant region
    float sum_sq = sum * sum;
    if (sum > 0.0f) { // Avoid division by zero
        for (size_t i = 0; i < max_rows; ++i) {
            size_t limit_j = attentionType ? (i + 1) : (size_t)t;
            limit_j = (std::min)(limit_j, (size_t)result.col); // Use result.col here, parenthesize
            for (size_t j = 0; j < limit_j; ++j) {
                // Derivative: (sum - transformed_element) / sum^2
                float transformed_element = y(i, j) + abs_min_val;
                result(i, j) = (sum - transformed_element) / sum_sq;
            }
        }
    }
    // Non-relevant elements remain 0 (already initialized)

    return result; // Return the derivative matrix
}