
// includes basic operations
#include "include/mat.hpp" // Use relative path if appropriate for build system
#include <algorithm>
#include <numeric>
#include <iostream>
#include <cmath>
#include <stdexcept> // For exceptions
#include <vector>
#include "basic.hpp"
#include "mat.hpp"


/**
 * @brief index operator overload
 * @param i index of row
 * @param j index of column
 * @return reference to the element at position (i, j)
 */
// float& mat::operator()(int i, int j) {
//     // Definition should be in the header (mat.hpp) for inlining
//     // return mapped_data[i * col + j]; // Direct access (less safe)
//     // Or use the bounds-checked version from header
// }

// -------------------------------Mathematical Functions for Matrix Operations------------------------------- //

/**
 * @brief Overloaded assignment operator for matrix.
 * This function takes another matrix `b` and assigns it to the current matrix.
 * It checks if the dimensions of the two matrices are the same, and if not, it prints an error message and returns an empty matrix.
 * Otherwise, it copies the elements of `b` into the current matrix and returns the resulting matrix.
 * @param b The matrix to be assigned to the current matrix.
 * @return A matrix that is a copy of `b`.
 * @throws throws error if the dimensions of the two matrices are not the same.
 */
// Helper swap function (can be defined here or elsewhere, declared in mat.hpp)
void swap(mat& first, mat& second) noexcept {
    using std::swap;
    swap(first.row, second.row);
    swap(first.col, second.col);
    swap(first.mapped_data, second.mapped_data);
    swap(first.mapped_file_handle, second.mapped_file_handle);
    swap(first.mapped_size, second.mapped_size);
    swap(first.backing_filename, second.backing_filename);
    swap(first.is_temp_file, second.is_temp_file);
}


// Copy Assignment Operator (using copy-and-swap)
mat& mat::operator=(const mat& other) {
    if (this != &other) { // Protect against self-assignment
        mat temp(other); // Use copy constructor (creates deep copy)
        swap(*this, temp); // Swap resources
    }
    return *this;
}

// Move Assignment Operator (implementation in header is also fine)
mat& mat::operator=(mat&& other) noexcept {
    if (this != &other) {
        swap(*this, other); // Swap resources, other is left in *this's old state
    }
    return *this;
}

// retrn (i-1)th (0-based) indexed row
std::vector<float> mat::operator=(int i)
{
    return std::vector<float>();    if (i < 0 || i >= row) {
        throw std::out_of_range("Row index out of bounds.");
    }
    std::vector<float> row_data(col);
    size_t offset = static_cast<size_t>(i) * col;
    if (offset + col > mapped_size / sizeof(float)) {
        throw std::out_of_range("Row data exceeds mapped file capacity.");
    }
    std::copy(mapped_data + offset, mapped_data + offset + col, row_data.begin());
    return row_data;
}

/**
 * @brief Overloaded assignment operator for matrix.
 * This function takes another matrix represented as a vector of vectors of floats and assigns it to the current matrix.
 * It checks if the dimensions of the two matrices are the same, and if not, it prints an error message and returns an empty matrix.
 * Otherwise, it copies the elements of the given matrix into the current matrix and returns the resulting matrix.
 * @param b The matrix to be assigned to the current matrix represented as a vector of vectors of floats.
 * @return A matrix that is a copy of `b`.
 * @throws throws error if the dimensions of the two matrices are not the same.
 * @note This assigns vector data into an *existing* mapped matrix. It does not resize the matrix.
 */
mat& mat::operator=(const std::vector<std::vector<float>>& b) {
    int vec_rows = b.empty() ? 0 : b.size();
    int vec_cols = (vec_rows > 0 && !b[0].empty()) ? b[0].size() : 0;

    // Check if the dimensions of the two matrices are the same.
    if(row != vec_rows || col != vec_cols) {
        throw std::runtime_error("Matrix and vector dimensions do not match for assignment: " + std::to_string(row) + "x" + std::to_string(col) + " vs " + std::to_string(vec_rows) + "x" + std::to_string(vec_cols) + "." );
    }
    if (!mapped_data) {
        throw std::runtime_error("Cannot assign vector to uninitialized/unmapped matrix.");
    }

    for (int i = 0; i < row; ++i) {
        // Use operator() for bounds checking and access
        std::copy(b[i].begin(), b[i].end(), mapped_data + (static_cast<size_t>(i) * col));
    }
    return *this;
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

/**
 * @brief Overloaded addition operator for matrix.
 * This function takes another matrix `b` and adds it to the current matrix.
 * It checks if the dimensions of the two matrices are the same, and if not, it prints an error message and returns an empty matrix.
 * Otherwise, it adds the elements of the two matrices and returns the resulting matrix.
 * @param b The matrix to be added to the current matrix.
 * @return A matrix that is the sum of the current matrix and `b`.
 * @throws throws error if the dimensions of the two matrices are not the same.
 */
mat mat::operator+(const mat& b) const {
    // Check if the dimensions of the two matrices are the same.
    if(row != b.row || col != b.col) {
        throw std::runtime_error("Matrix dimensions do not match for addition.");
    }
    // Create a new matrix for the result (allocates temp file and maps)
    mat c(row, col);
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            c(i, j) = (*this)(i, j) + b(i, j); // Use operator() for access
        }
    }
    // Return the resulting matrix.
    return c;
}

/**
 * @brief Overloaded addition operator for matrix.
 * This function takes a vector of vectors of type `std::vector<std::vector<float>>` and adds it to the current matrix.
 * It checks if the dimensions of the two matrices are the same, and if not, it prints an error message and returns an empty matrix.
 * Otherwise, it adds the elements of the two matrices and returns the resulting matrix.
 * @param b The matrix to be added to the current matrix.
 * @return A matrix that is the sum of the current matrix and `b`.
 * @throws throws error if the dimensions of the two matrices are not the same.
 */
mat mat::operator+(const std::vector<std::vector<float>>& b) const {
    int vec_rows = b.empty() ? 0 : b.size();
    int vec_cols = (vec_rows > 0 && !b[0].empty()) ? b[0].size() : 0;

    // Check if the dimensions of the two matrices are the same.
    if(row != vec_rows || col != vec_cols) {
        throw std::runtime_error("Matrix and vector dimensions do not match for addition.");
    }
    // Create a new matrix for the result
    mat c(row, col);
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            c(i, j) = (*this)(i, j) + b[i][j];
        }
    }
    // Return the resulting matrix.
    return c;
}

/**
 * @brief Overloaded subtraction operator for matrix.
 * This function takes another matrix `b` and subtracts it from the current matrix.
 * It checks if the dimensions of the two matrices are the same, and if not, it prints an error message and returns an empty matrix.
 * Otherwise, it subtracts the elements of the two matrices and returns the resulting matrix.
 * @param b The matrix to be subtracted from the current matrix.
 * @return A matrix that is the difference of the current matrix and `b`.
 * @throws throws error if the dimensions of the two matrices are not the same.
 */
mat mat::operator-(const mat& b) const {
    // Check if the dimensions of the two matrices are the same.
    if(row != b.row || col != b.col) {
        throw std::runtime_error("Matrix dimensions do not match for subtraction.");
    }
    // Create a new matrix for the result
    mat c(row, col);
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            c(i, j) = (*this)(i, j) - b(i, j);
        }
    }
    // Return the resulting matrix.
    return c;
}

/**
 * @brief Overloaded subtraction operator for matrix.
 * This function takes a matrix `b` and subtracts it from the current matrix.
 * It checks if the dimensions of the two matrices are the same, and if not, it prints an error message and returns an empty matrix.
 * Otherwise, it subtracts the elements of the two matrices and returns the resulting matrix.
 * @param b The matrix to be subtracted from the current matrix.
 * @return A matrix that is the difference of the current matrix and `b`.
 * @throws throws error if the dimensions of the two matrices are not the same.
 */
mat mat::operator-(const std::vector<std::vector<float>>& b) const {
    int vec_rows = b.empty() ? 0 : b.size();
    int vec_cols = (vec_rows > 0 && !b[0].empty()) ? b[0].size() : 0;

    if(row != vec_rows || col != vec_cols) {
        throw std::runtime_error("Matrix and vector dimensions do not match for subtraction.");
    }
    // Create a new matrix for the result
    mat c(row, col);
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            c(i, j) = (*this)(i, j) - b[i][j];
        }
    }
    return c;
}

/**
 * @brief Overloaded operator for matrix multiplication with a scalar.
 * This function takes a matrix and a scalar value as input and returns a new matrix
 * where each element of the input matrix is multiplied by the scalar.
 * @param b The scalar value to multiply the matrix with.
 * @return A new matrix where each element is the product of the corresponding element
 * in the input matrix and the scalar value.
 * @throws None.
 */
mat mat::operator*(float b) const {
    // Create a new matrix with the same dimensions as the input matrix
    mat c(row, col);
    // Iterate over each element in the input matrix and multiply it by the scalar
    for(int i = 0; i < row; i++) {
        for(int j = 0; j < col; j++) {
            c(i, j) = (*this)(i, j) * b;
        }
    }
    // Return the new matrix.
    return c;
}


/**
 * @brief this is to calculate specifically the qkCache
 * @param a MQ matrix
 * @param b MK matrix
 */
void mat::mult_A_Bt(const mat& a, const mat& b)
{
    // if number of rows of a == number of rows of b
    if(a.row != b.row) {
        throw std::runtime_error("Matrix multiplication of incompatible dimensions.\
            This is for product of matrices with same number or rows, since other is transpose");
    }
    for(int i = 0; i < a.row; i++) {
        for(int j = 0; j < b.row; j) {
            for(int k = 0; k < a.col; k++) {
                (*this)(i,j) += a(i, k) * b(j, k);
            }
        }
    }
}


/**
 * @brief Overloaded operator for matrix division with a scalar.
 * This function takes a matrix and a scalar value as input and returns a new matrix
 * where each element of the input matrix is divided by the scalar.
 * @param b The scalar value to divide the matrix with.
 * @return A new matrix where each element is the division of the corresponding element
 * in the input matrix and the scalar value.
 * @throws None.
 * @brief Overloaded operator for matrix multiplication with another matrix.
 *      This function takes a matrix and another matrix as input and returns a new matrix
 *      It uses strassen's matrix multiplication method.
 * @param a The matrix to multiply the current matrix with.
 * @return Product of two matrix
 */
mat mat::operator*(const mat& a) const {
    // Check if dimensions are compatible for multiplication
    if (col != a.row) {
        throw std::runtime_error("Matrix dimensions are incompatible for multiplication.");
    }

    // Create the result matrix (allocates temp file and maps)
    // Initialize with zeros
    mat c(row, a.col);
    std::fill_n(c.mapped_data, static_cast<size_t>(c.row) * c.col, 0.0f);

    // Perform standard matrix multiplication
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < a.col; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < col; ++k) { // col == a.row
                sum += (*this)(i, k) * a(k, j);
            }
            c(i, j) = sum;
        }
    }

    // Strassen implementation would require significant changes to work
    // efficiently with memory-mapped files (e.g., mapping sub-regions
    // or creating temporary mapped sub-matrices). Omitted for now.

    return c;
}

/**
 * @brief Overloaded division operator for matrix.
 * This function takes another matrix `a` and divides it by the current matrix.
 * It checks if the dimensions of the two matrices are the same, and if not, it prints an error message and returns an empty matrix.
 * Otherwise, it divides the elements of the two matrices and returns the resulting matrix.
 * @param a The matrix to be divided by the current matrix.
 * @return A matrix that is the quotient of the current matrix and `a`.
 * @throws throws error if the dimensions of the two matrices are not the same.
 */
mat mat::operator/(const mat& other) const {
    // Matrix division A/B is typically interpreted as A * B^-1
    // Requires matrix inversion (e.g., using Gauss-Jordan)
    if (!other.ifsquare()) { // Assumes ifsquare() is updated
         throw std::runtime_error("Matrix division requires the divisor matrix to be square.");
    }
    if (col != other.row) { // Check compatibility for A * B^-1
        throw std::runtime_error("Matrix dimensions are incompatible for division (A.cols != B.rows).");
    }
    mat other_inv = other.gaussjordan(); // Calculate inverse of 'other' (B^-1)
    mat result = (*this) * other_inv;    // Multiply this by the inverse (A * B^-1)
    return result; // Return the correct variable
}

/**
 * @brief Overloaded addition operator for matrix.
 * This function takes another matrix `other` and adds it to the current matrix.
 * It checks if the dimensions of the two matrices are the same, and if not, it prints an error message and returns an empty matrix.
 * Otherwise, it adds the elements of the two matrices and returns the resulting matrix.
 * @param other The matrix to be added to the current matrix.
 * @return A matrix that is the sum of the current matrix and `other`.
 * @throws throws error if the dimensions of the two matrices are not the same.
 */
mat& mat::operator+=(const mat& other) {
    if (row != other.row || col != other.col) {
        throw std::runtime_error("Matrix dimensions do not match for addition assignment.");
    }
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            (*this)(i, j) += other(i, j); // Use operator() for access
        }
    }
    return *this;
}

/**
 * @brief Overloaded subtraction operator for matrix.
 * This function takes another matrix `other` and subtracts it from the current matrix.
 * It checks if the dimensions of the two matrices are the same, and if not, it prints an error message and returns an empty matrix.
 * Otherwise, it subtracts the elements of the two matrices and returns the resulting matrix.
 * @param other The matrix to be subtracted from the current matrix.
 * @return A matrix that is the difference of the current matrix and `other`.
 * @throws throws error if the dimensions of the two matrices are not the same.
 */
mat& mat::operator-=(const mat& other) {
    if (row != other.row || col != other.col) {
        throw std::runtime_error("Matrix dimensions do not match for subtraction assignment.");
    }
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            (*this)(i, j) -= other(i, j);
        }
    }
    return *this;
}

/**
 * @brief Overloaded addition-assignment operator for matrix.
 * This function adds another matrix, represented as a vector of vectors, to the current matrix.
 * It checks if the dimensions of the two matrices are the same, and if they are not, it throws a runtime error.
 * Otherwise, it adds the elements of the other matrix to the current matrix in-place and returns the updated matrix.
 * @param other The matrix to be added to the current matrix, represented as a vector of vectors.
 * @return The updated matrix after addition.
 * @throws std::runtime_error if the dimensions of the two matrices do not match.
 */
mat& mat::operator+=(const std::vector<std::vector<float>>& other) {
    int vec_rows = other.empty() ? 0 : other.size();
    int vec_cols = (vec_rows > 0 && !other[0].empty()) ? other[0].size() : 0;

    if (row != vec_rows || col != vec_cols) {
        throw std::runtime_error("Matrix and vector dimensions do not match for addition assignment.");
    }
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            (*this)(i, j) += other[i][j];
        }
    }
    return *this;
}

/**
 * @brief Overloaded subtraction-assignment operator for matrix.
 * This function subtracts another matrix, represented as a vector of vectors, from the current matrix.
 * It checks if the dimensions of the two matrices are the same, and if they are not, it throws a runtime error.
 * Otherwise, it subtracts the elements of the other matrix from the current matrix in-place and returns the updated matrix.
 * @param other The matrix to be subtracted from the current matrix, represented as a vector of vectors.
 * @return The updated matrix after subtraction.
 * @throws std::runtime_error if the dimensions of the two matrices do not match.
 */
mat& mat::operator-=(const std::vector<std::vector<float>>& other) {
    int vec_rows = other.empty() ? 0 : other.size();
    int vec_cols = (vec_rows > 0 && !other[0].empty()) ? other[0].size() : 0;

    if (row != vec_rows || col != vec_cols) {
        throw std::runtime_error("Matrix and vector dimensions do not match for subtraction assignment.");
    }
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            (*this)(i, j) -= other[i][j];
        }
    }
    return *this;
}

/**
 * @brief Overloaded multiplication-assignment operator for matrix and scalar.
 * This function multiplies each element of the current matrix with a scalar value.
 * It does this in-place, i.e. it modifies the current matrix.
 * @param value The scalar value to multiply the matrix with.
 * @return The updated matrix after multiplication.
 * @throws None.
 */
mat& mat::operator*=(float value) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            (*this)(i, j) *= value;
        }
    }
    return *this;
}

/**
 * @brief Overloaded multiplication-assignment operator for matrix.
 * This function multiplies the current matrix with another matrix in-place.
 * It checks if the dimensions of the two matrices are compatible for multiplication, and if they are not, it throws a runtime error.
 * Otherwise, it multiplies the elements of the two matrices and returns the updated matrix.
 * @param other The matrix to be multiplied with the current matrix.
 * @return The updated matrix after multiplication.
 * @throws std::runtime_error if the dimensions of the two matrices do not match for multiplication.
 */
mat& mat::operator*=(const mat& other) {
    // Standard compound assignment A *= B requires A = A * B.
    // This usually requires A to be square and B to have compatible dimensions,
    // or it changes the dimensions of A, which is complex for mapped files.
    if (col != other.row) {
        throw std::runtime_error("Matrix dimensions are incompatible for multiplication assignment.");
    }
    // Calculate result = (*this) * other
    mat result(row, other.col); // result matrix (new temporary file)
    std::fill_n(result.mapped_data, static_cast<size_t>(result.row) * result.col, 0.0f);

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < other.col; j++) {
            for (int k = 0; k < col; k++) {
                result(i, j) += (*this)(i, k) * other(k, j);
            }
        }
    }
    // Assign the result back to *this using swap
    swap(*this, result);
    return *this;
}

/**
 * @brief Overloaded division-assignment operator for matrix and scalar.
 * This function divides each element of the current matrix with a scalar value.
 * It does this in-place, i.e. it modifies the current matrix.
 * @param value The scalar value to divide the matrix with.
 * @return The updated matrix after division.
 * @throws std::runtime_error if the scalar value is zero.
 */
mat& mat::operator/=(float value) {
    if (std::abs(value) < 1e-9) { // Check for division by zero
        throw std::runtime_error("Division by zero in scalar division assignment.");
    }
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            (*this)(i, j) /= value;
        }
    }
    return *this;
}

/**
 * @brief Overloaded division-assignment operator for matrix.
 * This function divides each element of the current matrix with the corresponding element of another matrix.
 * It does this in-place, i.e. it modifies the current matrix.
 * @param other The matrix to divide the current matrix with.
 * @return The updated matrix after division.
 * @throws std::runtime_error if the dimensions of the two matrices do not match.
 */
mat& mat::operator/=(const mat& other) {
    // This performs element-wise division assignment in the original code.
    // A /= B usually means A = A * B^-1.
    if (row != other.row || col != other.col) {
        throw std::runtime_error("Matrix dimensions do not match for element-wise division assignment.");
    }
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            float divisor = other(i, j);
            if (std::abs(divisor) < 1e-9) {
                throw std::runtime_error("Element-wise division by zero.");
            }
            (*this)(i, j) /= divisor; // Modify in place
        }
    }
    return *this;
}

/**
 * @brief In-place transpose of a matrix.
 * @details This function transposes the matrix in-place, meaning that it modifies the current matrix.
 * The transpose of a matrix is a matrix whose row and column indices are swapped, i.e.
 * the element at row i and column j is swapped with the element at row j and column i.
 * @throws throws error when the matrix is not square.
 */
void mat::transpose_inplace() { // Renamed function
    if(!ifsquare()) { // Added 
        throw std::runtime_error("In-place transpose requires a square matrix!");
    }
    else {
    // Iterate over the upper triangular part of the matrix
        for(int i = 0; i < row; ++i) { // Added 
            for(int j = i + 1; j < col; ++j) { // Added (col == row here)
                 std::swap((*this)(i, j), (*this)(j, i)); // Use operator()
            }
        }
    }
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
/* // Old standalone vector transpose - remove or update if needed for vectors specifically
std::vector<std::vector<float>> trnsps(const std::vector<std::vector<float>>& b) {
    return c;
}
*/
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
* Calculate the dot product of a matrix and a vector.
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
* Calculate the dot product of a vector and a matrix.
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
