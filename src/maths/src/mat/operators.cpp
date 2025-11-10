
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
