
// includes basic operations
#include "include/mat.hpp"
#include <algorithm>
#include <numeric>
#include <iostream>
#include <cmath>
#include "basic.hpp"


/**
 * @brief index operator overload
 * @param i index of row
 * @param j index of column
 * @return reference to the element at position (i, j)
 */
double& mat::operator()(int i, int j) {
    return a[i][j];
}

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
mat mat::operator=(mat b) {
    // Check if the dimensions of the two matrices are the same.
    if(a.size() != b.a.size() || a[0].size() != b.a[0].size()) {
        // If the dimensions are not the same, print an error message and return an empty matrix.
        throw std::runtime_error("Matrix sizes do not match");
    }
    // Create a new matrix with the same dimensions as the current matrix.
    mat c(a.size(), a[0].size());
    this->a = b.a;
    this->row = b.row;
    this->col = b.col;
    return c;
}

/**
 * @brief Overloaded assignment operator for matrix.
 * This function takes another matrix represented as a vector of vectors of doubles and assigns it to the current matrix.
 * It checks if the dimensions of the two matrices are the same, and if not, it prints an error message and returns an empty matrix.
 * Otherwise, it copies the elements of the given matrix into the current matrix and returns the resulting matrix.
 * @param b The matrix to be assigned to the current matrix represented as a vector of vectors of doubles.
 * @return A matrix that is a copy of `b`.
 * @throws throws error if the dimensions of the two matrices are not the same.
 */
mat mat::operator=(std::vector<std::vector<double>> b) {
    // Check if the dimensions of the two matrices are the same.
    if(a.size() != b.size() || a[0].size() != b[0].size()) {
        // If the dimensions are not the same, print an error message and return an empty matrix.
        std::cerr << "Matrix sizes do not match" << std::endl;
    }
    // Create a new matrix with the same dimensions as the current matrix.
    mat c(a.size(), a[0].size());
    this->a = b;
    this->row = b.size();
    this->col = b[0].size();
    return c;
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
mat mat::operator+(mat b) {
    // Check if the dimensions of the two matrices are the same.
    if(a.size() != b.a.size() || a[0].size() != b.a[0].size()) {
        // If the dimensions are not the same, print an error message and return an empty matrix.
        std::cerr << "Matrix sizes do not match" << std::endl;
    }
    // Create a new matrix with the same dimensions as the current matrix.
    mat c(this->a.size(), this->a[0].size());
    c.a = this->a + b.a;
    // Return the resulting matrix.
    return c;
}

/**
 * @brief Overloaded addition operator for matrix.
 * This function takes a vector of vectors of type `std::vector<std::vector<double>>` and adds it to the current matrix.
 * It checks if the dimensions of the two matrices are the same, and if not, it prints an error message and returns an empty matrix.
 * Otherwise, it adds the elements of the two matrices and returns the resulting matrix.
 * @param b The matrix to be added to the current matrix.
 * @return A matrix that is the sum of the current matrix and `b`.
 * @throws throws error if the dimensions of the two matrices are not the same.
 */
mat mat::operator+(std::vector<std::vector<double>> b) {
    // Check if the dimensions of the two matrices are the same.
    if(a.size() != b.size() || a[0].size() != b[0].size()) {
        // If the dimensions are not the same, print an error message and return an empty matrix.
        std::cerr << "Matrix sizes do not match" << std::endl;
    }
    // Create a new matrix with the same dimensions as the current matrix.
    mat c(a.size(), a[0].size());
    // Add the elements of the two matrices and store the result in `c`.
    c.a = this->a + b;
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
mat mat::operator-(mat b) {
    // Check if the dimensions of the two matrices are the same.
    if(a.size() != b.a.size() || a[0].size() != b.a[0].size()) {
        // If the dimensions are not the same, print an error message and return an empty matrix.
        std::cerr << "Matrix sizes do not match" << std::endl;
    }
    // Create a new matrix with the same dimensions as the current matrix.
    mat c(a.size(), a[0].size());
    // Subtract the elements of the two matrices and store the result in `c`.
    c.a = this->a - b.a;
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
mat mat::operator-(std::vector<std::vector<double>> b) {
    if(a.size() != b.size() || a[0].size() != b[0].size()) {
        // If the dimensions are not the same, print an error message and return an empty matrix.
        std::cerr << "Matrix sizes do not match" << std::endl;
    }
    // Create a new matrix with the same dimensions as the current matrix.
    mat c(a.size(), a[0].size());
    // Subtract the elements of the two matrices and store the result in `c`.
    c.a = this->a - b;
    // Return the resulting matrix.
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
mat mat::operator*(double b) {
    // Create a new matrix with the same dimensions as the input matrix
    mat c(a.size(), a[0].size());
    // Iterate over each element in the input matrix and multiply it by the scalar
    for(int i = 0; i < a.size(); i++) {
        for(int j = 0; j < a[0].size(); j++) {
            c.a[i][j] = a[i][j] * b;
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
 */
mat mat::operator/(double b) {
    // Create a new matrix with the same dimensions as the input matrix
    mat c(a.size(), a[0].size());
    // Iterate over each element in the input matrix and divide it by the scalar
    for(int i = 0; i < a.size(); i++) {
        for(int j = 0; j < a[0].size(); j++) {
            // Check if the element is zero before division
            if(a[i][j] != 0) {
                c.a[i][j] = a[i][j] / b;
            }
            c.a[i][j] = a[i][j] / b;
        }
    }
    // Return the new matrix.
    return c;
}

/**
 * @brief Overloaded operator for matrix multiplication with another matrix.
 *      This function takes a matrix and another matrix as input and returns a new matrix
 *      It uses strassen's matrix multiplication method.
 * @param a The matrix to multiply the current matrix with.
 * @return Product of two matrix
 */
mat mat::operator*(mat a) {
    if(this->row == this->col == a.row == a.col && this->row%2 == 0) {
        // use of strassens for same matrix multiplication
        if (this->row == 2) {            
            mat c(2);
            double p1 = this->a[0][0] * (a.a[0][1] - a.a[1][1]);
            double p2 = (this->a[0][0] + this->a[0][1]) * a.a[1][1];
            double p3 = (this->a[1][0] + this->a[1][1]) * a.a[0][0];
            double p4 = this->a[1][1] * (a.a[1][0] - a.a[0][0]);
            double p5 = (this->a[0][0] + this->a[1][1]) * (a.a[0][0] + a.a[1][1]);
            double p6 = (this->a[0][1] - this->a[1][1]) * (a.a[1][0] + a.a[1][1]);
            double p7 = (this->a[0][0] - this->a[1][0]) * (a.a[0][0] + a.a[0][1]);
            // calculate the four submatrices
            c.a[0][0] = p5 + p4 - p2 + p6;
            c.a[0][1] = p1 + p2;
            c.a[1][0] = p3 + p4;
            c.a[1][1] = p1 + p5 - p3 - p7;
            return c;
        }
        int half = this->row / 2;
        // split the matrices into submatrices
        mat a1(half, half), a2(half, half), a3(half, half), a4(half, half), b1(half, half), b2(half, half), b3(half, half), b4(half, half);
        for(int i = 0; i < half; i++) {
            for(int j = 0; j < half; j++) {
                a1.a[i][j] = this->a[i][j];
                a2.a[i][j] = this->a[i][j + half];
                a3.a[i][j] = this->a[i + half][j];
                a4.a[i][j] = this->a[i + half][j + half];
                b1.a[i][j] = a.a[i][j];
                b2.a[i][j] = a.a[i][j + half];
                b3.a[i][j] = a.a[i + half][j];
                b4.a[i][j] = a.a[i + half][j + half];
            }
        }
        // calculate the four submatrices
        mat c1 = a1 * b1 - a2 * b4;
        mat c2 = a1 * b2 + a2 * b3;
        mat c3 = a3 * b1 + a4 * b2;
        mat c4 = a3 * b4 - a4 * b3;
        mat c(half * 2, half * 2);
        // combine the four submatrices into one matrix
        return c;
    }
    else {
        mat c(this->col, a.col);
        for(int i = 0; i < this->row; i++) {
            for(int j = 0; j < a.col; j++) {
                for(int k = 0; k < this->col; k++) {
                    c.a[i][j] += this->a[i][k] * a.a[k][j];
                }
            }
        }
        return c;
    }
    throw std::runtime_error("Invalid Sizes");
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
mat mat::operator/(mat a) {
    if(this->row != a.row || this->row != a.a.size()) {
        // Check if the dimensions of the matrix and the vector are the same
        std::cerr << "Invalid Sizes" << std::endl;
    }
    // matrix division by matrix is similar to matrix multiplication by inverse matrix
    mat c(this->a.size());
    c = a * this->inverse();
    return c;
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
mat mat::operator+=(mat other) {
    if (this->row != other.row || this->col != other.col) {
        throw std::runtime_error("Matrix sizes do not match");
    }
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            this->a[i][j] += other.a[i][j];
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
mat mat::operator-=(mat other) {
    if (this->row != other.row || this->col != other.col) {
        throw std::runtime_error("Matrix sizes do not match");
    }
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            this->a[i][j] -= other.a[i][j];
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
mat mat::operator+=(std::vector<std::vector<double>> other) {
    if (this->row != other.size() || this->col != other[0].size()) {
        throw std::runtime_error("Matrix sizes do not match");
    }
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            this->a[i][j] += other[i][j];
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
mat mat::operator-=(std::vector<std::vector<double>> other) {
    if (this->row != other.size() || this->col != other[0].size()) {
        throw std::runtime_error("Matrix sizes do not match");
    }
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            this->a[i][j] -= other[i][j];
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
mat mat::operator*=(double value) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            this->a[i][j] *= value;
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
mat mat::operator*=(mat other) {
    if (this->col != other.row) {
        throw std::runtime_error("Matrix sizes do not match for multiplication");
    }
    mat c(this->row, other.col); // result matrix
    for (int i = 0; i < this->row; i++) {
        for (int j = 0; j < other.col; j++) {
            // multiply the elements of the two matrices
            for (int k = 0; k < this->col; k++) {
                c.a[i][j] += this->a[i][k] * other.a[k][j];
            }
        }
    }
    // replace the current matrix with the result matrix
    return c;
}

/**
 * @brief Overloaded division-assignment operator for matrix and scalar.
 * This function divides each element of the current matrix with a scalar value.
 * It does this in-place, i.e. it modifies the current matrix.
 * @param value The scalar value to divide the matrix with.
 * @return The updated matrix after division.
 * @throws std::runtime_error if the scalar value is zero.
 */
mat mat::operator/=(double value) {
    if (value == 0) {
        throw std::runtime_error("Division by zero");
    }
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            this->a[i][j] /= value;
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
mat mat::operator/=(mat other) {
    if (this->row != other.row || this->col != other.col) {
        throw std::runtime_error("Matrix sizes do not match");
    }
    mat c(this->row, this->col);
    for (int i = 0; i < this->row; i++) {
        for (int j = 0; j < this->col; j++) {
            c.a[i][j] = this->a[i][j] / other.a[i][j];
        }
    }
    return c;
}

/**
 * @brief Create an identity matrix of size n x n.
 * @param n Size of the matrix (number of rows and columns).
 * @return The identity matrix.
 */
mat mat::imat(int n) {
    // Initialize a matrix of size n x n
    mat A(n);
    // Set diagonal elements to 1
    for(int i = 0; i < n; i++)
        A.a[i][i] = 1;
    // Return the identity matrix
    return A;
}


/**
 * @brief Transpose of matrix.
 * @details This function transposes the matrix in-place, meaning that it modifies the current matrix.
 * The transpose of a matrix is a matrix whose row and column indices are swapped, i.e.
 * the element at row i and column j is swapped with the element at row j and column i.
 * @throws throws error when the matrix is not square.
 */
mat mat::transpose() {
    if(ifsquare() == 0) {
        throw std::runtime_error("Transpose is for square matrix only!");
    }
    mat b(*this);
    // Iterate over the upper triangular part of the matrix
    for(size_t i = 0; i < b.a.size(); ++i) {
        for(size_t j = i + 1; j < b.a[i].size(); ++j) {
            if(i != j) { std::swap(b.a[i][j], b.a[j][i]); }
        }
    }
    return b;
}

/**
 * @brief In-place transpose of a matrix.
 * @details This function transposes the matrix in-place, meaning that it modifies the current matrix.
 * The transpose of a matrix is a matrix whose row and column indices are swapped, i.e.
 * the element at row i and column j is swapped with the element at row j and column i.
 * @throws throws error when the matrix is not square.
 */
void mat::trnsps() {
    if(ifsquare() == 0) {
        throw std::runtime_error("Transpose is for square matrix only!");
    }
    else {
    // Iterate over the upper triangular part of the matrix
        for(size_t i = 0; i < a.size(); ++i) {
            for(size_t j = i + 1; j < a[i].size(); ++j) {
                if(i != j) { std::swap(a[i][j], a[j][i]); }
            }
        }
    }
}

/**
 * @brief Multiplies a matrix by a vector using hadamard matrix multiplication.
 * @param[in] a 2D vector of doubles representing a matrix
 * @param[in] b 1D vector of doubles representing a vector
 * @return a 1D vector of doubles representing the result of the matrix multiplication
 * @throws throws error if the sizes of the matrices are not compatible
 */
std::vector<double> matmul(std::vector<std::vector<double>> a, std::vector<double> b) {
    if(a[0].size() != b.size()) {
        throw std::runtime_error("Size of matrix and vector must be the same");
    }
    std::vector<double> c(a.size());
    for(int i = 0; i < a.size(); i++) {
        for(int j = 0; j < a[0].size(); j++) {
            c[i] += a[i][j] * b[j];
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
double mat::trace() {
    // Check if the matrix is square
    if(this->ifsquare() == false) {
        throw std::invalid_argument("Matrix is not square");
    }
    double tr = 0;  // Initialize the trace to 0
    int n = a.size(); // Get the number of rows and columns
    // Iterate over the main diagonal and add the elements to the trace
    for(int i = 0; i < n;i++) {
        tr +=  a[i][i];
    }
    return tr;
}

/**
 * @brief Compute the minor matrix of a matrix.
 * The minor matrix is a matrix whose elements are the determinants of 
 * the submatrices formed by removing the current row and column.
 * @param c matrix input
 * @return minor matrix
 */
mat minor(mat c) {
    mat b, min;
    // Loop over all elements of the matrix
    for(int i = 0; i < c.a.size(); i++) {
        for(int j = 0; j < c.a[i].size(); j++) {
            // Compute the submatrix by removing the current row and column
            b = submat(c, i, j);
            // Compute the determinant of the submatrix
            min.a[i][j] = b.det() * std::pow((-1), i+j);
        }
    }
    return min;
}

/**
 * @brief Compute the cofactor matrix of a matrix.
 * @details The cofactor matrix is a matrix whose elements are the cofactors of the elements of the matrix.
 * @return The cofactor matrix.
 */
mat mat::cofac() {
    mat b(this->a.size());
    mat c(this->a.size()-1);
    // Loop over all elements of the matrix
    for(int i = 0; i < this->a.size(); i++) {
        for(int j = 0; j < this->a[i].size(); j++) {
            // Compute the submatrix by removing the current row and column
            c = submat(this->a, i, j);
            // Compute the determinant of the submatrix
            b.a[i][j] = pow(-1, i+j) * c.det();
        }
    }
    return b;
}

/**
 * @brief Transpose a matrix.
 * @details This function takes a 2D vector (matrix) as input and returns its transpose.
 * The transpose of a matrix is obtained by swapping its rows and columns.
 * @param b 2D vector of doubles representing the matrix to be transposed.
 * @return A 2D vector of doubles representing the transposed matrix.
 */
std::vector<std::vector<double>> trnsps(std::vector<std::vector<double>> b) {
    if(b[0].size() != b.size()) {
        throw std::invalid_argument("Matrix is not square");
        return b;
    }
    // Create a copy of the input matrix to store the transposed matrix
    std::vector<std::vector<double>> c = b;
    // Iterate over each element in the matrix
    for(int i = 0; i < c.size(); i++) {
        for(int j = i+1; j < c.size(); j++) {
            // Swap elements to transpose the matrix
            if(i != j)
                std::swap(c[i][j], c[j][i]);
        }
    }
    // Return the transposed matrix
    return c;
}

/**
 * @brief Compute the adjoint of a matrix.
 * @details The adjoint of a matrix is the transpose of the cofactor matrix.
 * The cofactor matrix is a matrix whose elements are the cofactors of the elements of the matrix.
 * @return The adjoint of the matrix.
 */
mat mat::adjoint() {
    // Compute the cofactor matrix
    mat b(this->cofac());
    // Compute the transpose of the cofactor matrix
    return b.transpose();
}

/**
 * @brief Compute the covariance matrix of a dataset.
 * @details This function computes the covariance matrix for a matrix of data points.
 * Each column of the input matrix represents a variable, and each row represents a data point.
 * @param a A 2D vector of doubles representing the dataset.
 * @return A 2D vector of doubles representing the covariance matrix.
 */
mat covariance(mat a) {
    // Compute mean of each column
    mat cov(a.a.size());     // covariance matrix
    mat trnsp = a.transpose();  // transpose of matrix a
    std::vector<double> m(a.a.size());
    for (int i = 0; i < a.a.size(); i++) {
        // compute mean of column i
        m.push_back(std::accumulate(trnsp.a[i].begin(), trnsp.a[i].end(), 0.0)/a.a[i].size());
    }
    // subtract mean from each column
    for (size_t i = 0; i < a.a.size(); i++) {
        for (size_t j = 0; j < a.a[i].size(); j++) {
            trnsp.a[i][j] -= m[i];
        }
    }
    // compute transpose of matrix a
    mat a1 = a.transpose();
    if (a.a.size() == 0) {
        throw std::invalid_argument("Matrix size cannot be 0");
    }
    // compute covariance matrix
    cov = (a1*trnsp)*(1/(static_cast<double>(a.a[0].size())-1));
    return cov;
}

/**
 * @brief Calculate the covariance matrix of a dataset.
 * @details This function computes the covariance matrix for a matrix of data points.
 * Each column of the input matrix represents a variable, and each row represents a data point.
 * @param a A 2D vector of doubles representing the dataset.
 * @return A 2D vector of doubles representing the covariance matrix.
 */
std::vector<std::vector<double>> covmat(std::vector<std::vector<double>> a) {
    std::vector<double> m(a.size()); // Mean of columns of a
    std::vector<std::vector<double>> cov(a.size(), std::vector<double>(a.size(), 0));
    // Transpose of matrix a
    std::vector<std::vector<double>> c(a.size(), std::vector<double>(a[0].size(), 0));
    std::vector<std::vector<double>> d(a[0].size(), std::vector<double>(a.size(), 0));
    // Calculate transpose
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a.size(); j++) {
            c[j][i] = a[i][j];
            d[i][j] = a[i][j];
        }
    }
    // Calculate mean of columns of a
    for (int i = 0; i < a.size(); i++) {
        m[i] = std::accumulate(a[i].begin(), a[i].end(), 0.0)/a[i].size(); // Mean of rows of c
    }
    // Subtract mean from each column
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a.size(); j++) {
            c[i][j] -= m[i];
            d[i][j] -= m[j];
        }
    }
    // Calculate covariance matrix
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a.size(); j++) {
            for (int k = 0; k < a.size(); k++) {
                cov[i][j] += c[i][k] * d[k][j];
            }
        }
    }
    // Normalize the covariance matrix
    double normalizationFactor = 1 / (static_cast<double>(a.size()) - 1);
    for (int i = 0; i < cov.size(); i++) {
        for (int j = 0; j < cov[0].size(); j++) {
            cov[i][j] *= normalizationFactor;
        }
    }
    return cov;
}

/**
 * @brief Creates a matrix with random values between 0 and 1.
 * This function creates a matrix with the specified number of rows and columns, and fills it with random values between 0 and 1.
 * @param row The number of rows in the matrix.
 * @param col The number of columns in the matrix.
 * @return A matrix with the specified size and random values.
 */
mat mat::Random(int row, int col) {
    mat result(row, col);
    std::mt19937 gen(std::random_device{}()); // Create a random number generator
    std::normal_distribution<double> dist(-10.0, 10.0); // Create a distribution that generates random numbers between 0 and 1
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            result.a[i][j] = dist(gen); // Generate a random number and assign it to the current position in the matrix
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
std::vector<double> dot(mat a, std::vector<double> b) {
    if(a.a.empty() || a.a[0].size() != b.size())
        throw std::runtime_error("Row vectors of matrix should have the same size as the vector");
    
    std::vector<double> c(a.a.size());
    
    for(size_t i = 0; i < a.a.size(); i++) {
        c[i] = std::inner_product(a.a[i].begin(), a.a[i].end(), b.begin(), 0.0);
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
std::vector<double> dot(std::vector<double> a, mat b) {
    if(b.a.empty() || b.a[0].size() != a.size())
        throw std::runtime_error("Row vectors of matrix should have the same size as the vector");
    
    std::vector<double> c(b.a.size());
    
    for(size_t i = 0; i < b.a.size(); i++) {
        c[i] = std::inner_product(b.a[i].begin(), b.a[i].end(), a.begin(), 0.0);
    }
    
    return c;
}
