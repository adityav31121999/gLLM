
// properties and operations of matrix
#include "include/mat.hpp" // Use relative path if appropriate
#include <cmath>   // For std::abs
#include <limits>  // For numeric_limits

/**
 * @brief set value for ith row and jth column
 * @param i row number
 * @param j column number
 * @param val value to be inserted or set
 */
void mat::set(int i, int j, float val) {
    // Use operator() which includes bounds checking
    (*this)(i, j) = val;
}


/**
 * @brief Check if the matrix is square.
 * @return true if the matrix is square, false otherwise.
 */
bool mat::ifsquare() const { // Added const
    // Compare the number of rows and columns - No change needed here
    if(this->row == this->col)
        return true;
    return false;
}


/**
 * @brief Check if the matrix is symmetric.
 * @return true if the matrix is symmetric, false otherwise.
 */
bool mat::ifsymmetric() const { // Added const
    constexpr float tolerance = std::numeric_limits<float>::epsilon() * 100; // Tolerance for float comparison
    // Check if the matrix is square
    if(this->ifsquare()) {
        int n = this->row; // Use row member
        // Check symmetry
        for(int i = 0; i < n; i++) {
            for(int j = i+1; j < n; j++) {
                // Check if the elements are equal using operator() and tolerance
                if(std::abs((*this)(i, j) - (*this)(j, i)) > tolerance) {
                        return false;
                }
            }
        }
        return true;
    }
    return false;
}


/**
 * @brief Check if the matrix is the identity matrix.
 * @details This function takes a matrix as an argument and returns true if the matrix is the identity matrix, false otherwise.
 * @return true if the matrix is the identity matrix, false otherwise.
 */
bool mat::ifidentity() const { // Added const
    constexpr float tolerance = std::numeric_limits<float>::epsilon() * 100; // Tolerance for float comparison
    // Check if the matrix is square
    if (this->ifsquare()) {
        int n = this->row; // Use row member
        // Check if the matrix is the identity matrix
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                // Check elements using operator() and tolerance
                if ((i == j && std::abs((*this)(i, j) - 1.0f) > tolerance) || (i != j && std::abs((*this)(i, j)) > tolerance)) {
                    // If not, return false
                    return false;
                }
            }
        }
        // If the matrix is the identity matrix, return true
        return true;
    }
    // If the matrix is not square, return false
    return false;
}


/**
 * @brief Check if the matrix is diagonal.
 * @details This function takes a matrix as an argument and returns true if the matrix is diagonal, false otherwise.
 * @return true if the matrix is diagonal, false otherwise.
 */
bool mat::ifdiagonal() const { // Added const
    constexpr float tolerance = std::numeric_limits<float>::epsilon() * 100; // Tolerance for float comparison
    // Check if the matrix is square
    if (this->ifsquare()) {
        int n = this->row; // Use row member
        // Check if the matrix is diagonal
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                // Check if the non-diagonal elements are close to 0
                if (i != j && std::abs((*this)(i, j)) > tolerance) {
                    return false;
                }
                // Check if the diagonal elements are NOT close to 0 (optional, depends on definition)
                // If diagonal elements *can* be zero, remove this check:
                if (i == j && std::abs((*this)(i, j)) < tolerance) {
                    return false;
                }
            }
        }
        return true;
    }
    return false;
}


/**
 * @brief Check if the matrix is upper triangular.
 * @details This function takes a matrix as an argument and returns true if the matrix is upper triangular, false otherwise.
 * @return true if the matrix is upper triangular, false otherwise.
 */
bool mat::ifupper() const { // Added const
    constexpr float tolerance = std::numeric_limits<float>::epsilon() * 100; // Tolerance for float comparison
    int n = this->row; // Use row member (assuming square or checking relevant part)
    // Check if the elements below the diagonal are 0
    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i && j < this->col; j++) { // Ensure j stays within column bounds
            if (std::abs((*this)(i, j)) > tolerance) { // Use operator() and tolerance
                // If not, return false
                return false;
            }
        }
    }
    // If the matrix is upper triangular, return true
    return true;
}


/**
 * @brief Check if the matrix is lower triangular.
 * @details This function takes a matrix as an argument and returns true if the matrix is lower triangular, false otherwise.
 * @return true if the matrix is lower triangular, false otherwise.
 */
bool mat::iflower() const { // Added const
    constexpr float tolerance = std::numeric_limits<float>::epsilon() * 100; // Tolerance for float comparison
    // Check if elements *above* the main diagonal are zero
    for (int i = 0; i < this->row; i++) {
        // Start j from i+1 to check only elements above the diagonal
        for (int j = i + 1; j < this->col; j++) {
            if (std::abs((*this)(i, j)) > tolerance) {
                // If not, return false
                return false;
            }
        }
    }
    // If the matrix is lower triangular, return true
    return true;
}


/**
 * @brief Check if the matrix is skew-symmetric.
 * @details A matrix is skew-symmetric if for every element a[i][j], a[i][j] = -a[j][i].
 * @return true if the matrix is skew-symmetric, false otherwise.
 */
bool mat::ifskew() const { // Added const
    constexpr float tolerance = std::numeric_limits<float>::epsilon() * 100; // Tolerance for float comparison
    if (!this->ifsquare()) {
        return false; // Skew-symmetric must be square
    }
    int n = this->row; // Use row member
    // Iterate over the matrix elements
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            // Check diagonal elements are zero and off-diagonal are negative symmetric
            if ((i == j && std::abs((*this)(i, j)) > tolerance) ||
                (i != j && std::abs((*this)(i, j) + (*this)(j, i)) > tolerance)) {
                return false; // Return false if any element fails the skew-symmetric condition
            }
        }
    }
    return true; // Return true if all elements satisfy the skew-symmetric condition
}
