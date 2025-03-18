
// properties and operations of matrix
#include "include/mat.hpp"

/**
 * @brief Check if the matrix is square.
 * @return true if the matrix is square, false otherwise.
 */
bool mat::ifsquare() {
    // Compare the number of rows and columns
    if(this->row == this->col)
        return true;
    return false;
}


/**
 * @brief Check if the matrix is rectangular.
 * @return true if the matrix is rectangular (i.e. not square), false otherwise.
 */
bool mat::ifrectangular() {
    // Compare the number of rows and columns
    if(this->row != this->col)
        return true;
    return false;
}


/**
 * @brief Check if the matrix is symmetric.
 * @return true if the matrix is symmetric, false otherwise.
 */
bool mat::ifsymmetric() {
    // Check if the matrix is square
    if(this->ifsquare() == true) {
        int n = this->a.size();
        // Check symmetry
        for(int i = 0; i < n; i++) {
            for(int j = i+1; j < n; j++) {
                if(i != j) {
                    // Check if the elements are equal
                    if(this->a[i][j] != this->a[j][i])
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
bool mat::ifidentity() {
    // Check if the matrix is square
    if (this->ifsquare()) {
        int n = this->a.size();
        // Check if the matrix is the identity matrix
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                // Check if the elements are the same as the identity matrix
                if ((i == j && this->a[i][j] != 1) || (i != j && this->a[i][j] != 0)) {
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
bool mat::ifdiagonal() {
    // Check if the matrix is square
    if (this->ifsquare()) {
        int n = this->a.size();
        // Check if the matrix is diagonal
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                // Check if the non-diagonal elements are 0
                if (i != j && this->a[i][j] != 0) {
                    return false;
                }
                // Check if the diagonal elements are non-0
                if (i == j && this->a[i][j] == 0) {
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
bool mat::ifupper() {
    int n = this->a.size();
    // Check if the elements below the diagonal are 0
    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (this->a[i][j] != 0) {
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
bool mat::iflower() {
    int n = this->a.size();
    // Check if the diagonal and elements above the diagonal are 0
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i > j && this->a[i][j] != 0) {
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
bool mat::ifskew() {
    int n = this->a.size();
    // Iterate over the matrix elements
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            // Check if the element is the negative of its symmetric counterpart
            if (this->a[i][j] != -this->a[j][i]) {
                return false; // Return false if any element fails the skew-symmetric condition
            }
        }
    }
    return true; // Return true if all elements satisfy the skew-symmetric condition
}
