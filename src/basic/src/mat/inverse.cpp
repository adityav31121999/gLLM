
// ANYONE CAN USE IT
#include <algorithm>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include "include/mat.hpp"

// converts a row-major matrix to square matrix by adding extra columns, 
// and providing values to diagonal or antidiagonal elements 1 and remaining
// new elements 0.
void mat::row2Square(bool dia) {
    int newCol = row - col;
    int start = col;
    // resize matrix to make square
    for(int i = start-1; i < newCol; i++) {
        // set value to new diagonal elements
    }
}

// converts a column-major matrix to square matrix by adding extra rows, 
// and providing values to diagonal or antidiagonal elements 1 and remaining
// new elements 0.
void mat::col2Square(bool dia) {
    if(dia == 0) {
        // antidiagonal
    }
    else {
        // diagonal
    }
}


// Helper function to swap two rows in mapped memory
void swap_rows_in_mapped_memory(mat& m, int r1, int r2) {
    if (r1 == r2 || !m.mapped_data) return;
    int cols = m.col;
    size_t row_size_bytes = static_cast<size_t>(cols) * sizeof(float);
    // Use a temporary buffer for the swap
    std::vector<float> temp_row(cols);
    float* row1_ptr = m.mapped_data + static_cast<size_t>(r1) * cols;
    float* row2_ptr = m.mapped_data + static_cast<size_t>(r2) * cols;

    memcpy(temp_row.data(), row1_ptr, row_size_bytes);
    memcpy(row1_ptr, row2_ptr, row_size_bytes);
    memcpy(row2_ptr, temp_row.data(), row_size_bytes);
}

/**
 * @brief Calculate the inverse of the matrix using Gauss-Jordan Elimination
 * @return The inverse of the matrix
 * @throws std::runtime_error if the matrix is not square or is singular (non-invertible).
 */
mat mat::gaussjordan() const {
    // Check if the matrix is square (assuming ifsquare() is updated)
    if (!ifsquare()) {
        throw std::runtime_error("Gauss-Jordan inversion requires a square matrix.");
    }
    // Determinant check is implicitly handled by checking for zero pivots later.
    // Explicit det() call might be expensive and redundant here.
    // if (std::abs(det()) < 1e-9) { // Assuming det() is updated
    //     throw std::runtime_error("Matrix is singular, cannot calculate inverse.");
    // }

    int n = row;
    mat d = *this; // Create a working copy using the copy constructor (new temp file)
    mat b(n);      // Create the identity matrix (new temp file)

    // Initialize b as the identity matrix
    std::fill_n(b.mapped_data, static_cast<size_t>(n) * n, 0.0f);
    for (int i = 0; i < n; ++i) {
        b(i, i) = 1.0f;
    }

    // --- Gaussian Elimination (Forward Elimination + Backward Substitution) ---

    for (int i = 0; i < n; ++i) {
        // Find pivot (largest element in current column i, from row i downwards)
        int pivot_row = i;
        float max_val = std::abs(d(i, i));
        for (int k = i + 1; k < n; ++k) {
            if (std::abs(d(k, i)) > max_val) {
                max_val = std::abs(d(k, i));
                pivot_row = k;
            }
        }

        // Check for singularity (zero pivot)
        if (max_val < 1e-9) { // Use a small tolerance for floating-point comparison
            throw std::runtime_error("Matrix is singular (or nearly singular), cannot invert.");
        }

        // Swap rows if necessary to bring pivot to diagonal
        if (pivot_row != i) {
            swap_rows_in_mapped_memory(d, i, pivot_row);
            swap_rows_in_mapped_memory(b, i, pivot_row);
        }

        // Normalize the pivot row (make pivot element 1)
        float pivot_val = d(i, i); // Get pivot value *after* potential swap
        for (int j = 0; j < n; ++j) { // Normalize entire row
            d(i, j) /= pivot_val;
            b(i, j) /= pivot_val;
        }

        // Eliminate other entries in the current column i
        for (int k = 0; k < n; ++k) {
            if (k != i) {
                float factor = d(k, i);
                // Subtract factor * pivot_row from row k
                for (int j = 0; j < n; ++j) { // Apply to entire row
                    d(k, j) -= factor * d(i, j);
                    b(k, j) -= factor * b(i, j);
                }
            }
        }
    }

    // At this point, matrix 'd' should be the identity matrix (within float precision)
    // and matrix 'b' should be the inverse of the original matrix.

    return b;
}