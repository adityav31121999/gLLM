
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
    if (!this->ifsquare()) {
        throw std::runtime_error("Gauss-Jordan inversion requires a square matrix.");
    }
    // Determinant check is implicitly handled by checking for zero pivots later.
    // Explicit det() call might be expensive and redundant here.
    // if (std::abs(this->det()) < 1e-9) { // Assuming det() is updated
    //     throw std::runtime_error("Matrix is singular, cannot calculate inverse.");
    // }

    int n = this->row;
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
