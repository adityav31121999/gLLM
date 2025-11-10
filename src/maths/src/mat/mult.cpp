#include "include/mat.hpp"
#include <thread>
#include <mutex>
#include <numeric>

/**
 * @brief Computes the Hadamard product for a specified range of rows.
 * @details This is a helper function designed to be called by a thread to parallelize
 * the element-wise multiplication of two matrices.
 * @param a The first input matrix.
 * @param b The second input matrix.
 * @param result The matrix where the results for the specified range will be stored.
 * @param a_row The number of rows in matrix a.
 * @param a_col The number of columns in matrix a.
 * @param b_row The number of rows in matrix b.
 * @param b_col The number of columns in matrix b.
 * @param start_row The starting row index (inclusive) for this thread's computation.
 * @param end_row The ending row index (exclusive) for this thread's computation.
 */
void hadamard_range(const mat& a, const mat& b, mat& result, int a_row, int a_col, int b_row, int b_col, int start_row, int end_row) {
    if (a_row != b_row || a_col != b_col) {
        throw std::runtime_error("Matrix dimensions must match for Hadamard product.");
    }

    for (int i = start_row; i < end_row; ++i) {
        for (int j = 0; j < a_col; ++j) {
            result(i, j) = a(i, j) * b(i, j);
        }
    }
}

/**
 * @brief Computes the matrix product for a specified range of result rows.
 * @details This is a helper function designed to be called by a thread to parallelize
 * the multiplication of two matrices.
 * @param a The first input matrix (left-hand side).
 * @param b The second input matrix (right-hand side).
 * @param result The matrix where the results for the specified range will be stored.
 * @param a_col The number of columns in matrix a.
 * @param b_row The number of rows in matrix b.
 * @param b_col The number of columns in matrix b.
 * @param start_row The starting row index of the result matrix (inclusive) for this thread's computation.
 * @param end_row The ending row index of the result matrix (exclusive) for this thread's computation.
 */
void matrix_mult_range(const mat& a, const mat& b, mat& result, int a_col, int b_row, int b_col, int start_row, int end_row) {
    if (a_col != b_row) {
        throw std::runtime_error("Matrix dimensions are incompatible for multiplication.");
    }

    for (int i = start_row; i < end_row; ++i) {
        for (int j = 0; j < b_col; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < a_col; ++k) {
                sum += a(i, k) * b(k, j);
            }
            result(i, j) = sum;
        }
    }
}

/**
 * @brief Computes the Hadamard (element-wise) product of this matrix and another matrix.
 * @details This function performs an element-wise multiplication between the current matrix
 * and the `other` matrix. It uses multithreading to parallelize the computation over the rows.
 * @param other The matrix to multiply with element-wise.
 * @return A new matrix containing the result of the Hadamard product.
 * @throws std::runtime_error if the dimensions of the two matrices do not match.
 */
mat mat::hadamard(const mat& other) const {
    if (row != other.row || col != other.col) {
        throw std::runtime_error("Matrix dimensions must match for Hadamard product.");
    }

    mat result(row, col);

    // Determine the number of threads to use (adjust as needed)
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) {
        num_threads = 4; // Default to 4 threads if hardware_concurrency() fails
    }

    // Divide the rows among the threads
    std::vector<std::thread> threads;
    int rows_per_thread = row / num_threads;
    int extra_rows = row % num_threads;

    int start_row = 0;
    for (unsigned int i = 0; i < num_threads; ++i) {
        int end_row = start_row + rows_per_thread;
        if (i < extra_rows) {
            end_row++;
        }

        threads.emplace_back(hadamard_range, std::ref(*this), std::ref(other), std::ref(result), row, col, other.row, other.col, start_row, end_row);
        start_row = end_row;
    }

    // Join the threads
    for (auto& thread : threads) {
        thread.join();
    }
    return result;
}

/**
 * @brief Computes the Hadamard (element-wise) product of two matrices.
 * @details This is a convenience function that calls the member hadamard function on matrix `a`.
 * @param a The first matrix.
 * @param b The second matrix.
 * @return A new matrix containing the result of the Hadamard product.
 */
mat mat::hadamard(const mat& a, const mat& b) {
    return a.hadamard(b);
}

/**
 * @brief Multiplies this matrix by another matrix.
 * @details This function performs matrix multiplication (this * other). It uses multithreading
 * to parallelize the computation over the rows of the resulting matrix.
 * @param other The matrix to multiply with.
 * @return A new matrix containing the result of the multiplication.
 * @throws std::runtime_error if the inner dimensions of the matrices are incompatible for multiplication.
 */
mat mat::mult(const mat& other) const {
    if (col != other.row) {
        throw std::runtime_error("Matrix dimensions are incompatible for multiplication.");
    }

    mat result(row, other.col);
    std::fill_n(result.mapped_data, static_cast<size_t>(result.row) * result.col, 0.0f);

    // Determine the number of threads to use
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) {
        num_threads = 4; // Default to 4 threads if hardware_concurrency() fails
    }

    // Divide the rows among the threads
    std::vector<std::thread> threads;
    int rows_per_thread = row / num_threads;
    int extra_rows = row % num_threads;

    int start_row = 0;
    for (unsigned int i = 0; i < num_threads; ++i) {
        int end_row = start_row + rows_per_thread;
        if (i < extra_rows) {
            end_row++;
        }

        threads.emplace_back(matrix_mult_range, std::ref(*this), std::ref(other), std::ref(result), col, other.row, other.col, start_row, end_row);
        start_row = end_row;
    }

    // Join the threads
    for (auto& thread : threads) {
        thread.join();
    }

    return result;
}

/**
 * @brief Multiplies two matrices.
 * @details This is a convenience function that calls the member mult function on matrix `a`.
 * @param a The first matrix (left-hand side).
 * @param b The second matrix (right-hand side).
 * @return A new matrix containing the result of the multiplication.
 */
mat mat::mult(const mat& a, const mat& b) {
    return a.mult(b);
}
