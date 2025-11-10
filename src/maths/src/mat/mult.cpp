#include "include/mat.hpp"
#include <thread>
#include <mutex>
#include <numeric>

// Helper function to perform matrix multiplication for a portion of the rows
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

mat mat::mult(const mat& a, const mat& b) {
    return a.mult(b);
}

mat mat::mult(const std::vector<const mat*>& mats) {
    if (mats.empty()) {
        return mat();
    }

    // Check if all matrices are compatible for multiplication
    for (size_t i = 0; i < mats.size() - 1; ++i) {
        if (mats[i]->col != mats[i + 1]->row) {
            throw std::runtime_error("Matrix dimensions are incompatible for chained multiplication.");
        }
    }

    // Compute chained matrix multiplication
    mat result = *mats[0];
    for (size_t i = 1; i < mats.size(); ++i) {
        result = result.mult(*mats[i]);
    }

    return result;
}
