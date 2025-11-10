#include "include/mat.hpp"
#include <thread>
#include <mutex>
#include "include/mat.hpp"
#include <thread>
#include <mutex>
#include <numeric>  // For std::inner_product

// Helper function to compute Hadamard product for a portion of the matrix
void hadamard_range(const mat& a, const mat& b, mat& result, int a_row, int a_col, int b_row, int b_col, int start_row, int end_row) {
    if (a_row != b_row || a_col != b_col) {
        throw std::runtime_error("Matrix dimensions must match for Hadamard product.");
    }

    for (int i = start_row; i < end_row; ++i) {
        for (int j = 0; j < a_col; ++j) {
            result(i, j) = a(i, j) * b(i, j); // Access using operator() is fine
        }
    }
}

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

mat mat::hadamard(const mat& a, const mat& b) {
    return a.hadamard(b);
}

mat mat::hadamard(const std::vector<const mat*>& mats) {
    if (mats.empty()) {
        return mat();
    }

    // Check if all matrices have the same dimensions
    int num_rows = mats[0]->row;
    int num_cols = mats[0]->col;
    for (const mat* m : mats) {
        if (m->row != num_rows || m->col != num_cols) {
            throw std::runtime_error("All matrices must have the same dimensions for chained Hadamard product.");
        }
    }

    mat result = *mats[0]; // Dereference the first matrix
    for (size_t i = 1; i < mats.size(); ++i) {
        result = result.hadamard(*mats[i]); // Dereference subsequent matrices
    }

    return result;
}


/*mat mat::hadamard(const mat &)
{
    return mat();
}

mat mat::hadamard(const mat &, const mat &)
{
    return mat();
}

mat mat::hadamard(const std::vector<mat&>)
{
    return mat();
}*/
