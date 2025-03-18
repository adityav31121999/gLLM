
// subamtrix and row echelon
#include "include/mat.hpp"
#include "mat.hpp"
#include <stdexcept>

/**
 * @brief Extracts a submatrix by removing the specified row and column from the input matrix.
 * @param a The input matrix from which the submatrix is to be extracted.
 * @param i The index of the row to be removed.
 * @param j The index of the column to be removed.
 * @return A new matrix that is the result of removing the specified row and column from the input matrix.
 */
mat submat(mat a, unsigned int i, unsigned int j) {
    if(a.col != a.row) {
        throw std::runtime_error("Matrix Should Be Square.");
        return mat{};
    }
    // Create a new matrix with dimensions one less than the input matrix
    mat c(a.row - 1);
    // Copy the original matrix data
    std::vector<std::vector<double>> b(a.a);
    // Remove the i-th row
    b.erase(b.begin() + i);
    // Remove the j-th column from each remaining row
    for (auto& row : b) {
        row.erase(row.begin() + j);
    }
    // Assign the modified data to the new matrix
    c.a = b;
    // Return the new submatrix
    return c;
}


/**
 * @brief Extracts a submatrix by removing the specified row and column from the input matrix.
 * @details This function takes a matrix as input and removes the specified row and column to form a submatrix.
 * It returns a new matrix with the modified dimensions.
 * @param a The input matrix from which the submatrix is to be extracted.
 * @param i The index of the row to be removed.
 * @param j The index of the column to be removed.
 * @return A new matrix that is the result of removing the specified row and column from the input matrix.
 */
mat submat(std::vector<std::vector<double>> a, unsigned int i, unsigned int j) {
    if(a.size() != a[0].size()) {
        throw std::runtime_error("Matrix Should Be Square.");
        return mat(0);
    }

    // Create a new matrix with dimensions one less than the input matrix
    mat c(a.size() - 1);
    // Copy the original matrix data
    std::vector<std::vector<double>> b(a);
    // Remove the i-th row
    b.erase(b.begin() + i);
    // Remove the j-th column from each remaining row
    for (auto& row : b) {
        row.erase(row.begin() + j);
    }
    // Assign the modified data to the new matrix
    c.a = b;
    // Return the new submatrix
    return c;
}


/**
 * @brief Converts a matrix to its row echelon form using Gaussian elimination.
 * @param a The input matrix as a 2D vector of doubles.
 * @return A matrix in row echelon form.
 */
mat rowechelon(std::vector<std::vector<double>> a) {
    // Initialize matrix b with the same dimensions as the input matrix
    mat b(a.size(), a[0].size());
    
    for(int k = 0; k < b.a.size() - 1; k++) {
        // Check if the pivot element is zero
        if(b.a[k][k] == 0) {
            // Find a row below the current row with a non-zero element in the same column
            int i = k + 1;
            while(i < b.a.size() && b.a[i][k] == 0)
                i++;
            // Swap the current row with the found row
            if(i < b.a.size()) {
                swap(b.a[k], b.a[i]);
            }
        }
        // If pivot element is still zero, skip this column
        if(b.a[k][k] == 0)
            continue;
        
        // Normalize the pivot row
        for(int j = k; j < b.a[0].size(); j++) {
            b.a[k][j] /= b.a[k][k];
        }

        // Eliminate all elements below the pivot in the current column
        for(int i = k + 1; i < b.a.size(); i++) {
            double f = b.a[i][k];
            for(int j = k; j < b.a[0].size(); j++) {
                b.a[i][j] -= b.a[k][j] * f;
            }
        }
    }
    return b;
}


/**
 * @brief Converts a matrix to its row echelon form using Gaussian elimination.
 * @param a The input matrix as a 2D vector of doubles.
 * @return A matrix in row echelon form.
 */
mat rowechelon(mat a) {
    // Initialize matrix b with the same dimensions as the input matrix
    mat b(a.a.size(), a.a[0].size());

    // Iterate through each column of the matrix
    for (int k = 0; k < b.a.size() - 1; k++) {
        // Check if the pivot element is zero
        if (b.a[k][k] == 0) {
            // Find a row below the current row with a non-zero element in the same column
            int i = k + 1;
            while (i < b.a.size() && b.a[i][k] == 0)
                i++;
            // Swap the current row with the found row
            if (i < b.a.size()) {
                swap(b.a[k], b.a[i]);
            }
        }

        // If pivot element is still zero, skip this column
        if (b.a[k][k] == 0)
            continue;

        // Normalize the pivot row
        for(int j = k; j < b.a[0].size(); j++) {
            b.a[k][j] /= b.a[k][k];
        }

        // Eliminate all elements below the pivot in the current column
        for (int i = k + 1; i < b.a.size(); i++) {
            double f = b.a[i][k] / b.a[k][k];
            for (int j = k; j < b.a[0].size(); j++) {
                b.a[i][j] -= f * b.a[k][j];
            }
        }
    }
    return b;
}


/**
 * @brief Resizes the matrix to the specified dimensions. This function takes an 
 * integer number of rows and columns as input and resizes the matrix to the 
 * specified dimensions. It returns a new matrix with the modified dimensions.
 * @param row The number of rows to resize the matrix to.
 * @param col The number of columns to resize the matrix to.
 * @return A new matrix with the modified dimensions.
 */
mat mat::resize(int row, int col) {
    // Create a new matrix with the specified dimensions
    mat a;
    // Resize the matrix data
    a.a.resize(row, std::vector<double>(col, 0.0));
    // Set the number of rows and columns
    a.row = row;
    a.col = col;
    // Return the new matrix
    return a;
}
