
// determinant
#include "include/mat.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

/**
 * @brief Calculates the determinant of a 2x2 matrix: 
 *      |00 01|
 *      |10 11|
 * @return The determinant of the 2x2 matrix
 */
double mat::det2() {
    // Determinant formula: ad - bc
    return (this->a[0][0] * this->a[1][1] - this->a[0][1] * this->a[1][0]);
}

/**
 * @brief Calculates the determinant of a 3x3 matrix using the Sarrus rule: 
 *      |00 01 02|
 *      |10 11 12|
 *      |20 21 22|
 * @return The determinant of the 3x3 matrix
 */
double mat::det3() {
    // Sarrus rule for 3x3 matrix
    return this->a[0][0]*(this->a[1][1] * this->a[2][2] - this->a[1][2] * this->a[2][1])
          -this->a[0][1]*(this->a[0][0] * this->a[1][2] - this->a[0][2] * this->a[1][0])
          +this->a[0][2]*(this->a[0][0] * this->a[1][1] - this->a[0][1] * this->a[1][0]);
}

/**
 * @brief Calculates the determinant of a 3x3 matrix using the Sarrus rule: 
 *      |00 01 02 03|
 *      |10 11 12 13|
 *      |20 21 22 23|
 *      |30 31 32 33|
 * @return The determinant of the 3x3 matrix
 */
double mat::det4() {
    // Sarrus rule for 3x3 matrix
    return (this->a[0][0] * this->a[1][1] - this->a[0][1] * this->a[1][0]) * (this->a[2][2] * this->a[3][3] - this->a[2][3] * this->a[3][2])
          -(this->a[0][0] * this->a[1][2] - this->a[0][2] * this->a[1][0]) * (this->a[2][1] * this->a[3][3] - this->a[2][3] * this->a[3][1])
          +(this->a[0][0] * this->a[1][3] - this->a[0][3] * this->a[1][0]) * (this->a[2][1] * this->a[3][2] - this->a[2][2] * this->a[3][1])
          -(this->a[0][1] * this->a[1][2] - this->a[0][2] * this->a[1][1]) * (this->a[2][0] * this->a[3][3] - this->a[2][3] * this->a[3][0])
          +(this->a[0][1] * this->a[1][3] - this->a[0][3] * this->a[1][1]) * (this->a[2][0] * this->a[3][2] - this->a[2][2] * this->a[3][0])
          -(this->a[0][2] * this->a[1][3] - this->a[0][3] * this->a[1][2]) * (this->a[2][0] * this->a[3][1] - this->a[2][1] * this->a[3][0]);
}

/**
 * @brief Calculate the determinant of an n x n matrix using row reduction
 *      |00 01 02 03 ---------- 0n|
 *      |10 11 12 13 ---------- 1n|
 *      |20 21 22 23 ---------- 2n|
 *      |30 31 32 33 ---------- 3n|
 *      |.  .  .  .  ---------- ..|
 *      |.  .  .  .  ---------- ..|
 *      |n0 n1 n2 n3 ---------- nn|
 * @return The determinant of the matrix
 * @throws std::runtime_error if the matrix is not square
 */
double mat::detn() {
    if(this->ifsquare() == 1) {
        double det = 1;
        mat b(this->row);
        int n = this->row;
        int row_swap_counter = 0;
        // Initialize b with current matrix values
        b.a = this->a; 
        // Swap rows to avoid zero on the diagonal
        if(b.a[n-1][n-1] == 0) {
            std::swap(b.a[0], b.a[n-1]);
            row_swap_counter++;
        }
        for(int i = 0; i < n-1; i++) {
            if(b.a[i][i] == 0) {
                std::swap(b.a[i], b.a[i+1]);
                row_swap_counter++;
            }
        }
        // Convert matrix to upper triangular form
        int k = 0;
        double f = 0;
        while(k < n-1) {
            for(int i = k; i < n-1; i++) {
                if(b.a[i+1][k] == 0)
                    continue;
                for(int j = i; j < n-1; j++) { 
                    b.a[i+1][j] = b.a[i+1][j] - (b.a[k][j] * b.a[i+1][k] / b.a[k][k]);
                }
            }
            k++;
        }
        // Calculate determinant as product of diagonal elements
        for(int i = 0; i < n; i++) { det *= b.a[i][i]; }
        // Adjust sign based on number of row swaps
        return pow(-1, row_swap_counter) * det;
    }
    else
        throw std::runtime_error("Provide a square matrix!");
}

/**
 * @brief Calculate the determinant of a square matrix
 * @return The determinant of the matrix
 * @throws std::runtime_error if the matrix is not square
 */
double mat::det() {
    if(this->ifsquare()) {
        int n = this->row;
        double det = 0;
        switch(n) {
            case 1:
                det = this->a[0][0];
                break;
            case 2:
                det = this->det2();
                break;
            case 3:
                det = this->det3();
                break;
            case 4:
                det = this->det4();
                break;
            default:
                det = this->detn();
                break;
        }
        return det;
    }
    else
        throw std::runtime_error("Determinant is for square matrix only!");
}
