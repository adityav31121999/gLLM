
// ANYONE CAN USE IT
#include <algorithm>
#include <iostream>
#include "include/mat.hpp"

/**
 * @brief Multiplicative Inverse of the matrix by adjoint method
 * @return the inverse of the matrix
 */
mat mat::inverse() {
    if ((this->ifsquare() != 1) || (this->det() == 0)) {
        // if the matrix is square and the determinant is not zero
        // calculate the inverse as the adjoint of the matrix divided
        // by the determinant
        throw std::invalid_argument("Matrix is not invertible");
    }
    return this->adjoint()*(-1.0 / this->det());
}


/**
 * @brief Additive Inverse of a matrix (A + (-A) = 0)
 * @return -1 * the matrix
 */
mat mat::inva() {
    int n = this->row;
    int m = this->col;
    mat a(n, m);
    // loop through all elements of the matrix
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < m; j++) {
            // multiply each element by -1
            a.a[i][j] = this->a[i][j] * -1;
        }
    }
    return a;
}


/**
 * @brief Calculate the inverse of the matrix using Gauss-Jordan Elimination
 * @return The inverse of the matrix
 */
mat mat::gaussjordan() {
    if((this->ifsquare() != 0) || (this->det() == 0)) {
        std::cerr << "Only Square Matrix is Allowed!" << std::endl;
    }    
    mat d(this->row);         // copy of original matrix
    mat b(this->row);         // inverse matrix
    int n = this->row;
    b.imat(n);      // make identity matrix
    d.a = this->a;

    // check for diagonals elements, if zero: swap rows
    for(int i = 0; i < n; i++) {
        if(d.a[i][i] == 0) {
            // swap ith vector with (i-1)th vector
            std::swap(d.a[i], d.a[i+1]);
            std::swap(b.a[i], b.a[i+1]);
        }
    }

    // solve for inverse
    int k = 0;
    // lower triangle formation -> k: 0 to n-2
    while(k < n-1) {
        // k is constant
        // make diagonal element 1 of kth row
        for(int i = 0; i < n; i++) {
            d.a[k][i] /= d.a[k][k];     // original matrix
            b.a[k][i] /= d.a[k][k];     // inverse matrix
        }
        // make elemnts 0, except diagonal of kth column
        for(int i = k; i < n-1; i++) {
            // changes row at every interval
            for(int j = 0; j < n-1; j++) {
                if(d.a[j+1][k] == 0)
                    continue;
                // solves for every column in (i+1)th row
                d.a[i+1][j] -= d.a[k+1][k+1] * d.a[i][j];     // original matrix
                b.a[i+1][j] -= d.a[k+1][k+1] * b.a[i][j];     // inverse matrix
            }
            // d.c[i+1][0] -= d.a[k+1][k+1]*d.c[i][0];         // constant column vector
        }
        k++;
    }
    
    // upper triangle formation -> k: n-1 to 1
    k = n-1;
    while(k > 0) {
        // k is constant
        // make diagonal element 1 of kth row
        for(int i = 0; i < n; i++) {
            d.a[k][i] /= d.a[k][k];     // original matrix
            b.a[k][i] /= d.a[k][k];     // inverse matrix
        }
        // make elemnts 0, except diagonal of kth column
        for(int i = k; i >= 0; i--) {
            // changes row at every interval
            for(int j = n-1; j >= 0; j--) {
                // solves for every column in (i+1)th row
                if(d.a[j-1][k] == 0)
                    continue;
                d.a[i-1][j] -= d.a[k-1][k-1] * d.a[i][j];     // original matrix
                b.a[i-1][j] -= d.a[k-1][k-1] * b.a[i][j];     // inverse matrix
            }
            // d.c[i-1][0] -= d.a[k-1][k-1]*d.c[i][0];           // constant column vector
        }
        k--;
    }
    return b;
}
