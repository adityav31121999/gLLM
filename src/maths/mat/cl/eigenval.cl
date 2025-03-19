
// Eigen Value
#include "include/mat.hpp"
#include <cmath>

/**
 * @brief compute the eigen values and eigen vectors of a matrix
 * @param a matrix input
 * @return pair of eigen values and eigen vectors
 */
std::pair<mat*, mat*> eigen(mat *a) {
    int n = a->getrow();
    mat *e = new mat(n);            // eigen values
    mat *v = new mat(n);            // eigen vectors
    float *b = new float[n];      // temporary vector
    float *c = new float[n];      // temporary vector
    float *z = new float[n];      // temporary vector
    float s, h, f, g;
    int i, its, j, k, l;
    float anorm = 0.0;
    // compute the norm of the matrix
    for (i = 0; i < n; i++) {
        l = 0;
        for (j = 0; j < n; j++) { l += std::abs(a->a[i][j]); }
        if ((anorm == 0.0) || (l > anorm)) { anorm = l; }
    }
    // reduce a to Hessenberg form
    for (l = 0; l < n; l++) {
        l += 1;
        i = l - 1;
        if (i < n) {
            for (k = i; k < n; k++) {
                s = 0.0;
                for (j = i; j < n; j++) { s += a->a[j][l-1] * a->a[j][k]; }
                g = s;
                if (l > 1) {
                    h = 0.0;
                    for (j = i; j < l-1; j++) {
                        s += a->a[j][l-1] * a->a[j][i-1];
                        h += a->a[j][i-1] * a->a[j][i-1];
                    }
                    f = g - h * a->a[i-1][i-1];
                    g = a->a[i-1][l-1] - h * a->a[i-1][i-1];
                }
                if (l > 1) {
                    f = g - h * a->a[i-1][i-1];
                    for (j = i; j < n; j++) { a->a[j][l-1] -= f * a->a[j][i-1]; }
                }
                c[i] = g;
                if (l != n) {
                    if (l > 1) {
                        h = 0.0;
                        for (j = l; j < n; j++) {
                            s += a->a[i-1][j] * a->a[i-1][j];
                            h += a->a[i-1][j] * a->a[i-1][j];
                        }
                        f = g - h * a->a[i-1][i-1];
                        g = a->a[i-1][l-1] - h * a->a[i-1][i-1];
                    }
                    for (j = l; j < n; j++) { a->a[i-1][j] -= f * a->a[i-1][i-1]; }
                }
                if (l < n) { b[l-1] = g; }
                a->a[i-1][i-1] = g;
                g = 0.0;
                s = 0.0;
                if (l < n) {
                    for (j = l; j < n; j++) {
                        if (std::abs(a->a[l-1][j]) + anorm != anorm) { s += std::abs(a->a[l-1][j]); }
                    }
                    if (s != 0.0) {
                        for (j = l; j < n; j++) {
                            a->a[l-1][j] /= s;
                            g += a->a[l-1][j] * a->a[l-1][j];
                        }
                        f = a->a[l-1][l-1];
                        g = std::sqrt(g);
                        if (f < 0.0) { g = -g; }
                        h = f - g;
                        for (j = l; j < n; j++) { a->a[l-1][j] *= h; }
                        z[l-1] = h;
                    }
                }
                if (l > 1) {
                    for (j = 0; j < l-1; j++) {
                        g = 0.0;
                        for (k = 0; k < l; k++) { g += a->a[k][l-1] * a->a[k][j]; }
                        for (k = 0; k < l; k++) { a->a[k][j] -= g * a->a[k][l-1]; }
                    }
                }
                if (l < n) {
                    a->a[l-1][l-1] = f;
                    a->a[l-1][l] = 0.0;
                }
            }
        }
    }
    
    // accumulate the transformations used in the QR algorithm
    for (i = 0; i < n; i++) {
        v->a[i][i] = 1.0;
        for (j = n-1; j > i; j--) {
            f = 0.0;
            for (k = i; k < j+1; k++) { f += a->a[k][j] * v->a[k][i]; }
            f = -f / a->a[j][j+1];
            for (k = i; k < j+1; k++) { v->a[k][i] += f * a->a[k][j]; }
        }
    }
    // QR algorithm
    for (its = 0; its < 50; its++) {
        for (l = 0; l < n; l++) {
            b[l] = a->a[l][l];
            c[l] = b[l];
            if (l < n-1) {
                if (std::abs(c[l+1]) > std::abs(c[l])) { c[l] = c[l+1]; }
            }
        }
        s = 0.0;
        for (l = 0; l < n; l++) { s += std::abs(c[l]); }
        if (s == 0.0) { break; }
        f = s;
        for (l = 0; l < n; l++) {
            c[l] /= f;
            b[l] /= f;
        }
        for (j = 0; j < n; j++) {
            g = 0.0;
            for (k = 0; k < n; k++) { g += a->a[k][j] * c[k]; }
            for (k = 0; k < n; k++) { a->a[k][j] -= g * c[k]; }
        }
        for (i = 0; i < n; i++) { z[i] = b[i]; }
        for (j = 0; j < n; j++) {
            g = 0.0;
            for (k = 0; k < n; k++) { g += a->a[j][k] * c[k]; }
            for (k = 0; k < n; k++) { a->a[j][k] -= g * c[k]; }
        }
        for (i = 0; i < n; i++) { b[i] = z[i]; }
    }
    // form the matrix of eigen vectors
    v = a;
    // return the eigen values and eigen vectors
    return std::make_pair(e, v);
}


/**
 * @brief Compute the eigen vector of a matrix using the power iteration method
 * @details This function takes a matrix as input and returns its eigen vector.
 *  The power iteration method is used to find the eigen vector. The method works
 *  by repeatedly multiplying the matrix with the current eigen vector guess,
 *  and then normalizing the result. The eigen vector is then updated and the
 *  difference between the old and new eigen vector is calculated. The process
 *  is repeated until the difference is smaller than a certain tolerance.
 * @param a matrix input
 * @return eigen vector of matrix
 */
std::vector<float> eigenvec(mat *a) {
    int n = a->getrow();
    std::vector<float> eigenvector(n, 1.0); // Initial guess for eigenvector
    std::vector<float> temp(n, 0.0);
    float tolerance = 1e-10;
    float diff = 1.0;
    // Repeatedly multiply matrix 'a' with the current eigenvector guess
    while (diff > tolerance) {
        // Multiply matrix 'a' with the current eigenvector guess
        for (int i = 0; i < n; ++i) {
            temp[i] = 0.0;
            for (int j = 0; j < n; ++j) { temp[i] += a->a[i][j] * eigenvector[j]; }
        }
        // Normalize the result
        float norm = 0.0;
        for (int i = 0; i < n; ++i) { norm += temp[i] * temp[i]; }
        norm = sqrt(norm);
        // Update the eigenvector and calculate the difference
        diff = 0.0;
        for (int i = 0; i < n; ++i) {
            float new_val = temp[i] / norm;
            diff += fabs(new_val - eigenvector[i]);
            eigenvector[i] = new_val;
        }
    }
    return eigenvector;
}


/**
 * @brief Compute the eigenvalues of a matrix using the QR algorithm
 * This function takes a matrix as input and returns its eigenvalues using
 * the QR algorithm. The algorithm works by repeatedly applying a QR
 * decomposition to the matrix, and then using the resulting R matrix to
 * compute the eigenvalues. The algorithm is stopped after 50 iterations or
 * when the eigenvalues converge to a certain tolerance.
 * @param a matrix input
 * @return eigen values of matrix
 */
std::vector<float> eigenval(mat *a) {
    int n = a->getrow();
    std::vector<float> eigenvalue(n, 0.0);
    mat *b = new mat(n, n);
    int l, k, j, i, its;
    float f, g, s, z;
    b = a;
    // QR algorithm
    for (its = 0; its < 50; its++) {
        for (l = 0; l < n; l++) {
            // compute the eigenvalue of the current row
            g = 0.0;
            for (k = 0; k < n; k++) { g += b->a[l][k] * b->a[l][k]; }
            g = sqrt(g);
            eigenvalue[l] = g;
            // find the maximum eigenvalue
            if (l < n-1) {
                if (std::abs(eigenvalue[l+1]) > std::abs(eigenvalue[l])) { 
                    eigenvalue[l] = eigenvalue[l+1]; 
                }
            }
        }
        s = 0.0;
        for (l = 0; l < n; l++) { s += eigenvalue[l] * eigenvalue[l]; }
        if (s == 0.0) { break; }
        f = sqrt(s);
        for (l = 0; l < n; l++) {
            eigenvalue[l] /= f;
            for (k = 0; k < n; k++) { b->a[l][k] /= f; }
        }
        for (j = 0; j < n; j++) {
            g = 0.0;
            for (k = 0; k < n; k++) { g += b->a[k][j] * eigenvalue[k]; }
            for (k = 0; k < n; k++) { b->a[k][j] -= g * eigenvalue[k]; }
        }
        for (i = 0; i < n; i++) { z = b->a[i][i]; }
        for (j = 0; j < n; j++) {
            g = 0.0;
            for (k = 0; k < n; k++) { g += b->a[j][k] * eigenvalue[k]; }
            for (k = 0; k < n; k++) { b->a[j][k] -= g * eigenvalue[k]; }
        }
        for (i = 0; i < n; i++) { 
            b->a[i][i] = z; 
        }
    }
    return eigenvalue;
}

