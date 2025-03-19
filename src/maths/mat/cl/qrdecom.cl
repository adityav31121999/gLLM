
#include "include/mat.hpp"

/**
 * @brief Function to create an orthogonal matrix using the given matrix
 * @param a Input matrix
 * @return Pair of Q and R matrices representing the orthogonal matrix
 */
std::pair<mat*, mat*> makeOrthogonalMatrix(mat *a) {
    int m = a->row;
    int n = a->col;
    mat *q = new mat(m, m);
    mat *r = new mat(m, n);

    // Compute the QR decomposition
    for(int k = 0; k < std::min(m, n); k++) {
        // Compute the kth column of Q
        float *v = new float[m];
        for(int i = 0; i < m; i++) {
            v[i] = a->a[i][k];
        }

        float *u = householder(v, m, k);
        float *qk = householderTransform(u, m, k);

        // Compute the kth row of R
        float *rk = new float[n];
        for(int j = 0; j < n; j++) {
            float sum = 0;
            for(int i = 0; i < m; i++) {
                sum += a->a[i][j] * qk[i];
            }
            rk[j] = sum;
        }

        // Update Q and R
        for(int i = 0; i < m; i++) {
            q->a[i][k] = qk[i];
        }
        for(int j = 0; j < n; j++) {
            r->a[k][j] = rk[j];
        }
    }

    return std::make_pair(q, r);
}


/**
 * @brief QR decomposition of a matrix
 * This function performs a QR decomposition of the input matrix 'a'.
 * It decomposes the matrix into an orthogonal matrix Q and an upper triangular matrix R
 * such that a = Q * R.
 * @param a matrix input
 * @return pair of Q and R matrices
 */
std::pair<mat*, mat*> qrdecom(mat *a) {
    int m = a->getrow();
    int n = a->getcol();
    mat *q = new mat(m, m);
    mat *r = new mat(m, n);
    // Compute the QR decomposition
    for(int k = 0; k < std::min(m, n); k++) {
        // Compute the kth column of Q
        float *v = new float[m];
        for(int i = 0; i < m; i++) { v[i] = a->a[i][k]; }
        float *u = householder(v, m, k);
        float *qk = householderTransform(u, m, k);
        // Compute the kth row of R
        float *rk = new float[n];
        for(int j = 0; j < n; j++) {
            float sum = 0;
            for(int i = 0; i < m; i++) { sum += a->a[i][j] * qk[i]; }
            rk[j] = sum;
        }
        // Update Q and R
        for(int i = 0; i < m; i++) { q->a[i][k] = qk[i]; }
        for(int j = 0; j < n; j++) { r->a[k][j] = rk[j]; }
    }
    return std::make_pair(q, r);
}
