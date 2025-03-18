
#include "include/mat.hpp"

/**
 * @brief Householder transformation: This function computes the Householder transformation 
 * of a given vector v. The Householder transformation is a transformation used in linear 
 * algebra to transform a vector into a vector with the same norm, but with all elements
 * below the kth element equal to 0. The Householder vector u is computed as u = v + alpha * e, 
 * where e is the kth unit vector, and alpha is chosen such that the norm of u is equal to the 
 * norm of v.
 * @param v vector to be transformed
 * @param n size of vector
 * @param k index of the vector
 * @return Householder vector u
 */
double *householder(double *v, int n, int k) {
    double *u = new double[n];
    double sigma = 0;
    // Compute the norm of the vector v
    for(int i = k; i < n; i++) {
        sigma += v[i]*v[i];
    }
    // Compute the value of alpha
    double alpha = v[k];
    if(alpha >= 0) { 
        alpha = sqrt(sigma);
    } 
    else { 
        alpha = -sqrt(sigma); 
    }
    // Compute the value of beta
    double beta = alpha*alpha - v[k]*v[k];
    // Compute the Householder vector u
    u[k] = v[k] - alpha;
    // Copy the elements of v into u
    for(int i = k+1; i < n; i++) { 
        u[i] = v[i]; 
    }
    return u;
}


/**
 * @brief Apply Householder transformation: This function takes a Householder 
 * vector and applies the transformation to it. The result is the Householder 
 * vector qk.
 * @param u Householder vector
 * @param n size of vector
 * @param k index of the vector
 * @return Householder vector qk
 */
double *householderTransform(double *u, int n, int k) {
    double *qk = new double[n];
    double *v = new double[n];
    // Compute the result of the Householder transformation
    for(int i = 0; i < n; i++) { v[i] = 0; }
    v[k] = 1;
    double beta = u[k];
    double alpha = 2/beta;
    for(int i = k; i < n; i++) {
        double sum = 0;
        // Compute the sum of the elements of u and v
        for(int j = k; j < n; j++) { sum += u[j]*v[j]; }
        // Compute the result of the Householder transformation
        qk[i] = v[i] - alpha*sum*u[i];
    }
    return qk;
}


/**
 * @brief Singular Value Decomposition of a Matrix
 * This function performs singular value decomposition on a given matrix 'a'.
 * It decomposes the matrix into three matrices: u, s, and v such that a = u * s * v^t.
 * Here, s is a diagonal matrix containing the singular values of 'a'.
 * @tparam t data-type
 * @param a matrix input
 * @return A pair of matrices (u, v) resulting from the SVD
 */
template <typename t> std::pair<mat*, mat*> SVdecom(mat *a) {
    int n = a->row;  // number of rows
    int m = a->col;  // number of columns
    mat *u = new mat(n, n);  // matrix to store left singular vectors
    mat *v = new mat(m, m);  // matrix to store right singular vectors
    mat *s = new mat(n, m);  // matrix to store singular values
    // Initialize u and v as identity matrices
    u.imat(n);
    v.imat(m);
    // Compute a^t * a
    mat *at = a.transpose();
    mat *aTa = new mat(n, n);
    aTa.mult(a, at);  // a * a^t
    mat *AtA = new mat(m, m);
    AtA.mult(at, a);  // a^t * a
    // Compute eigenvalues and eigenvectors of a^t * a
    std::pair<mat*, mat*> eigs = eigen(AtA);
    mat *eigVal = eigs.first;  // eigenvalues (diagonal elements of s)
    mat *eigVec = eigs.second; // eigenvectors
    // Initialize s to the diagonal matrix containing the singular values
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < m; j++) {
            s->a[i][j] = 0;  // zero out non-diagonal elements
        }
        s->a[i][i] = eigVal->a[i][0];  // assign eigenvalue to diagonal
    }
    // Initialize u with eigenvectors of a^t * a
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            u->a[i][j] = eigVec->a[i][j];
        }
    }
    // Initialize v with eigenvectors of a * a^t
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < m; j++) {
            v->a[i][j] = eigVec->a[i][j];
        }
    }
    // Return the pair of matrices (u, v)
    return std::make_pair(u, v);
}

