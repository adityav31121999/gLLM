
/**
 * @brief OpenCL Kernel: Calculates the determinant of a 2x2 matrix.
 * @param flattenedMatrix Pointer to the 2x2 matrix elements in global GPU memory (row-major).
 * @param d_result Pointer to a float in global GPU memory where the result will be stored.
 * @param n Dimension of the matrix (must be 2).
 */
__kernel void det2(__global float* flattenedMatrix, __global float* d_result, int n) {
    // This kernel is intended to run with a single work-item.
    if (get_global_id(0) == 0) {
        if (n != 2) {
            // Basic error check within kernel
            return;
        }
        // Access elements: | a00 a01 |
        //                  | a10 a11 |
        float a00 = flattenedMatrix[IDX(0, 0, n)];
        float a01 = flattenedMatrix[IDX(0, 1, n)];
        float a10 = flattenedMatrix[IDX(1, 0, n)];
        float a11 = flattenedMatrix[IDX(1, 1, n)];

        // Determinant formula: ad - bc
        *d_result = (a00 * a11) - (a01 * a10);
    }
}

/**
 * @brief OpenCL Kernel: Calculates the determinant of a 3x3 matrix using Sarrus rule/cofactor expansion.
 * @param flattenedMatrix Pointer to the 3x3 matrix elements in global GPU memory (row-major).
 * @param d_result Pointer to a float in global GPU memory where the result will be stored.
 * @param n Dimension of the matrix (must be 3).
 */
__kernel void det3(__global float* flattenedMatrix, __global float* d_result, int n) {
    // This kernel is intended to run with a single work-item.
    if (get_global_id(0) == 0) {
        if (n != 3) {
            return; // Basic error check
        }
        // Access elements: | a00 a01 a02 |
        //                  | a10 a11 a12 |
        //                  | a20 a21 a22 |
        float a00 = flattenedMatrix[IDX(0, 0, n)]; float a01 = flattenedMatrix[IDX(0, 1, n)]; float a02 = flattenedMatrix[IDX(0, 2, n)];
        float a10 = flattenedMatrix[IDX(1, 0, n)]; float a11 = flattenedMatrix[IDX(1, 1, n)]; float a12 = flattenedMatrix[IDX(1, 2, n)];
        float a20 = flattenedMatrix[IDX(2, 0, n)]; float a21 = flattenedMatrix[IDX(2, 1, n)]; float a22 = flattenedMatrix[IDX(2, 2, n)];

        // Standard Sarrus / cofactor expansion for det3:
        float det_correct = a00 * (a11 * a22 - a12 * a21)
                          - a01 * (a10 * a22 - a12 * a20)
                          + a02 * (a10 * a21 - a11 * a20);

        *d_result = det_correct;
    }
}


/**
 * @brief OpenCL Kernel: Calculates the determinant of a 4x4 matrix using Laplace expansion.
 * @param flattenedMatrix Pointer to the 4x4 matrix elements in global GPU memory (row-major).
 * @param d_result Pointer to a float in global GPU memory where the result will be stored.
 * @param n Dimension of the matrix (must be 4).
 */
__kernel void det4(__global float* flattenedMatrix, __global float* d_result, int n) {
     // This kernel is intended to run with a single work-item.
    if (get_global_id(0) == 0) {
        if (n != 4) {
            return; // Basic error check
        }
        // Access elements using macro for readability
        float a00 = flattenedMatrix[IDX(0, 0, n)]; float a01 = flattenedMatrix[IDX(0, 1, n)]; float a02 = flattenedMatrix[IDX(0, 2, n)]; float a03 = flattenedMatrix[IDX(0, 3, n)];
        float a10 = flattenedMatrix[IDX(1, 0, n)]; float a11 = flattenedMatrix[IDX(1, 1, n)]; float a12 = flattenedMatrix[IDX(1, 2, n)]; float a13 = flattenedMatrix[IDX(1, 3, n)];
        float a20 = flattenedMatrix[IDX(2, 0, n)]; float a21 = flattenedMatrix[IDX(2, 1, n)]; float a22 = flattenedMatrix[IDX(2, 2, n)]; float a23 = flattenedMatrix[IDX(2, 3, n)];
        float a30 = flattenedMatrix[IDX(3, 0, n)]; float a31 = flattenedMatrix[IDX(3, 1, n)]; float a32 = flattenedMatrix[IDX(3, 2, n)]; float a33 = flattenedMatrix[IDX(3, 3, n)];

        // Direct translation of the formula (Laplace expansion or similar)
        *d_result = (a00 * a11 - a01 * a10) * (a22 * a33 - a23 * a32)
                  - (a00 * a12 - a02 * a10) * (a21 * a33 - a23 * a31)
                  + (a00 * a13 - a03 * a10) * (a21 * a32 - a22 * a31)
                  - (a01 * a12 - a02 * a11) * (a20 * a33 - a23 * a30)
                  + (a01 * a13 - a03 * a11) * (a20 * a32 - a22 * a30)
                  - (a02 * a13 - a03 * a12) * (a20 * a31 - a21 * a30);
    }
}

/**
 * @brief OpenCL Kernel: Calculate the determinant of an n x n matrix using Gaussian elimination (row reduction).
 *        NOTE: This kernel modifies the input matrix `flattenedMatrix` IN PLACE.
 *        NOTE: This single work-item kernel is unlikely to be faster than CPU for this task.
 * @param flattenedMatrix Pointer to the n x n matrix elements in global GPU memory (row-major). Will be modified.
 * @param d_result Pointer to a float in global GPU memory where the result will be stored.
 * @param n Dimension of the square matrix.
 */
__kernel void detn(__global float* flattenedMatrix, __global float* d_result, int n) {
    // This kernel is intended to run with a single work-item.
    if (get_global_id(0) == 0) {
        if (n <= 0) {
             *d_result = 0.0f; // Or some error indicator
             return;
        }
        if (n == 1) {
            *d_result = flattenedMatrix[0];
            return;
        }

        float det = 1.0f;
        int row_swap_counter = 0;
        float temp_val; // For swapping

        // Convert matrix to upper triangular form (modifies flattenedMatrix)
        for (int k = 0; k < n; k++) { // Iterate through pivot columns/rows
            // Find pivot (largest element in current column k below row k)
            int pivot_row = k;
            // Use fabs for floating point absolute value in OpenCL
            float max_val = fabs(flattenedMatrix[IDX(k, k, n)]);
            for (int i = k + 1; i < n; i++) {
                if (fabs(flattenedMatrix[IDX(i, k, n)]) > max_val) {
                    max_val = fabs(flattenedMatrix[IDX(i, k, n)]);
                    pivot_row = i;
                }
            }

            // Swap rows if necessary to bring pivot to diagonal
            if (pivot_row != k) {
                for (int j = k; j < n; j++) { // Swap elements from column k onwards
                    temp_val = flattenedMatrix[IDX(k, j, n)];
                    flattenedMatrix[IDX(k, j, n)] = flattenedMatrix[IDX(pivot_row, j, n)];
                    flattenedMatrix[IDX(pivot_row, j, n)] = temp_val;
                }
                row_swap_counter++;
            }

            // Check for singularity (zero pivot after swapping)
            // Use a small epsilon for floating-point comparison
            float pivot_val = flattenedMatrix[IDX(k, k, n)];
            if (fabs(pivot_val) < 1e-9f) { // Use fabs and float literal
                *d_result = 0.0f; // Matrix is singular
                return;
            }

            // Eliminate elements below the pivot in column k
            for (int i = k + 1; i < n; i++) {
                float factor = flattenedMatrix[IDX(i, k, n)] / pivot_val;
                // Subtract factor * pivot_row from current row i (from column k onwards)
                flattenedMatrix[IDX(i, k, n)] = 0.0f; // Set element below pivot to zero explicitly
                for (int j = k + 1; j < n; j++) {
                    flattenedMatrix[IDX(i, j, n)] -= factor * flattenedMatrix[IDX(k, j, n)];
                }
            }
        } // End Gaussian elimination loop (k)

        // Calculate determinant as product of diagonal elements
        for (int i = 0; i < n; i++) {
            det *= flattenedMatrix[IDX(i, i, n)];
        }

        // Adjust sign based on number of row swaps
        if (row_swap_counter % 2 != 0) {
            det = -det;
        }
        // Using pow is also valid: det *= pow(-1.0f, (float)row_swap_counter);

        *d_result = det;
    }
}


/**
 * @brief OpenCL Kernel: Calculates the trace of a square matrix (sum of diagonal elements).
 * @param flattenedMatrix Pointer to the n x n matrix elements in global GPU memory (row-major).
 * @param d_result Pointer to a float in global GPU memory where the result will be stored.
 * @param n Dimension of the square matrix.
 */
__kernel void trace(__global float* flattenedMatrix, __global float* d_result, int n) {
    // This kernel is intended to run with a single work-item.
     if (get_global_id(0) == 0) {
        if (n <= 0) {
             *d_result = 0.0f; // Or some error indicator
             return;
        }

        float trace_sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            trace_sum += flattenedMatrix[IDX(i, i, n)]; // Add diagonal element
        }
        *d_result = trace_sum;
    }
}
