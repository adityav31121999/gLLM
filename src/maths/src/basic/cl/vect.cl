__kernel void matrixMultiplyKernel(__global const float* A, __global const float* B,
                                   __global float* C, int rowsA, int colsA, int colsB)
{
    // Get the global row and column indices for the output matrix C
    int row = get_global_id(1); // Corresponds to y-dimension in launch
    int col = get_global_id(0); // Corresponds to x-dimension in launch

    // Check if the work-item is within the bounds of the output matrix C
    if (row < rowsA && col < colsB) {
        float sum = 0.0f;
        for (int i = 0; i < colsA; ++i) {
            sum += A[row * colsA + i] * B[i * colsB + col];
        }
        // Store the result in C[row][col]
        C[row * colsB + col] = sum;
    }
}

__kernel void vectorAddKernel(__global const float* A, __global const float* B,
                              __global float* C, int len)
{
    // Get the global index for the 1D vector
    int idx = get_global_id(0); // Corresponds to x-dimension in launch

    // Check if the work-item is within the bounds of the vector length
    if (idx < len) {
        C[idx] = A[idx] + B[idx];
    }
}
