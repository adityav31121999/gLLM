__kernel void kernelTransposeMatrix(
    __global const float *A,
    __global float *y,
    const int M,
    const int N)
{
    // Get global indices for this work-item
    int row = get_global_id(0); // Row index of y
    int col = get_global_id(1); // Column index of y

    // Check if within bounds
    if (row < N && col < M) {
        // y[row][col] = A[col][row]
        y[row * M + col] = A[col * N + row];
    }
}


__kernel void matrix_multiply( // C = A * B, using float4 for A and B
    __global const float *A_f,
    __global const float *B_f,
    __global float *C_f,
    const int M,
    const int N,
    const int K)
{
    // Get global indices for this work-item
    int row = get_global_id(0); // Row index of C
    int col = get_global_id(1); // Column index of C

    // Check if within bounds
    if (row < M && col < N) {
        float sum = 0.0f;
        // Compute dot product for C[row][col]
        // A is M x K, B is K x N
        // C[row][col] = sum(A[row][k] * B[k][col]) for k from 0 to K-1

        // Use float4 for faster access if K is a multiple of 4
        __global const float4* A_f4 = (__global const float4*)A_f;
        __global const float4* B_f4 = (__global const float4*)B_f;

        for (int k_f4 = 0; k_f4 < K / 4; k_f4++) {
            float4 a_vec = A_f4[row * (K / 4) + k_f4];
            float4 b_vec = (float4)(B_f[k_f4 * 4 * N + col], B_f[(k_f4 * 4 + 1) * N + col], B_f[(k_f4 * 4 + 2) * N + col], B_f[(k_f4 * 4 + 3) * N + col]);
            sum += dot(a_vec, b_vec);
        }
        // Handle remaining elements if K is not a multiple of 4
        for (int k = (K / 4) * 4; k < K; k++) {
            sum += A_f[row * K + k] * B_f[k * N + col];
        }
        // Store result in C
        C_f[row * N + col] = sum;
    }
}


__kernel void vector_matrix_multiply( // y = A * x, using float4 for A and x
    __global const float *A_f,
    __global const float *x_f,
    __global float *y_f,
    const int M,
    const int N)
{
    // Get global index for this work-item
    int row = get_global_id(0);

    // Check if within bounds
    if (row < M) {
        float sum = 0.0f;
        // Compute dot product for y[row]
        // A is M x N, x is N x 1
        // y[row] = sum(A[row][col] * x[col]) for col from 0 to N-1

        // Use float4 for faster access if N is a multiple of 4
        __global const float4* A_f4 = (__global const float4*)A_f;
        __global const float4* x_f4 = (__global const float4*)x_f;

        for (int col_f4 = 0; col_f4 < N / 4; col_f4++) {
            sum += dot(A_f4[row * (N / 4) + col_f4], x_f4[col_f4]);
        }
        // Handle remaining elements if N is not a multiple of 4
        for (int col = (N / 4) * 4; col < N; col++) {
            sum += A_f[row * N + col] * x_f[col];
        }
        // Store result in y
        y_f[row] = sum;
    }
}

__kernel void dot_matrix_vector(__global float* vec, __global float* matrix, __global float* vecOutput, int veclength, int matrixRow, int matrixColumn) {
    // Calculate the global work-item ID, corresponding to the row index 'i' of the matrix
    // and the index 'i' of the output vector 'vecOutput'.
    int i = get_global_id(0);

    // Check dimensions consistency (optional, better done on host)
    // if (veclength != matrixColumn) return; // Basic check

    // Check if the work-item ID is within the bounds of the output vector size (matrixRow)
    if (i < matrixRow) {
        float sum = 0.0f;
        // Calculate the dot product of the i-th row of 'matrix' with 'vec'
        for (int j = 0; j < matrixColumn; ++j) { // matrixColumn is N (veclength)
            // Access element matrix[i][j]
            sum += matrix[IDX(i, j, matrixColumn)] * vec[j];
        }
        // Store the result in the i-th element of the output vector
        vecOutput[i] = sum;
    }
}

__kernel void vectorxMatTkernel(
    __global const float *x,      // Input vector of length N
    __global const float *A,      // Input matrix A of size M x N (row-major, non-transposed)
    __global float *y,            // Output vector of length M
    const int M,                  // Number of rows in A (length of y)
    const int N)                  // Number of columns in A (length of x)
{
    const int gid = get_global_id(0);
    const int lid = get_local_id(0);
    const int group_size = get_local_size(0);

    __local float local_sum[256]; // Local memory for reduction, adjust size as needed

    if (gid < M) {
        float sum = 0.0f;
        // Compute dot product of x with the gid-th row of A (equivalent to gid-th column of A^T)
        for (int k = lid; k < N; k += group_size) {
            sum += x[k] * A[gid * N + k];
        }
        local_sum[lid] = sum;

        // Synchronize threads in the workgroup
        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform reduction in local memory
        for (int offset = group_size / 2; offset > 0; offset /= 2) {
            if (lid < offset) {
                local_sum[lid] += local_sum[lid + offset];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // Write result to global memory
        if (lid == 0) {
            y[gid] = local_sum[0];
        }
    }
}

__kernel void matxMatTkernel(
    __global const float *A,      // Input matrix A of size M x K (row-major, non-transposed)
    __global const float *B,      // Input matrix B of size N x K (row-major, non-transposed)
    __global float *C,            // Output matrix C of size M x N (row-major)
    const int M,                  // Number of rows in A
    const int N,                  // Number of rows in B (columns in B^T)
    const int K)                  // Number of columns in A and B
{
    const int i = get_global_id(0); // Row index for C
    const int j = get_global_id(1); // Column index for C
    const int local_i = get_local_id(0);
    const int local_j = get_local_id(1);
    const int local_size_i = get_local_size(0);
    const int local_size_j = get_local_size(1);

    __local float tile_A[16][16]; // Local memory tiles, adjust size as needed
    __local float tile_B[16][16];

    float sum = 0.0f;

    if (i < M && j < N) {
        // Loop over tiles
        for (int t = 0; t < K; t += local_size_i) {
            // Load tile_A: A[i][t + local_j]
            if (t + local_j < K) {
                tile_A[local_i][local_j] = A[i * K + (t + local_j)];
            } else {
                tile_A[local_i][local_j] = 0.0f;
            }

            // Load tile_B: B[j][t + local_i] (for B^T, we need B[j][k])
            if (t + local_i < K) {
                tile_B[local_j][local_i] = B[j * K + (t + local_i)];
            } else {
                tile_B[local_j][local_i] = 0.0f;
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            // Compute partial sum for this tile
            for (int k = 0; k < local_size_i; ++k) {
                sum += tile_A[local_i][k] * tile_B[local_j][k];
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // Write result to C
        C[i * N + j] = sum;
    }
}

__kernel void dot_vector_matrix_vector(__global float* vec1, __global float* matrix,
            __global float* vec2, __global float* d_scalarOutput, int vec1length,
            int matrixRow, int matrixColumn, int vec2length) 
{

    // Assume N = vec1length = matrixRow = matrixColumn = vec2length
    // Host should verify these dimensions before launch.
    int N = vec1length;

    // Allocate shared memory dynamically based on launch configuration
    // Size needs to be at least N, but reduction works best with get_local_size(0) (power of 2)
    __local float sdata[1024]; // Maximum local memory allocation. Host has to ensure N <= 1024.

    // Work-item ID within the workgroup
    int tid = get_local_id(0);

    // --- Step 1: Calculate intermediate value temp[tid] = dot(vec1, matrix[tid]) ---
    //    Then multiply by vec2[tid] and store in shared memory for reduction.
    if (tid < N) {
        float temp_i = 0.0f;
        // Calculate dot(vec1, matrix[tid])
        for (int k = 0; k < N; ++k) {
            // Access matrix[tid][k]
            temp_i += vec1[k] * matrix[IDX(tid, k, N)];
        }
        // Calculate the contribution to the final dot product: temp[tid] * vec2[tid]
        // Store this contribution in shared memory.
        sdata[tid] = temp_i * vec2[tid];
    } 
    else {
        // Work-items outside the range [0, N-1] but within the workgroup need to initialize
        // their shared memory slot if they participate in the reduction.
        if (tid < get_local_size(0)) { // Check if tid is a valid thread index in the workgroup
            sdata[tid] = 0.0f;
        }
    }

    // Synchronize all work-items within the workgroup to ensure all sdata values are written
    barrier(CLK_LOCAL_MEM_FENCE);

    // --- Step 2: Parallel Reduction in Shared Memory ---
    // Reduce the values stored in sdata using get_local_size(0) work-items.
    for (unsigned int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        // Synchronize after each step of the reduction
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // --- Step 3: Write Result ---
    // Work-item 0 writes the final sum (located in sdata[0]) to the output pointer in global memory
    if (tid == 0) {
        *d_scalarOutput = sdata[0];
    }
}
