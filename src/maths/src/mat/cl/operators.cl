// Helper macro for indexing flattened matrix (assuming row-major)
#define IDX(row, col, dim) ((row) * (dim) + (col))

// Enable extensions for atomics and potentially double precision (which might include float atomics)
// #pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
// #pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable
// #pragma OPENCL EXTENSION cl_khr_fp64 : enable // For double support
// #pragma OPENCL EXTENSION cl_khr_float_atomics : enable // Not supported on target, using manual implementation

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
    } else {
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
