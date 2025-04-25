#pragma OPENCL EXTENSION cl_khr_fp32 : enable // For float precision
// #pragma OPENCL EXTENSION cl_khr_fp64 : enable // Uncomment for double precision
#pragma OPENCL EXTENSION cl_khr_int32_base_atomics : enable // For atomic operations if used

// Define MAXFLOAT if not implicitly available
#ifndef MAXFLOAT
#define MAXFLOAT 3.402823466e+38F
#endif

// ========================================================================== //
//                                  Sigmoid                                   //
// ========================================================================== //

// Sigmoid of single value (Inefficient on GPU, consider moving to CPU)
__kernel void clSigmoid(float x, __global float* result) {
    *result = 1.0f / (1.0f + exp(-x));
}

// Sigmoid of vector (Element-wise)
__kernel void clSigmoid1d(__global float* x, __global float* out, int size)
{
    int i = get_global_id(0);
    if (i < size) {
        out[i] = 1.0f / (1.0f + exp(-x[i]));
    }
}

// Sigmoid of 2d vector / matrix (Element-wise)
__kernel void clSigmoid2d(__global float* x, __global float* out, int rows, int cols)
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        out[idx] = 1.0f / (1.0f + exp(-x[idx]));
    }
}

// Sigmoid derivative of single value (Inefficient on GPU, consider moving to CPU)
__kernel void clSigmoidder(float x, __global float* result) {
    float sigmoid_x = 1.0f / (1.0f + exp(-x));
    *result = sigmoid_x * (1.0f - sigmoid_x);
}

// Sigmoid derivative of vector (Element-wise)
__kernel void clSigmoid1dder(__global float* x, __global float* out, int size)
{
    int i = get_global_id(0);
    if (i < size) {
        float sigmoid_x = 1.0f / (1.0f + exp(-x[i]));
        out[i] = sigmoid_x * (1.0f - sigmoid_x);
    }
}

// Sigmoid derivative of 2d vector / matrix (Element-wise)
__kernel void clSigmoid2dder(__global float* x, __global float* out, int rows, int cols)
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        float sigmoid_x = 1.0f / (1.0f + exp(-x[idx]));
        out[idx] = sigmoid_x * (1.0f - sigmoid_x);
    }
}


// ========================================================================== //
//                                 Softmax                                    //
// ========================================================================== //

// Helper for parallel max reduction within a work-group
void parallel_reduce_max(__local float* buffer, size_t local_size) {
    size_t local_id = get_local_id(0);
    for (size_t stride = local_size / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            buffer[local_id] = fmax(buffer[local_id], buffer[local_id + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

// Helper for parallel sum reduction within a work-group
void parallel_reduce_sum(__local float* buffer, size_t local_size) {
    size_t local_id = get_local_id(0);
    for (size_t stride = local_size / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            buffer[local_id] += buffer[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

// Softmax of vector (using parallel reduction within a single work-group)
// NOTE: This kernel is efficient only when launched as a single work-group covering the entire size.
// For very large 'size' exceeding max work-group size or local memory, a multi-pass reduction is needed.
__kernel void clSoftmax1d(__global float* x, __global float* out, float temp, int size)
{
    int global_id = get_global_id(0); // Used for global data access
    int local_id = get_local_id(0);   // Used for local memory access and reduction
    size_t local_size = get_local_size(0); // Work-group size

    // Local memory buffer for reduction (size must be >= local_size)
    __local float local_buffer[256]; // Example size, adjust based on max expected local_size

    // Step 1: Find max value using parallel reduction within the work-group
    float my_val = (global_id < size) ? x[global_id] : -MAXFLOAT; // Use identity for padded elements
    local_buffer[local_id] = my_val;

    barrier(CLK_LOCAL_MEM_FENCE); // Wait for all threads to load their data

    // Perform parallel max reduction
    parallel_reduce_max(local_buffer, local_size);

    // Thread 0 now has the max value for the work-group
    float max_val = local_buffer[0];

    barrier(CLK_LOCAL_MEM_FENCE); // Wait for max_val to be visible to all threads

    // Step 2: Compute exponentials and sum using parallel reduction
    float shifted_exp = (global_id < size) ? exp((x[global_id] - max_val) / temp) : 0.0f; // Use identity for padded elements
    local_buffer[local_id] = shifted_exp;

    barrier(CLK_LOCAL_MEM_FENCE); // Wait for all threads to compute exponentials

    // Perform parallel sum reduction
    parallel_reduce_sum(local_buffer, local_size);

    // Thread 0 now has the sum of exponentials for the work-group
    float sum_val = local_buffer[0];

    barrier(CLK_LOCAL_MEM_FENCE); // Wait for sum_val to be visible

    // Step 3: Normalize elements
    if (global_id < size) {
        if (sum_val > 0.0f) {
            out[global_id] = shifted_exp / sum_val; // Use the computed shifted_exp for THIS thread
        } else if (size > 0) {
             // Handle sum == 0 case. Output uniform distribution? Or 0.0f?
             // Matching previous LOTA pattern:
             out[global_id] = 1.0f / size;
        } else { // size == 0
             out[global_id] = 0.0f;
        }
    }
}

// Softmax of 2d vector / matrix (row-wise, using parallel reduction per row work-group)
// NOTE: Assumes NDRange is launched with global_size={rows, cols_launch_size}, local_size={1, local_size_cols}
// where local_size_cols is the work-group size for column reduction and cols_launch_size is a multiple of it.
__kernel void clSoftmax2d(__global float* x, __global float* out, float temp, int rows, int cols)
{
    int row = get_global_id(0); // Global row index
    int local_col_id = get_local_id(1); // Local column index within the work-group
    size_t local_size_cols = get_local_size(1); // Work-group size along columns

    // Local memory buffer for reduction (size must be >= local_size_cols)
    __local float local_max_buffer[256]; // Example size, adjust based on max expected local_size_cols
    __local float local_sum_buffer[256]; // Example size, adjust based on max expected local_size_cols

    if (row < rows) { // Ensure row index is valid
        // Step 1: Find max value in the current row using parallel reduction
        float my_val = -MAXFLOAT;
        if (local_col_id < cols) { // Check boundary against number of columns
            my_val = x[row * cols + local_col_id];
        }
        local_max_buffer[local_col_id] = my_val;

        barrier(CLK_LOCAL_MEM_FENCE); // Wait for all threads to load their data

        // Perform parallel max reduction using the helpers (which assume get_local_id(0))
        // Need to adapt reduction logic or write a helper for get_local_id(1)
        // Let's inline the reduction steps for clarity for the 2D case using local_col_id
        size_t current_local_size = local_size_cols;
        for (size_t stride = current_local_size / 2; stride > 0; stride /= 2) {
            if (local_col_id < stride) {
                local_max_buffer[local_col_id] = fmax(local_max_buffer[local_col_id], local_max_buffer[local_col_id + stride]);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }


        // Thread 0 in column dimension now has the max value for the row
        float max_val = local_max_buffer[0];

        barrier(CLK_LOCAL_MEM_FENCE); // Wait for max_val to be visible

        // Step 2: Compute exponentials and sum using parallel reduction
        float shifted_exp = 0.0f;
        if (local_col_id < cols) { // Check boundary
            shifted_exp = exp((x[row * cols + local_col_id] - max_val) / temp);
        }
        local_sum_buffer[local_col_id] = shifted_exp;

        barrier(CLK_LOCAL_MEM_FENCE); // Wait for all threads to compute exponentials

        // Perform parallel sum reduction over columns
        current_local_size = local_size_cols;
        for (size_t stride = current_local_size / 2; stride > 0; stride /= 2) {
            if (local_col_id < stride) {
                local_sum_buffer[local_col_id] += local_sum_buffer[local_col_id + stride];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }


        // Thread 0 in column dimension now has the sum of exponentials for the row
        float sum_val = local_sum_buffer[0];

        barrier(CLK_LOCAL_MEM_FENCE); // Wait for sum_val to be visible

        // Step 3: Normalize elements in the row
        int global_col = get_global_id(1); // Global column index
        if (global_col < cols) { // Check global boundary
             int idx = row * cols + global_col;
             if (sum_val > 0.0f) {
                 // Need the shifted_exp value for THIS thread's global_col
                 // Recalculate shifted_exp for the current global_col
                 float current_shifted_exp = exp((x[idx] - max_val) / temp);
                 out[idx] = current_shifted_exp / sum_val;
             } else if (cols > 0) {
                 // Handle sum == 0, uniform distribution per row?
                 out[idx] = 1.0f / cols;
             } else { // cols == 0
                 out[idx] = 0.0f; // Or handle error
             }
        }
    }
}


// Softmax derivative of vector (diagonal Jacobian, s_i*(1-s_i)),
// CALCULATING SOFTMAX INTERNALLY using parallel reduction.
// NOTE: Efficient only when launched as a single work-group covering 'size'.
__kernel void clSoftmax1dder(__global float* x, __global float* out, float temp, int size)
{
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    size_t local_size = get_local_size(0);

    __local float local_buffer[256]; // Max possible local_size

    // Step 1: Find max value (parallel reduction)
    float my_val = (global_id < size) ? x[global_id] : -MAXFLOAT;
    local_buffer[local_id] = my_val;
    barrier(CLK_LOCAL_MEM_FENCE);
    parallel_reduce_max(local_buffer, local_size);
    float max_val = local_buffer[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Step 2: Compute exponentials and sum (parallel reduction)
    float shifted_exp = (global_id < size) ? exp((x[global_id] - max_val) / temp) : 0.0f;
    local_buffer[local_id] = shifted_exp;
    barrier(CLK_LOCAL_MEM_FENCE);
    parallel_reduce_sum(local_buffer, local_size);
    float sum_val = local_buffer[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Step 3: Calculate Softmax for THIS element and then its derivative
    if (global_id < size) {
        float s_i = 0.0f;
        if (sum_val > 0.0f) {
            s_i = shifted_exp / sum_val; // Use the pre-calculated shifted_exp for this thread
        } else if (size > 0) {
            // Handle sum == 0 case. Softmax becomes uniform? Derivative might be 0.
             // Let's follow the original LOTA pattern where derivative is 0 if sum is 0.
             s_i = 0.0f; // Derivative will be s_i * (1-s_i) = 0
        } else { // size == 0
             s_i = 0.0f; // Derivative will be 0
        }

        // Calculate the derivative: s_i * (1 - s_i)
        out[global_id] = s_i * (1.0f - s_i);
    }
}

// Softmax derivative of 2d vector / matrix (diagonal Jacobian, s_ij*(1-s_ij)),
// CALCULATING SOFTMAX INTERNALLY per row using parallel reduction.
// NOTE: Assumes NDRange is launched with global_size={rows, cols_launch_size}, local_size={1, local_size_cols}.
__kernel void clSoftmax2dder(__global float* x, __global float* out, float temp, int rows, int cols)
{
    int row = get_global_id(0); // Global row index
    int local_col_id = get_local_id(1); // Local column index within the work-group
    size_t local_size_cols = get_local_size(1); // Work-group size along columns

    // Local memory buffer for reduction (size must be >= local_size_cols)
    __local float local_max_buffer[256]; // Example size
    __local float local_sum_buffer[256]; // Example size

    if (row < rows) { // Ensure row index is valid
        // Step 1: Find max value in the current row (parallel reduction)
        float my_val = -MAXFLOAT;
        if (local_col_id < cols) { // Check boundary against number of columns
            my_val = x[row * cols + local_col_id];
        }
        local_max_buffer[local_col_id] = my_val;
        barrier(CLK_LOCAL_MEM_FENCE);
        // Inline reduction for local_col_id
        size_t current_local_size = local_size_cols;
        for (size_t stride = current_local_size / 2; stride > 0; stride /= 2) {
            if (local_col_id < stride) {
                local_max_buffer[local_col_id] = fmax(local_max_buffer[local_col_id], local_max_buffer[local_col_id + stride]);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        float max_val = local_max_buffer[0];
        barrier(CLK_LOCAL_MEM_FENCE);

        // Step 2: Compute exponentials and sum (parallel reduction)
        float shifted_exp_for_sum = 0.0f;
        if (local_col_id < cols) { // Check boundary
            shifted_exp_for_sum = exp((x[row * cols + local_col_id] - max_val) / temp);
        }
        local_sum_buffer[local_col_id] = shifted_exp_for_sum;
        barrier(CLK_LOCAL_MEM_FENCE);
        // Inline reduction for local_col_id
        current_local_size = local_size_cols;
        for (size_t stride = current_local_size / 2; stride > 0; stride /= 2) {
            if (local_col_id < stride) {
                local_sum_buffer[local_col_id] += local_sum_buffer[local_col_id + stride];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        float sum_val = local_sum_buffer[0];
        barrier(CLK_LOCAL_MEM_FENCE);

        // Step 3: Calculate Softmax for THIS element and then its derivative
        int global_col = get_global_id(1); // Global column index
        if (global_col < cols) { // Check global boundary
             int idx = row * cols + global_col;
             float s_ij = 0.0f;

             if (sum_val > 0.0f) {
                 // Recalculate shifted_exp for the current global_col
                 float current_shifted_exp = exp((x[idx] - max_val) / temp);
                 s_ij = current_shifted_exp / sum_val;
             } else if (cols > 0) {
                 // Handle sum == 0 case. Derivative is 0.
                  s_ij = 0.0f; // Derivative will be s_ij * (1-s_ij) = 0
             } else { // cols == 0
                 s_ij = 0.0f; // Derivative will be 0
             }

            // Derivative: s_ij * (1 - s_ij)
            out[idx] = s_ij * (1.0f - s_ij);
        }
    }
}

// Softmax derivative of vector (diagonal Jacobian, s_i*(1-s_i)),
// TAKING SOFTMAX OUTPUT 's' AS INPUT (alternative, potentially more efficient if 's' is already computed)
__kernel void clSoftmaxd1dder_from_s(__global float* s, __global float* out, int size)
{
    int i = get_global_id(0);
    if (i < size) {
        // Derivative: s_i * (1 - s_i)
        float s_i = s[i];
        out[i] = s_i * (1.0f - s_i);
    }
}

// Softmax derivative of 2d vector / matrix (diagonal Jacobian, s_ij*(1-s_ij)),
// TAKING SOFTMAX OUTPUT 's' AS INPUT (alternative)
__kernel void clSoftmaxd2dder_from_s(__global float* s, __global float* out, int rows, int cols)
{
    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        // Derivative: s_ij * (1 - s_ij)
        float s_ij = s[idx];
        out[idx] = s_ij * (1.0f - s_ij);
    }
}


// ========================================================================== //
//                                   ReLU                                     //
// ========================================================================== //

// ReLU of single value (Inefficient on GPU, consider moving to CPU)
__kernel void clReLU(float x, __global float* result) {
    *result = fmax(0.0f, x);
}

// ReLU of vector (Element-wise)
__kernel void clReLU1d(__global float* x, __global float* out, int size)
{
    int i = get_global_id(0);
    if (i < size) {
        out[i] = fmax(0.0f, x[i]);
    }
}

// ReLU of 2d vector/matrix (Element-wise)
__kernel void clReLU2d(__global float* x, __global float* out, int rows, int cols)
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        out[idx] = fmax(0.0f, x[idx]);
    }
}

// ReLU derivative of single value (Inefficient on GPU, consider moving to CPU)
__kernel void clReLUder(float x, __global float* result) {
    *result = (x > 0.0f) ? 1.0f : 0.0f;
}

// ReLU derivative of vector (Element-wise)
__kernel void clReLUder1d(__global float* x, __global float* out, int size)
{
    int i = get_global_id(0);
    if (i < size) {
        out[i] = (x[i] > 0.0f) ? 1.0f : 0.0f;
    }
}

// ReLU derivative of 2d vector/matrix (Element-wise)
__kernel void clReLUder2d(__global float* x, __global float* out, int rows, int cols)
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        out[idx] = (x[idx] > 0.0f) ? 1.0f : 0.0f;
    }
}

// ========================================================================== //
//                                   LOTA                                     //
// ========================================================================== //

// Helper for parallel min reduction within a work-group
void parallel_reduce_min(__local float* buffer, size_t local_size) {
    size_t local_id = get_local_id(0);
    for (size_t stride = local_size / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            buffer[local_id] = fmin(buffer[local_id], buffer[local_id + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

// LOTA of vector (using parallel reduction within a single work-group)
// NOTE: Efficient only when launched as a single work-group covering 'size'.
__kernel void clLOTA1d(__global float* y, __global float* out, int size) {
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    size_t local_size = get_local_size(0);

    __local float local_buffer[256]; // Max possible local_size

    // Step 1: Find min value using parallel reduction
    float my_val = (global_id < size) ? y[global_id] : MAXFLOAT; // Use identity for padded elements
    local_buffer[local_id] = my_val;

    barrier(CLK_LOCAL_MEM_FENCE);
    parallel_reduce_min(local_buffer, local_size); // Use custom min reduction

    float min_val = local_buffer[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Step 2: Compute sum of shifted values using parallel reduction
    float shifted_val = (global_id < size) ? (y[global_id] - min_val) : 0.0f; // Use identity for padded elements
    local_buffer[local_id] = shifted_val;

    barrier(CLK_LOCAL_MEM_FENCE);
    parallel_reduce_sum(local_buffer, local_size);

    float sum_val = local_buffer[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Step 3: Apply LOTA formula
    if (global_id < size) {
        if (sum_val > 0.0f) {
            out[global_id] = shifted_val / sum_val; // Use the computed shifted_val
        }
        else {
            out[global_id] = 1.0f / size;
        }
    }
}

// LOTA of 2d vector / matrix (using parallel reduction within a single work-group covering rows*cols)
// NOTE: Efficient only when launched as a single work-group covering rows*cols.
__kernel void clLOTA2d(__global float* y, __global float* out, int rows, int cols) {
    int global_id = get_global_id(0); // Flat index over rows*cols
    int local_id = get_local_id(0);
    size_t local_size = get_local_size(0);
    int size = rows * cols;

    __local float local_buffer[256]; // Max possible local_size

    // Step 1: Find min value using parallel reduction over the entire flattened matrix
    float my_val = (global_id < size) ? y[global_id] : MAXFLOAT;
    local_buffer[local_id] = my_val;

    barrier(CLK_LOCAL_MEM_FENCE);
    parallel_reduce_min(local_buffer, local_size);
    float min_val = local_buffer[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Step 2: Compute sum of shifted values using parallel reduction
    float shifted_val = (global_id < size) ? (y[global_id] - min_val) : 0.0f;
    local_buffer[local_id] = shifted_val;

    barrier(CLK_LOCAL_MEM_FENCE);
    parallel_reduce_sum(local_buffer, local_size);
    float sum_val = local_buffer[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Step 3: Apply LOTA formula
    if (global_id < size) {
        if (sum_val > 0.0f) {
            out[global_id] = shifted_val / sum_val;
        }
        else {
            out[global_id] = 1.0f / size;
        }
    }
}


// LOTA of 2d vector / matrix with masking (using parallel reduction over masked elements in a single work-group)
// NOTE: Efficient only when launched as a single work-group covering rows*cols.
__kernel void clLOTA2dmasking(__global float* y, __global float* out, int rows, int cols, int attentionType) {
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    size_t local_size = get_local_size(0);
    int size = rows * cols;

    __local float local_buffer[256]; // Max possible local_size

    // Step 1: Find min value over masked elements using parallel reduction
    float my_val = MAXFLOAT; // Identity for min
    if (global_id < size) {
        int row = global_id / cols;
        int col = global_id % cols;
        if (attentionType == 0 || col < row) { // Mask condition
             my_val = y[global_id];
        }
    }
    local_buffer[local_id] = my_val;

    barrier(CLK_LOCAL_MEM_FENCE);
    parallel_reduce_min(local_buffer, local_size); // Use custom min reduction
    float min_val = local_buffer[0];

    // Handle case where no valid entries were found (min_val is still MAXFLOAT)
    if (min_val == MAXFLOAT) {
         min_val = 0.0f; // Match original CUDA logic
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Step 2: Compute sum of shifted masked values using parallel reduction
    float shifted_val = 0.0f; // Identity for sum
     if (global_id < size) {
        int row = global_id / cols;
        int col = global_id % cols;
        if (attentionType == 0 || col < row) { // Mask condition
            shifted_val = y[global_id] - min_val;
        }
    }
    local_buffer[local_id] = shifted_val;

    barrier(CLK_LOCAL_MEM_FENCE);
    parallel_reduce_sum(local_buffer, local_size);
    float sum_val = local_buffer[0];
    barrier(CLK_LOCAL_MEM_FENCE);


    // Step 3: Apply LOTA formula to masked elements, set others to 0
    if (global_id < size) {
        int row = global_id / cols;
        int col = global_id % cols;

        if (attentionType == 0 || col < row) { // Mask condition
             out[global_id] = (sum_val > 0.0f) ? (y[global_id] - min_val) / sum_val : 0.0f; // Original used 0.0f
        }
        else {
            out[global_id] = 0.0f; // Elements not included in calculation are set to 0.0f
        }
    }
}

/**
 * @brief OpenCL kernel function for LOTA derivative (using size, parallel reduction)
 * NOTE: Efficient only when launched as a single work-group covering 'size'.
 * @param[in] y input array in global memory
 * @param[out] out output array in global memory
 * @param[in] size size of array
 */
__kernel void clLOTA1dder(__global float* y, __global float* out, int size) {
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    size_t local_size = get_local_size(0);

    __local float local_buffer[256]; // Max possible local_size

    // Step 1: Find min value using parallel reduction
    float my_val = (global_id < size) ? y[global_id] : MAXFLOAT;
    local_buffer[local_id] = my_val;
    barrier(CLK_LOCAL_MEM_FENCE);
    parallel_reduce_min(local_buffer, local_size);
    float min_val = local_buffer[0];
    barrier(CLK_LOCAL_MEM_FENCE);


    // Step 2: Compute sum of shifted values using parallel reduction
    float shifted_val_for_sum = (global_id < size) ? (y[global_id] - min_val) : 0.0f;
    local_buffer[local_id] = shifted_val_for_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    parallel_reduce_sum(local_buffer, local_size);
    float sum_val = local_buffer[0];
    barrier(CLK_LOCAL_MEM_FENCE);


    // Step 3: Apply LOTA derivative formula: (sum - y_i + min) / (sum * sum)
    if (global_id < size) {
        if (sum_val > 0.0f) {
            out[global_id] = (sum_val - y[global_id] + min_val) / (sum_val * sum_val);
        }
        else {
            out[global_id] = 0.0f;
        }
    }
}

/**
 * @brief OpenCL kernel function for LOTA derivative (using rows/cols, parallel reduction)
 * NOTE: Efficient only when launched as a single work-group covering rows*cols.
 * @param[in] y input array in global memory
 * @param[out] out output array in global memory
 * @param[in] rows number of rows
 * @param[in] cols number of cols
 */
__kernel void clLOTA2dder(__global float* y, __global float* out, int rows, int cols) {
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    size_t local_size = get_local_size(0);
    int size = rows * cols;

    __local float local_buffer[256]; // Max possible local_size

    // Step 1: Find min value using parallel reduction
    float my_val = (global_id < size) ? y[global_id] : MAXFLOAT;
    local_buffer[local_id] = my_val;
    barrier(CLK_LOCAL_MEM_FENCE);
    parallel_reduce_min(local_buffer, local_size);
    float min_val = local_buffer[0];
    barrier(CLK_LOCAL_MEM_FENCE);


    // Step 2: Compute sum of shifted values using parallel reduction
    float shifted_val_for_sum = (global_id < size) ? (y[global_id] - min_val) : 0.0f;
    local_buffer[local_id] = shifted_val_for_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    parallel_reduce_sum(local_buffer, local_size);
    float sum_val = local_buffer[0];
    barrier(CLK_LOCAL_MEM_FENCE);


    // Step 3: Apply LOTA derivative formula: (sum - y_i + min) / (sum * sum)
    if (global_id < size) {
        if (sum_val > 0.0f) {
             out[global_id] = (sum_val - y[global_id] + min_val) / (sum_val * sum_val);
        }
        else {
            out[global_id] = 0.0f;
        }
    }
}

/**
 * @brief OpenCL kernel function for LOTA derivative (using rows/cols and attentionType, parallel reduction)
 * NOTE: Efficient only when launched as a single work-group covering rows*cols.
 * @param[in] y input array in global memory
 * @param[out] out output array in global memory
 * @param[in] rows number of rows
 * @param[in] cols number of cols
 * @param[in] attentionType flag for attention type (0 for false, non-zero for true)
 */
__kernel void clLOTA2ddermasking(__global float* y, __global float* out, int rows, int cols, int attentionType) {
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    size_t local_size = get_local_size(0);
    int size = rows * cols;

    __local float local_buffer[256]; // Max possible local_size

    // Step 1: Find min value over masked elements using parallel reduction
    float my_val = MAXFLOAT; // Identity for min
    if (global_id < size) {
        int row = global_id / cols;
        int col = global_id % cols;
        if (attentionType == 0 || col < row) { // Mask condition
             my_val = y[global_id];
        }
    }
    local_buffer[local_id] = my_val;
    barrier(CLK_LOCAL_MEM_FENCE);
    parallel_reduce_min(local_buffer, local_size);
    float min_val = local_buffer[0];

     // Handle case where no valid entries were found (min_val is still MAXFLOAT)
    if (min_val == MAXFLOAT) {
         min_val = 0.0f; // Match original CUDA logic
    }

    barrier(CLK_LOCAL_MEM_FENCE);


    // Step 2: Compute sum of shifted masked values using parallel reduction
    float shifted_val_for_sum = 0.0f; // Identity for sum
     if (global_id < size) {
        int row = global_id / cols;
        int col = global_id % cols;
        if (attentionType == 0 || col < row) { // Mask condition
            shifted_val_for_sum = y[global_id] - min_val;
        }
    }
    local_buffer[local_id] = shifted_val_for_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    parallel_reduce_sum(local_buffer, local_size);
    float sum_val = local_buffer[0];
    barrier(CLK_LOCAL_MEM_FENCE);


    // Step 3: Apply LOTA derivative formula to masked elements, set others to 0
    if (global_id < size) {
        int row = global_id / cols;
        int col = global_id % cols;

        if (attentionType == 0 || col < row) { // Mask condition
             out[global_id] = (sum_val > 0.0f) ? (sum_val - y[global_id] + min_val) / (sum_val * sum_val) : 0.0f; // Original used 0.0f
        }
        else {
            out[global_id] = 0.0f; // Elements not included in calculation are set to 0.0f
        }
    }
}
