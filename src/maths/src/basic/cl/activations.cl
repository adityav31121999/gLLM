
// Helper macro for indexing flattened matrix (assuming row-major)
#define IDX(row, col, dim) ((row) * (dim) + (col))

// Enable extensions for atomics and potentially double precision (which might include float atomics)
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable // For double support, might help with float atomics on some platforms
// #pragma OPENCL EXTENSION cl_khr_float_atomics : enable // Ignored by target platform, implementing manually

__kernel void clSigmoid(float x, __global float* result) {
    *result = 1.0f / (1.0f + exp(-x));
}

__kernel void clSigmoid1d(__global float* x, __global float* out, int size)
{
    int i = get_global_id(0);
    if (i < size) {
        out[i] = 1.0f / (1.0f + exp(-x[i]));
    }
}

__kernel void clSigmoid2d(__global float* x, __global float* out, int rows, int cols)
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        out[idx] = 1.0f / (1.0f + exp(-x[idx]));
    }
}

__kernel void clSigmoidder(float x, __global float* result) {
    float sigmoid_x = 1.0f / (1.0f + exp(-x));
    *result = sigmoid_x * (1.0f - sigmoid_x);
}

__kernel void clSigmoid1dder(__global float* x, __global float* out, int size)
{
    int i = get_global_id(0);
    if (i < size) {
        float sigmoid_x = 1.0f / (1.0f + exp(-x[i]));
        out[i] = sigmoid_x * (1.0f - sigmoid_x);
    }
}

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

void parallel_reduce_max(__local float* buffer, uint local_size) {
    uint local_id = get_local_id(0);
    for (uint stride = local_size / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            buffer[local_id] = fmax(buffer[local_id], buffer[local_id + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

void parallel_reduce_sum(__local float* buffer, uint local_size) {
    uint local_id = get_local_id(0);
    for (uint stride = local_size / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            buffer[local_id] += buffer[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel void clSoftmax1d(__global float* x, __global float* out, float temp, int size)
{
    int global_id = get_global_id(0); // Used for global data access
    int local_id = get_local_id(0);   // Used for local memory access and reduction
    uint local_size = get_local_size(0); // Work-group size

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

__kernel void clSoftmax2d(__global float* x, __global float* out, float temp, int rows, int cols)
{
    int row = get_global_id(0); // Global row index
    int local_col_id = get_local_id(1); // Local column index within the work-group
    uint local_size_cols = get_local_size(1); // Work-group size along columns

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
        uint current_local_size = local_size_cols;
        for (uint stride = current_local_size / 2; stride > 0; stride /= 2) {
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
        for (uint stride = current_local_size / 2; stride > 0; stride /= 2) {
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

__kernel void clSoftmax1dder(__global float* x, __global float* out, float temp, int size)
{
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    uint local_size = get_local_size(0);

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

__kernel void clSoftmax2dder(__global float* x, __global float* out, float temp, int rows, int cols)
{
    int row = get_global_id(0); // Global row index
    int local_col_id = get_local_id(1); // Local column index within the work-group
    uint local_size_cols = get_local_size(1); // Work-group size along columns

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
        uint current_local_size = local_size_cols;
        for (uint stride = current_local_size / 2; stride > 0; stride /= 2) {
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
        for (uint stride = current_local_size / 2; stride > 0; stride /= 2) {
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

__kernel void clSoftmaxd1dder_from_s(__global float* s, __global float* out, int size)
{
    int i = get_global_id(0);
    if (i < size) {
        // Derivative: s_i * (1 - s_i)
        float s_i = s[i];
        out[i] = s_i * (1.0f - s_i);
    }
}

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

__kernel void clReLU(float x, __global float* result) {
    *result = fmax(0.0f, x);
}

__kernel void clReLU1d(__global float* x, __global float* out, int size)
{
    int i = get_global_id(0);
    if (i < size) {
        out[i] = fmax(0.0f, x[i]);
    }
}

__kernel void clReLU2d(__global float* x, __global float* out, int rows, int cols)
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        out[idx] = fmax(0.0f, x[idx]);
    }
}

__kernel void clReLUder(float x, __global float* result) {
    *result = (x > 0.0f) ? 1.0f : 0.0f;
}

__kernel void clReLUder1d(__global float* x, __global float* out, int size)
{
    int i = get_global_id(0);
    if (i < size) {
        out[i] = (x[i] > 0.0f) ? 1.0f : 0.0f;
    }
}

__kernel void clReLUder2d(__global float* x, __global float* out, int rows, int cols)
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        out[idx] = (x[idx] > 0.0f) ? 1.0f : 0.0f;
    }
}

void parallel_reduce_min(__local float* buffer, uint local_size) {
    uint local_id = get_local_id(0);
    for (uint stride = local_size / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            buffer[local_id] = fmin(buffer[local_id], buffer[local_id + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel void clLOTA1d(__global float* y, __global float* out, int size) {
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    uint local_size = get_local_size(0);

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

__kernel void clLOTA2d(__global float* y, __global float* out, int rows, int cols) {
    int global_id = get_global_id(0); // Flat index over rows*cols
    int local_id = get_local_id(0);
    uint local_size = get_local_size(0);
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

__kernel void clLOTA2dmasking(__global float* y, __global float* out, int rows, int cols, int limit_dim, int isSelfAttention) {
    // isSelfAttention: 1 if self (lower triangle incl. diagonal), 0 if cross (square up to limit_dim)
    // This kernel performs parallel reduction over the masked elements.
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    uint local_size = get_local_size(0);
    int total_elements_in_buffer = rows * cols;

    // Ensure local_buffer is large enough for the work-group size.
    // Host must ensure local_size <= 256 (or whatever this is set to).
    __local float local_buffer[256];

    // Determine if the current element is active for calculations based on the mask
    bool is_active_element = false;
    float current_y_value = 0.0f; // Value of y[global_id] if active

    if (global_id < total_elements_in_buffer) {
        int r = global_id / cols;
        int c = global_id % cols;
        if (r < limit_dim && c < limit_dim) { // Within the t x t subgrid
            if (isSelfAttention == 1) { // Self-attention: lower triangle including diagonal
                if (c <= r) is_active_element = true;
            } else { // Cross-attention (isSelfAttention == 0): square
                is_active_element = true;
            }
        }
        if (is_active_element) {
            current_y_value = y[global_id];
        }
    }

    // --- Step 1: Find min_val over the masked (active) region ---
    // Each work-item contributes its y_value if active, or MAXFLOAT otherwise.
    local_buffer[local_id] = is_active_element ? current_y_value : MAXFLOAT;
    barrier(CLK_LOCAL_MEM_FENCE);
    parallel_reduce_min(local_buffer, local_size); // Reduce to find the minimum among active elements
    float min_val_masked = local_buffer[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // If min_val_masked is still MAXFLOAT, it means no active elements were found in this work-group's processing range.
    // A true global min would require inter-work-group communication or a final reduction step on the host / separate kernel.
    // For now, if it's MAXFLOAT, assume no active elements globally for this reduction pass, set to 0.
    if (min_val_masked == MAXFLOAT) {
        min_val_masked = 0.0f;
    }
    float abs_min_val_masked = fabs(min_val_masked);

    // --- Step 2: Count active elements for uniform distribution if sum is zero ---
    local_buffer[local_id] = is_active_element ? 1.0f : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    parallel_reduce_sum(local_buffer, local_size);
    float active_elements_count = local_buffer[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // --- Step 3: Compute sum of (y_value + abs_min_val_masked) for active elements ---
    // Each work-item contributes (y_value + abs_min_val_masked) if active, or 0.0f otherwise.
    local_buffer[local_id] = is_active_element ? (current_y_value + abs_min_val_masked) : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    parallel_reduce_sum(local_buffer, local_size);
    float sum_shifted_masked = local_buffer[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // --- Step 4: Apply LOTA formula to active elements ---
    if (global_id < total_elements_in_buffer) {
        if (is_active_element) {
            if (sum_shifted_masked > 1e-9f) { // Use a small epsilon for float comparison
                out[global_id] = (current_y_value + abs_min_val_masked) / sum_shifted_masked;
            } else if (active_elements_count > 0) {
                // Sum is zero (or very small), distribute uniformly among active elements
                out[global_id] = 1.0f / active_elements_count;
            } else {
                // No active elements, or sum is zero with no active elements (should not happen if is_active_element is true)
                out[global_id] = 0.0f; // Default to 0
            }
        } else {
            // Element is outside the relevant region, set output to 0
            out[global_id] = 0.0f;
        }
    }
}

__kernel void clLOTA1dder(__global float* y, __global float* out, int size) {
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    uint local_size = get_local_size(0);

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

__kernel void clLOTA2dder(__global float* y, __global float* out, int rows, int cols) {
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    uint local_size = get_local_size(0);
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

__kernel void clLOTA2ddermasking(__global float* y, __global float* out, int rows, int cols, int limit_dim, int isSelfAttention) {
    // isSelfAttention: 1 if self, 0 if cross.
    // Similar parallel reduction structure as clLOTA2dmasking.
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    uint local_size = get_local_size(0);
    int total_elements_in_buffer = rows * cols;

    __local float local_buffer[256];

    // Ensure local_buffer size is sufficient (Host should ensure local_size <= 256)
    // if (local_size > 256) return; // Optional safety check

    // Step 1: Find min value over masked elements using parallel reduction (INLINED)
    float my_val = MAXFLOAT; // Identity for min
    bool is_active_element = false;
    float current_y_value = 0.0f;

    if (global_id < total_elements_in_buffer) {
        int row = global_id / cols;
        int col = global_id % cols;
        if (row < limit_dim && col < limit_dim) {
            if (isSelfAttention == 1) { // Self-attention: lower triangle including diagonal
                if (col <= row) is_active_element = true;
            } else { // Cross-attention: square
                is_active_element = true;
            }
        }
        if (is_active_element) {
            current_y_value = y[global_id];
            my_val = current_y_value;
        }
    }
    if (local_id < local_size) { // Ensure local_id is within bounds for local_buffer
        local_buffer[local_id] = my_val;
    }


    barrier(CLK_LOCAL_MEM_FENCE); // Ensure all values loaded

    // --- Inlined Min Reduction ---
    // Ensure loop bounds respect the actual local_size, not just the buffer size
    for (uint stride = local_size / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            // Bounds check for local_buffer access
            if ((local_id + stride) < local_size) { // Check read index
                 local_buffer[local_id] = fmin(local_buffer[local_id], local_buffer[local_id + stride]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // --- End Inlined Min Reduction ---

    // Only work-item 0 reads the final result, so no bounds check needed here
    float min_val_masked = local_buffer[0];

     // Handle case where no valid entries were found (min_val is still MAXFLOAT)
    if (min_val_masked == MAXFLOAT) {
        min_val_masked = 0.0f;
    }
    float abs_min_val_masked = fabs(min_val_masked);
    barrier(CLK_LOCAL_MEM_FENCE); // Ensure min_val_masked is visible to all

    // Step 2: Compute sum of shifted masked values using parallel reduction (INLINED)
    float shifted_val_for_sum = 0.0f; // Identity for sum
    if (is_active_element) {
        // For derivative, sum is of (y_value + abs_min_val_masked)
        shifted_val_for_sum = current_y_value + abs_min_val_masked;
    }

    if (local_id < local_size) { // Ensure local_id is within bounds for local_buffer
        local_buffer[local_id] = shifted_val_for_sum;
    }

    barrier(CLK_LOCAL_MEM_FENCE); // Ensure all shifted values loaded

    // --- Inlined Sum Reduction ---
    // Ensure loop bounds respect the actual local_size
    for (uint stride = local_size / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            // Bounds check for local_buffer access
            if ((local_id + stride) < local_size) { // Check read index
                local_buffer[local_id] += local_buffer[local_id + stride];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // --- End Inlined Sum Reduction ---

    // Only work-item 0 reads the final result
    float sum_shifted_masked = local_buffer[0];
    barrier(CLK_LOCAL_MEM_FENCE); // Ensure sum_shifted_masked is visible to all

    // Step 3: Apply LOTA derivative formula to masked elements, set others to 0
    if (global_id < total_elements_in_buffer) {
        if (is_active_element) {
             // Apply derivative formula: (sum - y_i + min) / (sum * sum)
             // Use small epsilon for float comparison robustness
            float transformed_yi = current_y_value + abs_min_val_masked;
            out[global_id] = (sum_shifted_masked > 1e-9f) ? (sum_shifted_masked - transformed_yi) / (sum_shifted_masked * sum_shifted_masked) : 0.0f;
        } else {
            // Elements not included in calculation are set to 0.0f
            out[global_id] = 0.0f;
        }
    }
}
