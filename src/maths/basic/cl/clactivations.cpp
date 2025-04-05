
#include "include/basic.hpp"
#include <CL/cl.hpp>

//----------------SIGMOID----------------//

/**
 * @brief OpenCL function for sigmoid
 * @param[in] x input
 * @param[out] result sigmoid(x)
 */
const char* sigmoidKernelSource = R"(
    __kernel void clSigmoid(float x, __global float* result) {
        *result = 1.0f / (1.0f + exp(-x));
    }
)";

const char* sigmoidderKernelSource = R"(
    __kernel void clSigmoidder(float x, __global float* result) {
        float sigmoid_x = 1.0f / (1.0f + exp(-x));
        *result = sigmoid_x * (1.0f - sigmoid_x);
    }
)";

/**
 * @brief Sigmoid activation function for 2D array.
 * @param[in] x input array
 * @param[out] out output array
 * @param[in] rows number of rows
 * @param[in] cols number of columns
 */
const char* sigmoid2DKernelSource = R"(
    __kernel void clSigmoid(__global float* x, __global float* out, int rows, int cols) {
        int row = get_global_id(0);
        int col = get_global_id(1);
        if (row < rows && col < cols) {
            out[row * cols + col] = 1.0f / (1.0f + exp(-x[row * cols + col]));
        }
    }
)";

/**
 * @brief Sigmoid derivative for 1D array.
 * @param[in] x input array
 * @param[out] out output array
 * @param[in] size size of the array
 */
const char* sigmoid1DderKernelSource = R"(
    __kernel void clSigmoidder(__global float* x, __global float* out, int size) {
        int i = get_global_id(0);
        if (i < size) {
            float sigmoid_x = 1.0f / (1.0f + exp(-x[i]));
            out[i] = sigmoid_x * (1.0f - sigmoid_x);
        }
    }
)";

/**
 * @brief Sigmoid derivative for 2D array.
 * @param[in] x input array
 * @param[out] out output array
 * @param[in] rows number of rows
 * @param[in] cols number of columns
 */
const char* sigmoidDer2DKernelSource = R"(
    __kernel void clSigmoidder(__global float* x, __global float* out, int rows, int cols) {
        int row = get_global_id(0);
        int col = get_global_id(1);
        if (row < rows && col < cols) {
            float sigmoid_x = 1.0f / (1.0f + exp(-x[row * cols + col]));
            out[row * cols + col] = sigmoid_x * (1.0f - sigmoid_x);
        }
    }
)";

//----------------SOFTMAX----------------//

/**
 * @brief Softmax activation function for 1D array.
 * @param[in] x input array
 * @param[out] out output array
 * @param[in] temp temperature parameter
 * @param[in] size size of the array
 */
const char* softmax1DKernelSource = R"(
    __kernel void clSoftmax(__global float* x, __global float* out, float temp, int size) {
        int i = get_global_id(0);
        
        // First pass: compute exponentials and store them in output
        if (i < size) {
            out[i] = exp(x[i] / temp);
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
        
        // Second pass: compute sum (only first thread)
        float sum = 0.0f;
        if (i == 0) {
            for (int j = 0; j < size; j++) {
                sum += out[j];
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
        
        // Third pass: normalize by sum
        if (i < size) {
            if (sum > 0.0f) { // Avoid division by zero
                out[i] = out[i] / sum;
            } 
            else {
                out[i] = 1.0f / size; // Equal distribution if all values are the same
            }
        }
    }
)";

/**
 * @brief Softmax activation function for 2D array.
 * @param[in] x input array
 * @param[out] out output array
 * @param[in] temp temperature parameter
 * @param[in] rows number of rows
 * @param[in] cols number of columns
 */
const char* softmax2DKernelSource = R"(
    __kernel void clSoftmax(__global float* x, __global float* out, float temp, int rows, int cols) {
        int row = get_global_id(0);
        int col = get_global_id(1);
        if (row < rows && col < cols) {
            float sum = 0.0f;
            for (int j = 0; j < cols; j++) {
                sum += exp(x[row * cols + j] / temp);
            }
            if (sum > 0.0f) { // Avoid division by zero
                out[row * cols + col] = exp(x[row * cols + col] / temp) / sum;
            } 
            else {
                out[row * cols + col] = 1.0f / cols; // Or handle as error/default
            }
        }
    }
)";


/**
 * @brief Softmax derivative for 1D array.
 * @param[in] x input array (original input values, not softmax output)
 * @param[out] out output derivative array (Jacobian diagonal)
 * @param[in] temp temperature parameter
 * @param[in] size size of the array
 */
const char* softmaxDer1DKernelSource = R"(
    __kernel void clSoftmaxder(__global float* x, __global float* out, float temp, int size) {
        int i = get_global_id(0);
        if (i < size) {
            // First compute softmax values
            __local float softmax_vals[256]; // Assuming max size is 256, adjust as needed
            
            // Compute exp(x/temp) for all elements
            softmax_vals[i] = exp(x[i] / temp);
            barrier(CLK_LOCAL_MEM_FENCE);
            
            // Compute sum of exponentials
            float sum = 0.0f;
            for (int j = 0; j < size; j++) {
                sum += softmax_vals[j];
            }
            
            // Normalize to get softmax values
            for (int j = 0; j < size; j++) {
                softmax_vals[j] /= sum;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            // Calculate the derivative
            float result = softmax_vals[i] * (1.0f - softmax_vals[i]);
            
            // Subtract the softmax of each other element
            for (int j = 0; j < size; j++) {
                if (i != j) {
                    result -= softmax_vals[j];
                }
            }
            out[i] = result;
        }
    }
)";

/**
 * @brief Softmax derivative for 2D array.
 * @param[in] x input array (should be the softmax output)
 * @param[out] out output derivative array (Jacobian diagonal for each row)
 * @param[in] temp temperature parameter
 * @param[in] rows number of rows
 * @param[in] cols number of columns
 */
const char* softmaxDer2DKernelSource = R"(
    __kernel void clSoftmaxder(__global float* x, __global float* out, float temp, int rows, int cols) {
        int row = get_global_id(0);
        int col = get_global_id(1);
        if (row < rows && col < cols) {
            // Calculate the derivative of softmax(x, temp) w.r.t x
            // Derivative for softmax_ij w.r.t input_ij is softmax_ij * (1 - softmax_ij)
            // This calculates the diagonal of the Jacobian matrix for each row.
            float softmax_vals[cols];
            for (int i = 0; i < cols; i++) {
                softmax_vals[i] = exp(x[row * cols + i] / temp);
            }
            float sum = 0.0f;
            for (int i = 0; i < cols; i++) {
                sum += softmax_vals[i];
            }
            for (int i = 0; i < cols; i++) {
                softmax_vals[i] /= sum;
            }
            float softmax_ij = softmax_vals[col];
            out[row * cols + col] = softmax_ij * (1.0f - softmax_ij);
        }
    }
)";

//----------------ReLU----------------//

/**
 * @brief ReLU activation function for single element.
 * @param[in] x input
 * @param[out] result output
 */
const char* reluKernelSource = R"(
    __kernel void clReLU(float x, __global float* result) {
        *result = max(0.0f, x);
    }
)";

/**
 * @brief ReLU activation function for 1D array.
 * @param[in] x input array
 * @param[out] out output array
 * @param[in] size size of the array 
 */
const char* relu1DKernelSource = R"(
    __kernel void clReLU(__global float* x, __global float* out, int size) {
        int i = get_global_id(0);
        if (i < size) {
            out[i] = max(0.0f, x[i]);
        }
    }
)";

/**
 * @brief ReLU activation function for 2D array.
 * @param[in] x input array
 * @param[out] out output array
 * @param[in] rows number of rows
 * @param[in] cols number of columns
 */
const char* relu2DKernelSource = R"(
    __kernel void clReLU(__global float* x, __global float* out, int rows, int cols) {
        int row = get_global_id(0);
        int col = get_global_id(1);
        if (row < rows && col < cols) {
            out[row * cols + col] = max(0.0f, x[row * cols + col]);
        }
    }
)";


/**
 * @brief ReLU derivative for single element.
 * @param[in] x input
 * @param[out] result output
 */
const char* reluDerKernelSource = R"(
    __kernel void clReLUder(float x, __global float* result) {
        *result = (x > 0.0f) ? 1.0f : 0.0f;
    }
)";

/**
 * @brief ReLU derivative for 1D array.
 * @param[in] x input array
 * @param[out] out output array
 * @param[in] size size of the array 
 */
const char* relu1DderKernelSource = R"(
    __kernel void clReLUder(__global float* x, __global float* out, int size) {
        int i = get_global_id(0);
        if (i < size) {
            out[i] = (x[i] > 0.0f) ? 1.0f : 0.0f;
        }
    }
)";

/**
 * @brief ReLU derivative for 2D array.
 * @param[in] x input array
 * @param[out] out output array
 * @param[in] rows number of rows
 * @param[in] cols number of columns
 */
const char* relu2DDerKernelSource = R"(
    __kernel void clReLUder(__global float* x, __global float* out, int rows, int cols) {
        int row = get_global_id(0);
        int col = get_global_id(1);
        if (row < rows && col < cols) {
            out[row * cols + col] = (x[row * cols + col] > 0.0f) ? 1.0f : 0.0f;
        }
    }
)";

//----------------LOTA----------------//

/**
 * @brief LOTA activation function for 1D array.
 * @param[in] y input array
 * @param[out] out output array
 * @param[in] size size of the array 
 */
const char* lota1DKernelSource = R"(
    __kernel void clLOTA(__global float* y, __global float* out, int size) {
        int i = get_global_id(0);
        if (i < size) {
            // Find minimum value in the array
            float min_val = y[0];
            for (int j = 1; j < size; j++) {
                min_val = fmin(min_val, y[j]);
            }
            
            // Calculate sum for normalization
            float sum = 0.0f;
            for (int j = 0; j < size; j++) {
                sum += (y[j] - min_val);
            }
            
            // Normalize if sum is not zero
            if (sum > 0.0f) {
                out[i] = (y[i] - min_val) / sum;
            } else {
                out[i] = 1.0f / size; // Equal distribution if all values are the same
            }
        }
    }
)";

/**
 * @brief LOTA activation function for 2D array.
 * @param[in] y input array
 * @param[out] out output array
 * @param[in] rows number of rows
 * @param[in] cols number of columns
 * @param[in] attentionType 0 for square processing (col < cols), non-zero for triangular (col < row)
 */
const char* lota2DKernelSource = R"(
    __kernel void clLOTA(__global float* y, __global float* out, int rows, int cols, int attentionType) {
        int row = get_global_id(0);
        int col = get_global_id(1);
        // Determine processing boundary based on attentionType
        bool process = attentionType ? (col < row) : (col < cols);
        
        // First pass: compute min value for each row (only first thread in each row)
        if (col == 0 && row < rows) {
            float min_val = FLT_MAX;
            // Find minimum value in this row
            for (int j = 0; j < cols; j++) {
                if (attentionType && j >= row) continue; // Skip if using triangular attention
                if (j < cols) {
                    min_val = fmin(min_val, y[row * cols + j]);
                }
            }
            
            // Store min_val in a temporary location (first element of the row)
            out[row * cols] = min_val;
        }
        barrier(CLK_GLOBAL_MEM_FENCE); // Ensure all threads see the min value
        
        // Second pass: compute sum for normalization (only first thread in each row)
        if (col == 0 && row < rows) {
            float min_val = out[row * cols];
            float sum = 0.0f;
            
            for (int j = 0; j < cols; j++) {
                if (attentionType && j >= row) continue; // Skip if using triangular attention
                if (j < cols) {
                    sum += (y[row * cols + j] - min_val);
                }
            }
            
            // Store sum in a temporary location (second element of the row)
            if (cols > 1) {
                out[row * cols + 1] = sum;
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE); // Ensure all threads see the sum
        
        // Third pass: normalize values
        if (row < rows && process && col < cols) {
            float min_val = out[row * cols];
            float sum = (cols > 1) ? out[row * cols + 1] : 0.0f;
            
            if (sum > 0.0f) {
                out[row * cols + col] = (y[row * cols + col] - min_val) / sum;
            } else {
                // If sum is zero (all values are the same), use equal distribution
                int count = attentionType ? row : cols;
                out[row * cols + col] = 1.0f / count;
            }
        }
    }
)";

/**
 * @brief LOTA activation function for 2D array with limit.
 * @param[in] y input array
 * @param[out] out output array
 * @param[in] rows number of rows
 * @param[in] cols number of columns
 * @param[in] limit upper limit for the activation (used when attentionType is 0)
 * @param[in] attentionType 0 for square processing (col < limit), non-zero for triangular (col < row)
 */
const char* lota2DWithLimitKernelSource = R"(
    __kernel void clLOTA(__global float* y, __global float* out, int rows, int cols, int limit, int attentionType) {
        int row = get_global_id(0);
        int col = get_global_id(1);
        // Determine processing boundary based on attentionType
        bool process = attentionType ? (col < row) : (col < limit); // Use limit for non-attention case
        
        // First pass: apply limit constraint and find min value for each row
        if (row < rows && col < cols) {
            // Apply limit constraint
            if (process) {
                out[row * cols + col] = (y[row * cols + col] > 0.0f && y[row * cols + col] < limit) ? y[row * cols + col] : 0.0f;
            } else {
                out[row * cols + col] = 0.0f;
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
        
        // Second pass: compute min value for each row (only first thread in each row)
        if (col == 0 && row < rows) {
            float min_val = FLT_MAX;
            // Find minimum non-zero value in this row
            for (int j = 0; j < cols; j++) {
                if (attentionType && j >= row) continue; // Skip if using triangular attention
                if (j < cols && out[row * cols + j] > 0.0f) {
                    min_val = fmin(min_val, out[row * cols + j]);
                }
            }
            
            if (min_val == FLT_MAX) min_val = 0.0f; // No positive values found
            
            // Store min_val in a temporary location
            out[row * cols + cols - 2] = min_val;
            
            // Compute sum for normalization
            float sum = 0.0f;
            for (int j = 0; j < cols; j++) {
                if (attentionType && j >= row) continue; // Skip if using triangular attention
                if (j < cols && out[row * cols + j] > 0.0f) {
                    out[row * cols + j] -= min_val; // Subtract min value
                    sum += out[row * cols + j];
                }
            }
            
            // Store sum in a temporary location
            out[row * cols + cols - 1] = sum;
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
        
        // Third pass: normalize values
        if (row < rows && process && col < cols) {
            float sum = out[row * cols + cols - 1];
            
            if (sum > 0.0f && out[row * cols + col] > 0.0f) {
                out[row * cols + col] /= sum;
            }
        }
    }
)";


/**
 * @brief LOTA derivative for 1D array.
 * @param[in] y input array
 * @param[out] out output array
 * @param[in] size size of the array 
 */
const char* lota1DDerKernelSource = R"(
    __kernel void clLOTAder(__global float* y, __global float* out, int size) {
        int i = get_global_id(0);
        if (i < size) {
            // Find minimum value in the array
            float min_val = y[0];
            for (int j = 1; j < size; j++) {
                min_val = fmin(min_val, y[j]);
            }
            
            // Calculate sum for normalization
            float sum = 0.0f;
            for (int j = 0; j < size; j++) {
                sum += (y[j] - min_val);
            }
            
            // Calculate derivative
            if (sum > 0.0f) {
                // The derivative is (sum - (y[i] - min_val)) / sum^2 = (sum - y[i] + min_val) / sum^2
                out[i] = (sum - y[i] + min_val) / (sum * sum);
            } else {
                out[i] = 0.0f;
            }
        }
    }
)";

/**
 * @brief LOTA derivative for 2D array.
 * @param[in] y input array
 * @param[out] out output array
 * @param[in] rows number of rows
 * @param[in] cols number of columns
 * @param[in] attentionType 0 for square processing (col < cols), non-zero for triangular (col < row)
 */
const char* lota2DDerKernelSource = R"(
    __kernel void clLOTAder(__global float* y, __global float* out, int rows, int cols, int attentionType) {
        int row = get_global_id(0);
        int col = get_global_id(1);
        // Determine processing boundary based on attentionType
        bool process = attentionType ? (col < row) : (col < cols);
        
        // First pass: compute min value for each row (only first thread in each row)
        if (col == 0 && row < rows) {
            float min_val = y[row * cols];
            int valid_count = 0;
            
            // Find minimum value in this row
            for (int j = 0; j < cols; j++) {
                if (attentionType && j >= row) continue; // Skip if using triangular attention
                if (j < cols) {
                    min_val = fmin(min_val, y[row * cols + j]);
                    valid_count++;
                }
            }
            
            // Store min_val in a temporary location
            out[row * cols] = min_val;
            
            // Compute sum for normalization
            float sum = 0.0f;
            for (int j = 0; j < cols; j++) {
                if (attentionType && j >= row) continue; // Skip if using triangular attention
                if (j < cols) {
                    sum += (y[row * cols + j] - min_val);
                }
            }
            
            // Store sum in a temporary location
            if (cols > 1) {
                out[row * cols + 1] = sum;
                out[row * cols + 2] = (float)valid_count;
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
        
        // Second pass: compute derivatives
        if (row < rows && process && col < cols) {
            float min_val = out[row * cols];
            float sum = (cols > 1) ? out[row * cols + 1] : 0.0f;
            
            if (sum > 0.0f) {
                // The derivative is (sum - (y[i] - min_val)) / sum^2 = (sum - y[i] + min_val) / sum^2
                out[row * cols + col] = (sum - y[row * cols + col] + min_val) / (sum * sum);
            } else {
                out[row * cols + col] = 0.0f;
            }
        }
    }
)";

/**
 * @brief LOTA derivative for 2D array with limit.
 * @param[in] y input array
 * @param[out] out output array
 * @param[in] rows number of rows
 * @param[in] cols number of columns
 * @param[in] limit upper limit for the activation (used when attentionType is 0)
 * @param[in] attentionType 0 for square processing (col < limit), non-zero for triangular
 */
const char* lota2DDerWithLimitKernelSource = R"(
    __kernel void clLOTAder(__global float* y, __global float* out, int rows, int cols, int limit, int attentionType) {
        int row = get_global_id(0);
        int col = get_global_id(1);
        // Determine processing boundary based on attentionType
        bool process = attentionType ? (col < row) : (col < limit); // Use limit for non-attention case
        
        // First pass: determine valid values based on limit constraint
        bool valid = false;
        if (row < rows && col < cols && process) {
            valid = (y[row * cols + col] > 0.0f && y[row * cols + col] < limit);
            // Store validity in output (will be overwritten later)
            out[row * cols + col] = valid ? 1.0f : 0.0f;
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
        
        // Second pass: compute min value for each row (only first thread in each row)
        if (col == 0 && row < rows) {
            float min_val = FLT_MAX;
            int valid_count = 0;
            
            // Find minimum non-zero value that meets the limit constraint
            for (int j = 0; j < cols; j++) {
                if (attentionType && j >= row) continue; // Skip if using triangular attention
                if (j < cols && out[row * cols + j] > 0.0f) {
                    min_val = fmin(min_val, y[row * cols + j]);
                    valid_count++;
                }
            }
            
            if (min_val == FLT_MAX) min_val = 0.0f; // No valid values found
            
            // Store min_val and valid_count in temporary locations
            out[row * cols + cols - 3] = min_val;
            out[row * cols + cols - 2] = (float)valid_count;
            
            // Compute sum for normalization
            float sum = 0.0f;
            for (int j = 0; j < cols; j++) {
                if (attentionType && j >= row) continue; // Skip if using triangular attention
                if (j < cols && out[row * cols + j] > 0.0f) {
                    sum += (y[row * cols + j] - min_val);
                }
            }
            
            // Store sum in a temporary location
            out[row * cols + cols - 1] = sum;
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
        
        // Third pass: compute derivatives
        if (row < rows && process && col < cols) {
            float min_val = out[row * cols + cols - 3];
            float sum = out[row * cols + cols - 1];
            bool is_valid = (y[row * cols + col] > 0.0f && y[row * cols + col] < limit);
            
            if (is_valid && sum > 0.0f) {
                // The derivative is (sum - (y[i] - min_val)) / sum^2 = (sum - y[i] + min_val) / sum^2
                out[row * cols + col] = (sum - y[row * cols + col] + min_val) / (sum * sum);
            } else {
                out[row * cols + col] = 0.0f;
            }
        }
    }
)";
