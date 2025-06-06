// Helper macro for indexing flattened matrix (assuming row-major)
#define IDX(row, col, dim) ((row) * (dim) + (col))

// Enable extensions for atomics and potentially double precision (which might include float atomics)
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable // For double support, might help with float atomics on some platforms

__kernel void l1PenaltyKernel(__global const float* weights,
                              __global float* result, // Output buffer for partial sums (size = num_groups)
                              __local float* temp_sum, // Local memory buffer (size = local_size)
                              const int size)         // Total number of weights
{
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int group_id = get_group_id(0);
    int local_size = get_local_size(0);

    // Initialize local memory for this work-item
    temp_sum[local_id] = 0.0f;

    // Each work-item accumulates its portion of the sum
    // Loop necessary if total size > global work size
    for (int i = global_id; i < size; i += get_global_size(0)) {
        temp_sum[local_id] += fabs(weights[i]);
    }

    // Synchronize within the work-group to ensure all items finished accumulation
    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform reduction in local memory
    for (int stride = local_size / 2; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            temp_sum[local_id] += temp_sum[local_id + stride];
        }
        // Synchronize after each step of the reduction
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // First work-item in the group writes the partial sum for this group to global memory
    if (local_id == 0) {
        result[group_id] = temp_sum[0];
    }
}

__kernel void l2PenaltyKernel(__global const float* weights,
                              __global float* result,    // Output buffer for partial sums (size = num_groups)
                              __local float* temp_sum,  // Local memory buffer (size = local_size)
                              const int size)          // Total number of weights
{
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int group_id = get_group_id(0);
    int local_size = get_local_size(0);

    // Initialize local memory
    temp_sum[local_id] = 0.0f;

    // Each work-item accumulates its portion of the sum of squares
    for (int i = global_id; i < size; i += get_global_size(0)) {
        float w = weights[i];
        temp_sum[local_id] += w * w;
    }

    // Synchronize within the work-group
    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform reduction in local memory
    for (int stride = local_size / 2; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            temp_sum[local_id] += temp_sum[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // First work-item in the group writes the partial sum to global memory
    if (local_id == 0) {
        result[group_id] = temp_sum[0];
    }
}

__kernel void absDiffKernel(__global const float* outputs,
                            __global const float* targets,
                            __global float* result,     // Output buffer for partial sums (size = num_groups)
                            __local float* temp_sum,   // Local memory buffer (size = local_size)
                            const int size)           // Size of output/target vectors
{
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int group_id = get_group_id(0);
    int local_size = get_local_size(0);

    // Initialize local memory
    temp_sum[local_id] = 0.0f;

    // Each work-item accumulates its portion of the sum of absolute differences
    for (int i = global_id; i < size; i += get_global_size(0)) {
        temp_sum[local_id] += fabs(outputs[i] - targets[i]);
    }

    // Synchronize within the work-group
    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform reduction in local memory
    for (int stride = local_size / 2; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            temp_sum[local_id] += temp_sum[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // First work-item in the group writes the partial sum to global memory
    if (local_id == 0) {
        result[group_id] = temp_sum[0];
    }
}

__kernel void squaredDiffKernel(__global const float* outputs,
                                __global const float* targets,
                                __global float* result,     // Output buffer for partial sums (size = num_groups)
                                __local float* temp_sum,   // Local memory buffer (size = local_size)
                                const int size)           // Size of output/target vectors
{
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int group_id = get_group_id(0);
    int local_size = get_local_size(0);

    // Initialize local memory
    temp_sum[local_id] = 0.0f;

    // Each work-item accumulates its portion of the sum of squared differences
    for (int i = global_id; i < size; i += get_global_size(0)) {
        float diff = outputs[i] - targets[i];
        temp_sum[local_id] += diff * diff;
    }

    // Synchronize within the work-group
    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform reduction in local memory
    for (int stride = local_size / 2; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            temp_sum[local_id] += temp_sum[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // First work-item in the group writes the partial sum to global memory
    if (local_id == 0) {
        result[group_id] = temp_sum[0];
    }
}

__kernel void kernelComputeGradMLPInput(__global const float* deltas,
                                        __global const float* weights,
                                        __global float* grad_input,
                                        const int current_layer_size,
                                        const int input_size)
{
    int input_idx = get_global_id(0); // Corresponds to input dimension index 'i'

    if (input_idx < input_size) {
        float sum = 0.0f;
        // Sum over the neurons 'j' in the current layer
        for (int j = 0; j < current_layer_size; ++j) {
            // weights are row-major: weights[row * num_cols + col]
            // Here: row = j (current layer neuron), num_cols = input_size, col = input_idx (input dimension)
            sum += deltas[j] * weights[j * input_size + input_idx];
        }
        grad_input[input_idx] = sum;
    }
}

__kernel void kernelOutputDeltaSigmoid(__global const float* activations,
                                       __global const float* expected,
                                       __global float* deltas,
                                       const int size)
{
    int idx = get_global_id(0);
    if (idx < size) {
        float activation = activations[idx];
        float error = activation - expected[idx];
        // Sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))
        float sigmoid_der = activation * (1.0f - activation);
        deltas[idx] = error * sigmoid_der;
    }
}

__kernel void kernelHiddenDeltaSigmoid(__global const float* next_layer_deltas,
                                       __global const float* weights, // Weights W_ji where j=next, i=current
                                       __global const float* activations,
                                       __global float* deltas,
                                       const int current_layer_size,
                                       const int next_layer_size)
{
    int neuron_idx = get_global_id(0); // Index 'i' for the current layer neuron

    if (neuron_idx < current_layer_size) {
        float error_sum = 0.0f;
        // Sum contribution from each neuron 'j' in the next layer
        // weights are row-major: weights[row * num_cols + col]
        // Here: row=j (next layer neuron), num_cols=current_layer_size, col=neuron_idx (current layer neuron)
        // We need weight W_ji which connects current neuron 'i' to next neuron 'j'
        for (int j = 0; j < next_layer_size; j++) {
            error_sum += next_layer_deltas[j] * weights[j * current_layer_size + neuron_idx];
        }

        // Calculate delta using the sigmoid derivative for the current neuron 'i'
        float activation = activations[neuron_idx];
        float sigmoid_der = activation * (1.0f - activation);
        deltas[neuron_idx] = error_sum * sigmoid_der;
    }
}

__kernel void kernelLastLayerDeltaSigmoid(__global const float* grad_output,
                                          __global const float* activations,
                                          __global float* deltas,
                                          const int size)
{
    int idx = get_global_id(0);
    if (idx < size) {
        float activation = activations[idx];
        // Sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))
        float sigmoid_der = activation * (1.0f - activation);
        // If ReLU: float relu_der = (activation > 0.0f) ? 1.0f : 0.0f; deltas[idx] = grad_output[idx] * relu_der;
        deltas[idx] = grad_output[idx] * sigmoid_der;
    }
}

__kernel void kernelUpdateWeightsAndGradients(__global const float* deltas,
                                              __global const float* prev_activations,
                                              __global float* weights,
                                              __global float* gradients, // Assumed to be a valid buffer
                                              const float learning_rate,
                                              const int current_layer_size,
                                              const int prev_layer_size)
{
    // Use 2D indexing
    int col = get_global_id(0); // Index for previous layer neuron (input to weight) 'j'
    int row = get_global_id(1); // Index for current layer neuron (output of weight) 'i'

    if (row < current_layer_size && col < prev_layer_size) {
        int weight_idx = row * prev_layer_size + col; // row-major index for W_ij

        // Calculate gradient: delta_i * activation_j
        float grad = deltas[row] * prev_activations[col];

        // Store gradient
        gradients[weight_idx] = grad;

        // Update weight using gradient descent
        weights[weight_idx] -= learning_rate * grad;
    }
}

__kernel void kernelUpdateWeights(__global const float* deltas,
                                  __global const float* prev_activations,
                                  __global float* weights,
                                  const float learning_rate,
                                  const int current_layer_size,
                                  const int prev_layer_size)
{
    // Use 2D indexing
    int col = get_global_id(0); // Index for previous layer neuron (input to weight) 'j'
    int row = get_global_id(1); // Index for current layer neuron (output of weight) 'i'

    if (row < current_layer_size && col < prev_layer_size) {
        int weight_idx = row * prev_layer_size + col; // row-major index for W_ij

        // Calculate gradient implicitly and update weight
        weights[weight_idx] -= learning_rate * deltas[row] * prev_activations[col];
    }
}


__kernel void kernelUpdateWeightsL1(__global float* weights,
                                    __global const float* deltas,
                                    __global const float* prev_activations,
                                    const float learning_rate,
                                    const float lambda,
                                    const int current_layer_size,
                                    const int prev_layer_size)
{
    // Use 2D indexing
    int col = get_global_id(0); // Index for previous layer neuron 'j'
    int row = get_global_id(1); // Index for current layer neuron 'i'

    if (row < current_layer_size && col < prev_layer_size) {
        int weight_idx = row * prev_layer_size + col; // row-major index for W_ij
        float gradient = deltas[row] * prev_activations[col];
        float weight_val = weights[weight_idx];

        // L1 regularization update (using subgradient)
        // sign(w) = 1 if w > 0, -1 if w < 0, 0 if w = 0 (subgradient is [-1, 1])
        // The CUDA code implements the update correctly based on sign.
        if (weight_val > 0.0f) {
            weights[weight_idx] -= learning_rate * (gradient + lambda);
        } 
        else if (weight_val < 0.0f) {
            weights[weight_idx] -= learning_rate * (gradient - lambda);
        } 
        else {
            weights[weight_idx] -= learning_rate * (gradient - lambda); // Matches CUDA's 'else' block
        }
        // An alternative matching the CUDA code exactly:
        /*
        if (weights[weight_idx] > 0) {
            weights[weight_idx] -= learning_rate * (lambda + gradient);
        } 
        else { // Covers <= 0
            weights[weight_idx] -= learning_rate * (-lambda + gradient);
        }
        */
    }
}

__kernel void kernelUpdateWeightsL2(__global float* weights,
                                    __global const float* deltas,
                                    __global const float* prev_activations,
                                    const float learning_rate,
                                    const float lambda,
                                    const int current_layer_size,
                                    const int prev_layer_size)
{
    // Use 2D indexing
    int col = get_global_id(0); // Index for previous layer neuron 'j'
    int row = get_global_id(1); // Index for current layer neuron 'i'

    if (row < current_layer_size && col < prev_layer_size) {
        int weight_idx = row * prev_layer_size + col; // row-major index for W_ij
        float gradient = deltas[row] * prev_activations[col];

        // L2 regularization update (weight decay)
        weights[weight_idx] -= learning_rate * (gradient + lambda * weights[weight_idx]);
        // Alternative form (often used): weights[weight_idx] *= (1.0f - learning_rate * lambda); weights[weight_idx] -= learning_rate * gradient;
        // The form used here matches the CUDA kernel.
    }
}

__kernel void kernelUpdateInputMLP(__global float* input,
                                   __global const float* weights,
                                   __global const float* deltas,
                                   const float learning_rate,
                                   const int first_hidden_layer_size,
                                   const int input_size)
{
    int input_idx = get_global_id(0); // Index 'i' for the input vector dimension

    if (input_idx < input_size) {
        float grad_input_component = 0.0f;
        // Sum over the neurons 'j' in the first hidden layer
        for (int j = 0; j < first_hidden_layer_size; j++) {
            // weights are row-major: weights[row * num_cols + col]
            // Here: row = j (hidden neuron), num_cols = input_size, col = input_idx (input dimension)
            grad_input_component += deltas[j] * weights[j * input_size + input_idx];
        }
        // Update the input vector element
        input[input_idx] -= learning_rate * grad_input_component;
    }
}

__kernel void kernelLayerForward(__global const float* inputs,
                                 __global const float* weights,
                                 __global float* outputs,
                                 const int input_size,
                                 const int output_size)
{
    int neuron_idx = get_global_id(0); // Index 'j' for the output neuron

    if (neuron_idx < output_size) {
        float sum = 0.0f;
        // Perform dot product: inputs . weights_row_j
        // weights are row-major: weights[row * num_cols + col]
        // Here: row = neuron_idx, num_cols = input_size, col = i (input dimension)
        for (int i = 0; i < input_size; i++) {
            sum += inputs[i] * weights[neuron_idx * input_size + i];
        }
        // Store the weighted sum (pre-activation value)
        outputs[neuron_idx] = sum;
        // Activation function (e.g., sigmoid, ReLU) would typically be applied in a separate kernel or here.
        // outputs[neuron_idx] = 1.0f / (1.0f + exp(-sum)); // Example Sigmoid
    }
}

__kernel void kernelMseReduction(__global const float* expected,
                                 __global const float* output,
                                 __global float* partial_mse, // Output buffer for partial sums
                                 __local float* temp_sum,    // Local memory buffer
                                 const int size)
{
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int group_id = get_group_id(0);
    int local_size = get_local_size(0);

    // Initialize local memory
    temp_sum[local_id] = 0.0f;

    // Each work-item accumulates its portion of the sum of squared errors
    for (int i = global_id; i < size; i += get_global_size(0)) {
        float diff = expected[i] - output[i];
        temp_sum[local_id] += diff * diff;
    }

    // Synchronize within the work-group
    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform reduction in local memory
    for (int stride = local_size / 2; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            temp_sum[local_id] += temp_sum[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // First work-item in the group writes the partial sum to global memory
    if (local_id == 0) {
        partial_mse[group_id] = temp_sum[0];
    }
}

__kernel void kernelRpropUpdate(__global float* weights,
                                __global const float* gradients,
                                __global float* prev_gradients, // Read and Write
                                __global float* delta_weights,  // Read and Write
                                const float eta_plus,
                                const float eta_minus,
                                const float delta_max,
                                const float delta_min,
                                const int size)
{
    int idx = get_global_id(0);

    if (idx < size) {
        float grad = gradients[idx];
        float prev_grad = prev_gradients[idx];
        float delta = delta_weights[idx];
        float weight_update = 0.0f;

        // Rprop update logic
        float sign_change = grad * prev_grad;

        if (sign_change > 0.0f) {
            // No sign change: Increase step size, apply update
            delta = fmin(delta * eta_plus, delta_max);
            weight_update = -copysign(delta, grad); // Update direction is opposite to gradient
            prev_gradients[idx] = grad; // Store current gradient for next iteration
        }
        else if (sign_change < 0.0f) {
            delta = fmax(delta * eta_minus, delta_min);
            weight_update = 0.0f; // No weight update this step
            prev_gradients[idx] = 0.0f; // Reset prev_grad to prevent double penalty
            delta = fmax(delta * eta_minus, delta_min);
            weights[idx] += copysign(delta, prev_grad); // Revert step based on prev_grad sign
            prev_gradients[idx] = 0.0f; // Reset prev_grad
            delta_weights[idx] = delta; // Store updated delta
            return; // Exit early, no further update this step

        }
        else {
            weight_update = -copysign(delta, grad);
            prev_gradients[idx] = grad; // Store current gradient
        }

        // Apply the calculated weight update
        weights[idx] += weight_update;
        // Store the updated step size
        delta_weights[idx] = delta;
    }
}
