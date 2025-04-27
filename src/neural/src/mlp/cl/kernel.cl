/**
 * Kernel to calculate the L1 penalty (sum of absolute values) using parallel reduction.
 * Writes partial sums per work-group to the result buffer.
 * Host code needs to sum the 'result' buffer.
 */
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

/*
* Kernel to calculate the L2 penalty (sum of squares) using parallel reduction.
* Writes partial sums per work-group to the result buffer.
* Host code needs to sum the 'result' buffer.
*/
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

/*
* Kernel to calculate the sum of absolute differences using parallel reduction.
* Writes partial sums per work-group to the result buffer.
* Host code needs to sum the 'result' buffer.
*/
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

/*
* Kernel to calculate the sum of squared differences using parallel reduction.
* Writes partial sums per work-group to the result buffer.
* Host code needs to sum the 'result' buffer. This is the numerator of MSE.
*/
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


/* =========================================================================
 * New Kernels Converted from CUDA
 * ========================================================================= */

/**
 * @brief Calculates the gradient w.r.t. the input of an MLP layer.
 *        grad_input[i] = sum_j (deltas[j] * weights[j * input_size + i])
 * @param deltas Deltas of the current layer (output layer of the conceptual block)
 * @param weights Weights connecting the input to the current layer (Row-major: [current_layer_size x input_size])
 * @param grad_input Gradient w.r.t. the input (to be computed)
 * @param current_layer_size Size of the current layer (number of deltas/rows in weights)
 * @param input_size Size of the input layer (number of columns in weights)
 */
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


/**
* @brief OpenCL kernel for calculating output layer deltas (Sigmoid derivative assumed).
*        delta = (activation - expected) * activation * (1.0f - activation)
* @param activations Activations of the output layer
* @param expected Expected output values
* @param deltas Deltas to be computed for the output layer
* @param size Size of the output layer
*/
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


/**
* @brief OpenCL kernel for calculating hidden layer deltas (Sigmoid derivative assumed)
* @param next_layer_deltas Deltas from the next layer
* @param weights Weights connecting the current layer to the next layer (Row-major: [next_layer_size x current_layer_size])
* @param activations Activations of the current layer
* @param deltas Deltas to be computed for the current layer
* @param current_layer_size Size of the current layer
* @param next_layer_size Size of the next layer
*/
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


/**
* @brief OpenCL kernel to compute delta for the last MLP layer given the gradient w.r.t. its output.
*        Assumes Sigmoid activation for the output layer.
*        delta = grad_output * activation * (1.0f - activation)
* @param grad_output Gradient w.r.t the output of this layer
* @param activations Activations of this layer
* @param deltas Deltas to be computed for this layer
* @param size Size of this layer
*/
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


/**
* @brief OpenCL kernel for updating weights and calculating gradients.
*        gradient = delta[current] * activation[previous]
*        weight -= learning_rate * gradient
* @param deltas Deltas for the current layer (size: current_layer_size)
* @param prev_activations Activations from the previous layer (or inputs) (size: prev_layer_size)
* @param weights Weights to be updated (Row-major: [current_layer_size x prev_layer_size])
* @param gradients Gradients to be calculated (Row-major: [current_layer_size x prev_layer_size])
* @param learning_rate Learning rate
* @param current_layer_size Size of the current layer (number of rows in weights)
* @param prev_layer_size Size of the previous layer (number of columns in weights)
*/
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

/**
* @brief OpenCL kernel for updating weights (without calculating gradients explicitly).
*        weight -= learning_rate * delta[current] * activation[previous]
* @param deltas Deltas for the current layer (size: current_layer_size)
* @param prev_activations Activations from the previous layer (or inputs) (size: prev_layer_size)
* @param weights Weights to be updated (Row-major: [current_layer_size x prev_layer_size])
* @param learning_rate Learning rate
* @param current_layer_size Size of the current layer (number of rows in weights)
* @param prev_layer_size Size of the previous layer (number of columns in weights)
*/
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


/**
 * @brief OpenCL kernel for L1 regularization weight update.
 *        gradient = delta[current] * activation[previous]
 *        weight -= learning_rate * (gradient + lambda * sign(weight))
 * @param weights Weights to be updated (Row-major: [current_layer_size x prev_layer_size])
 * @param deltas Deltas for the current layer
 * @param prev_activations Activations from the previous layer
 * @param learning_rate Learning rate
 * @param lambda L1 regularization parameter
 * @param current_layer_size Size of the current layer
 * @param prev_layer_size Size of the previous layer
 */
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
        } else if (weight_val < 0.0f) {
            weights[weight_idx] -= learning_rate * (gradient - lambda);
        } else {
            // If weight is exactly zero, the subgradient term is [-lambda, +lambda] * learning_rate
            // A common approach is to apply thresholding: only update if |gradient| > lambda
            // Or simply apply the update based on the gradient sign (proximal gradient step)
            // Let's match the CUDA code's logic which implicitly handles w=0 by checking >0 or <0.
            // If w=0, the second condition (else) applies: weights[idx] -= lr*(-lambda + grad)
            // This seems slightly asymmetric. Let's use the explicit sign check from CUDA:
             weights[weight_idx] -= learning_rate * (gradient - lambda); // Matches CUDA's 'else' block
             // Note: A more robust proximal step for w=0 would be:
             // float update = learning_rate * gradient;
             // if (update > learning_rate * lambda) weight[idx] = -(update - learning_rate * lambda);
             // else if (update < -learning_rate * lambda) weight[idx] = -(update + learning_rate * lambda);
             // else weight[idx] = 0;
             // But we stick to the provided CUDA logic for now.
        }
        // An alternative matching the CUDA code exactly:
        /*
        if (weights[weight_idx] > 0) {
            weights[weight_idx] -= learning_rate * (lambda + gradient);
        } else { // Covers <= 0
            weights[weight_idx] -= learning_rate * (-lambda + gradient);
        }
        */
    }
}

/**
 * @brief OpenCL kernel for L2 regularization weight update.
 *        gradient = delta[current] * activation[previous]
 *        weight -= learning_rate * (gradient + lambda * weight)
 * @param weights Weights to be updated (Row-major: [current_layer_size x prev_layer_size])
 * @param deltas Deltas for the current layer
 * @param prev_activations Activations from the previous layer
 * @param learning_rate Learning rate
 * @param lambda L2 regularization parameter
 * @param current_layer_size Size of the current layer
 * @param prev_layer_size Size of the previous layer
 */
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


/**
 * @brief OpenCL kernel for updating the input vector based on first layer deltas.
 *        input[i] -= learning_rate * sum_j (deltas[j] * weights[j * input_size + i])
 *        This computes the gradient w.r.t input and applies an update step.
 * @param input Input vector to be updated (size: input_size)
 * @param weights Weights of the first hidden layer (Row-major: [first_hidden_layer_size x input_size])
 * @param deltas Deltas of the first hidden layer (size: first_hidden_layer_size)
 * @param learning_rate Learning rate
 * @param first_hidden_layer_size Size of the first hidden layer (number of rows in weights)
 * @param input_size Size of the input vector (number of columns in weights)
 */
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


/**
 * @brief OpenCL kernel for MLP layer forward propagation (Matrix-Vector multiplication).
 *        outputs[j] = sum_i (inputs[i] * weights[j * input_size + i])
 *        (Assumes no bias term for simplicity, matching CUDA kernel)
 * @param inputs Input data vector (size: input_size)
 * @param weights Weights for the current layer (Row-major: [output_size x input_size])
 * @param outputs Output data vector (size: output_size)
 * @param input_size Size of the input layer (number of columns in weights)
 * @param output_size Size of the output layer (number of rows in weights)
 */
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


/**
 * @brief OpenCL kernel for calculating the Mean Squared Error (MSE) using reduction.
 *        Computes sum of squared errors per work group.
 *        Host code needs to sum the partial results and divide by size.
 * @param expected Pointer to the expected output data on the device.
 * @param output Pointer to the actual output data on the device.
 * @param partial_mse Pointer to buffer for partial MSE sums (size = num_groups).
 * @param temp_sum Local memory buffer for reduction (size = local_size).
 * @param size The number of output neurons.
 */
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


/**
 * @brief OpenCL kernel for Rprop weight update.
 * @param weights Weights to be updated
 * @param gradients Current gradients
 * @param prev_gradients Previous gradients (updated in place)
 * @param delta_weights Step sizes for each weight (updated in place)
 * @param eta_plus Increase factor (e.g., 1.2)
 * @param eta_minus Decrease factor (e.g., 0.5)
 * @param delta_max Maximum step size
 * @param delta_min Minimum step size
 * @param size Number of weights/gradients
 */
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
            // Sign change: Decrease step size, zero previous gradient, *do not* update weight this step
            delta = fmax(delta * eta_minus, delta_min);
            weight_update = 0.0f; // No weight update this step
            prev_gradients[idx] = 0.0f; // Reset prev_grad to prevent double penalty
            // Note: Some Rprop variants might revert the previous step here, but the CUDA code doesn't seem to.
            // The CUDA code does: weights[idx] += copysignf(delta, prev_grad); which reverts the *previous* update magnitude using the *new* delta and *old* gradient sign.
            // Let's match the CUDA logic more closely:
            // delta = fmax(delta * eta_minus, delta_min); // Decrease step size
            // weights[idx] += copysign(delta_weights[idx], prev_grad); // Revert previous step magnitude (using OLD delta?) - CUDA code uses *new* delta here. Let's match that.
            // weights[idx] += copysign(delta, prev_grad); // Revert previous step magnitude using *new* delta
            // prev_gradients[idx] = 0; // Set to zero
            // delta_weights[idx] = delta; // Store updated delta
            // return; // Skip the final weight update and delta store below for this case.

            // Let's re-implement matching the CUDA structure:
            delta = fmax(delta * eta_minus, delta_min);
            // Revert previous step's weight change magnitude (using the *new* delta)
            // The previous update was likely -copysign(delta_weights[idx], prev_grad)
            // So reverting means adding copysign(delta_weights[idx], prev_grad)
            // The CUDA code adds copysign(delta, prev_grad). Let's use that.
            weights[idx] += copysign(delta, prev_grad); // Revert step based on prev_grad sign
            prev_gradients[idx] = 0.0f; // Reset prev_grad
            delta_weights[idx] = delta; // Store updated delta
            return; // Exit early, no further update this step

        }
        else { // sign_change == 0.0f (one or both gradients are zero)
            // Apply update using current step size
            weight_update = -copysign(delta, grad);
            prev_gradients[idx] = grad; // Store current gradient
        }

        // Apply the calculated weight update
        weights[idx] += weight_update;
        // Store the updated step size
        delta_weights[idx] = delta;
    }
}
