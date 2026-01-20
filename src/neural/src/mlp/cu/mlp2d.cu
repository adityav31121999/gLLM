#ifdef USE_CU
#include "include/mlp.hpp"

// ---------------- forward pass ---------------- //

/**
 * @brief CUDA kernel for MLP forward propagation for a batch.
 * @param inputs Input data (batch_size x input_size)
 * @param weights Weights for the current layer (output_size x input_size)
 * @param outputs Output data (batch_size x output_size)
 * @param input_size Size of the input layer feature vector.
 * @param output_size Size of the output layer feature vector.
 * @param rows The batch size.
 */
__global__ void kernelLayerForward(const float* inputs, const float* weights, float* outputs, 
    int input_size, int output_size, int rows)
{
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (neuron_idx < output_size && row_idx < rows) {
        const float* current_input = inputs + row_idx * input_size;
        float sum = 0.0f;

        // Perform matrix-vector multiplication for one sample in the batch
        for (int i = 0; i < input_size; i++) {
            sum += current_input[i] * weights[neuron_idx * input_size + i];
        }
        outputs[row_idx * output_size + neuron_idx] = sum;
    }
}

// ---------------- backward pass ---------------- //

/**
 * @brief CUDA kernel for calculating output layer deltas with BCE loss.
 *        delta = output - expected
 * @param output Output layer values (batch_size x size)
 * @param expected Expected output values (batch_size x size)
 * @param delta Output deltas to be computed (batch_size x size)
 * @param size Size of the output feature vector.
 * @param rows The batch size.
 */
__global__ void kernelOutputDelta(const float* output, const float* expected, float* delta, int size, int rows) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < size && row_idx < rows) {
        int offset = row_idx * size + idx;
        // For Sigmoid activation with Binary Cross-Entropy (BCE) loss, the gradient
        // w.r.t. pre-activation is simply (activation - expected).
        delta[offset] = output[offset] - expected[offset];
    }
}

/**
* @brief CUDA kernel for calculating output layer deltas with Sigmoid activation.
*        delta = (activation - expected) * activation * (1.0f - activation)
* @param activations Activations of the output layer (batch_size x size)
* @param expected Expected output values (batch_size x size)
* @param deltas Deltas to be computed for the output layer (batch_size x size)
* @param size Size of the output feature vector.
* @param rows The batch size.
*/
__global__ void kernelOutputDeltaSigmoid(const float* activations, const float* expected, float* deltas, int size, int rows)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < size && row_idx < rows) {
        int offset = row_idx * size + idx;
        float activation = activations[offset];
        float error = activation - expected[offset];
        float sigmoid_der = activation * (1.0f - activation); // Derivative of Sigmoid
        deltas[offset] = error * sigmoid_der;
    }
}


/**
* @brief CUDA kernel for calculating hidden layer deltas (Sigmoid derivative assumed).
* @param next_layer_deltas Deltas from the next layer (batch_size x next_layer_size)
* @param weights Weights connecting to the next layer (next_layer_size x current_layer_size)
* @param activations Activations of the current layer (batch_size x current_layer_size)
* @param deltas Deltas to be computed for the current layer (batch_size x current_layer_size)
* @param current_layer_size Size of the current layer's feature vector.
* @param next_layer_size Size of the next layer's feature vector.
* @param rows The batch size.
*/
__global__ void hiddenDeltaKernel(const float* next_layer_deltas, const float* weights, const float* activations,
    float* deltas, int current_layer_size, int next_layer_size, int rows)
{
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x; // Index 'i' for the current layer neuron
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (neuron_idx < current_layer_size && row_idx < rows) {
        float error_sum = 0.0f;
        const float* next_delta_batch = next_layer_deltas + row_idx * next_layer_size;
        
        // Sum up the weighted deltas from the next layer for one sample in the batch
        for (int j = 0; j < next_layer_size; j++) {
            error_sum += next_delta_batch[j] * weights[j * current_layer_size + neuron_idx];
        }

        int current_offset = row_idx * current_layer_size + neuron_idx;
        float activation = activations[current_offset];
        float sigmoid_der = activation * (1.0f - activation);
        deltas[current_offset] = error_sum * sigmoid_der;
    }
}

/**
 * @brief Calculates the gradient w.r.t. the input of an MLP layer for a batch.
 *        grad_input[r][i] = sum_j (deltas[r][j] * weights[j][i])
 * @param deltas Deltas of the current layer (batch_size x current_layer_size)
 * @param weights Weights connecting the input to the current layer (current_layer_size x input_size)
 * @param grad_input Gradient w.r.t. the input (to be computed) (batch_size x input_size)
 * @param current_layer_size Size of the current layer's feature vector.
 * @param input_size Size of the input layer's feature vector.
 * @param rows The batch size.
*/
__global__ void kernelComputeGradMLPInput(const float* deltas, const float* weights, float* grad_input,
    int current_layer_size, int input_size, int rows)
{
    int input_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (input_idx < input_size && row_idx < rows) {
        float sum = 0.0f;
        const float* delta_batch = deltas + row_idx * current_layer_size;
        for (int j = 0; j < current_layer_size; ++j) {
            sum += delta_batch[j] * weights[j * input_size + input_idx];
        }
        grad_input[row_idx * input_size + input_idx] = sum;
    }
}

/**
 * @brief CUDA kernel for updating the input batch.
 * @param input Input batch to be updated (batch_size x input_size)
 * @param weights Weights of the first hidden layer (first_hidden_layer_size x input_size)
 * @param deltas Deltas of the first hidden layer (batch_size x first_hidden_layer_size)
 * @param learning_rate Learning rate.
 * @param first_hidden_layer_size Size of the first hidden layer's feature vector.
 * @param input_size Size of the input layer's feature vector.
 * @param rows The batch size.
 */
__global__ void kernelUpdateInputMLP(float* input, const float* weights,
                                   const float* deltas, float learning_rate,
                                   int first_hidden_layer_size, int input_size,
                                   int rows)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // input dimension index
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < input_size && row_idx < rows) {
        float grad_input_component = 0.0f;
        const float* delta_batch = deltas + row_idx * first_hidden_layer_size;

        // Sum over the neurons 'j' in the first hidden layer
        for (int j = 0; j < first_hidden_layer_size; j++) {
            grad_input_component += delta_batch[j] * weights[j * input_size + i];
        }
        
        int input_offset = row_idx * input_size + i;
        input[input_offset] -= learning_rate * grad_input_component;
    }
}

// ---------------- update weigthts ---------------- //

/**
 * @brief CUDA kernel for updating weights based on batch gradients.
 * @param deltas Deltas for the current layer (batch_size x current_layer_size)
 * @param prev_activations Activations from the previous layer (batch_size x prev_layer_size)
 * @param weights Weights to be updated (current_layer_size x prev_layer_size)
 * @param learning_rate Learning rate.
 * @param current_layer_size Size of the current layer's feature vector.
 * @param prev_layer_size Size of the previous layer's feature vector.
 * @param rows The batch size.
 */
__global__ void kernelUpdateWeights(const float* deltas, const float* prev_activations, float* weights,
    float learning_rate, int current_layer_size, int prev_layer_size, int rows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // prev_layer neuron index ('col')
    int j = blockIdx.y * blockDim.y + threadIdx.y; // current_layer neuron index ('row')
    
    if (j < current_layer_size && i < prev_layer_size) {
        int weight_idx = j * prev_layer_size + i;
        
        float gradient = 0.0f;
        // Accumulate gradient over the batch
        for (int r = 0; r < rows; ++r) {
            gradient += deltas[r * current_layer_size + j] * prev_activations[r * prev_layer_size + i];
        }

        weights[weight_idx] -= learning_rate * gradient;
    }
}


/**
* @brief CUDA kernel for updating weights and calculating gradients from a batch.
* @param deltas Deltas for the current layer (batch_size x current_layer_size)
* @param prev_activations Activations from the previous layer (batch_size x prev_layer_size)
* @param weights Weights to be updated (current_layer_size x prev_layer_size)
* @param gweights Gradients to be calculated (current_layer_size x prev_layer_size)
* @param learning_rate Learning rate.
* @param current_layer_size Size of the current layer's feature vector.
* @param prev_layer_size Size of the previous layer's feature vector.
* @param rows The batch size.
*/
__global__ void kernelUpdateWeightsAndGradients(const float* deltas, const float* prev_activations, float* weights,
                                    float* gweights, float learning_rate, int current_layer_size, int prev_layer_size, int rows)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // prev_layer neuron index ('col')
    int j = blockIdx.y * blockDim.y + threadIdx.y; // current_layer neuron index ('row')

    if (j < current_layer_size && i < prev_layer_size) {
        int weight_idx = j * prev_layer_size + i;
        
        float gradient = 0.0f;
        // Accumulate gradient over the batch
        for (int r = 0; r < rows; ++r) {
            gradient += deltas[r * current_layer_size + j] * prev_activations[r * prev_layer_size + i];
        }
        
        if (gweights != nullptr) {
            gweights[weight_idx] = gradient;
        }
        weights[weight_idx] -= learning_rate * gradient;
    }
}   


/**
 * @brief CUDA kernel for L1 regularization weight update from a batch.
 * @param deltas Deltas for the current layer (batch_size x current_layer_size)
 * @param prev_activations Activations from the previous layer (batch_size x prev_layer_size)
 * @param weights Weights to be updated (current_layer_size x prev_layer_size)
 * @param learning_rate Learning rate.
 * @param lambda L1 regularization parameter.
 * @param current_layer_size Size of the current layer's feature vector.
 * @param prev_layer_size Size of the previous layer's feature vector.
 * @param rows The batch size.
 */
__global__ void kernelUpdateWeightsL1(const float* deltas, const float* prev_activations, float* weights, float learning_rate, 
    float lambda, int current_layer_size, int prev_layer_size, int rows) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // prev_layer neuron index ('col')
    int j = blockIdx.y * blockDim.y + threadIdx.y; // current_layer neuron index ('row')
    
    if (j < current_layer_size && i < prev_layer_size) {
        int weight_idx = j * prev_layer_size + i;

        float error_gradient = 0.0f;
        for (int r = 0; r < rows; ++r) {
            error_gradient += deltas[r * current_layer_size + j] * prev_activations[r * prev_layer_size + i];
        }
        
        float current_weight = weights[weight_idx];
        float sign = (current_weight > 0.0f) ? 1.0f : ((current_weight < 0.0f) ? -1.0f : 0.0f);
        float l1_gradient = lambda * sign;
        
        float total_gradient = error_gradient + l1_gradient;
        
        weights[weight_idx] -= learning_rate * total_gradient;
    }
}

/**
 * @brief CUDA kernel for L2 regularization weight update from a batch.
 * @param deltas Deltas for the current layer (batch_size x current_layer_size)
 * @param prev_activations Activations from the previous layer (batch_size x prev_layer_size)
 * @param weights Weights to be updated (current_layer_size x prev_layer_size)
 * @param gweights Gradients to be calculated (current_layer_size x prev_layer_size)
 * @param learning_rate Learning rate.
 * @param lambda L2 regularization parameter.
 * @param current_layer_size Size of the current layer's feature vector.
 * @param prev_layer_size Size of the previous layer's feature vector.
 * @param rows The batch size.
 */
__global__ void kernelUpdateWeightsL2(const float* deltas, const float* prev_activations, float* weights, float* gweights,
    float learning_rate, float lambda, int current_layer_size, int prev_layer_size, int rows) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // prev_layer neuron index ('col')
    int j = blockIdx.y * blockDim.y + threadIdx.y; // current_layer neuron index ('row')
    
    if (j < current_layer_size && i < prev_layer_size) {
        int weight_idx = j * prev_layer_size + i;
        
        float error_gradient = 0.0f;
        for (int r = 0; r < rows; ++r) {
            error_gradient += deltas[r * current_layer_size + j] * prev_activations[r * prev_layer_size + i];
        }

        float l2_gradient = lambda * weights[weight_idx];
        float total_gradient = error_gradient + l2_gradient;
        
        if (gweights != nullptr) {
            gweights[weight_idx] = total_gradient;
        }
        weights[weight_idx] -= learning_rate * total_gradient;
    }
}

/**
 * @brief CUDA kernel for Elastic Net (L1+L2) regularization weight update from a batch.
 * @param deltas Deltas for the current layer (batch_size x current_layer_size)
 * @param prev_activations Activations from the previous layer (batch_size x prev_layer_size)
 * @param weights Weights to be updated (current_layer_size x prev_layer_size)
 * @param gweights Gradients of the weights (output, can be nullptr).
 * @param learning_rate The learning rate.
 * @param lambda_l1 The L1 regularization parameter.
 * @param lambda_l2 The L2 regularization parameter.
 * @param current_layer_size Size of the current layer's feature vector.
 * @param prev_layer_size Size of the previous layer's feature vector.
 * @param rows The batch size.
 */
__global__ void kernelUpdateElasticNet(const float* deltas, const float* prev_activations, 
        float* weights, float* gweights, float learning_rate, float lambda_l1, 
        float lambda_l2, int current_layer_size, int prev_layer_size, int rows)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // neuron index in previous layer ('col')
    int j = blockIdx.y * blockDim.y + threadIdx.y; // neuron index in current layer ('row')

    if (i < prev_layer_size && j < current_layer_size) {
        int weight_idx = j * prev_layer_size + i;
        
        float error_gradient = 0.0f;
        for (int r = 0; r < rows; ++r) {
            error_gradient += deltas[r * current_layer_size + j] * prev_activations[r * prev_layer_size + i];
        }
        
        float current_weight = weights[weight_idx];
        float l2_gradient = lambda_l2 * current_weight;
        float sign = (current_weight > 0.0f) ? 1.0f : ((current_weight < 0.0f) ? -1.0f : 0.0f);
        float l1_gradient = lambda_l1 * sign;
        
        float total_gradient = error_gradient + l2_gradient + l1_gradient;
        
        if (gweights != nullptr) {
            gweights[weight_idx] = total_gradient;
        }
        weights[weight_idx] -= learning_rate * total_gradient;
    }
}


/**
 * @brief CUDA kernel for Rprop weight update. This kernel operates on pre-computed batch gradients.
 * @param weights Weights to be updated.
 * @param gradients Current batch gradients.
 * @param prev_gradients Previous gradients from the last update.
 * @param delta_weights Step sizes for each weight.
 * @param eta_plus Increase factor for step size.
 * @param eta_minus Decrease factor for step size.
 * @param delta_max Maximum step size.
 * @param delta_min Minimum step size.
 * @param size Total number of weights.
 */
__global__ void rpropUpdateKernel(float* weights, const float* gradients, float* prev_gradients, 
    float* delta_weights, float eta_plus, float eta_minus, 
    float delta_max, float delta_min, int size) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float grad = gradients[idx];
        float prev_grad = prev_gradients[idx];
        float delta = delta_weights[idx];

        // Apply Rprop update rule
        if (grad * prev_grad > 0) {
            // Same sign - increase step size
            delta = fminf(delta * eta_plus, delta_max);
            weights[idx] -= copysignf(delta, grad);
            prev_gradients[idx] = grad;
        } 
        else if (grad * prev_grad < 0) {
            // Sign changed - decrease step size
            delta = fmaxf(delta * eta_minus, delta_min);
            // Revert previous step
            weights[idx] += copysignf(delta, prev_grad);
            prev_gradients[idx] = 0; // Set to zero to avoid oscillation
        } 
        else {
            // First iteration or zero gradient
            weights[idx] -= copysignf(delta, grad);
            prev_gradients[idx] = grad;
        }

        // Store updated delta
        delta_weights[idx] = delta;
    }
}

#endif