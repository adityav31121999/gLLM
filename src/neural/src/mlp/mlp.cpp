// mlp.cpp: constructor for mlp class
#include "include/mlp.hpp" // Includes basic.hpp which defines OpenCLContext if USE_OPENCL
#include <stdexcept>

// --- Non-OpenCL Constructor ---
#ifndef USE_OPENCL

/**
 * @brief Constructor used when OpenCL is NOT enabled.
 * (in = out = neurons)
 * @param in input dimension, same for output dimension
 * @param layers number of layers
 * @param epochs number of epochs for training (Note: epochs might be better handled in training loop)
 * @param learning learning rate for the network (Note: learning rate might be better handled in training loop)
 */
mlp::mlp(unsigned int in, unsigned int layers, unsigned int epochs /*= 10*/, float learning /*= 0.01*/)
    : status(false) // Initialize status
{
    // Validate inputs
    if (in == 0 || layers == 0) { // layers must be >= 1 for weights[0] to be valid
        throw std::invalid_argument("MLP dimensions (in, layers) must be positive.");
    }

    // Resize member containers
    input.resize(in, 0.0f);
    output.resize(in, 0.0f);
    expected.resize(in, 0.0f);
    // weights has size layers+1 (input->h1, h1->h2, ..., hN->output)
    weights.resize(layers + 1, std::vector<std::vector<float>>(in, std::vector<float>(in, 0.0f)));
    hlayers.resize(layers, std::vector<float>(in, 0.0f)); // Intermediate layer outputs (pre-activation)
    activations.resize(layers, std::vector<float>(in, 0.0f)); // Activated outputs of hidden layers
    // gweights should match weights dimensions
    gweights.resize(layers + 1, std::vector<std::vector<float>>(in, std::vector<float>(in, 0.0f)));

    // Initialize weights (e.g., with random values)
    initializeWeights(in, layers);

    // Note: epochs and learning rate are passed but not stored as members in the current design.
    // They are typically used by the training functions.
}

#else // USE_OPENCL is defined

// --- OpenCL Constructor ---

/**
 * @brief Constructor used when OpenCL IS enabled.
 * (in = out = neurons)
 * @param context Reference to the shared OpenCL context object.
 * @param in input dimension, same for output dimension
 * @param layers number of layers
 * @param epochs number of epochs for training (Note: epochs might be better handled in training loop)
 * @param learning learning rate for the network (Note: learning rate might be better handled in training loop)
 */
mlp::mlp(OpenCLContext& context, unsigned int in, unsigned int layers, unsigned int epochs /*= 10*/, float learning /*= 0.01*/)
    : clContext(context), // Initialize the OpenCL context reference member
      status(false)       // Initialize status
{
    // Validate inputs
    if (in == 0 || layers == 0) { // layers must be >= 1 for weights[0] to be valid
        throw std::invalid_argument("MLP dimensions (in, layers) must be positive.");
    }

    // Resize member containers (same as non-OpenCL version)
    input.resize(in, 0.0f);
    output.resize(in, 0.0f);
    expected.resize(in, 0.0f);
    // weights has size layers+1 (input->h1, h1->h2, ..., hN->output)
    weights.resize(layers + 1, std::vector<std::vector<float>>(in, std::vector<float>(in, 0.0f)));
    hlayers.resize(layers, std::vector<float>(in, 0.0f)); // Intermediate layer outputs (pre-activation)
    activations.resize(layers, std::vector<float>(in, 0.0f)); // Activated outputs of hidden layers
    // gweights should match weights dimensions
    gweights.resize(layers + 1, std::vector<std::vector<float>>(in, std::vector<float>(in, 0.0f)));

    // Initialize weights (e.g., with random values)
    initializeWeights(in, layers);

    // Note: epochs and learning rate are passed but not stored as members in the current design.
    // They are typically used by the training functions.

    // Optional: Add a check or log message confirming OpenCL context is received
    std::cout << "MLP constructed with OpenCL context using device: "
              << clContext.device.getInfo<CL_DEVICE_NAME>() << std::endl;
}

#endif // USE_OPENCL

// clear all values
void mlp::clearValues() {
    std::fill(input.begin(), input.end(), 0.0f);
    std::fill(output.begin(), output.end(), 0.0f);
    std::fill(expected.begin(), expected.end(), 0.0f);

    // Clear 2D vectors
    for (auto& layer : hlayers) {
        std::fill(layer.begin(), layer.end(), 0.0f);
    }
    for (auto& layer : activations) {
        std::fill(layer.begin(), layer.end(), 0.0f);
    }

    // Clear 3D vectors
    for (auto& layer : weights) {
        for (auto& matrix : layer) {
            std::fill(matrix.begin(), matrix.end(), 0.0f);
        }
    }
    for (auto& layer : gweights) {
        for (auto& matrix : layer) {
            std::fill(matrix.begin(), matrix.end(), 0.0f);
        }
    }
}
