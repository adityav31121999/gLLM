
#include "include/mlp.hpp"
#include <stdexcept>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstring>

#ifndef USE_OPENCL

/**
 * @brief Constructor used when OpenCL is NOT enabled (in = out = neurons)
 * @param layerSizes Vector containing the number of neurons in each layer (e.g., {input_size, hidden1_size, output_size}).
 * @param epochs number of epochs for training (Note: epochs might be better handled in training loop)
 * @param learning learning rate for the network (Note: learning rate might be better handled in training loop)
 */
mlp::mlp(const std::vector<unsigned int>& layerSizes, unsigned int epochs, float learning, float lambda_L1, float lambda_L2)
    : status(false),
      layer_sizes(layerSizes),
      epochs(epochs),
      learning_rate(learning),
      lambda_l1(lambda_L1),
      lambda_l2(lambda_L2)
{
    if (layerSizes.size() < 2) {
        throw std::invalid_argument("MLP must have at least an input and an output layer (size >= 2).");
    }
    for(unsigned int size : layerSizes) {
        if (size == 0) {
            throw std::invalid_argument("MLP layer sizes must be positive.");
        }
    }

    num_layers = layerSizes.size();
    input.resize(layer_sizes[0], 0.0f);
    output.resize(layer_sizes.back(), 0.0f);
    expected.resize(layer_sizes.back(), 0.0f);

    activations.resize(num_layers);
    for(unsigned int i = 0; i < num_layers; ++i) {
        activations[i].resize(layer_sizes[i], 0.0f);
    }

    if (num_layers > 1) {
        hlayers.resize(num_layers - 1);
        for (unsigned int i = 0; i < num_layers - 1; ++i) {
            hlayers[i].resize(layer_sizes[i+1], 0.0f);
        }
    }

    weights.resize(num_layers - 1);
    gweights.resize(num_layers - 1);
    moments.resize(num_layers - 1);
    velocity.resize(num_layers - 1);
    for (unsigned int i = 0; i < num_layers - 1; ++i) {
        weights[i] = mat(layer_sizes[i+1], layer_sizes[i]);
        gweights[i] = mat(layer_sizes[i+1], layer_sizes[i]);
        moments[i] = mat(layer_sizes[i+1], layer_sizes[i]);
        velocity[i] = mat(layer_sizes[i+1], layer_sizes[i]);
    }

    params = 0;
    for (unsigned int i = 0; i < num_layers - 1; i++) {
        params += weights[i].row * weights[i].col;
    }
    params *= 4;
    params += (num_layers*activations[0].size()*2) + 3*input.size();
    // initializeAdamMoments();
    std::cout << "MLP constructed." << std::endl;
}

#else

/**
 * @brief Constructor used when OpenCL IS enabled (in = out = neurons)
 * @param context Reference to the shared OpenCL context object.
 * @param layerSizes Vector containing the number of neurons in each layer (input, hidden..., output).
 * @param epochs number of epochs for training (Note: epochs might be better handled in training loop)
 * @param learning learning rate for the network (Note: learning rate might be better handled in training loop)
 */
mlp::mlp(OpenCLContext& context, const std::vector<unsigned int>& layerSizes, unsigned int epochs, float learning, float lambda_L1, float lambda_L2)
    : clContext(context),
      status(false),
      layer_sizes(layerSizes),
      epochs(epochs),
      learning_rate(learning),
      lambda_l1(lambda_L1),
      lambda_l2(lambda_L2)
{
    // Validate inputs
    if (layerSizes.size() < 2) {
        throw std::invalid_argument("MLP must have at least an input and an output layer (size >= 2).");
    }
    for(unsigned int size : layerSizes) {
        if (size == 0) {
            throw std::invalid_argument("MLP layer sizes must be positive.");
        }
    }
    num_layers = layerSizes.size();

    input.resize(layer_sizes[0], 0.0f);
    output.resize(layer_sizes.back(), 0.0f);
    expected.resize(layer_sizes.back(), 0.0f);

    activations.resize(num_layers);
    for(unsigned int i = 0; i < num_layers; ++i) {
        activations[i].resize(layer_sizes[i], 0.0f);
    }

    if (num_layers > 1) {
        hlayers.resize(num_layers - 1);
        for (unsigned int i = 0; i < num_layers - 1; ++i) {
            hlayers[i].resize(layer_sizes[i+1], 0.0f);
        }
    }

    weights.resize(num_layers - 1);
    gweights.resize(num_layers - 1);
    moments.resize(num_layers - 1);
    velocity.resize(num_layers - 1);
    for (unsigned int i = 0; i < num_layers - 1; ++i) {
        weights[i] = mat(layer_sizes[i+1], layer_sizes[i]);
        gweights[i] = mat(layer_sizes[i+1], layer_sizes[i]);
        moments[i] = mat(layer_sizes[i+1], layer_sizes[i]);
        velocity[i] = mat(layer_sizes[i+1], layer_sizes[i]);
    }

    params = 0;
    for (unsigned int i = 0; i < num_layers - 1; i++) {
        params += weights[i].row * weights[i].col;
    }
    params *= 4;
    params += (num_layers*activations[0].size()*2) + 3*input.size();
    // initializeAdamMoments();
    std::cout << "MLP constructed with OpenCL." << std::endl;
}

#endif // USE_OPENCL

// inititalise adam moments and velocity for mlp
void mlp::initializeAdamMoments() {
    // moments.clear();
    // velocity.clear();
    for (size_t i = 0; i < weights.size(); ++i) {
        // Assuming weights[i] is already properly constructed/mapped
        // moments.emplace_back(weights[i].row, weights[i].col); // Create new mat for moment
        // velocity.emplace_back(weights[i].row, weights[i].col); // Create new mat for moment
        // Set all elements to zero for initialization
        for (int r = 0; r < moments.back().row; ++r) {
            for (int c = 0; c < moments.back().col; ++c) {
                moments.back()(r, c) = 0.0f;
                velocity.back()(r, c) = 0.0f;
            }
        }
    }
}

// clear all values
void mlp::clearValues() {
    std::fill(input.begin(), input.end(), 0.0f);
    std::fill(output.begin(), output.end(), 0.0f);
    std::fill(expected.begin(), expected.end(), 0.0f);

    // Clear 2D vectors (hlayers, activations - if they are still vectors)
    for (auto& layer : hlayers) {
        std::fill(layer.begin(), layer.end(), 0.0f);
    }
    for (auto& layer : activations) {
        std::fill(layer.begin(), layer.end(), 0.0f);
    }

    // Clear mat vectors (weights, gweights)
    for (auto& layer : weights) {
        if (layer.mapped_data && layer.mapped_size > 0) {
            memset(layer.mapped_data, 0, layer.mapped_size); // Efficiently zero out mapped memory
        }
    }
    for (auto& layer : gweights) {
        if (layer.mapped_data && layer.mapped_size > 0) {
            memset(layer.mapped_data, 0, layer.mapped_size); // Efficiently zero out mapped memory
        }
    }
}

/**
 * @brief serialise mlp weights and gweights to train bin file
 * @param locationWithFileName train .bin file name and location
 */
void mlp::serialise4train(const std::string& locationWithFileName) {
    std::ofstream outFile(locationWithFileName, std::ios::binary | std::ios::out);
    if (!outFile.is_open()) {
        throw std::runtime_error("Failed to open file for writing MLP: " + locationWithFileName);
    }

    // serialise each file
    for(size_t i = 0; i < weights.size(); ++i) { // Iterate based on actual size
        write2filefrommat(weights[i], locationWithFileName);
    }
    for(size_t i = 0; i < gweights.size(); ++i) { // Iterate based on actual size
        write2filefrommat(gweights[i], locationWithFileName);
    }

    outFile.close();
    if (!outFile) {
        throw std::runtime_error("MLP Serialization: Error occurred while closing file: " + locationWithFileName);
    }
}

/**
 * @brief serialise mlp weights to specific bin file for use
 * @param filename bin file name and location
 * @param layers number of layers of weights matrix
 * @param dimensions square dimension of each weights matrix
 */
void mlp::serialise4use(const std::string& locationWithFileName) {
    std::ofstream outFile(locationWithFileName, std::ios::binary | std::ios::out);
    if (!outFile.is_open()) {
        throw std::runtime_error("Failed to open file for writing MLP: " + locationWithFileName);
    }

    // serialise each file
    for(size_t i = 0; i < weights.size(); ++i) { // Iterate based on actual size
        write2filefrommat(weights[i], locationWithFileName);
    }

    outFile.close();
    if (!outFile) {
        throw std::runtime_error("MLP Serialization: Error occurred while closing file: " + locationWithFileName);
    }
}
