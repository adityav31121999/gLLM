#include "include/mlp.hpp"
#include <stdexcept>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstring>

// --- Non-OpenCL Constructor ---
#ifndef USE_CL

/**
 * @brief Constructor used when OpenCL is NOT enabled.
 * (in = out = neurons)
 * @param layerSizes Vector containing the number of neurons in each layer (e.g., {input_size, hidden1_size, output_size}).
 * @param epochs number of epochs for training (Note: epochs might be better handled in training loop)
 * @param learning learning rate for the network (Note: learning rate might be better handled in training loop)
 */
mlp2d::mlp2d(const int inH, const int inW, const int outWidth, const std::vector<unsigned int>& layerSizes, unsigned int epochs, float learning)
    : status(false),
      layer_sizes(layerSizes),
      epochs(epochs),
      learning_rate(learning)
{
    if (layerSizes.size() < 2) {
        throw std::invalid_argument("mlp2d must have at least an input and an output layer (size >= 2).");
    }
    for(unsigned int size : layerSizes) {
        if (size == 0) {
            throw std::invalid_argument("mlp2d layer sizes must be positive.");
        }
    }
    num_layers = layerSizes.size();

    input.resize(inH, std::vector<float>(inW, 0.0f));
    output.resize(inH, std::vector<float>(outWidth, 0.0f));
    expected.resize(inH, std::vector<float>(outWidth, 0.0f));

    activations.resize(num_layers);

    for(unsigned int i = 0; i < num_layers; ++i) {
        activations[i].resize(inH, std::vector<float>(layer_sizes[i], 0.0f));
    }

    if (num_layers > 1) {
        hlayers.resize(num_layers - 1);
        for (unsigned int i = 0; i < num_layers - 1; ++i) {
            hlayers[i].resize(inH, std::vector<float>(layer_sizes[i], 0.0f));
        }
    }

    weights.resize(num_layers - 1);
    gweights.resize(num_layers - 1);
    for (unsigned int i = 0; i < num_layers - 1; ++i) {
        weights[i] = mat(layer_sizes[i+1], layer_sizes[i]);
        gweights[i] = mat(layer_sizes[i+1], layer_sizes[i]);
    }

    params = 0;
    for (unsigned int i = 0; i < num_layers - 1; i++) {
        params += weights[i].row * weights[i].col;
    }
    params *= 2;
    // params += (num_layers*activations[0].size()*2) + 3*input.size();
    // initializeAdamMoments();
    // std::cout << "mlp2d constructed." << std::endl;
}


/**
 * @brief Constructor used when OpenCL IS enabled.
 * (in = out = neurons)
 * @param context Reference to the shared OpenCL context object.
 * @param layerSizes Vector containing the number of neurons in each layer (input, hidden..., output).
 * @param epochs number of epochs for training (Note: epochs might be better handled in training loop)
 * @param learning learning rate for the network (Note: learning rate might be better handled in training loop)
 */
mlp2d::mlp2d(const int inH, const int inW, const int outWidth, const std::string& inAtt, const std::vector<unsigned int>& layerSizes, unsigned int epochs, float learning)
    : status(false),
      layer_sizes(layerSizes),
      epochs(epochs),
      learning_rate(learning)
{
    // Validate inputs
    if (layerSizes.size() < 2) {
        throw std::invalid_argument("mlp2d must have at least an input and an output layer (size >= 2).");
    }
    for(unsigned int size : layerSizes) {
        if (size == 0) {
            throw std::invalid_argument("mlp2d layer sizes must be positive.");
        }
    }
    num_layers = layerSizes.size();

    input.resize(inH, std::vector<float>(inW, 0.0f));
    output.resize(inH, std::vector<float>(outWidth, 0.0f));
    expected.resize(inH, std::vector<float>(outWidth, 0.0f));

    for(unsigned int i = 0; i < num_layers; ++i) {
        activations[i].resize(inH, std::vector<float>(layer_sizes[i], 0.0f));
    }

    if (num_layers > 1) {
        hlayers.resize(num_layers - 1);
        for (unsigned int i = 0; i < num_layers - 1; ++i) {
            hlayers[i].resize(inH, std::vector<float>(layer_sizes[i], 0.0f));
        }
    }

    weights.resize(num_layers - 1);
    gweights.resize(num_layers - 1);
    for (unsigned int i = 0; i < num_layers - 1; ++i) {
        std::string inmlp2d = inAtt + "mlp2dL" + std::to_string(i);
        weights[i] = mat(inmlp2d, layer_sizes[i+1], layer_sizes[i]);
        inmlp2d = inAtt + "mlp2dgL" + std::to_string(i);
        gweights[i] = mat(inmlp2d, layer_sizes[i+1], layer_sizes[i]);
    }

    params = 0;
    for (unsigned int i = 0; i < num_layers - 1; i++) {
        params += weights[i].row * weights[i].col;
    }
    params *= 2;
    // params += (num_layers*activations[0].size()*2 + 3*input.size());
    // initializeAdamMoments();
    // std::cout << "mlp2d constructed with OpenCL -> " << params << std::endl;
}

#else


/**
 * @brief Constructor used when OpenCL IS enabled.
 * (in = out = neurons)
 * @param context Reference to the shared OpenCL context object.
 * @param layerSizes Vector containing the number of neurons in each layer (input, hidden..., output).
 * @param epochs number of epochs for training (Note: epochs might be better handled in training loop)
 * @param learning learning rate for the network (Note: learning rate might be better handled in training loop)
 */
mlp2d::mlp2d(OpenCLContext& context, const int inH, const int inW, const int outWidth, const std::vector<unsigned int>& layerSizes, unsigned int epochs, float learning)
    : clContext(context),
      status(false),
      layer_sizes(layerSizes),
      epochs(epochs),
      learning_rate(learning)
{
    // Validate inputs
    if (layerSizes.size() < 2) {
        throw std::invalid_argument("mlp2d must have at least an input and an output layer (size >= 2).");
    }
    for(unsigned int size : layerSizes) {
        if (size == 0) {
            throw std::invalid_argument("mlp2d layer sizes must be positive.");
        }
    }
    num_layers = layerSizes.size();

    input.resize(inH, std::vector<float>(inW, 0.0f));
    output.resize(inH, std::vector<float>(outWidth, 0.0f));
    expected.resize(inH, std::vector<float>(outWidth, 0.0f));

    for(unsigned int i = 0; i < num_layers; ++i) {
        activations[i].resize(inH, std::vector<float>(layer_sizes[i], 0.0f));
    }

    if (num_layers > 1) {
        hlayers.resize(num_layers - 1);
        for (unsigned int i = 0; i < num_layers - 1; ++i) {
            hlayers[i].resize(inH, std::vector<float>(layer_sizes[i], 0.0f));
        }
    }

    weights.resize(num_layers - 1);
    gweights.resize(num_layers - 1);
    for (unsigned int i = 0; i < num_layers - 1; ++i) {
        weights[i] = mat(layer_sizes[i+1], layer_sizes[i]);
        gweights[i] = mat(layer_sizes[i+1], layer_sizes[i]);
    }

    params = 0;
    for (unsigned int i = 0; i < num_layers - 1; i++) {
        params += weights[i].row * weights[i].col;
    }
    params *= 2;
    // params += (num_layers*activations[0].size()*2 + 3*input.size());
    // initializeAdamMoments();
    // std::cout << "mlp2d constructed with OpenCL -> " << params << std::endl;
}

/**
 * @brief Constructor used when OpenCL IS enabled.
 * (in = out = neurons)
 * @param context Reference to the shared OpenCL context object.
 * @param layerSizes Vector containing the number of neurons in each layer (input, hidden..., output).
 * @param epochs number of epochs for training (Note: epochs might be better handled in training loop)
 * @param learning learning rate for the network (Note: learning rate might be better handled in training loop)
 */
mlp2d::mlp2d(OpenCLContext& context, const int inH, const int inW, const int outWidth, const std::string& inAtt, const std::vector<unsigned int>& layerSizes, unsigned int epochs, float learning)
    : clContext(context),
      status(false),
      layer_sizes(layerSizes),
      epochs(epochs),
      learning_rate(learning)
{
    // Validate inputs
    if (layerSizes.size() < 2) {
        throw std::invalid_argument("mlp2d must have at least an input and an output layer (size >= 2).");
    }
    for(unsigned int size : layerSizes) {
        if (size == 0) {
            throw std::invalid_argument("mlp2d layer sizes must be positive.");
        }
    }
    num_layers = layerSizes.size();

    input.resize(inH, std::vector<float>(inW, 0.0f));
    output.resize(inH, std::vector<float>(outWidth, 0.0f));
    expected.resize(inH, std::vector<float>(outWidth, 0.0f));

    for(unsigned int i = 0; i < num_layers; ++i) {
        activations[i].resize(inH, std::vector<float>(layer_sizes[i], 0.0f));
    }

    if (num_layers > 1) {
        hlayers.resize(num_layers - 1);
        for (unsigned int i = 0; i < num_layers - 1; ++i) {
            hlayers[i].resize(inH, std::vector<float>(layer_sizes[i], 0.0f));
        }
    }

    weights.resize(num_layers - 1);
    gweights.resize(num_layers - 1);
    for (unsigned int i = 0; i < num_layers - 1; ++i) {
        std::string inmlp2d = inAtt + "mlp2dL" + std::to_string(i);
        weights[i] = mat(inmlp2d, layer_sizes[i+1], layer_sizes[i]);
        inmlp2d = inAtt + "mlp2dgL" + std::to_string(i);
        gweights[i] = mat(inmlp2d, layer_sizes[i+1], layer_sizes[i]);
    }

    params = 0;
    for (unsigned int i = 0; i < num_layers - 1; i++) {
        params += weights[i].row * weights[i].col;
    }
    params *= 2;
    // params += (num_layers*activations[0].size()*2 + 3*input.size());
    // initializeAdamMoments();
    // std::cout << "mlp2d constructed with OpenCL -> " << params << std::endl;
}

#endif // USE_CL


// clear all values
void mlp2d::clearValues() {
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
 * @brief serialise mlp2d weights and gweights to train bin file
 * @param locationWithFileName train .bin file name and location
 */
void mlp2d::serialise4train(const std::string& locationWithFileName) {
    std::ofstream outFile(locationWithFileName, std::ios::binary | std::ios::out);
    if (!outFile.is_open()) {
        throw std::runtime_error("Failed to open file for writing mlp2d: " + locationWithFileName);
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
        throw std::runtime_error("mlp2d Serialization: Error occurred while closing file: " + locationWithFileName);
    }
}

/**
 * @brief serialise mlp2d weights to specific bin file for use
 * @param filename bin file name and location
 * @param layers number of layers of weights matrix
 * @param dimensions square dimension of each weights matrix
 */
void mlp2d::serialise4use(const std::string& locationWithFileName) {
    std::ofstream outFile(locationWithFileName, std::ios::binary | std::ios::out);
    if (!outFile.is_open()) {
        throw std::runtime_error("Failed to open file for writing mlp2d: " + locationWithFileName);
    }

    // serialise each file
    for(size_t i = 0; i < weights.size(); ++i) { // Iterate based on actual size
        write2filefrommat(weights[i], locationWithFileName);
    }

    outFile.close();
    if (!outFile) {
        throw std::runtime_error("mlp2d Serialization: Error occurred while closing file: " + locationWithFileName);
    }
}
