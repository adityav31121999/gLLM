
// mlp.cpp: constructor for mlp class
#include "include/mlp.hpp"
#include <stdexcept>


/**
 * @brief Constructor used in specifically for FFN in LLM
 * (in = out = neurons)
 * @param in input dimension, same for output dimension
 * @param layers number of layers
 * @param neurons number of neurons in each layer
 * @param epochs number of epochs for training
 * @param learning learning rate for the network
 */
mlp::mlp(unsigned int in, unsigned int layers, unsigned int epochs, double learning) {
    // all variables and containers
    this->dim = in;
    this->layers = layers;
    this->neurons = dim;
    this->epochs = epochs;
    this->learning = learning;
    input.resize(dim, 0.0);
    output.resize(dim, 0.0);
    expected.resize(dim, 0.0);
    weights.resize(layers, std::vector<std::vector<double>>(neurons, std::vector<double>(neurons, 0.0)));
    hlayers.resize(layers, std::vector<double>(neurons, 0.0));
    activations.resize(layers, std::vector<double>(neurons, 0.0));
    gweights.resize(layers, std::vector<std::vector<double>>(neurons, std::vector<double>(neurons, 0.0)));
    initializeWeights();
    totalParams = neurons * neurons * layers;
}
