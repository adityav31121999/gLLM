
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
mlp::mlp(unsigned int in, unsigned int layers, unsigned int epochs, float learning) {
    // all variables and containers
    input.resize(in, 0.0);
    output.resize(in, 0.0);
    expected.resize(in, 0.0);
    weights.resize(layers, std::vector<std::vector<float>>(in, std::vector<float>(in, 0.0)));
    hlayers.resize(layers, std::vector<float>(in, 0.0));
    activations.resize(layers, std::vector<float>(in, 0.0));
    gweights.resize(layers, std::vector<std::vector<float>>(in, std::vector<float>(in, 0.0)));
    initializeWeights(in, layers);
#ifdef USE_OPENCL
    // define the size of each buffer
    
#elif USE_CUDA
    // define the size of each pointer to buffer
#endif
}
