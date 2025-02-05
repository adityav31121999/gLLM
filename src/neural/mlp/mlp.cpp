
// mlp.cpp: constructor for mlp class
#include "include/mlp.hpp"
#include <iostream>
#include <stdexcept>

/**
 * @brief Default constructor for the mlp class. This constructor initializes the
 * multi-layer perceptron with the given parameters.
 * @param in number of inputs
 * @param out number of outputs
 * @param layers number of layers
 * @param neurons number of neurons in each layer
 * @param epochs number of epochs for training
 * @param learning learning rate for the network
 */
mlp::mlp(unsigned int in, unsigned int out, unsigned int epochs, double learning)
{
    // all variables and containers
    this->in = in;
    this->out = out;
    layers = in + out;
    neurons = in * out;
    this->epochs = epochs;
    this->learning = learning;
    input.resize(in, 0.0);
    output.resize(out, 0.0);
    expected.resize(out, 0.0);
    iweights.resize(neurons, std::vector<double>(in, 0.0));
    oweights.resize(out, std::vector<double>(neurons, 0.0));
    weights.resize(layers - 1, std::vector<std::vector<double>>(neurons, std::vector<double>(neurons, 0.0)));
    hlayers.resize(layers, std::vector<double>(neurons, 0.0));
    activations.resize(layers, std::vector<double>(neurons, 0.0));
    giweights.resize(neurons, std::vector<double>(in, 0.0));
    goweights.resize(out, std::vector<double>(neurons, 0.0));
    gweights.resize(layers - 1, std::vector<std::vector<double>>(neurons, std::vector<double>(neurons, 0.0)));
    initializeWeights();
}


/**
 * @brief Constructor for the mlp class. This constructor initializes the
 * multi-layer perceptron with the given parameters and sets the input,
 * expected output and output vectors.
 * @param input input vector
 * @param expected expected output vector
 * @param output output vector
 * @param layers number of layers
 * @param neurons number of neurons in each layer
 * @param epochs number of epochs for training
 * @param learning learning rate for the network
 */
mlp::mlp(std::vector<double> input, std::vector<double> expected, std::vector<double> output,
        unsigned int epochs, double learning) 
{
    if(expected.size() != output.size())
        throw std::runtime_error("-_-SIZE OF OUTPUT AND EXPECTED SHOULD MATCH-_-");
    // all variables and containers
    this->input = input;
    this->expected = expected;
    this->output = output;
    in = input.size();
    out = output.size();
    this->layers = in + out;
    this->neurons = in * out;
    this->epochs = epochs;
    this->learning = learning;
    iweights.resize(neurons, std::vector<double>(in, 0.0));
    oweights.resize(out, std::vector<double>(neurons, 0.0));
    weights.resize(layers - 1, std::vector<std::vector<double>>(neurons, std::vector<double>(neurons, 0.0)));
    hlayers.resize(layers, std::vector<double>(neurons, 0.0));
    activations.resize(layers, std::vector<double>(neurons, 0.0));
    giweights.resize(neurons, std::vector<double>(in, 0.0));
    goweights.resize(out, std::vector<double>(neurons, 0.0));
    gweights.resize(layers - 1, std::vector<std::vector<double>>(neurons, std::vector<double>(neurons, 0.0)));
    initializeWeights();
}
