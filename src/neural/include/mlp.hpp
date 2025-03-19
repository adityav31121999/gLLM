
/**
 * @file mlp.hpp
 * Header file for the Multi-layer Perceptron (MLP) class and its related functions.
 * This file contains the declaration of the MLP class, which is used to create
 * and manage a multi-layer perceptron neural network. The file also includes
 * necessary headers and dependencies required for the MLP class.
 * Dependencies:
 * - <maths.hpp>: For activation functions used in the neural network.
 * The MLP class provides methods to initialize the network, perform forward
 * propagation, and apply activation functions to the network layers.
 */

#ifndef MLP_HPP
#define MLP_HPP 1

#include <vector>
#include <maths.hpp>

/**
 * @brief Multi-layer Perceptron class (with No BIASES) specifically designed for LLMs
 */
class mlp {
public:
// member variables
    bool status;                // 1 if completely trained, 0 otherwise
// member containers
    std::vector<double> input;      // input vector
    std::vector<double> output;     // output vector
    std::vector<double> expected;   // expected output vectors
    std::vector<std::vector<std::vector<double>>> weights;      // weights for matrix layer (i + h + o)
    std::vector<std::vector<double>> hlayers;       // hidden layers
    std::vector<std::vector<double>> activations;   // activations for each layer
    std::vector<std::vector<std::vector<double>>> gweights;     // gradient of weights for matrix layer gradient(i + h + o)

// member functions
    // default constructor
    mlp() = default;
    mlp(unsigned int in, unsigned int layers, unsigned int epochs = 10, double learning = 0.01);

    void forward(int in, int layers);
    void backprop2in(int layers, int in, double learning);
    void backward(int layers, int in, double learning);
    void backprop(int layers, int in, double learning);
    void backwithL1(int layers, int in, double learning);
    void backwithL2(int layers, int in, double learning);
    void rprop(std::vector<std::vector<double>>&, int layers, int in, double learning, int epochs);
    void train(double& mse, int in, int layers, double learning);
    void train(std::vector<std::vector<double>>&, double& mse, int in, int layers, double learning);
    void validate(int in, int layers);
    void test(int in, int layers);
    void initializeWeights(int in, int layers);

    // default destructor
    ~mlp() = default;
};

// mlp-related functions
double getL1Penalty(std::vector<std::vector<std::vector<double>>>&);
double getL2Penalty(std::vector<std::vector<std::vector<double>>>&);
double computeLossWithL1(std::vector<double>&, std::vector<double>&, mlp&, double);
double computeLossWithL2(std::vector<double>&, std::vector<double>&, mlp&, double);
double dropoutGeneralisation(std::vector<double>&, std::vector<double>&, mlp&, double);

#endif
