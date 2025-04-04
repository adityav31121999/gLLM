
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
    std::vector<float> input;      // input vector
    std::vector<float> output;     // output vector
    std::vector<float> expected;   // expected output vectors
    std::vector<std::vector<std::vector<float>>> weights;      // weights for matrix layer (i + h + o)
    std::vector<std::vector<float>> hlayers;       // hidden layers
    std::vector<std::vector<float>> activations;   // activations for each layer
    std::vector<std::vector<std::vector<float>>> gweights;     // gradient of weights for matrix layer gradient(i + h + o)

// member functions
    // default constructor
    mlp() = default;
    mlp(unsigned int in, unsigned int layers, unsigned int epochs = 10, float learning = 0.01);

    void forward(int in, int layers);
    void backward(int layers, int in, float learning);
    void backprop(int layers, int in, float learning);
    void backwithL1(int layers, int in, float learning);
    void backwithL2(int layers, int in, float learning);
    void backprop2in(int layers, int in, float learning);
    void rprop(std::vector<std::vector<float>>&, int layers, int in, float learning, int epochs);
    void train(float& mse, int in, int layers, float learning);
    void train(std::vector<std::vector<float>>&, float& mse, int in, int layers, float learning);
    void validate(int in, int layers);
    void test(int in, int layers);
    void initializeWeights(int in, int layers);

#ifdef USE_CUDA

    // cuda implementation for mlp
    void cuForward(int in, int layers);
    void cuBackward(int layers, int in, float learning);
    void cuBackprop(int layers, int in, float learning);
    void cuBackwithL1(int layers, int in, float learning);
    void cuBackwithL2(int layers, int in, float learning);
    void cuBackprop2in(int layers, int in, float learning);
    void cuRprop(std::vector<std::vector<float>>&, int layers, int in, float learning, int epochs);
    void cuTrain(float& mse, int in, int layers, float learning);
    void cuTrain(std::vector<std::vector<float>>&, float& mse, int in, int layers, float learning);
    void cuValidate(int in, int layers);
    void cuTest(int in, int layers);
    float cugetL1Penalty(std::vector<std::vector<std::vector<float>>>&);
    float cugetL2Penalty(std::vector<std::vector<std::vector<float>>>&);
    float cucomputeLossWithL1(std::vector<float>&, std::vector<float>&, mlp&, float);
    float cucomputeLossWithL2(std::vector<float>&, std::vector<float>&, mlp&, float);
    float cudropoutGeneralisation(std::vector<float>&, std::vector<float>&, mlp&, float);

#elif USE_OPENCL

    // opencl implementation for mlp
    void clForward(int in, int layers);
    void clBackward(int layers, int in, float learning);
    void clBackprop(int layers, int in, float learning);
    void clBackwithL1(int layers, int in, float learning);
    void clBackwithL2(int layers, int in, float learning);
    void clBackprop2in(int layers, int in, float learning);
    void clRprop(std::vector<std::vector<float>>&, int layers, int in, float learning, int epochs);
    void clTrain(float& mse, int in, int layers, float learning);
    void clTrain(std::vector<std::vector<float>>&, float& mse, int in, int layers, float learning);
    void clValidate(int in, int layers);
    void clTest(int in, int layers);
    float clgetL1Penalty(std::vector<std::vector<std::vector<float>>>&);
    float clgetL2Penalty(std::vector<std::vector<std::vector<float>>>&);
    float clcomputeLossWithL1(std::vector<float>&, std::vector<float>&, mlp&, float);
    float clcomputeLossWithL2(std::vector<float>&, std::vector<float>&, mlp&, float);
    float cldropoutGeneralisation(std::vector<float>&, std::vector<float>&, mlp&, float);

#endif

    // default destructor
    ~mlp() = default;
};

// mlp-related functions
float getL1Penalty(std::vector<std::vector<std::vector<float>>>&);
float getL2Penalty(std::vector<std::vector<std::vector<float>>>&);
float computeLossWithL1(std::vector<float>&, std::vector<float>&, mlp&, float);
float computeLossWithL2(std::vector<float>&, std::vector<float>&, mlp&, float);
float dropoutGeneralisation(std::vector<float>&, std::vector<float>&, mlp&, float);

#endif
