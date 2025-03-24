
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
    // default destructor
    ~mlp() = default;
};

// mlp-related functions
float getL1Penalty(std::vector<std::vector<std::vector<float>>>&);
float getL2Penalty(std::vector<std::vector<std::vector<float>>>&);
float computeLossWithL1(std::vector<float>&, std::vector<float>&, mlp&, float);
float computeLossWithL2(std::vector<float>&, std::vector<float>&, mlp&, float);
float dropoutGeneralisation(std::vector<float>&, std::vector<float>&, mlp&, float);



#ifdef USE_CUDA

    // cuda equivalent functions for mlp
    __global__ void forwardcu(const float* input, float* output, float* weights, float* hlayers, float* activations, int in, int layers, 
                                int input_size, int* layer_offsets, int* weight_offsets);
    __global__ void backpropcu(const float* in, const float* out, const float* w, const float* h, const float* a, int i, int l, float learning);
    __global__ void backprop2incu(const float* in, const float* out, const float* w, const float* h, const float* a, int i, int l, float learning);
    __global__ void backpropL1cu(const float* in, const float* out, const float* w, const float* h, const float* a, int i, int l, float learning);
    __global__ void backpropL2cu(const float* in, const float* out, const float* w, const float* h, const float* a, int i, int l, float learning);
    __global__ void rpropcu(const float* in, const float* out, const float* w, const float* h, const float* a, int i, int l, float learning, int epochs);
    __global__ void traincu(const float* in, const float* out, const float* w, const float* h, const float* a, float* mse, int i, int l, float learning);
    __global__ void train2cu(const float* in, const float* out, const float* w, const float* h, const float* a, float* mse, int i, int l, float learning);
    __global__ void validatecu(const float* in, const float* out, const float* w, const float* h, const float* a, int i, int l);
    __global__ void testcu(const float* in, const float* out, const float* w, const float* h, const float* a, int i, int l);
    __global__ void L1Penaltycu(const float* w, float* penalty, int i, int l);
    __global__ void L2Penaltycu(const float* w, float* penalty, int i, int l);

#elif USE_OPENCL

    // opencl equivalent functions for mlp
    __kernel void forwardcl(__global float* in, __global float* out, __global float* w, __global float* h, __global float* a, int i, int l);
    __kernel void backpropcl(__global float* in, __global float* out, __global float* w, __global float* h, __global float* a, int i, int l, float learning);
    __kernel void backprop2incl(__global float* in, __global float* out, __global float* w, __global float* h, __global float* a, int i, int l, float learning);
    __kernel void backpropL1cl(__global float* in, __global float* out, __global float* w, __global float* h, __global float* a, int i, int l, float learning);
    __kernel void backpropL2cl(__global float* in, __global float* out, __global float* w, __global float* h, __global float* a, int i, int l, float learning);
    __kernel void rpropcl(__global float* in, __global float* out, __global float* w, __global float* h, __global float* a, int i, int l, float learning, int epochs);
    __kernel void traincl(__global float* in, __global float* out, __global float* w, __global float* h, __global float* a, __global float* mse, int i, int l, float learning);
    __kernel void train2cl(__global float* in, __global float* out, __global float* w, __global float* h, __global float* a, __global float* mse, int i, int l, float learning);
    __kernel void validatecl(__global float* in, __global float* out, __global float* w, __global float* h, __global float* a, int i, int l);
    __kernel void testcl(__global float* in, __global float* out, __global float* w, __global float* h, __global float* a, int i, int l);
    __kernel void L1Penaltycl(__global float* w, __global float* penalty, int i, int l);
    __kernel void L2Penaltycl(__global float* w, __global float* penalty, int i, int l);

#endif


#endif
