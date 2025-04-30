/**
 * @file mlp.hpp
 * Header file for the Multi-layer Perceptron (MLP) class and its related functions.
 * This file contains the declaration of the MLP class, which is used to create
 * and manage a multi-layer perceptron neural network. The file also includes
 * necessary headers and dependencies required for the MLP class.
 * Dependencies:
 * - <maths.hpp>: For activation functions and the shared OpenCLContext.
 * The MLP class provides methods to initialize the network, perform forward
 * propagation, and apply activation functions to the network layers.
 */

#ifndef MLP_HPP
#define MLP_HPP 1

#include <maths.hpp> // This includes basic.hpp which defines OpenCLContext
#include <string>
#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>
#include <map>

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
    // default constructor - Deleted because clContext reference needs initialization
    // mlp() = default; // Cannot default construct if there's a reference member

    // Constructor(s) modified to accept OpenCLContext when needed
#ifdef USE_OPENCL
    OpenCLContext& clContext; // <-- THIS CALL TRIGGERS THE PROCESS
    // Constructor when OpenCL is enabled
    mlp(OpenCLContext& context, unsigned int in, unsigned int layers, unsigned int epochs = 10, float learning = 0.01);
#else
    // Constructor when OpenCL is disabled
    mlp(unsigned int in, unsigned int layers, unsigned int epochs = 10, float learning = 0.01);
#endif


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

#elif USE_OPENCL

// opencl implementation for mlp methods (will use clContext member)
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

#else

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

#endif
    void initializeWeights(int in, int layers);
    // default destructor
    ~mlp() = default; // Default destructor is fine
};

// mlp-related functions (non-member)
float getL1Penalty(std::vector<std::vector<std::vector<float>>>&);
float getL2Penalty(std::vector<std::vector<std::vector<float>>>&);
float computeLossWithL1(std::vector<float>&, std::vector<float>&, mlp&, float);
float computeLossWithL2(std::vector<float>&, std::vector<float>&, mlp&, float);
float dropoutGeneralisation(std::vector<float>&, std::vector<float>&, mlp&, float);
std::vector<float> flattenWeights(const std::vector<std::vector<std::vector<float>>>& weights);
std::vector<float> flatten(const std::vector<std::vector<float>>& vec2d);
std::vector<float> flatten(mat&);
void transposeFlattenMatrix(const std::vector<std::vector<float>>& input, std::vector<float>& output_flat, int rows, int cols);
void unflatten(const std::vector<float>& flat, std::vector<std::vector<float>>& vec2d, size_t rows, size_t cols);

#ifdef USE_CUDA

// cuda implementation (kernels remain the same)
    __global__ void matrixMultiplyKernel(const float* A, const float* B, float* C, int rowsA, int colsA, int colsB);
    __global__ void vectorAddKernel(const float* A, const float* B, float* C, int len);
    __global__ void l1PenaltyKernel(float* weights, float* result, int size);
    __global__ void l2PenaltyKernel(float* weights, float* result, int size);
    __global__ void absDiffKernel(float* outputs, float* targets, float* result, int size);
    __global__ void squaredDiffKernel(float* outputs, float* targets, float* result, int size);
    __global__ void cuMSEKernel(float* expected, float* output, float* mse, int size);
    __global__ void kernelOutputDelta(float* output, float* expected, float* delta, int size); // Note: CUDA likely uses specific name like kernelOutputDeltaSigmoid
    __global__ void hiddenDeltaKernel(float* next_layer_deltas, float* weights, float* activations,
            float* deltas, int current_layer_size, int next_layer_size); // Note: CUDA likely uses specific name like kernelHiddenDeltaSigmoid
    __global__ void kernelComputeGradMLPInput(const float* deltas, const float* weights, float* grad_input,
            int current_layer_size, int input_size);
    __global__ void kernelLastLayerDelta(const float* grad_output, const float* activations, float* deltas, int size); // Note: CUDA likely uses specific name like kernelLastLayerDeltaSigmoid
    __global__ void updateWeightsKernel(float* deltas, float* prev_activations, float* weights, float learning_rate,
            int current_layer_size, int prev_layer_size); // Note: CUDA likely uses specific name like kernelUpdateWeights
    __global__ void updateWeightsKernel(float* deltas, float* prev_activations, float* weights, float* gradients,
            float learning_rate, int current_layer_size, int prev_layer_size); // Note: CUDA likely uses specific name like kernelUpdateWeightsAndGradients
    __global__ void updateWeightsL1Kernel(float* weights, float* deltas, float* prev_activations, float learning_rate,
            float lambda, int current_layer_size, int prev_layer_size); // Note: CUDA likely uses specific name like kernelUpdateWeightsL1
    __global__ void updateWeightsL2Kernel(float* weights, float* deltas, float* prev_activations, float learning_rate,
            float lambda, int current_layer_size, int prev_layer_size); // Note: CUDA likely uses specific name like kernelUpdateWeightsL2
    __global__ void rpropUpdateKernel(float* weights, float* gradients, float* prev_gradients, float* delta_weights,
            float eta_plus, float eta_minus, float delta_max, float delta_min, int size); // Note: CUDA likely uses specific name like kernelRpropUpdate
    __global__ void updateInputVectorKernel(float* input, float* weights, float* deltas, float learning_rate, int size); // Note: CUDA likely uses specific name like kernelUpdateInputMLP
    __global__ void layerForwardKernel(float* inputs, float* weights, float* outputs,  int input_size, int output_size); // Note: CUDA likely uses specific name like kernelLayerForward
    float cugetL1Penalty(std::vector<std::vector<std::vector<float>>>&);
    float cugetL2Penalty(std::vector<std::vector<std::vector<float>>>&);
    float cucomputeLossWithL1(std::vector<float>&, std::vector<float>&, mlp&, float);
    float cucomputeLossWithL2(std::vector<float>&, std::vector<float>&, mlp&, float);
    float cudropoutGeneralisation(std::vector<float>&, std::vector<float>&, mlp&, float);

#elif USE_OPENCL

    float clgetL1Penalty(OpenCLContext& clContext, std::vector<std::vector<std::vector<float>>>& weights);
    float clgetL2Penalty(OpenCLContext& clContext, std::vector<std::vector<std::vector<float>>>& weights);
    float clcomputeLossWithL1(OpenCLContext& clContext, std::vector<float>& expected, std::vector<float>& output, mlp& network, float lambda);
    float clcomputeLossWithL2(OpenCLContext& clContext, std::vector<float>& expected, std::vector<float>& output, mlp& network, float lambda);
    float cldropoutGeneralisation(OpenCLContext& clContext, std::vector<float>& expected, std::vector<float>& output, mlp& network, float dropout_rate);

#endif // USE_CUDA / USE_OPENCL

#endif // MLP_HPP
