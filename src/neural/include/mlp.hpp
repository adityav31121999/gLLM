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
#include <map>

#include <string>
#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>
#include <map>

#ifdef USE_OPENCL
    #include <CL/cl.hpp>
    #include <map>
#endif

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
#elif USE_OPENCL
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program; // Holds the compiled program from clcompute.cl and others
    std::map<std::string, cl::Kernel> kernels; // Map to store kernel objects by name
    cl::Device default_device; // Store the device being used
// opencl implementation for mlp
    float clMSE(const std::vector<float>& expected_vec, const std::vector<float>& output_vec, int in);
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
    void CL_CHECK(cl_int err, const char* file, int line);
    cl_mem cl_create_buffer(cl_context context, cl_mem_flags flags, size_t size, void* host_ptr, cl_int& err);
    void cl_write_buffer(cl_command_queue queue, cl_mem buffer, size_t size, const void* ptr, cl_bool blocking = CL_TRUE);
    void cl_read_buffer(cl_command_queue queue, cl_mem buffer, size_t size, void* ptr, cl_bool blocking = CL_TRUE);
    void cl_fill_buffer(cl_command_queue queue, cl_mem buffer, const void* pattern, size_t pattern_size, size_t offset, size_t size);
    void cl_set_kernel_arg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void* arg_value);
    void cl_enqueue_nd_range_kernel(cl_command_queue queue, cl_kernel kernel, cl_uint work_dim, const size_t* global_work_offset, const size_t* global_work_size, const size_t* local_work_size);
    void cl_finish(cl_command_queue queue);
    void cl_release_mem_object(cl_mem memobj);
    #define CL_CHECK(err) CL_CHECK(err, __FILE__, __LINE__)
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
std::vector<float> flattenWeights(const std::vector<std::vector<std::vector<float>>>& weights);
std::vector<float> flatten(const std::vector<std::vector<float>>& vec2d);
std::vector<float> flatten(mat&);
void transposeFlattenMatrix(const std::vector<std::vector<float>>& input, std::vector<float>& output_flat, int rows, int cols);
void unflatten(const std::vector<float>& flat, std::vector<std::vector<float>>& vec2d, size_t rows, size_t cols);

#ifdef USE_CUDA
// cuda implementation
    __global__ void matrixMultiplyKernel(const float* A, const float* B, float* C, int rowsA, int colsA, int colsB);
    __global__ void vectorAddKernel(const float* A, const float* B, float* C, int len);
    __global__ void l1PenaltyKernel(float* weights, float* result, int size);
    __global__ void l2PenaltyKernel(float* weights, float* result, int size);
    __global__ void absDiffKernel(float* outputs, float* targets, float* result, int size);
    __global__ void squaredDiffKernel(float* outputs, float* targets, float* result, int size);
    __global__ void cuMSEKernel(float* expected, float* output, float* mse, int size);
    __global__ void kernelOutputDelta(float* output, float* expected, float* delta, int size);
    __global__ void hiddenDeltaKernel(float* next_layer_deltas, float* weights, float* activations, 
            float* deltas, int current_layer_size, int next_layer_size);
    __global__ void kernelComputeGradMLPInput(const float* deltas, const float* weights, float* grad_input,
            int current_layer_size, int input_size);
    __global__ void kernelLastLayerDelta(const float* grad_output, const float* activations, float* deltas, int size);
    __global__ void updateWeightsKernel(float* deltas, float* prev_activations, float* weights, float learning_rate, 
            int current_layer_size, int prev_layer_size);
    __global__ void updateWeightsKernel(float* deltas, float* prev_activations, float* weights, float* gradients, 
            float learning_rate, int current_layer_size, int prev_layer_size);
    __global__ void updateWeightsL1Kernel(float* weights, float* deltas, float* prev_activations, float learning_rate, 
            float lambda, int current_layer_size, int prev_layer_size);
    __global__ void updateWeightsL2Kernel(float* weights, float* deltas, float* prev_activations, float learning_rate, 
            float lambda, int current_layer_size, int prev_layer_size);
    __global__ void rpropUpdateKernel(float* weights, float* gradients, float* prev_gradients, float* delta_weights, 
            float eta_plus, float eta_minus, float delta_max, float delta_min, int size);
    __global__ void updateInputVectorKernel(float* input, float* weights, float* deltas, float learning_rate, int size);
    __global__ void layerForwardKernel(float* inputs, float* weights, float* outputs,  int input_size, int output_size);
    __global__ void kernelComputeGradMLPInput(const float* deltas, const float* weights, float* grad_input,
            int current_layer_size, int input_size);
    float cugetL1Penalty(std::vector<std::vector<std::vector<float>>>&);
    float cugetL2Penalty(std::vector<std::vector<std::vector<float>>>&);
    float cucomputeLossWithL1(std::vector<float>&, std::vector<float>&, mlp&, float);
    float cucomputeLossWithL2(std::vector<float>&, std::vector<float>&, mlp&, float);
    float cudropoutGeneralisation(std::vector<float>&, std::vector<float>&, mlp&, float);
#elif USE_OPENCL
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program; // Holds the compiled program from clcompute.cl and others
    std::map<std::string, cl::Kernel> kernels; // Map to store kernel objects by name
    cl::Device default_device; // Store the device being used
// opencl implementation
    float clgetL1Penalty(std::vector<std::vector<std::vector<float>>>&);
    float clgetL2Penalty(std::vector<std::vector<std::vector<float>>>&);
    float clcomputeLossWithL1(std::vector<float>&, std::vector<float>&, mlp&, float);
    float clcomputeLossWithL2(std::vector<float>&, std::vector<float>&, mlp&, float);
    float cldropoutGeneralisation(std::vector<float>&, std::vector<float>&, mlp&, float);
#endif

#endif
