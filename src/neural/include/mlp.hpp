#ifndef MLP_HPP
#define MLP_HPP 1
#include <maths.hpp>
#include <string>
#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>
#include <map>

#ifdef USE_CU
#include <cuda_runtime.h>
#include "include/cppsup.hpp"
#elif USE_CL
#include <CL/cl.h>
#include "include/clsup.hpp"
#endif

#define LAYERS_MLP 4                        // layers of mlp
#define LEARNING_MAX 0.01f                  // maximum learning rate allowable
#define LEARNING_MIN 0.00001f               // minimum learning rate allowable
#define LR_PATIENCE 10                      // for learning rate adaptability by plateau
#define MAX_GRAD_CLIP 0.5f                  // maximum gradient clipping
#define EPOCHS 100                          // number of epochs for training of token

/**
 * @brief Multi-layer Perceptron class (with No BIASES) specifically designed for LLMs
 */
class mlp {
public:
// member variables
    bool status;                // 1 if completely trained, 0 otherwise
    unsigned int in;            // input size
    unsigned int out;           // output size
    unsigned int num_layers;    // Total number of layers (including input and output)
    std::vector<unsigned int> layer_sizes;      // Number of neurons in each layer
    unsigned int epochs;        // Training epochs

    float learning_rate;        // Learning rate
    float lambda_l1;            // lambda for lasso
    float lambda_l2;            // lambda for ridge

    std::vector<mat> weights;       // Weight matrices between layers (using memory-mapped mat)
    std::vector<std::vector<float>> normGamma;   // gamma for layer normalisation
    std::vector<std::vector<float>> normBeta;    // beta for layer normalisation

// member containers
    std::vector<float> input;      // input vector
    std::vector<float> output;     // output vector
    std::vector<float> expected;   // expected output vectors
    std::vector<std::vector<float>> hlayers;        // hidden layers (intermediate, might stay in RAM)
    std::vector<std::vector<float>> activations;    // activations for each layer
    std::vector<std::vector<float>> normalised;     // nomrlised (activations) or whatever
    std::vector<mat> gweights;     // Gradient matrices corresponding to weights (using memory-mapped mat)
    int params;                    // parameters in mlp

    // member containers
    std::vector<std::vector<float>> inBatch;        // input vector
    std::vector<std::vector<float>> outBatch;       // output vector
    std::vector<std::vector<float>> expBatch;       // expected output vectors
    std::vector<std::vector<std::vector<float>>> hBatch;    // hidden layers (intermediate, might stay in RAM)
    std::vector<std::vector<std::vector<float>>> actBatch;  // activations for each layer
    std::vector<std::vector<mat>> gwBatch;          // Gradient matrices corresponding to weights (using memory-mapped mat)

    // Constructor(s) modified to accept OpenCLContext when needed
#ifdef USE_CL
    OpenCLContext& clContext;
    // Constructor when OpenCL is enabled
    mlp() = default;
    mlp(OpenCLContext& context, const std::vector<unsigned int>& layerSizes, unsigned int epochs = 10, float learning = 0.01);
    mlp(OpenCLContext& context, const std::string& inBlock, const std::vector<unsigned int>& layerSizes, unsigned int epochs = 10, float learning = 0.01);
#else
    mlp() = default;
    // Constructor when OpenCL is disabled
    mlp(const std::vector<unsigned int>& layerSizes, unsigned int epochs = 10, float learning = 0.01);
    mlp(const std::string& inBlock, const std::vector<unsigned int>& layerSizes, unsigned int epochs = 10, float learning = 0.01);
#endif

    // Explicitly define copy constructor and copy assignment operator
    mlp(const mlp& other);
    mlp& operator=(const mlp& other);

#ifdef USE_CU

// cuda implementation for mlp
    void cuForward();
    void cuBackward(float learning);
    void cuBackprop(float learning);
    void cuBackwithL1(float learning);
    void cuBackwithL2(float learning);
    void cuBackwithELasticNet(float learning);
    void cuBackprop2in(float learning);
    void cuRprop(std::vector<std::vector<float>>&, float learning, int epochs);
    void cuTrain(float& mse,float learning);
    void cuTrain(std::vector<std::vector<float>>&, float& mse, float learning);

#elif USE_CL

// opencl implementation for mlp methods (will use clContext member)
    void clForward();
    void clBackward(float learning);
    void clBackprop(float learning);
    void clBackwithL1(float learning);
    void clBackwithL2(float learning);
    void clBackwithElasticNet(float learning);
    void clBackprop2in(float learning);
    void clRprop(std::vector<std::vector<float>>&, float learning, int epochs);
    void clTrain(float& mse, float learning);
    void clTrain(std::vector<std::vector<float>>&, float& mse, float learning);

#else

    void forward();
    void backward(float learning);
    void backprop(float learning);
    void backwithL1(float learning);
    void backwithL2(float learning);
    void backwithElastic(float learning);
    void backprop2in(float learning);
    void rprop(std::vector<std::vector<float>>&, float learning, int epochs);
    void train(float& mse, float learning);
    void train(std::vector<std::vector<float>>&, float& mse, float learning);

#endif

    void initializeWeights();

    // New method for initializing weights from a shared map region
    void initializeWeightsFromSharedMap(
        MappedFile* shared_handle, float* shared_base_ptr,
        const std::string& path_to_shared_file,
        size_t& current_byte_offset_in_shared_map, // Pass offset by reference to update it
        bool training_mode_for_mlp) { 

        weights.clear(); // Clear any existing weights
        if (training_mode_for_mlp) gweights.clear();

        for (size_t i = 0; i < num_layers - 1; ++i) {
            unsigned int rows = layer_sizes[i + 1];
            unsigned int cols = layer_sizes[i];
            
            weights.emplace_back(); // Add a default-constructed mat
            weights.back().assign_shared_segment(shared_handle, shared_base_ptr,
                                               current_byte_offset_in_shared_map,
                                               rows, cols, path_to_shared_file);
            current_byte_offset_in_shared_map += static_cast<size_t>(rows) * cols * sizeof(float);
        }
    }

    void clearValues();
    void serialise(unsigned long long offset, const std::string& locationWithFilename);
    void deserialise(unsigned long long offset, const std::string& locationWithFilename);
    void serialise4train(const std::string& locationWithFileName);
    void serialise4use(const std::string& locationWithFileName);
    
    ~mlp() = default; // Default destructor is fine
};


/**
 * @brief Multi-layer Perceptron class (with No BIASES) with 2D input-output 
 *  specifically designed for LLMs
 */
class mlp2d {
public:
// member variables
    bool status;                // 1 if completely trained, 0 otherwise
    int inHeight;               // input height
    int inWidth;                // input width
    unsigned int num_layers;    // Total number of layers (including input and output)
    std::vector<unsigned int> layer_sizes;      // Number of neurons in each layer
    int outWidth;               // output width
    unsigned int epochs;        // Training epochs
    float learning_rate;        // Learning rate
    float lambda_l1;            // L1 regularization parameter
    float lambda_l2;            // L2 regularization parameter
    unsigned int t;             // Time step for Adam (number of updates), initialized to 0

    std::vector<mat> weights;      // Weight matrices between layers (using memory-mapped mat)

// member containers
    std::vector<std::vector<float>> input;      // input vector
    std::vector<std::vector<float>> output;     // output vector
    std::vector<std::vector<float>> expected;   // expected output vectors
    std::vector<std::vector<std::vector<float>>> hlayers;       // hidden layers (intermediate, might stay in RAM)
    std::vector<std::vector<std::vector<float>>> activations;   // activations for each layer
    std::vector<mat> gweights;     // Gradient matrices corresponding to weights (using memory-mapped mat)
    int params;                    // parameters in mlp

    std::vector<std::vector<std::vector<float>>> inBatch;       // input vector
    std::vector<std::vector<std::vector<float>>> outBatch;      // output vector
    std::vector<std::vector<std::vector<float>>> expBatch;      // expected output vectors
    std::vector<std::vector<std::vector<std::vector<float>>>> hBatch;       // hidden layers (intermediate, might stay in RAM)
    std::vector<std::vector<std::vector<std::vector<float>>>> actBatch;     // activations for each layer
    std::vector<std::vector<mat>> gwBatch;                      // Gradient matrices corresponding to weights (using memory-mapped mat)

#ifdef USE_CL
    OpenCLContext& clContext;
    mlp2d() = default;
    mlp2d(OpenCLContext& context, const int inH, const int inW, const int outWidth, const std::vector<unsigned int>& layerSizes,
        unsigned int epochs = 10, float learning = 0.01);
    mlp2d(OpenCLContext& context, const int inH, const int inW, const int outWidth, const std::string& inBlock,
        const std::vector<unsigned int>& layerSizes, unsigned int epochs = 10, float learning = 0.01);
#else
    mlp2d() = default;
    mlp2d(const int inH, const int inW, const int outWidth, const std::vector<unsigned int>& layerSizes, unsigned int epochs = 10,
        float learning = 0.01);
    mlp2d(const int inH, const int inW, const int outWidth, const std::string& inBlock, const std::vector<unsigned int>& layerSizes,
        unsigned int epochs = 10, float learning = 0.01);
#endif

    // Explicitly define copy constructor and copy assignment operator
    mlp2d(const mlp2d& other);
    mlp2d& operator=(const mlp2d& other);

#ifdef USE_CU

// cuda implementation for mlp
    void cuForward();
    void cuBackward(float learning);
    void cuBackprop(float learning);
    void cuBackwithL1(float learning);
    void cuBackwithL2(float learning);
    void cuBackwithELasticNet(float learning);
    void cuBackprop2in(float learning);
    void cuRprop(std::vector<std::vector<float>>&, float learning, int epochs);
    void cuTrain(float& mse, float learning);
    void cuTrain(std::vector<std::vector<float>>&, float& mse, float learning);

#elif USE_CL

// opencl implementation for mlp methods (will use clContext member)
    void clForward();
    void clBackward(float learning);
    void clBackprop(float learning);
    void clBackwithL1(float learning);
    void clBackwithL2(float learning);
    void clBackwithElasticNet(float learning);
    void clBackprop2in(float learning);
    void clRprop(std::vector<std::vector<std::vector<std::vector<float>>>>&, float learning, int epochs);
    void clTrain(float& mse, float learning);
    void clTrain(std::vector<std::vector<std::vector<std::vector<float>>>>&, float& mse, float learning);

#else

    void forward();
    void backward(float learning);
    void backprop(float learning);
    void backwithL1(float learning);
    void backwithL2(float learning);
    void backwithElastic(float learning);
    void backprop2in(float learning);
    void rprop(std::vector<std::vector<std::vector<float>>>&, float learning, int epochs);
    void train(float& mse, float learning);
    void train(std::vector<std::vector<std::vector<float>>>&, float& mse, float learning);

#endif

    void initializeWeights();

    // New method for initializing weights from a shared map region
    void initializeWeightsFromSharedMap(
        MappedFile* shared_handle, float* shared_base_ptr,
        const std::string& path_to_shared_file,
        size_t& current_byte_offset_in_shared_map, // Pass offset by reference to update it
        bool training_mode_for_mlp) { 

        weights.clear(); // Clear any existing weights
        if (training_mode_for_mlp) gweights.clear();

        for (size_t i = 0; i < num_layers - 1; ++i) {
            unsigned int rows = layer_sizes[i + 1];
            unsigned int cols = layer_sizes[i];
            
            weights.emplace_back(); // Add a default-constructed mat
            weights.back().assign_shared_segment(shared_handle, shared_base_ptr,
                                               current_byte_offset_in_shared_map,
                                               rows, cols, path_to_shared_file);
            current_byte_offset_in_shared_map += static_cast<size_t>(rows) * cols * sizeof(float);
        }
    }

    void clearValues();
    void serialise(unsigned long long offset, const std::string& locationWithFilename);
    void deserialise(unsigned long long offset, const std::string& locationWithFilename);
    void serialise4train(const std::string& locationWithFileName);
    void serialise4use(const std::string& locationWithFileName);
    
    ~mlp2d() = default; // Default destructor is fine
};

// -------------------------------
//          definitions
// -------------------------------

// Inline implementations for copy constructor and copy assignment operator
inline mlp::mlp(const mlp& other) :
#ifdef USE_CL
    clContext(other.clContext), // Initialize OpenCL context reference
#endif
    status(other.status),
    num_layers(other.num_layers),
    layer_sizes(other.layer_sizes),
    epochs(other.epochs),
    learning_rate(other.learning_rate),
    input(other.input),
    output(other.output),
    expected(other.expected),
    weights(other.weights),         // Relies on mat's copy constructor and std::vector's copy constructor
    hlayers(other.hlayers),         // std::vector copy constructor
    activations(other.activations), // std::vector copy constructor
    gweights(other.gweights),       // Relies on mat's copy constructor and std::vector's copy constructor
    params(other.params)
{}

inline mlp& mlp::operator=(const mlp& other) {
    if (this == &other) {
        return *this; // Self-assignment check
    }

#ifdef USE_CL
    // Assign to the OpenCLContext object clContext refers to.
    // This requires OpenCLContext to be assignable (which we made it).
    clContext = other.clContext;
#endif

    status = other.status;
    num_layers = other.num_layers;
    layer_sizes = other.layer_sizes;
    epochs = other.epochs;
    learning_rate = other.learning_rate;
    input = other.input;
    output = other.output;
    expected = other.expected;
    weights = other.weights;         // Relies on mat's assignment operator and std::vector's assignment
    hlayers = other.hlayers;         // std::vector assignment
    activations = other.activations; // std::vector assignment
    gweights = other.gweights;       // Relies on mat's assignment operator and std::vector's assignment
    params = other.params;

    return *this;
}


// Inline implementations for copy constructor and copy assignment operator
inline mlp2d::mlp2d(const mlp2d& other) :
#ifdef USE_CL
    clContext(other.clContext), // Initialize OpenCL context reference
#endif
    status(other.status),
    num_layers(other.num_layers),
    layer_sizes(other.layer_sizes),
    epochs(other.epochs),
    learning_rate(other.learning_rate),
    input(other.input),
    output(other.output),
    expected(other.expected),
    weights(other.weights),         // Relies on mat's copy constructor and std::vector's copy constructor
    hlayers(other.hlayers),         // std::vector copy constructor
    activations(other.activations), // std::vector copy constructor
    gweights(other.gweights),       // Relies on mat's copy constructor and std::vector's copy constructor
    params(other.params)
{}

inline mlp2d& mlp2d::operator=(const mlp2d& other) {
    if (this == &other) {
        return *this; // Self-assignment check
    }

#ifdef USE_CL
    // Assign to the OpenCLContext object clContext refers to.
    // This requires OpenCLContext to be assignable (which we made it).
    clContext = other.clContext;
#endif

    status = other.status;
    num_layers = other.num_layers;
    layer_sizes = other.layer_sizes;
    epochs = other.epochs;
    learning_rate = other.learning_rate;
    input = other.input;
    output = other.output;
    expected = other.expected;
    weights = other.weights;         // Relies on mat's assignment operator and std::vector's assignment
    hlayers = other.hlayers;         // std::vector assignment
    activations = other.activations; // std::vector assignment
    gweights = other.gweights;       // Relies on mat's assignment operator and std::vector's assignment
    params = other.params;

    return *this;
}

// -------------------------------
//          declarations
// -------------------------------

// mlp-related functions (non-member)

std::vector<float> flattenWeights(const std::vector<mat>& weights);
void write2filefrommlp(const mlp&, const std::string&);
void write2filefrommlp(const mlp2d&, const std::string&);

float getL1Penalty(const std::vector<mat>& weights);
float getL2Penalty(const std::vector<mat>& weights);

float computeLossWithL1(std::vector<float>&, std::vector<float>&, mlp&, float);
float computeLossWithL2(std::vector<float>&, std::vector<float>&, mlp&, float);
float computeLossWithElasticNet(std::vector<float>& outputs, std::vector<float>& targets, mlp& network, float lambda_l1, float lambda_l2);
float dropoutGeneralisation(std::vector<float>&, std::vector<float>&, mlp&, float);

float computeLossWithL1(std::vector<float>&, std::vector<float>&, mlp2d&, float);
float computeLossWithL2(std::vector<float>&, std::vector<float>&, mlp2d&, float);
float computeLossWithElasticNet(std::vector<float>& outputs, std::vector<float>& targets, mlp2d& network, float lambda_l1, float lambda_l2);
float dropoutGeneralisation(std::vector<float>&, std::vector<float>&, mlp2d&, float);

#ifdef USE_CU

#include <cuda.h>
#include <cuda_runtime.h>

// cuda implementation (kernels remain the same)

    __global__ void vectorAddKernel(const float* A, const float* B, float* C, int len);
    __global__ void matrixMultiplyKernel(const float* A, const float* B, float* C, int rowsA, int colsA, int colsB);
    __global__ void layerForwardKernel(float* inputs, float* weights, float* outputs,  int input_size, int output_size);
    __global__ void layerForwardKernel(float* inputs, float* weights, float* outputs,  int inrow, int incol, int put_size);

    __global__ void l1PenaltyKernel(float* weights, float* result, int size);
    __global__ void l2PenaltyKernel(float* weights, float* result, int size);
    __global__ void absDiffKernel(float* outputs, float* targets, float* result, int size);
    __global__ void squaredDiffKernel(float* outputs, float* targets, float* result, int size);
    __global__ void cuMSEKernel(float* expected, float* output, float* mse, int size);
    __global__ void kernelOutputDelta(const float* output, const float* expected, float* delta, int size);

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
    __global__ void updateWeightsL2Kernel(float* deltas, float* prev_activations, float* weights, float* gweights,
                    float learning_rate, float lambda, int current_layer_size, int prev_layer_size);
    __global__ void kernelUpdateElasticNet(float* deltas, float* prev_activations, float* weights, float* gweights, float learning_rate, float lambda_l1, 
                    float lambda_l2, int current_layer_size, int prev_layer_size);
    __global__ void rpropUpdateKernel(float* weights, float* gradients, float* prev_gradients, float* delta_weights,
                    float eta_plus, float eta_minus, float delta_max, float delta_min, int size);
    __global__ void updateInputVectorKernel(float* input, float* weights, float* deltas, float learning_rate, int size);


    __global__ void hiddenDeltaKernel2d(float* next_layer_deltas, float* weights, float* activations,
                    float* deltas, int current_layer_size, int next_layer_size);
    __global__ void kernelComputeGradMLPInput2d(const float* deltas, const float* weights, float* grad_input,
                    int current_layer_size, int input_size);
    __global__ void kernelLastLayerDelta2d(const float* grad_output, const float* activations, float* deltas, int size);
    __global__ void updateWeightsKernel(float* deltas, float* prev_activations, float* weights, float learning_rate,
                    int current_layer_size, int prev_layer_size);
    __global__ void updateWeightsKernel2d(float* deltas, float* prev_activations, float* weights, float* gradients,
                    float learning_rate, int current_layer_size, int prev_layer_size);
    __global__ void updateWeightsL1Kernel2d(float* weights, float* deltas, float* prev_activations, float learning_rate, 
                    float lambda, int current_layer_size, int prev_layer_size);
    __global__ void updateWeightsL2Kernel2d(float* deltas, float* prev_activations, float* weights, float* gweights,
                    float learning_rate, float lambda, int current_layer_size, int prev_layer_size);
    __global__ void kernelUpdateElasticNet2d(float* deltas, float* prev_activations, float* weights, float* gweights,
                    float learning_rate, float lambda_l1, float lambda_l2, int current_layer_size, int prev_layer_size);
    __global__ void rpropUpdateKernel2d(float* weights, float* gradients, float* prev_gradients, float* delta_weights,
                    float eta_plus, float eta_minus, float delta_max, float delta_min, int size);
    __global__ void updateInputVectorKernel2d(float* input, float* weights, float* deltas, float learning_rate, int size);

    float cugetL1Penalty(std::vector<std::vector<std::vector<float>>>&);
    float cugetL2Penalty(std::vector<std::vector<std::vector<float>>>&);
    float cucomputeLossWithL1(std::vector<float>&, std::vector<float>&, mlp&, float);
    float cucomputeLossWithL2(std::vector<float>&, std::vector<float>&, mlp&, float);
    float cudropoutGeneralisation(std::vector<float>&, std::vector<float>&, mlp&, float);

#endif

#endif // MLP_HPP