#ifndef MLP_HPP
#define MLP_HPP 1

#include <maths.hpp>
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
    unsigned int num_layers;    // Total number of layers (including input and output)
    std::vector<unsigned int> layer_sizes;      // Number of neurons in each layer
    unsigned int epochs;        // Training epochs
    float learning_rate;        // Learning rate
    float lambda_l1;            // L1 regularization parameter
    float lambda_l2;            // L2 regularization parameter
    unsigned int t;             // Time step for Adam (number of updates), initialized to 0

    // float beta1;                // Adam hyperparameter (decay rate for first moment)
    // float beta2;                // Adam hyperparameter (decay rate for second moment)
    // float epsilon;              // Adam hyperparameter (small value to prevent division by zero)

// member containers
    std::vector<float> input;      // input vector
    std::vector<float> output;     // output vector
    std::vector<float> expected;   // expected output vectors
    std::vector<mat> weights;      // Weight matrices between layers (using memory-mapped mat)
    std::vector<std::vector<float>> hlayers;       // hidden layers (intermediate, might stay in RAM)
    std::vector<std::vector<float>> activations;   // activations for each layer
    std::vector<mat> gweights;     // Gradient matrices corresponding to weights (using memory-mapped mat)
    int params;                    // parameters in mlp

    std::vector<mat> moments; // First moments for weights
    std::vector<mat> velocity; // Second moments for weights

    // Constructor(s) modified to accept OpenCLContext when needed
#ifdef USE_OPENCL
    OpenCLContext& clContext; // <-- THIS CALL TRIGGERS THE PROCESS
    // Constructor when OpenCL is enabled
    mlp(OpenCLContext& context, const std::vector<unsigned int>& layerSizes, unsigned int epochs, float learning, float lambda_L1, float lambda_L2);
#elif USE_CUDA || USE_CPU
    mlp() = default;
    // Constructor when OpenCL is disabled
    mlp(const std::vector<unsigned int>& layerSizes, unsigned int epochs, float learning, float lambda_L1, float lambda_L2);
#endif

    // Explicitly define copy constructor and copy assignment operator
    mlp(const mlp& other);
    mlp& operator=(const mlp& other);

    mlp(const mlp&& other);
    mlp& operator=(const mlp&& other);

#ifdef USE_CUDA

// cuda implementation for mlp
    void cuForward(int in, int layers);
    void cuBackward(int layers, int in, float learning);
    void cuBackprop(int layers, int in, float learning);
    void cuBackwithL1(int layers, int in, float learning);
    void cuBackwithL2(int layers, int in, float learning);
    void cuBackwithELasticNet(int layers, int in, float learning);
    void cuBackprop2in(int layers, int in, float learning);
    void cuRprop(std::vector<std::vector<float>>&, int layers, int in, float learning, int epochs);
    void cuTrain(float& mse, int in, int layers, float learning);
    void cuTrain(std::vector<std::vector<float>>&, float& mse, int in, int layers, float learning);
    void cuValidate(int in, int layers);
    void cuTest(int in, int layers);
    void cuAdamUpdate(unsigned long long t_adam, float beta1, float beta2, float epsilon, float learning_rate);

#elif USE_OPENCL

// opencl implementation for mlp methods (will use clContext member)
    void clForward(int in, int layers);
    void clBackward(int layers, int in, float learning);
    void clBackprop(int layers, int in, float learning);
    void clBackwithL1(int layers, int in, float learning);
    void clBackwithL2(int layers, int in, float learning);
    void clBackwithElasticNet(int layers, int in, float learning);
    void clBackprop2in(int layers, int in, float learning);
    void clRprop(std::vector<std::vector<float>>&, int layers, int in, float learning, int epochs);
    void clTrain(float& mse, int in, int layers, float learning);
    void clTrain(std::vector<std::vector<float>>&, float& mse, int in, int layers, float learning);
    void clValidate(int in, int layers);
    void clTest(int in, int layers);
    void clAdamUpdate(OpenCLContext& clContext, unsigned long long t_adam_param, float beta1, float beta2, float epsilon, float learning_rate); 

#else

    void forward(int in, int layers);
    void backward(int layers, int in, float learning);
    void backprop(int layers, int in, float learning);
    void backwithL1(int layers, int in, float learning);
    void backwithL2(int layers, int in, float learning);
    void backwithElasticNet(int in, int layers, float learning);
    void backprop2in(int layers, int in, float learning);
    void rprop(std::vector<std::vector<float>>&, int layers, int in, float learning, int epochs);
    void train(float& mse, int in, int layers, float learning);
    void train(std::vector<std::vector<float>>&, float& mse, int in, int layers, float learning);
    void validate(int in, int layers);
    void test(int in, int layers);
    void adamUpdate(unsigned long long t_adam_param, float beta1, float beta2, float epsilon, float learning_rate);

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
            // TODO: Handle gweights similarly if they are also file-backed in the shared map
        }
    }

    void initializeAdamMoments();
    void clearValues();
    void serialise(unsigned long long offset, const std::string& locationWithFilename);
    void deserialise(unsigned long long offset, const std::string& locationWithFilename);
    void serialise4train(const std::string& locationWithFileName);
    void serialise4use(const std::string& locationWithFileName);
    
    ~mlp() = default; // Default destructor is fine
};

// Inline implementations for copy constructor and copy assignment operator
inline mlp::mlp(const mlp& other) :
#ifdef USE_OPENCL
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
    params(other.params),
    moments(other.moments),
    velocity(other.velocity)
{}

inline mlp& mlp::operator=(const mlp& other) {
    if (this == &other) {
        return *this; // Self-assignment check
    }

#ifdef USE_OPENCL
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
    moments = other.moments;
    velocity = other.velocity;
    return *this;
}

void write2filefrommlp(const mlp&, const std::string&);

// mlp-related functions (non-member)
float getL1Penalty(const std::vector<mat>& weights); // Updated signature
float getL2Penalty(const std::vector<mat>& weights); // Updated signature
float computeLossWithL1(std::vector<float>&, std::vector<float>&, mlp&, float);
float computeLossWithL2(std::vector<float>&, std::vector<float>&, mlp&, float);
float computeLossWithElasticNet(std::vector<float>& outputs, std::vector<float>& targets, mlp& network, float lambda_l1, float lambda_l2);
float dropoutGeneralisation(std::vector<float>&, std::vector<float>&, mlp&, float);
std::vector<float> flattenWeights(const std::vector<mat>& weights); // Updated signature

#ifdef USE_CUDA

// cuda implementation (kernels remain the same)
    __global__ void matrixMultiplyKernel(const float* A, const float* B, float* C, int rowsA, int colsA, int colsB);
    __global__ void vectorAddKernel(const float* A, const float* B, float* C, int len);
    __global__ void l1PenaltyKernel(float* weights, float* result, int size);
    __global__ void l2PenaltyKernel(float* weights, float* result, int size);
    __global__ void absDiffKernel(float* outputs, float* targets, float* result, int size);
    __global__ void squaredDiffKernel(float* outputs, float* targets, float* result, int size);
    __global__ void cuMSEKernel(float* expected, float* output, float* mse, int size);
    __global__ void kernelOutputDelta(const float* output, const float* expected, float* delta, int size); // Note: CUDA likely uses specific name like kernelOutputDeltaSigmoid
    __global__ void hiddenDeltaKernel(float* next_layer_deltas, float* weights, float* activations,
            float* deltas, int current_layer_size, int next_layer_size); // Note: CUDA likely uses specific name like kernelHiddenDeltaSigmoid
    __global__ void kernelComputeGradMLPInput(const float* deltas, const float* weights, float* grad_input,
            int current_layer_size, int input_size);
    __global__ void kernelLastLayerDelta(const float* grad_output, const float* activations, float* deltas, int size); // Note: CUDA likely uses specific name like kernelLastLayerDeltaSigmoid
    __global__ void updateWeightsKernel(float* deltas, float* prev_activations, float* weights, float learning_rate,
            int current_layer_size, int prev_layer_size); // Note: CUDA likely uses specific name like kernelUpdateWeights
    __global__ void updateWeightsKernel(float* deltas, float* prev_activations, float* weights, float* gradients,
            float learning_rate, int current_layer_size, int prev_layer_size); // Note: CUDA likely uses specific name like kernelUpdateWeightsAndGradients
    __global__ void updateWeightsL1Kernel(float* deltas, float* prev_activations, float* weights, float* gweights,
            float learning_rate, float lambda, int current_layer_size, int prev_layer_size);
    __global__ void updateWeightsL2Kernel(float* deltas, float* prev_activations, float* weights, float* gweights,
            float learning_rate, float lambda, int current_layer_size, int prev_layer_size);
    __global__ void rpropUpdateKernel(float* weights, float* gradients, float* prev_gradients, float* delta_weights,
            float eta_plus, float eta_minus, float delta_max, float delta_min, int size);
    __global__ void updateInputVectorKernel(float* input, float* weights, float* deltas, float learning_rate, int size); // Note: CUDA likely uses specific name like kernelUpdateInputMLP
    __global__ void layerForwardKernel(float* inputs, float* weights, float* outputs,  int input_size, int output_size); // Note: CUDA likely uses specific name like kernelLayerForward
    __global__ void kernelUpdateElasticNet(float* deltas, float* prev_activations, float* weights, float* gweights, float learning_rate, float lambda_l1, 
                    float lambda_l2, int current_layer_size, int prev_layer_size);
    float cugetL1Penalty(std::vector<std::vector<std::vector<float>>>&);
    float cugetL2Penalty(std::vector<std::vector<std::vector<float>>>&);
    float cucomputeLossWithL1(std::vector<float>&, std::vector<float>&, mlp&, float);
    float cucomputeLossWithL2(std::vector<float>&, std::vector<float>&, mlp&, float);
    float cudropoutGeneralisation(std::vector<float>&, std::vector<float>&, mlp&, float);

#endif

#endif // MLP_HPP
