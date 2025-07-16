#ifndef BLOCK_HPP
#define BLOCK_HPP 1

/**
 * @file CLass BLOCK for complete attention using 2d array of attention class. This
 * helps in maintining continuity for context in next blcok keeps producing tokens
 * till context limit for each block is reached. After all blocks are processed, the 
 * last of EVs are then used to continue the context for next iteration on command.
 *                                                                   Complete Attention
 *                                                                           |
 * -------------------------------------------------------------------------\/---------
 * (Attention Head - Attention Head ----- - Attention Head -> E') --> Partial attention
 * (Attention Head - Attention Head ----- - Attention Head -> E') --> Partial attention
 * (Attention Head - Attention Head ----- - Attention Head -> E') --> Partial attention
 *      |                   |                   |             |             |
 *      |                   |                   |             |             |
 *      |                   |                   |             |             |
 * (Attention Head - Attention Head ----- - Attention Head -> E') --> Partial attention
 * -----/\------------------/\------------------/\-------------------------------------
 *      |                   |                   |
 *      ---------Parallel-------.........--------
 */

#include <vector>
#include <maths.hpp>
#include <map>
#include "mlp.hpp"
#include "attention.hpp"

// Helper struct to manage device pointers for one head's worth of data
struct HeadDevicePointers {
    // Attention related
    float *d_expected_h = nullptr, *d_EH = nullptr, *d_EV = nullptr;
    float *d_grad_EH = nullptr, *d_grad_EV_scaled = nullptr;
    float *d_grad_dh = nullptr, *d_grad_dv = nullptr;
    float *d_KdotQ = nullptr, *d_head = nullptr;
    float *d_K = nullptr, *d_Q = nullptr;
    float *d_pre_MH = nullptr, *d_pre_MV = nullptr;
    float *d_MH_a = nullptr, *d_MV_a = nullptr, *d_MQ_a = nullptr, *d_MK_a = nullptr;
    float *d_grad_MH = nullptr, *d_grad_MV = nullptr;
    float *d_grad_head = nullptr;
    float *d_lota_deriv = nullptr;
    float *d_grad_KdotQ = nullptr;
    float *d_grad_K = nullptr, *d_grad_Q = nullptr;
    float *d_grad_MQ = nullptr, *d_grad_MK = nullptr;

    // MLP Internals
    std::vector<float*> d_hor_activations;
    std::vector<float*> d_hor_weights;
    std::vector<float*> d_hor_gweights;
    std::vector<float*> d_hor_deltas;
    std::vector<float*> d_ver_activations;
    std::vector<float*> d_ver_weights;
    std::vector<float*> d_ver_gweights;
    std::vector<float*> d_ver_deltas;

    HeadDevicePointers() = default; // Default constructor for vector initialization
};


/**
 * @brief block for complete attention (local context)
 */
class block {
public:
    int x, y;               // x layers with y heads in each layer
    int tokenCount;         // number of tokens in local context
    float error;            // error for block, mean of all Attention Heads
    float learning;         // learning rate for block
    float lambda_L1;        // L1 regularization strength
    float lambda_L2;        // L2 regularization strength
    bool isSelfAttention;   // if its self (1) or cross (0) attention
    bool inTraining;        // = 1 for training, = 0 for in use
    std::string str;        // to check whether new token is "@#O" or part of conversation
    std::vector<std::vector<std::vector<std::vector<float>>>> EV;
    mat tokForBlock;        // tokens for block
    std::vector<std::vector<attention>> b;  // block complete attention
    FILE* blockFile = nullptr;              // bin file for block
    std::string blockFilePath;              // path to model file
    long long int params;                   // parameters in block
    long long int blockOffset;              // offset for block in training bin file

#ifdef USE_OPENCL
    OpenCLContext& clcontext;
    block(OpenCLContext& context, int x_layers, int y_heads, int n_tokens, int d_embed, int h_internal, int l_mlp, 
        long long int vocab, bool attentionType, bool trainMode, int blockCount, const std::string& blockFilePath_param,
        float& learning, float lambda_L1, float lambda_L2);
#elif USE_CUDA || USE_CPU
    block() = default;
    block(int x_layers, int y_heads, int n_tokens, int d_embed, int h_internal, int l_mlp, long long int vocab, 
        bool attentionType, bool trainMode, int blockCount, const std::string blockFilePath_param, float& learning,
        float lambda_L1, float lambda_L2);
#endif

    // assignment operator to copy block
    block operator=(const block& other) {
        x = other.x;
        y = other.y;
        error = other.error;
        learning = other.learning;
        lambda_L1 = other.lambda_L1;
        lambda_L2 = other.lambda_L2;
        isSelfAttention = other.isSelfAttention;
        inTraining = other.inTraining;
        str = other.str;
        EV = other.EV;
        tokForBlock = other.tokForBlock;
        b = other.b;
        blockFile = other.blockFile;
        blockFilePath = other.blockFilePath;
        params = other.params;
        blockOffset = other.blockOffset;
    #ifdef USE_OPENCL
        clcontext = other.clcontext;
    #endif
        return *this;
    }

#ifdef USE_CUDA

    // for single parallel
    void cu1parallelForprop(int& in, int& tokenCount, int i, int& layers);
    void cu1ParallelForprop(std::vector<std::vector<std::vector<float>>>& EVp, int& in, int& tokenCount, int& blockCount, int i, int& layers, int& n);
    // for complete block
    void cuForprop(int& in, int& tokenCount, int& layers);
    void cuForprop(std::vector<std::vector<std::vector<std::vector<float>>>>& EVp, int& in, int& tokenCount, int& blockCount, int& layers, int& n);
    // for single parallel
    // partial attention backward
    void cupartialbackward1stBlock(std::vector<float>& expectedH, int& in, int& layers, int layno, float& learning, float& lambda_l1, float& lambda_l2);
    void cupartialbackward1stBlock(std::vector<std::vector<float>>& expectedH, int& in, int& layers, int layno, float& learning, float& lambda_l1, float& lambda_l2);
    void cupartialbackward1stBlock(std::vector<std::vector<std::vector<float>>>& expectedV, int& in, int& layers, int k, float& learning, float& lambda_l1, float& lambda_l2);
    void cupartialbackward(std::vector<float>& expectedH, int& in, int& layers, int layno, float& learning, float& lambda_l1, float& lambda_l2);
    void cupartialbackward(std::vector<std::vector<float>>& expectedH, int& in, int& layers, int layno, float& learning, float& lambda_l1, float& lambda_l2);
    void cupartialbackward(std::vector<std::vector<std::vector<float>>>& expectedV, int& in, int& layers, int k, int blocknumber, float& learning, float& lambda_l1, float& lambda_l2);
    // parallel partialbackward(i)
    void cubackward1stBlock(std::vector<float>& expectedH, int& in, int& layers, float& learning, float& lambda_l1, float& lambda_l2);
    void cubackward1stBlock(std::vector<std::vector<float>>& expectedH, int& in, int& layers, float& learning, float& lambda_l1, float& lambda_l2);
    void cubackward1stBlock(std::vector<std::vector<std::vector<std::vector<float>>>>& expectedV, int& in, int& layers, float& learning, float& lambda_l1, float& lambda_l2);
    void cubackward(std::vector<float>& expectedH, int& in, int& layers, int blockCount, float& learning, float& lambda_l1, float& lambda_l2);
    void cubackward(std::vector<std::vector<float>>& expectedH, int& in, int& layers, int blockCount, float& learning, float& lambda_l1, float& lambda_l2);
    void cubackward(std::vector<std::vector<std::vector<std::vector<float>>>>& expectedV, int& in, int& layers, int blockCount, float& learning, float& lambda_l1, float& lambda_l2);

#elif USE_OPENCL

    // for single parallel
    void cl1parallelForprop(int& in, int& tokenCount, int i, int& layers);
    void cl1ParallelForprop(std::vector<std::vector<std::vector<float>>>& EVp, int& in, int& tokenCount, int& blockCount, int i, int& layers, int& n);
    // for complete block
    void clForprop(int& in, int& tokenCount, int& layers);
    void clForprop(std::vector<std::vector<std::vector<std::vector<float>>>>& EVp, int& in, int& tokenCount, int& blockCount, int& layers, int& n);
    // for single parallel
    // partial attention backward
    void clpartialbackward1stBlock(std::vector<float>& expectedH, int& in, int& layers, int& layno, float& learning, float& lambda_l1, float& lambda_l2);
    void clpartialbackward1stBlock(std::vector<std::vector<float>>& expectedH, int& in, int& layers, int& layno, float& learning, float& lambda_l1, float& lambda_l2);
    void clpartialbackward1stBlock(std::vector<std::vector<std::vector<float>>>& expectedV, int& in, int& layers, int& k, float& learning, float& lambda_l1, float& lambda_l2);
    void clpartialbackward(std::vector<float>& expectedH, int& in, int& layers, int& layno, float& learning, float& lambda_l1, float& lambda_l2);
    void clpartialbackward(std::vector<std::vector<float>>& expectedH, int& in, int& layers, int& layno, float& learning, float& lambda_l1, float& lambda_l2);
    void clpartialbackward(std::vector<std::vector<std::vector<float>>>& expectedV, int& in, int& layers, int& k, int& blocknumber, float& learning, float& lambda_l1, float& lambda_l2);
    // parallel partialbackward(i)
    void clbackward1stBlock(std::vector<float>& expectedH, int& in, int& layers, float& learning, float& lambda_l1, float& lambda_l2);
    void clbackward1stBlock(std::vector<std::vector<float>>& expectedH, int& in, int& layers, float& learning, float& lambda_l1, float& lambda_l2);
    void clbackward1stBlock(std::vector<std::vector<std::vector<std::vector<float>>>>& expectedV, int& in, int& layers, float& learning, float& lambda_l1, float& lambda_l2);
    void clbackward(std::vector<float>& expectedH, int& in, int& layers, int& blockCount, float& learning, float& lambda_l1, float& lambda_l2);
    void clbackward(std::vector<std::vector<float>>& expectedH, int& in, int& layers, int& blockCount, float& learning, float& lambda_l1, float& lambda_l2);
    void clbackward(std::vector<std::vector<std::vector<std::vector<float>>>>& expectedV, int& in, int& layers, int& blockCount, float& learning, float& lambda_l1, float& lambda_l2);

#else

    // KdotQ for parallels
    void parallelKdotQ(int& columnNumber, int& tokenCount);
    // partial attention forprop
    void partialforprop(int& in, int& tokenCount, int i, int& layers);
    void partialforprop(std::vector<std::vector<std::vector<float>>>& EVp, int& in, int& tokenCount, int& blockCount, int i, int& layers, 
                int& n);
    // parallel partialforprop(i)
    void forprop(int& in, int& tokenCount, int& layers, int blockCount);
    void forprop(std::vector<std::vector<std::vector<std::vector<float>>>>& EVp, int& in, int& tokenCount, int& blockCount, int& layers, 
                int& n);
    // partial attention backward
    void partialbackward1stBlock(std::vector<float>& expectedH, int& in, int& layers, int layno, float& learning);
    void partialbackward1stBlock(std::vector<std::vector<float>>& expectedH, int& in, int& layers, int layno, float& learning);
    void partialbackward1stBlock(std::vector<std::vector<std::vector<float>>>& expectedV, int& in, int& layers, int k, float& learning);
    void partialbackward(std::vector<float>& expectedH, int& in, int& layers, int layno, float& learning);
    void partialbackward(std::vector<std::vector<float>>& expectedH, int& in, int& layers, int layno, float& learning);
    void partialbackward(std::vector<std::vector<std::vector<float>>>& expectedV, int& layers, int k, int blocknumber, float& learning);
    // parallel partialbackward(i)
    void backward1stBlock(std::vector<float>& expectedH, int& in, int& layers, float& learning);
    void backward1stBlock(std::vector<std::vector<float>>& expectedH, int& in, int& layers, float& learning);
    void backward1stBlock(std::vector<std::vector<std::vector<std::vector<float>>>>& expectedV, int& in, int& layers, float& learning);
    void backward(std::vector<float>& expectedH, int& in, int& layers, int blockCount, float& learning);
    void backward(std::vector<std::vector<float>>& expectedH, int& in, int& layers, int blockCount, float& learning);
    void backward(std::vector<std::vector<std::vector<std::vector<float>>>>& expectedV, int& in, int& layers, int blockCount, float& learning);

#endif

    void randomValuesForBlock(float min, float max);
    void setVerticalRetention(std::vector<std::vector<std::vector<std::vector<float>>>>& EV);
    void clearValues();
    void serialise(const std::string& locationWithFilename);
    void deserialise(const std::string& locationWithFilename);    

    ~block() = default;
};

#endif
