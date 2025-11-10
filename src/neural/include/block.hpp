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

/**
 * @brief block for complete attention (local context), each row is called as partial attention,
 * each column is called parallel (since they are performed in parallel on GPUs),
 * and the concatenated/summed output of block is called as complete attention. Blokc has a 
 * 2D matrix of all attention unit.
 */
class block {
public:
    int x, y;               // x layers with y heads in each layer
    int tokenCount;         // number of tokens in local context
    float error;            // error for block, mean of all Attention Heads
    bool isSelfAttention;   // if its self (1) or cross (0) attention
    bool inTraining;        // = 1 for training, = 0 for in use
    std::string str;        // to check whether new token is "@#O" or part of conversation
    std::vector<std::vector<std::vector<std::vector<float>>>> EV;
    mat tokForBlock;        // tokens for block
    std::vector<float> gradToken; // gradient for token when backprop from block to embedding
    std::vector<std::vector<attention>> b;  // block complete attention
    FILE* blockFile = nullptr;              // bin file for block
    std::string blockFilePath;              // path to model file
    unsigned long long params;              // parameters in block
    unsigned long long blockOffset;         // offset for block in training bin file

#ifdef USE_OPENCL
    OpenCLContext& clcontext;
    block(OpenCLContext& context, int x_layers, int y_heads, int n_tokens, int d_embed, int h_internal, int l_mlp, 
        unsigned long long vocab, bool attentionType, bool trainMode, int blockCount, const std::string& blockFilePath_param,
        float& learning);
    block(OpenCLContext& context, const std::string& blockName, int x_layers, int y_heads, int n_tokens, int d_embed, int h_internal, int l_mlp, 
        unsigned long long vocab, bool attentionType, bool trainMode, int blockCount, const std::string& blockFilePath_param,
        float& learning);
#elif USE_CUDA || USE_CPU
    block() = default;
    block(int x_layers, int y_heads, int n_tokens, int d_embed, int h_internal, int l_mlp, 
        unsigned long long vocab, bool attentionType, bool trainMode, int blockCount, const std::string& blockFilePath_param,
        float& learning);
    block(const std::string& blockName, int x_layers, int y_heads, int n_tokens, int d_embed, int h_internal, int l_mlp, 
        unsigned long long vocab, bool attentionType, bool trainMode, int blockCount, const std::string& blockFilePath_param,
        float& learning);
#endif

    // assignment operator to copy block
    block operator=(const block& other) {
        x = other.x;
        y = other.y;
        error = other.error;
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
    void cuForpropev(int& in, int& tokenCount, int& layers);
    void cuForpropev(std::vector<std::vector<std::vector<std::vector<float>>>>& EVp, int& in, int& tokenCount, int& blockCount, int& layers, int& n);
    // for single parallel, partial attention backward
    void cupartialbackward1stBlock(std::vector<float>& expectedH, int& in_dim, int& layers_mlp, int& layno_col_idx, float& learning, float& lambda_l1, float& lambda_l2);
    void cupartialbackward1stBlock(std::vector<std::vector<float>>& expectedH, int& in, int& layers, int& layno, float& learning, float& lambda_l1, float& lambda_l2);
    void curpartialbackward1stBlock(std::vector<std::vector<float>>& expectedH, int& in, int& layers, int& layno, float& learning, float& lambda_l1, float& lambda_l2);
    void cupartialbackward(std::vector<float> &expectedH, int& in_dim, int& layers_mlp, int& layno_col_idx, float& learning, float& lambda_l1, float& lambda_l2);
    void cupartialbackward(std::vector<std::vector<float>>& expectedH, int& in, int& layers, int& layno, float& learning, float& lambda_l1, float& lambda_l2);
    // parallel partialbackward(i)
    void cubackward1stBlock(std::vector<float>& expectedH, int& in, int& layers, float& learning, float& lambda_l1, float& lambda_l2);
    void cubackward1stBlock(std::vector<std::vector<float>>& expectedH, int& in, int& layers, float& learning, float& lambda_l1, float& lambda_l2);
    void curbackward1stBlock(std::vector<std::vector<float>>& expectedH, int& in, int& layers, float& learning, float& lambda_l1, float& lambda_l2);
    void cubackward(std::vector<float>& expectedH, int& in, int& layers, int& blockCount, float& learning, float& lambda_l1, float& lambda_l2);
    void cubackward(std::vector<std::vector<float>>& expectedH, int& in, int& layers, int& blockCount, float& learning, float& lambda_l1, float& lambda_l2);

#elif USE_OPENCL

    // for single parallel
    void cl1parallelForprop(int& in, int& tokenCount, int i, int& layers);
    void cl1ParallelForprop(std::vector<std::vector<std::vector<float>>>& EVp, int& in, int& tokenCount, int& blockCount, int i, int& layers, int& n);
    void cl1parallelForpropev(int& in, int& tokenCount, int i, int& layers);
    void cl1ParallelForpropev(std::vector<std::vector<std::vector<float>>>& EVp, int& in, int& tokenCount, int& blockCount, int i, int& layers, int& n);
    // for complete block
    void clForprop(int& in, int& tokenCount, int& layers);
    void clForprop(std::vector<std::vector<std::vector<std::vector<float>>>>& EVp, int& in, int& tokenCount, int& blockCount, int& layers, int& n);
    void clForpropev(int& in, int& tokenCount, int& layers);
    void clForpropev(std::vector<std::vector<std::vector<std::vector<float>>>>& EVp, int& in, int& tokenCount, int& blockCount, int& layers, int& n);
    // for single parallel, partial attention backward
    void clpartialbackward1stBlock(std::vector<float>& expectedH, int& in_dim, int& layers_mlp, int& layno_col_idx, float& learning, float& lambda_l1, float& lambda_l2);
    void clpartialbackward1stBlock(std::vector<std::vector<float>>& expectedH, int& in, int& layers, int& layno, float& learning, float& lambda_l1, float& lambda_l2);
    void clrpartialbackward1stBlock(std::vector<std::vector<float>>& expectedH, int& in, int& layers, int& layno, float& learning, float& lambda_l1, float& lambda_l2);
    void clpartialbackward(std::vector<float> &expectedH, int& in_dim, int& layers_mlp, int& layno_col_idx, float& learning, float& lambda_l1, float& lambda_l2);
    void clpartialbackward(std::vector<std::vector<float>>& expectedH, int& in, int& layers, int& layno, float& learning, float& lambda_l1, float& lambda_l2);
    // parallel partialbackward(i)
    void clbackward1stBlock(std::vector<float>& expectedH, int& in, int& layers, float& learning, float& lambda_l1, float& lambda_l2);
    void clbackward1stBlock(std::vector<std::vector<float>>& expectedH, int& in, int& layers, float& learning, float& lambda_l1, float& lambda_l2);
    void clrbackward1stBlock(std::vector<std::vector<float>>& expectedH, int& in, int& layers, float& learning, float& lambda_l1, float& lambda_l2);
    void clbackward(std::vector<float>& expectedH, int& in, int& layers, int& blockCount, float& learning, float& lambda_l1, float& lambda_l2);
    void clbackward(std::vector<std::vector<float>>& expectedH, int& in, int& layers, int& blockCount, float& learning, float& lambda_l1, float& lambda_l2);

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
    void partialbackward(std::vector<float>& expectedH, int& in, int& layers, int layno, float& learning);
    void partialbackward(std::vector<std::vector<float>>& expectedH, int& in, int& layers, int layno, float& learning);
    std::vector<std::vector<float>> rpartialbackward1stBlock(std::vector<std::vector<float>>& expectedH, int& in, int& layers, int layno, float& learning);
    // parallel partialbackward(i)
    void backward1stBlock(std::vector<float>& expectedH, int& in, int& layers, float& learning);
    void backward1stBlock(std::vector<std::vector<float>>& expectedH, int& in, int& layers, float& learning);
    void backward(std::vector<float>& expectedH, int& in, int& layers, int blockCount, float& learning);
    void backward(std::vector<std::vector<float>>& expectedH, int& in, int& layers, int blockCount, float& learning);
    std::vector<std::vector<float>> rbackward1stBlock(std::vector<std::vector<float>>& expectedH, int& in, int& layers, float& learning);

#endif

    void randomValuesForBlock(float x1, float x2, int n);
    void setVerticalRetention(std::vector<std::vector<std::vector<std::vector<float>>>>& EV);
    void clearValues();
    void serialise(const std::string& locationWithFilename);
    void deserialise(const std::string& locationWithFilename);    

    ~block() = default;
};

#endif
