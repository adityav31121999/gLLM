
/**
 * @file CLass BLOCK for complete attention using 2d array of attention class. This
 * helps in maintining continuity for context in next blcok keeps producing tokens
 * till context limit for each block is reached. After all blocks are processed, the 
 * last of EVs are then used to continue the context for next iteration on command.
 *                                                                Complete Attention
 *                                                                        |
 * ----------------------------------------------------------------------\/------------
 * (Attention Head - Attention Head ----- - Attention Head -> E') --> Partial attention
 * (Attention Head - Attention Head ----- - Attention Head -> E') --> Partial attention
 * (Attention Head - Attention Head ----- - Attention Head -> E') --> Partial attention
 *      |                   |                   |             |             |
 *      |                   |                   |             |             |
 *      |                   |                   |             |             |
 * (Attention Head - Attention Head ----- - Attention Head -> E') <-- Partial attention
 * -----/\-----------------------------------------------------------------------------
 *      |
 *  Parallel
 */

#ifndef BLOCK_HPP
#define BLOCK_HPP 1

#include <vector>
#include <maths.hpp>
#include <map>
#include "mlp.hpp"
#include "attention.hpp"


/**
 * @brief block for complete attention (local context)
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
    std::vector<std::vector<attention>> b;  // block complete attention
    FILE* blockFile = nullptr;              // bin file for block
    std::string blockFilePath;              // path to model file
    long long int params;                   // parameters in block
    long long int blockOffset;              // offset for block in training bin file

#ifdef USE_OPENCL
    OpenCLContext& clcontext;
    block(OpenCLContext& context, int x, int y, int n, int d, int h, int l, long long int vocab, bool attentionType, bool inTraining, int blockCount, const std::string& blockFilePath);
#elif USE_CUDA || USE_CPU
    block() = default;
    block(int x, int y, int n, int d, int h, int l, long long int vocab, bool attentionType, bool inTraining, int blockCount, const std::string blockFilePath_param);
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

// cuda equivalent functions for block (for parallels in loop)
    void cuParallelKdotQ(int& columnNumber, int& blockNumber, int& promptCount, int& tokenCount, bool isSelfAttention);
    void cuParallelUseKdotQ(const std::vector<std::vector<float>>& tokenEmbed, int& columnNumber, int& tokenCount, int& promptCount, bool isSelfAttention);
    void cuParallelUseKdotQ(const std::vector<std::vector<std::vector<float>>>& EVp, int& columnNumber, int& blockNumber, int& tokenCount, int& promptCount, bool isSelfAttention);
    // for single parallel
    void cu1parallelForprop(int& in, int& tokenCount, int i, int& layers);
    void cu1ParallelForprop(std::vector<std::vector<std::vector<float>>>& EVp, int& in, int& tokenCount, int& blockCount, int i, int& layers, int& n);
    // for complete block
    void cuForprop(int& in, int& tokenCount, int& layers);
    void cuForprop(std::vector<std::vector<std::vector<std::vector<float>>>>& EVp, int& in, int& tokenCount, int& blockCount, int& layers, int& n);
    // for single parallel
    void cu1ParallelBackward1stBlock(std::vector<float>& expectedH, int& in, int& layers, int layno);
    void cu1ParallelBackward1stBlock(std::vector<std::vector<std::vector<float>>>& expectedV, int& in, int& layers, int layno);
    void cu1ParallelBackward(std::vector<float>& expectedH, int& in, int& layers, int layno);
    void cu1ParallelBackward(std::vector<std::vector<std::vector<float>>>& expectedV, int& in, int& layers, int layno);
    // for complete block
    void cuBackward1stBlock(std::vector<float>& expectedH, int& in, int& layers);
    void cuBackward1stBlock(std::vector<std::vector<float>>& expectedH, int& in, int& layers);
    void cuBackward1stBlock(std::vector<std::vector<std::vector<std::vector<float>>>>& expectedV, int& in, int& layers);
    void cuBackward(std::vector<float>& expectedH, int& blockCount, int& in, int& layers);
    void cuBackward(std::vector<std::vector<float>>& expectedH, int& blockCount, int& in, int& layers);
    void cuBackward(std::vector<std::vector<std::vector<std::vector<float>>>>& expectedV, int& blockCount, int& in, int& layers);
    // for inference
    void cuInferParallel(const mat& tokens, int& in, int& tokenCount, int& layers, int& parallelNumber);
    void cuInferParallel(const std::vector<mat>& expectedV, const mat& tokForBlock, int& in, int& tokenCount, int& blockCount, int& layers, int& n, int& parallelNumber);
    void cuInfer(const mat& tokens, int& in, int& tokenCount, int& layers);
    void cuInfer(const std::vector<std::vector<mat>>& expectedV, const mat& tokForBlock, int& in, int& tokenCount, int& blockCount, int& layers, int& n, int& parallelNumber);

#elif USE_OPENCL

// opencl equivalent functions for block (for parallels in loop)
    void clParallelKdotQ(int& columnNumber, int& blockNumber, int& promptCount, int& tokenCount, bool isSelfAttention);
    void clParallelUseKdotQ(const std::vector<std::vector<float>>& tokenEmbed, int& columnNumber, int& tokenCount, int& promptCount, bool isSelfAttention);
    void clParallelUseKdotQ(const std::vector<std::vector<std::vector<float>>>& EVp, int& columnNumber, int& blockNumber, int& tokenCount, int& promptCount, bool isSelfAttention);
    // for single parallel
    void cl1parallelForprop(int& in, int& tokenCount, int i, int& layers);
    void cl1ParallelForprop(std::vector<std::vector<std::vector<float>>>& EVp, int& in, int& tokenCount, int& blockCount, int i, int& layers, int& n);
    // for complete block
    void clForprop(int& in, int& tokenCount, int& layers);
    void clForprop(std::vector<std::vector<std::vector<std::vector<float>>>>& EVp, int& in, int& tokenCount, int& blockCount, int& layers, int& n);
    // for single parallel
    void cl1ParallelBackward1stBlock(std::vector<float>& expectedH, int& in, int& layers, int layno);
    void cl1ParallelBackward1stBlock(std::vector<std::vector<std::vector<float>>>& expectedV, int& in, int& layers, int layno);
    void cl1ParallelBackward(std::vector<float>& expectedH, int& in, int& layers, int layno);
    void cl1ParallelBackward(std::vector<std::vector<std::vector<float>>>& expectedV, int& in, int& layers, int layno);
    // for complete block
    void clBackward1stBlock(std::vector<float>& expectedH, int& in, int& layers);
    void clBackward1stBlock(std::vector<std::vector<float>>& expectedH, int& in, int& layers);
    void clBackward1stBlock(std::vector<std::vector<std::vector<std::vector<float>>>>& expectedV, int& in, int& layers);
    void clBackward(std::vector<float>& expectedH, int& blockCount, int& in, int& layers);
    void clBackward(std::vector<std::vector<float>>& expectedH, int& blockCount, int& in, int& layers);
    void clBackward(std::vector<std::vector<std::vector<std::vector<float>>>>& expectedV, int& blockCount, int& in, int& layers);
    // for inference
    void clInferParallel(const mat& tokens, int& in, int& tokenCount, int& layers, int& parallelNumber);
    void clInferParallel(std::vector<mat>& expectedV, const mat& tokForBlock, int& in, int& tokenCount, int& blockCount, int& layers, int& n, int& parallelNumber);

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
    void partialbackward1stBlock(std::vector<float>& expectedH, int& in, int& layers, int layno);
    void partialbackward1stBlock(std::vector<std::vector<std::vector<float>>>& expectedV, int& in, int& layers, int layno);
    void partialbackward(std::vector<float>& expectedH, int& in, int& layers, int layno);
    void partialbackward(std::vector<std::vector<std::vector<float>>>& expectedV, int& in, int& layers, int layno);
    // parallel partialbackward(i)
    void backward1stBlock(std::vector<float>& expectedH, int& in, int& layers, int blockCount);
    void backward1stBlock(std::vector<std::vector<float>>& expectedH, int& in, int& layers, int blockCount);
    void backward1stBlock(std::vector<std::vector<std::vector<std::vector<float>>>>& expectedV, int& in, int& layers, int blockCount);
    void backward(std::vector<float>& expectedH, int& in, int& layers, int blockCount);
    void backward(std::vector<std::vector<float>>& expectedH, int& in, int& layers, int blockCount);
    void backward(std::vector<std::vector<std::vector<std::vector<float>>>>& expectedV, int& in, int& layers, int blockCount);
    // for inference
    void inferParallel(const mat& tokens, int &in, int &tokenCount, int &layers, int& parallelNumber);
    void inferParallel(std::vector<mat>& expectedV, const mat& tokForBlock, int& in, int& tokenCount, int& blockCount, int& layers, int& n, int& parallelNumber);

#endif
    void randomValuesForBlock(float min, float max);
    void setVerticalRetention(std::vector<std::vector<std::vector<std::vector<float>>>>& EV);
    void clearValues();
    void serialise(const std::string& locationWithFilename);
    void deserialise(const std::string& locationWithFilename);    

    ~block() = default;
};

#endif
