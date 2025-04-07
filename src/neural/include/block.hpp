
/**
 * @file CLass block for complete attention using 2d array of attention class. This
 * helps in maintining continuity for context in next blcok keeps producing tokens
 * till context limit for each block is reached. After all blocks are processed, the 
 * last of EVs are then used to continue the context for next iteration on command.
 *                                Complete Attention
 * ---------------------------------------------------------------------------------
 * (Incomplete Attention - ----- - Incomplete Attention -> E') <-- Partial attention
 * (Incomplete Attention - ----- - Incomplete Attention -> E') <-- Partial attention
 * (Incomplete Attention - ----- - Incomplete Attention -> E') <-- Partial attention 
 *      |           |                   |           |                       |
 *      |           |                   |           |                       |
 *      |           |                   |           |                       |
 * (Incomplete Attention - ----- - Incomplete Attention -> E') <-- Partial attention 
 * ---------------------------------------------------------------------------------
 */

#ifndef BLOCK_HPP
#define BLOCK_HPP 1

#include <vector>
#include <maths.hpp>
#include "mlp.hpp"
#include "attention.hpp"


/**
 * @brief block for complete attention
 */
class block {
public:
    int x, y;               // x layers with y heads in each layer
    float error;            // error for block, mean of all incomplete attentions
    bool isSelfAttention;   // if its self (1) or cross (0) attention
    std::string str;        // to check whether new token is "@#O" or part of conversation
    std::vector<float> EH;  // common horizontal vector for calculating probability of output
    // hold all vertical retention vector EVs from each attention heads
    std::vector<std::vector<std::vector<std::vector<float>>>> EV;
    std::vector<float> probability;             // probability space for next token
    std::vector<std::vector<attention>> b;      // block complete attention

    // default constructor
    block() = default;
    block(int x, int y, int n, int d, int h, int l, int vocab);
    block(int x, int y, int n, int d, int h, int l, int vocab, bool attentionType);

    void computeKdotQforTrainParallel();
    void computeKdotQforTrainLayer();
    void computeKdotQforUseParallel();
    void computeKdotQforUseLayer();
    // partial attention forprop
    void partialforprop(int& in, int& tokenCount, int i, int& layers);
    void partialforprop(std::vector<std::vector<std::vector<float>>>& EVp, int& in, int& tokenCount, int& blockCount, int i, int& layers, int& n);
    // parallel partialforprop(i)
    void forprop(int& in, int& tokenCount, int& layers);
    void forprop(std::vector<std::vector<std::vector<std::vector<float>>>>& EVp, int& in, int& tokenCount, int& blockCount, int& layers, int& n);
    // set retention vectors for vertical pass
    void setVerticalRetention(std::vector<std::vector<std::vector<std::vector<float>>>>& EV);
    // partial attention backward
    void partialbackward(std::vector<float>& expectedH, int& in, int& layers, int layno);
    void partialbackward(std::vector<std::vector<std::vector<float>>>& expectedV, int& in, int& layers, int layno);
    void partialbackward1stBlock(std::vector<float>& expectedH, int& in, int& layers, int layno);
    void partialbackward1stBlock(std::vector<std::vector<std::vector<float>>>& expectedV, int& in, int& layers, int layno);
    // parallel partialbackward(i)
    void backward(std::vector<float>& expectedH, int& in, int& layers);
    void backward(std::vector<std::vector<float>>& expectedH, int& in, int& layers);
    void backward(std::vector<std::vector<std::vector<std::vector<float>>>>& expectedV, int& in, int& layers);
    void backward1stBlock(std::vector<float>& expectedH, int& in, int& layers);
    void backward1stBlock(std::vector<std::vector<float>>& expectedH, int& in, int& layers);
    void backward1stBlock(std::vector<std::vector<std::vector<std::vector<float>>>>& expectedV, int& in, int& layers);

#ifdef USE_CUDA     // cuda equivalent functions for block
    void cuPartialforprop(int& in, int& tokenCount, int i, int& layers);
    void cuPartialforprop(std::vector<std::vector<std::vector<float>>>& EVp, int& in, int& tokenCount, int& blockCount, int i, int& layers, int& n);
    void cuForprop(int& in, int& tokenCount, int& layers);
    void cuForprop(std::vector<std::vector<std::vector<std::vector<float>>>>& EVp, int& in, int& tokenCount, int& blockCount, int& layers, int& n);
    void cupartialbackward(std::vector<float>& expectedH, int& in, int& layers, int layno);
    void cupartialbackward(std::vector<std::vector<std::vector<float>>>& expectedV, int& in, int& layers, int layno);
    void cupartialbackward1stBlock(std::vector<float>& expectedH, int& in, int& layers, int layno);
    void cupartialbackward1stBlock(std::vector<std::vector<std::vector<float>>>& expectedV, int& in, int& layers, int layno);
    void cubackward(std::vector<float>& expectedH, int& in, int& layers);
    void cubackward(std::vector<std::vector<float>>& expectedH, int& in, int& layers);
    void cubackward(std::vector<std::vector<std::vector<std::vector<float>>>>& expectedV, int& in, int& layers);
    void cubackward1stBlock(std::vector<float>& expectedH, int& in, int& layers);
    void cubackward1stBlock(std::vector<std::vector<float>>& expectedH, int& in, int& layers);
    void cubackward1stBlock(std::vector<std::vector<std::vector<std::vector<float>>>>& expectedV, int& in, int& layers);
#elif USE_OPENCL    // opencl equivalent functions for block
    void clPartialforprop(int& in, int& tokenCount, int i, int& layers);
    void clPartialforprop(std::vector<std::vector<std::vector<float>>>& EVp, int& in, int& tokenCount, int& blockCount, int i, int& layers, int& n);
    void clForprop(int& in, int& tokenCount, int& layers);
    void clForprop(std::vector<std::vector<std::vector<std::vector<float>>>>& EVp, int& in, int& tokenCount, int& blockCount, int& layers, int& n);
    void clpartialbackward(std::vector<float>& expectedH, int& in, int& layers, int layno);
    void clpartialbackward(std::vector<std::vector<std::vector<float>>>& expectedV, int& in, int& layers, int layno);
    void clpartialbackward1stBlock(std::vector<float>& expectedH, int& in, int& layers, int layno);
    void clpartialbackward1stBlock(std::vector<std::vector<std::vector<float>>>& expectedV, int& in, int& layers, int layno);
    void clbackward(std::vector<float>& expectedH, int& in, int& layers);
    void clbackward(std::vector<std::vector<float>>& expectedH, int& in, int& layers);
    void clbackward(std::vector<std::vector<std::vector<std::vector<float>>>>& expectedV, int& in, int& layers);
    void clbackward1stBlock(std::vector<float>& expectedH, int& in, int& layers);
    void clbackward1stBlock(std::vector<std::vector<float>>& expectedH, int& in, int& layers);
    void clbackward1stBlock(std::vector<std::vector<std::vector<std::vector<float>>>>& expectedV, int& in, int& layers);
#endif

    // default destructor
    ~block() = default;
};

#endif
