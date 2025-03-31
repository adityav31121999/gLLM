
/**
 * @file CLass block for complete attention using 2d array of attention class. This
 * helps in maintining continuity for context in next blcok keeps producing tokens
 * till context limit for each block is reached. After all blocks are processed, the 
 * last of EVs are then used to continue the context for next iteration on command.
 *                                Complete Attention
 * ---------------------------------------------------------------------------------
 * (Incomplete Attention - ----- - Incomplete Attention -> E') <-- Partial attention (layer)
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
    std::string str;        // to check whether new token is @#O or part of conversation
    // std::vector<float> EH;              // Common Approximation vector to add all tokens in horizontal pass
    // context retention tokens from each head
    std::vector<std::vector<std::vector<std::vector<float>>>> EV;
    // std::vector<float> expectedH;       // expected output from horizontal pass for backprop
    // expected output from vertical pass for backprop
    // std::vector<std::vector<std::vector<std::vector<float>>>> expectedV;
    std::vector<float> probability;             // probability space for next token
    std::vector<std::vector<attention>> b;      // block complete attention

    // default constructor
    block() = default;
    block(int x, int y, int n, int d, int h, int l, int vocab);
    block(int x, int y, int n, int d, int h, int l, int vocab, bool attentionType);

    // partial attention forprop
    void partialforprop(int& in, int& tokenCount, int i, int& layers);
    void partialforprop(std::vector<std::vector<std::vector<float>>>& EVp, int& in, int& tokenCount, int& blockCount, int i, int& layers, int& n);

    // parallel partialforprop(i)
    void forprop(int& in, int& tokenCount, int& layers);
    void forprop(std::vector<std::vector<std::vector<std::vector<float>>>>& EVp, int& in, int& tokenCount, int& blockCount, int& layers, int& n);

    // set retention vectors for vertical pass
    void setVerticalRetention(std::vector<std::vector<std::vector<std::vector<float>>>>& EV);

    // partial attention backward
    void partialbackward(std::vector<std::vector<std::vector<float>>>& expectedV, std::vector<float>& expectedH, int& in, int& layers, int layno);
    void partialbackward(std::vector<std::vector<std::vector<float>>>& expectedV, int& in, int& layers, int layno);
    void partialbackward(std::vector<float>& expectedH, int& in, int& layers, int layno);

    // parallel partialbackward(i)
    void backward(std::vector<std::vector<std::vector<std::vector<float>>>>& expectedV, std::vector<std::vector<float>>& expectedH, int& in, int& layers);
    void backward(std::vector<std::vector<std::vector<std::vector<float>>>>& expectedV, std::vector<float>& expectedH, int& in, int& layers);
    void backward(std::vector<std::vector<std::vector<std::vector<float>>>>& expectedV, int& in, int& layers);
    void backward(std::vector<float>& expectedH, int& in, int& layers);

    // parallel forprop(i) and backward(i)
    void train(std::vector<float>& expected, int& in, int& tokenCount, int& layers);
    void train(std::vector<float>& expected, int& in, int& tokenCount, int& layers, int& blockCount);

#ifdef USE_CUDA
    // cuda equivalent functions for block
#elif USE_OPENCL
    // opencl equivalent functions for block
#endif

    // default destructor
    ~block() = default;
};

#endif
