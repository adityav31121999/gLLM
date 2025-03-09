
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
    double error;       // error for block, mean of all incomplete attentions
    std::vector<std::vector<attention>> b;      // block complete attention
    std::vector<std::vector<std::vector<double>>> holdEVs;  // inbetween tokens transfer
    std::string str;    // string for end of tokens
    // default constructor
    block() = default;
    block(int x, int y, int n, int d, int h, int l);

    void partialforprop(std::vector<std::vector<double>>, int tokenCount, int i);         // partial attention forprop
    void forprop(std::vector<std::vector<double>>, int tokenCount);   // parallel partialforprop(i)
    void partialbackward(std::vector<double> tExp, int tokenCount, int i, double);        // partial attention backward
    void backward(std::vector<double> tExp, int tokenCount, double);                // parallel partialbackward(i)
    void train(std::vector<std::vector<double>>, int, double);           // parallel forprop(i) and backward(i)

    // default destructor
    ~block() = default;
};

#endif
