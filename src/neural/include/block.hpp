
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
    std::string str;    // to check whether new token is @#O or part of conversation
    std::vector<std::vector<attention>> b;      // block complete attention
    std::vector<std::vector<std::vector<double>>> holdEVs;      // inbetween tokens transfer
    std::vector<std::vector<std::vector<double>>> changeVs;     // hold vertical change vectors
    std::vector<std::vector<std::vector<double>>> holdmVs;      // hold mlp ver outputs
    std::vector<std::vector<std::vector<double>>> holddvs;      // hold dvs of all the heads

    // default constructor
    block() = default;
    block(int x, int y, int n, int d, int h, int l);

    // partial attention forprop
    void partialforprop(std::vector<std::vector<double>>&, std::vector<std::vector<double>>&, std::vector<std::vector<double>>& , std::vector<std::vector<double>>&, int&, int&, int&, int&);
    // parallel partialforprop(i)
    void forprop(std::vector<std::vector<double>>&, std::vector<std::vector<std::vector<double>>>&, std::vector<std::vector<std::vector<double>>>&, std::vector<std::vector<std::vector<double>>>&, int&, int&, int&);
    // partial attention backward
    void partialbackward(std::vector<double>&, std::vector<std::vector<double>>&, std::vector<std::vector<double>>&, std::vector<std::vector<double>>&, int&);
    // parallel partialbackward(i)
    void backward(std::vector<double>&, std::vector<std::vector<std::vector<double>>>&, std::vector<std::vector<std::vector<double>>>&, std::vector<std::vector<std::vector<double>>>&);
    // parallel forprop(i) and backward(i)
    void train(std::vector<std::vector<double>>&, std::vector<std::vector<std::vector<double>>>&, std::vector<std::vector<std::vector<double>>>&, std::vector<std::vector<std::vector<double>>>&, int&, int&, int&);

    // default destructor
    ~block() = default;
};

#endif
