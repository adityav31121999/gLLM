
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
#define VOCABSIZE 128000 // vocabulary size

/**
 * @brief block for complete attention
 */
class block {
public:
    double error;       // error for block, mean of all incomplete attentions
    std::string str;    // to check whether new token is @#O or part of conversation
    std::vector<double> EH;         // Common Approximation vector to add all tokens
    std::vector<double> probability;            // probability space for next token
    std::vector<std::vector<attention>> b;      // block complete attention
    std::vector<std::vector<std::vector<double>>> holdEVs;      // inbetween tokens transfer
    std::vector<std::vector<std::vector<double>>> changeVs;     // hold vertical change vectors
    std::vector<std::vector<std::vector<double>>> holdmVs;      // hold mlp ver outputs
    std::vector<std::vector<std::vector<double>>> holddvs;      // hold dvs of all the heads

    // default constructor
    block() = default;
    block(int x, int y, int n, int d, int h, int l);
    void computeAttention(std::vector<std::vector<double>>& KdotQ, std::vector<std::vector<double>>& Keys, std::vector<std::vector<double>>& Queries, int tokenCount);
    // partial attention forprop
    void partialforprop(std::vector<std::vector<double>>& tokenEmbed, std::vector<std::vector<double>>& KdotQ, std::vector<std::vector<double>>& K,
                        std::vector<std::vector<double>>& Q, std::vector<std::vector<double>>& dv, std::vector<std::vector<double>>& EV,
                        std::vector<std::vector<double>>& changeV, int& in, int& tokenCount, int& i, int& layers);
    void partialforprop(std::vector<std::vector<double>>& tokenEmbed, std::vector<std::vector<double>>& KdotQ, std::vector<std::vector<double>>& K,
                        std::vector<std::vector<double>>& Q, std::vector<std::vector<double>>& dv, std::vector<std::vector<double>>& EVp,
                        std::vector<std::vector<double>>& EVc, std::vector<std::vector<double>>& changeV, int& in, int& tokenCount, int blockCount, 
                        int& i, int& layers, int& n);
    // parallel partialforprop(i)
    void forprop(std::vector<std::vector<double>>& tokenEmbed, std::vector<std::vector<double>>& KdotQ, std::vector<std::vector<double>>& K,
                        std::vector<std::vector<double>>& Q, std::vector<std::vector<std::vector<double>>>& dv, 
                        std::vector<std::vector<std::vector<double>>>& EV, std::vector<std::vector<std::vector<double>>>& changeV, int& in, 
                        int& tokenCount, int& layers);
    void forprop(std::vector<std::vector<double>>& tokenEmbed, std::vector<std::vector<double>>& KdotQ, std::vector<std::vector<double>>& K,
                        std::vector<std::vector<double>>& Q, std::vector<std::vector<std::vector<double>>>& dv, 
                        std::vector<std::vector<std::vector<double>>>& EV, std::vector<std::vector<std::vector<double>>>& changeV, int& in, 
                        int& tokenCount, int& layers, int& blockCount);
    // partial attention backward
    void partialbackward(std::vector<double>& expected, std::vector<std::vector<double>>& changeV, std::vector<std::vector<double>>& dv, 
                        std::vector<std::vector<double>>& EV, int& in, int& layers, int layno);
    // parallel partialbackward(i)
    void backward(std::vector<double>& expected, std::vector<std::vector<std::vector<double>>>& changeV, std::vector<std::vector<std::vector<double>>>& dv, 
                        std::vector<std::vector<std::vector<double>>>& EV, int& in, int& layers);
    // parallel forprop(i) and backward(i)
    void train(std::vector<std::vector<double>>& tokenEmbed, std::vector<double>& expected, std::vector<std::vector<double>>& KdotQ, std::vector<std::vector<double>>& K,
                        std::vector<std::vector<double>>& Q, std::vector<std::vector<std::vector<double>>>& dv, std::vector<std::vector<std::vector<double>>>& EV, 
                        std::vector<std::vector<std::vector<double>>>& changeV, int& in, int& tokenCount, int& layers);
    void train(std::vector<std::vector<double>>& tokenEmbed, std::vector<double>& expected, std::vector<std::vector<double>>& KdotQ, std::vector<std::vector<double>>& K,
                        std::vector<std::vector<double>>& Q, std::vector<std::vector<std::vector<double>>>& dv, std::vector<std::vector<std::vector<double>>>& EVp, 
                        std::vector<std::vector<std::vector<double>>>& EVc, std::vector<std::vector<std::vector<double>>>& changeV, int& in, int& tokenCount, 
                        int& layers, int& blockCount);
    
    // default destructor
    ~block() = default;
};

#endif
