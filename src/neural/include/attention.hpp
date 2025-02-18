/**
 * (Attention - Attention - ----- - Attention -> E') <-- Partial attention 
 * (Attention - Attention - ----- - Attention -> E') <-- Partial attention 
 * (Attention - Attention - ----- - Attention -> E') <-- Partial attention 
 *      |           |                   |                       |
 *      |           |                   |                       |
 *      |           |                   |                       |
 * (Attention - Attention - ----- - Attention -> E') <-- Partial attention 
 * --------------------------------------------------------------------------
 *                                         Total Sum:-> Complete Attention
 */
#ifndef ATTENTION_HPP
#define ATTENTION_HPP 1

#include <vector>
#include <maths.hpp>
#include "include/mlp.hpp"

/**
 * @brief ATTENTION CLASS for calculating incomplete attention.
 * An array of incomplete attention is Partial Attention (LAYER) 
 * and an array of partial attention (BLOCK) is complete attention. 
 */
class attention {
public:
    // total embedding to be calculated using attention mechanism
    int tokenEmbed;
    // attention head matrix
    std::vector<std::vector<double>> head;
    // matrix for query, key, value vertical and value horizontal
    mat MQ, MK, MV, MH;
    // delta of each embedding for weighted sums in vertical and horizontal direction
    std::vector<std::vector<double>> dH, dV;
    std::vector<double> E1, E2, d1, d2;
    // mlp for incomplete attention
    mlp v, h;       // vertical and horizontal operation

    attention();
    attention(int tokenEmbed);
    void forward();
    void backward();
    void train();
    ~attention() = default;
};

/**
 * @brief block for complete attention
 */
class block {
public:
    int n;          // number of incomplete attentions in each partial attention
    std::vector<std::vector<attention>> b;  // block for complete attention
    block(int n) {
        this->n = n;
        b = std::vector<std::vector<attention>>(n, std::vector<attention>(n));
        
    }
    void forward();
    void backward();
    void train();
    ~block() = default;
};

#endif
