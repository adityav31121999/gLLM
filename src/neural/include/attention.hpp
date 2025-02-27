
/**
 * (Incomplete Attention - ----- - Incomplete Attention -> E') <-- Partial attention 
 * (Incomplete Attention - ----- - Incomplete Attention -> E') <-- Partial attention 
 * (Incomplete Attention - ----- - Incomplete Attention -> E') <-- Partial attention 
 *      |           |                   |           |                       |
 *      |           |                   |           |                       |
 *      |           |                   |           |                       |
 * (Incomplete Attention - ----- - Incomplete Attention -> E') <-- Partial attention 
 * ---------------------------------------------------------------------------------
 *                                                   Total Sum:-> Complete Attention
 * vector of complete attention -> full attention
 */
#ifndef ATTENTION_HPP
#define ATTENTION_HPP 1

#include <vector>
#include <maths.h>
#include "mlp.hpp"

/**
 * @brief ATTENTION CLASS for calculating incomplete attention.
 * An array of incomplete attention is Partial Attention (LAYER) 
 * and an array of partial attention (BLOCK) is complete attention.
 */
class attention {
public:
    int n;              // total tokens for each attention head
    int d;              // token dimension
    std::vector<std::vector<double>> head;      // attention head matrix
    mat MQ;             // query matrix
    mat MK;             // key matrix
    mat MV;             // vertical value for deltas
    mat MH;             // horizontal value for deltas
    std::vector<std::vector<double>> tokens;    // Tokens for attention head
    std::vector<std::vector<double>> next;      // Vertical tokens for attention head to be transferred to next block
    std::vector<std::vector<double>> dH;        // Weighted Sums Horizontally
    std::vector<std::vector<double>> dV;        // Weighted Sums Vertically
    mlp v;              // next block transfer for cross attention
    mlp h;              // horizontal transfer for self+cross attention
    std::vector<double> EH;         // Next Embedding in same block
    std::vector<double> EV;         // Next Embedding for next Block
    std::vector<double> dh;         // dH sum
    std::vector<double> dv;         // dV sum
    std::vector<double> changeH;    // change in Horizontal process
    std::vector<double> changeV;    // change in Vertical process

    attention();
    attention(int n, int d);

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
    std::vector<std::vector<attention>> b;      // block for complete attention
    std::vector<std::vector<std::vector<double>>> inbetween;    // inbetween tokens transfer

    block(int n) {
        this->n = n;
        b = std::vector<std::vector<attention>>(n, std::vector<attention>(n));
        // 
    }

    void forward();
    void backward();
    void train();
    
    ~block() = default;
};

#endif
