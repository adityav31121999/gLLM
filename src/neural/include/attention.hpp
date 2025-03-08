
/**
 * @file Attention class for calculating attention in neural networks. Attention is 
 * a mechanism that allows a neural network to focus on a specific part of the input 
 * sequence.
 * incomplete attention block
 *          \/
 * ---------------------------------------------------------------------------------
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
#include <maths.hpp>
#include "mlp.hpp"

/**
 * @brief ATTENTION CLASS for calculating incomplete attention.
 * An array of incomplete attention is Partial Attention (LAYER) 
 * and an array of partial attention (BLOCK) is complete attention.
 */
class attention {
public:
// variables
    int n;              // total tokens for each attention head
    int d;              // token dimension
    int h;              // height of MQ, MK and columns of MV, MH
    int l;              // layers of mlp
    double error;       // error for attention
    double learning;    // learning rate
// operands
    mlp ver;            // next block transfer for cross attention
    mlp hor;            // horizontal transfer for self+cross attention
    mat MQ;             // query matrix
    mat MK;             // key matrix
    mat MV;             // vertical value for deltas
    mat MH;             // horizontal value for deltas
// containers
    std::vector<std::vector<double>> head;      // attention head matrix -> KEYs x QUERYs -> [K(i).Q(j)] <- scalar
    std::vector<std::vector<double>> KEYS;      // KEY vectors -> tokens * MK
    std::vector<std::vector<double>> QUERYS;    // QUERY vectors -> tokens * MQ
    std::vector<std::vector<std::vector<double>>> dH;        // Weighted Sums Horizontally
    std::vector<std::vector<std::vector<double>>> dV;        // Weighted Sums Vertically
    std::vector<double> EH;         // Next Embedding in same block
    std::vector<double> EV;         // Next Embedding for next Block
    std::vector<double> dh;         // dH sum
    std::vector<double> dv;         // dV sum
    std::vector<double> mh;         // ReLU of hor output
    std::vector<double> mv;         // ReLU of ver output
    std::vector<double> changeH;    // change in Horizontal process as expected vector for backpropagation in hor mlp
    std::vector<double> changeV;    // change in Vertical process as expected vector for backpropagation in ver mlp
    std::vector<double> tExp;       // expected token/change

// functions
    // default constructor
    attention() = default;
    attention(int n, int d, int h, int l);

    void forprop(std::vector<std::vector<double>>, int);                 // CASE default
    void backward(std::vector<double>);
    void train(std::vector<std::vector<double>>, int);

    // default destructor
    ~attention() = default;
};


/**
 * @brief block for complete attention
 */
class block {
public:
    int x;          // number of partial attentions in complete attention
    int y;          // number of incomplete attention for each partial attention
    int n, d, h, l;     // inputs for attention class constructor
    int tokenCount;     // token count
    int totalParams;    // total parameters of complete attention
    double error;       // error for block, mean of all incomplete attentions
    std::vector<std::vector<double>> tokens;    // tokens for attention head
    std::vector<std::vector<attention>> b;      // block complete attention
    std::vector<std::vector<std::vector<double>>> holdEVs;  // inbetween tokens transfer
    std::vector<std::vector<double>> expected;  // tokens for attention head

    // default constructor
    block() = default;
    block(int x, int y, int n, int d, int h, int l);

    void partialforprop(int i);     // partial attention forprop
    void forprop();                 // parallel partialforprop(i)
    void partialbackward(int i);    // partial attention backward
    void backward();                // parallel partialbackward(i)
    void train();

    // default destructor
    ~block() = default;
};

// ADD CUDA FUNCTIONS FOR RUNNING OPERATIONS IN PARALLEL



// ADD OPENCL FUNCTIONS FOR RUNNING OPERATIONS IN PARALLEL



#endif
