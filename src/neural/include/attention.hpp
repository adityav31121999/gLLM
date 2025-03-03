
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
#include <maths.hpp>
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
    int h;              // height of MQ, MK and columns of MV, MH
    int l;              // layers of mlp
    int tokenCount;     // token count
    int totalParams;    // total parameters in one incomplete attention
    double error;       // error for attention
    mlp ver;            // next block transfer for cross attention
    mlp hor;            // horizontal transfer for self+cross attention
    mat MQ;             // query matrix
    mat MK;             // key matrix
    mat MV;             // vertical value for deltas
    mat MH;             // horizontal value for deltas
    std::vector<std::vector<double>> tokens;    // tokens for attention head
    std::vector<std::vector<double>> KEYS;      // KEY vectors -> tokens * MK
    std::vector<std::vector<double>> QUERYS;    // QUERY vectors -> tokens * MQ
    std::vector<std::vector<double>> head;      // attention head matrix -> KEYs x QUERYS -> [K(i).Q(j)] <- scalar
    std::vector<std::vector<double>> next;      // Vertical tokens for attention head to be transferred to next block
    std::vector<std::vector<double>> tokB;      // Shady tokens obtained from next and previous blocks
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

    // default constructor
    attention() = default;
    attention(int n, int d, int h, int l);

    void forward();
    void backward();
    void train();
    void countParams();

    ~attention() = default;
};

/**
 * @brief partial attention class
 */
class pattention {
public:
    int x;          // number of incomplete attentions in partial attention
    int n, d, h, l;     // inputs for attention class constructor
    int tokenCount;     // token count
    int totalParams;    // total parameters of complete attention
    double error;       // error for block, mean of all incomplete attentions
    std::vector<std::vector<attention>> b;      // block for complete attention
    std::vector<std::vector<std::vector<double>>> holdEVs;    // inbetween tokens transfer

    pattention() = default;
    pattention(int x, int n, int d, int h, int l);

    void forward();
    void backward();
    void train();
    void countParams();

    ~pattention() = default;
};


/**
 * @brief block for complete attention
 */
class block {
public:
    int x;          // number of incomplete attentions in each partial attention
    int y;          // number of layers of partial attention for complete attention block
    int n, d, h, l;     // inputs for attention class constructor
    int tokenCount;     // token count
    int totalParams;    // total parameters of complete attention
    double error;       // error for block, mean of all incomplete attentions
    std::vector<std::vector<attention>> b;      // block for complete attention
    std::vector<std::vector<std::vector<double>>> holdEVs;    // inbetween tokens transfer

    block() = default;
    block(int x, int y, int n, int d, int h, int l);

    void paForward(int k);
    void forward();
    void backward();
    void train();
    void countParams();
    
    ~block() = default;
};

#endif
