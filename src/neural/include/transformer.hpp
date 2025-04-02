
// transformer.hpp: header source for transformer class
#ifndef TRANSFORMER_HPP
#define TRANSFORMER_HPP 1

#include "mlp.hpp"
#include "attention.hpp"
#include "block.hpp"
#include <string>
#include <cmath>
#include <fstream>
#include <vector>
#include <iostream>

/**
 * @brief Common Transformer class for token/chunk prediction and context 
 * retention and grammatical restriction. This can be single block or multi-block
 * architecture.
 */
class transformer {
public:
    int m;          // number of blocks
    int total;      // total tokenLimit -> m * n
    int y;          // number of incomplete attentions in each partial attention
    int x;          // number of layers of partial attention for complete attention block
    int n;          // total tokens for each attention head or context window
    int d;          // token dimension
    int h;          // height of MQ, MK and columns of MV, MH
    int l;          // layers of mlp
    float learning;        // learning rate for MLPs
    int epochs;             // number of epochs for MLPs
    // int reps;               // repetitions for conversation when totalTokens is reahced and more needed till TERMINATE is met
    std::vector<block> t;   // attention block (1 or many)
    std::vector<std::string> tinput;    // token input
    std::vector<std::string> expected;  // expected token output
    std::vector<std::string> toutput;   // predicted token output
    std::vector<std::string> token;     // Hold all input, generated or predicted tokens till TERMINATOR MEETS (Input + Expected/Output + Terminator)
    std::vector<std::vector<float>> tokenEmbed;         // token embedding
    std::vector<std::vector<float>> embeddings;         // glove embedding with 64D
    // std::vector<std::vector<std::vector<float>>> keys;           // hold all EVs for backpropagation
    // std::vector<std::vector<std::vector<float>>> queries;        // hold all dVs for backpropagation
    FILE* prompts;          // text file to hold all the prompts
    FILE* responses;        // text file to hold all the responses
    int totalParams;        // total parameters of transformer
    int blockCount;         // which block is working
    int tokenCount;         // how many tokens have been generated
    int totalTokens;        // total tokens generated
    float error;            // error for transformer
    bool isSelf;            // if self attention or cross attention
    bool toNextBlock;       // transfer to next block
    bool toUp;              // backpropagation to upward blocks
    bool isTerminate;       // when '@#0' is calculated, to end the forward propagation

    // default constructor
    transformer() = default;
    transformer(int x, int y, int n, int d, int h, int l, int vocab);
    transformer(int m, int x, int y, int n, int d, int h, int l, int vocab);
    transformer(int x, int y, int n, int d, int h, int l, int vocab, bool attentionType);
    transformer(int m, int x, int y, int n, int d, int h, int l, int vocab, bool attentionType);

    void setLearning(float learning);       // set learning rate for MLPs
    void setEpochs(int epochs);             // set epochs for MLPs
    void setReps(int reps);                 // set repetitions for conversation
    void setAttention(bool attentionType);      // set self attention or cross attention
    void setPrompts(std::string prompts);   // set prompts file

// training
    void forward();
    void backward(std::vector<float>& expectedH);
    void backward(std::vector<float>& expectedH, int& blockCount);
    void backward(std::vector<std::vector<float>>& expectedH);
    void backward(std::vector<std::vector<float>>& expectedH, int& blockCount);
    void backward(std::vector<std::vector<std::vector<float>>>& expectedH);
    void backward(std::vector<std::vector<std::vector<float>>>& expectedH, int& blockCount);
    void train();           // train with feedforward()
    void instruct();        // instruct the transformer to do something
    void computeOutput(std::vector<float>& output, std::vector<float>& prediction, int voc);    // compute output

#ifdef USE_CUDA
    // cuda implementation
    // training
    void cuForward();
    void cuBackward(std::vector<float>& expectedH);
    void cuBackward(std::vector<float>& expectedH, int& blockCount);
    void cuBackward(std::vector<std::vector<float>>& expectedH);
    void cuBackward(std::vector<std::vector<float>>& expectedH, int& blockCount);
    void cuBackward(std::vector<std::vector<std::vector<float>>>& expectedH);
    void cuBackward(std::vector<std::vector<std::vector<float>>>& expectedH, int& blockCount);
    void cuTrain();           // train with feedforward()
    void cuInstruct();        // instruct the transformer to do something
    void cuComputeOutput(std::vector<float>& output, std::vector<float>& prediction, int voc);    // compute output
#elif USE_OPENCL
    // opencl implementation
    // training
    void clForward();
    void clBackward(std::vector<float>& expectedH);
    void clBackward(std::vector<float>& expectedH, int& blockCount);
    void clBackward(std::vector<std::vector<float>>& expectedH);
    void clBackward(std::vector<std::vector<float>>& expectedH, int& blockCount);
    void clBackward(std::vector<std::vector<std::vector<float>>>& expectedH);
    void clBackward(std::vector<std::vector<std::vector<float>>>& expectedH, int& blockCount);
    void clTrain();           // train with feedforward()
    void clInstruct();        // instruct the transformer to do something
    void clComputeOutput(std::vector<float>& output, std::vector<float>& prediction, int voc);    // compute output
#endif

// run transformer
    void run();             // run the transformer

    // default destructor
    ~transformer() = default;
};


// compute functions for dot, KdotQ and other values

void computeDot(std::vector<float>& T, std::vector<std::vector<float>>& M, float& dot);
void computeDot(std::vector<float>& T1, std::vector<float>& T2, std::vector<std::vector<float>>& M, float& dot);
void computeDot(std::vector<float>& Ti, mat M, std::vector<float>& Tj, float& dot);
void computeKdotQ(std::vector<std::vector<float>>& KdotQ, std::vector<std::vector<float>>& K, std::vector<std::vector<float>>& Q, 
    int& currentTokenCount, int& promptCount);
void computeKdotQ(std::vector<std::vector<float>>& KdotQ, std::vector<std::vector<float>>& tokenEmbed, mat& M, int& currentTokenCount, 
    int& promptCount);

#endif
