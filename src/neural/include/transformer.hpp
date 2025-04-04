
// transformer.hpp: header source for transformer class
#ifndef TRANSFORMER_HPP
#define TRANSFORMER_HPP 1

#include "mlp.hpp"
#include "attention.hpp"
#include "block.hpp"
#include <string>
#include <cmath>
#include <vector>
#include <iostream>

/**
 * @brief Common Transformer class for token/chunk prediction and context 
 * retention and grammatical restriction. This can be single block or multi-block
 * architecture.
 */
class transformer {
public:
    int m;                  // number of blocks
    int y;                  // number of incomplete attentions in each partial attention
    int x;                  // number of layers of partial attention for complete attention block
    int n;                  // total tokens for each attention head or context window
    int d;                  // token dimension
    int h;                  // height of MQ, MK and columns of MV, MH
    int l;                  // layers of mlp
    int totalParams;        // total parameters of transformer
    int blockCount;         // which block is working
    int tokenCount;         // how many tokens have been generated
    int totalTokens;        // total tokens generated
    int epochs;             // number of epochs for MLPs and Blocks
    int epochCount;         // epoch counter
    int trainCount;         // total training count
    float learning;         // learning rate for MLPs
    float error;            // error for transformer
    bool isSelf;            // if self attention or cross attention
    bool isTerminate;       // when '@#0' is calculated, to end the forward propagation

    std::vector<block> t;               // attention block (1 or many)
    std::vector<std::string> tinput;    // token input
    std::vector<std::string> expected;  // expected token output
    std::vector<std::string> toutput;   // predicted token output
    std::vector<std::string> token;     // Hold all input, generated or predicted tokens till TERMINATOR MEETS (Input + Expected/Output + Terminator)
    std::vector<std::vector<float>> tokenEmbed;         // token embedding
    std::vector<std::vector<float>> input;              // input embeddings
    std::vector<std::vector<float>> output;             // output embeddings
    std::vector<std::vector<float>> embeddings;         // all glove embeddings with 64D
    FILE* promptNresponse;  // prompt and response text file
    std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> EVs;        // when model in use

    // default constructor
    transformer() = default;
    transformer(int x, int y, int n, int d, int h, int l, int vocab);
    transformer(int m, int x, int y, int n, int d, int h, int l, int vocab);
    transformer(int x, int y, int n, int d, int h, int l, int vocab, bool attentionType);
    transformer(int m, int x, int y, int n, int d, int h, int l, int vocab, bool attentionType);

    void setDims(int m, int x, int y, int n, int d, int h, int l);      // set dimension of transformer
    void setLearning(float learning);       // set learning rate for MLPs
    void setEpochs(int epochs);             // set epochs for MLPs
    void setAttention(bool attentionType);  // set self attention (1) or cross attention (0)
    void setInput(std::vector<std::vector<float>>&);
    void setOutput(std::vector<std::vector<float>>&);
    void setEmbedding(std::string&);

    void computeKdotQs(int& promptCount, int& currentTokenCount, int& blockCount, bool& isSelf);
    void forward(int& blockCount, int& currentTokenCount, int& promptCount);
    void backward(std::vector<float>& expectedH);
    void backward(std::vector<float>& expectedH, int& blockCount);
    void backward(std::vector<std::vector<float>>& expectedH);
    void backward(std::vector<std::vector<float>>& expectedH, int& blockCount);
    void train(int& promptCount, int& currentTokenCount, int& blockCount, bool& isSelf, std::vector<float>& expected);
    void train(int& promptCount, int& currentTokenCount, int& blockCount, bool& isSelf, std::vector<std::vector<float>>& expected);
    void computeOutput(std::vector<float>& output, std::vector<std::vector<float>>& embeddings, int& voc, int& index);
    void run();

#ifdef USE_CUDA
    // cuda implementation
    void cuForward();
    void cuBackward(std::vector<float>& expectedH);
    void cuBackward(std::vector<float>& expectedH, int& blockCount);
    void cuBackward(std::vector<std::vector<float>>& expectedH);
    void cuBackward(std::vector<std::vector<float>>& expectedH, int& blockCount);
    void cuBackward(std::vector<std::vector<std::vector<float>>>& expectedH);
    void cuBackward(std::vector<std::vector<std::vector<float>>>& expectedH, int& blockCount);
    void cuTrain();
    void cuComputeOutput(std::vector<float>& output, std::vector<float>& prediction, int voc);
    void cuRun();
#elif USE_OPENCL
    // opencl implementation
    void clForward();
    void clBackward(std::vector<float>& expectedH);
    void clBackward(std::vector<float>& expectedH, int& blockCount);
    void clBackward(std::vector<std::vector<float>>& expectedH);
    void clBackward(std::vector<std::vector<float>>& expectedH, int& blockCount);
    void clBackward(std::vector<std::vector<std::vector<float>>>& expectedH);
    void clBackward(std::vector<std::vector<std::vector<float>>>& expectedH, int& blockCount);
    void clTrain();           // train with feedforward()
    void clComputeOutput(std::vector<float>& output, std::vector<float>& prediction, int voc);
    void clRun();
#endif

    // default destructor
    ~transformer() = default;
};


// compute functions for dot, KdotQ and other values

void computeDot(std::vector<float>& Ti, mat M, std::vector<float>& Tj, float& dot);
void computeDot(std::vector<float>& T, std::vector<std::vector<float>>& M, float& dot);
void computeDot(std::vector<float>& T1, std::vector<float>& T2, std::vector<std::vector<float>>& M, float& dot);
void computeKdotQ(std::vector<std::vector<float>>& KdotQ, std::vector<std::vector<float>>& tokenEmbed, mat& M, int& currentTokenCount, 
    int& promptCount, bool& attentionType);
void computeKdotQ(std::vector<std::vector<float>>& KdotQ, std::vector<std::vector<float>>& K, std::vector<std::vector<float>>& Q, 
    int& currentTokenCount, int& promptCount, bool& attentionType);
void computeKeys(std::vector<float>& t, std::vector<std::vector<float>>& m, std::vector<float>& k);
void computeQuerys(std::vector<float>& t, std::vector<std::vector<float>>& m, std::vector<float>& q);

#ifdef USE_CUDA
    // cuda implementation
    void cuComputeDot(std::vector<float>& T, std::vector<std::vector<float>>& M, float& dot);
    void cuComputeDot(std::vector<float>& T1, std::vector<float>& T2, std::vector<std::vector<float>>& M, float& dot);
    void cuComputeDot(std::vector<float>& Ti, mat M, std::vector<float>& Tj, float& dot);
    void cuComputeKdotQ(std::vector<std::vector<float>>& KdotQ, std::vector<std::vector<float>>& tokenEmbed, mat& M, int& currentTokenCount, 
        int& promptCount, bool& attentionType);
    void cuComputeKdotQ(std::vector<std::vector<float>>& KdotQ, std::vector<std::vector<float>>& K, std::vector<std::vector<float>>& Q, 
        int& currentTokenCount, int& promptCount, bool& attentionType);
#elif USE_OPENCL
    // opencl implementation
    void clComputeDot(std::vector<float>& T, std::vector<std::vector<float>>& M, float& dot);
    void clComputeDot(std::vector<float>& T1, std::vector<float>& T2, std::vector<std::vector<float>>& M, float& dot);
    void clComputeDot(std::vector<float>& Ti, mat M, std::vector<float>& Tj, float& dot);
    void clComputeKdotQ(std::vector<std::vector<float>>& KdotQ, std::vector<std::vector<float>>& tokenEmbed, mat& M, int& currentTokenCount, 
        int& promptCount, bool& attentionType);
    void clComputeKdotQ(std::vector<std::vector<float>>& KdotQ, std::vector<std::vector<float>>& K, std::vector<std::vector<float>>& Q, 
        int& currentTokenCount, int& promptCount, bool& attentionType);
#endif

#endif
