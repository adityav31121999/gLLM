
// transformer.hpp: header source for transformer class
#ifndef TRANSFORMER_HPP
#define TRANSFORMER_HPP 1

#include <string>
#include <cmath>
#include <vector>
#include <iostream>
#include "mlp.hpp"
#include "attention.hpp"
#include "block.hpp"

/**
 * @brief Common Transformer class for token/chunk prediction and context 
 * retention and grammatical restriction. This can be single block or multi-block
 * architecture.
 */
class transformer {
public:
// these are constant values, once constructor is called, they cannot be changed
    int m;                  // number of blocks
    int y;                  // number of incomplete attentions in each partial attention
    int x;                  // number of layers of partial attention for complete attention block
    int n;                  // total tokens for each attention head or context window
    int d;                  // token dimension
    int h;                  // height of MQ, MK and columns of MV, MH
    int l;                  // layers of mlp
    int epochs;             // number of epochs for MLPs and Blocks
    float learning;         // learning rate for MLPs
    bool isSelf;            // if self attention or cross attention
    int totalParams;        // total parameters of transformer

// these are variables that change during training
    int blockCount;         // which block is working
    int epochCount;         // epoch counter
    int promptCount;        // number of tokens in the prompt
    int currentTokenCount;  // current count of tokens in full context

// these are variables that change during runtime
    int trainCount;         // total training count
    float error;            // error for transformer
    bool isTerminate;       // when '@#0' is calculated, to end the forward propagation

    std::vector<block> t;               // attention block (1 or many)
    std::vector<std::vector<float>> tokenEmbed;         // token embedding
    std::vector<std::vector<float>> input;              // input embeddings
    std::vector<std::vector<float>> output;             // output embeddings
    std::vector<std::vector<float>> embeddings;         // all glove embeddings with 64D
    FILE* promptNresponse;  // prompt and response text file
    // when model is in use, hold EV of all the blocks here
    std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> EVs;

    // default constructor
    transformer() = default;
    transformer(int x, int y, int n, int d, int h, int l, int vocab);
    transformer(int m, int x, int y, int n, int d, int h, int l, int vocab);
    transformer(int x, int y, int n, int d, int h, int l, int vocab, bool attentionType);
    transformer(int m, int x, int y, int n, int d, int h, int l, int vocab, bool attentionType);
    transformer(int m, int x, int y, int n, int d, int h, int l, int vocab, bool attentionType, bool& CASE);

    void setDims(int m, int x, int y, int n, int d, int h, int l);      // set dimension of transformer
    void setLearning(float learning);       // set learning rate for MLPs
    void setEpochs(int epochs);             // set epochs for MLPs
    void setAttention(bool attentionType);  // set self attention (1) or cross attention (0)
    void setInput(std::vector<std::vector<float>>& input);      // set input embeddings for running transformer
    void setOutput(std::vector<std::vector<float>>& output);    // set output embeddings for instructing and finetuning
    void setEmbedding(std::string& embedding);      // set token embeddings

    void computeKdotQs(int& promptCount, int& currentTokenCount, int& blockCount, bool& isSelf);
    void forward(int& blockCount, int& currentTokenCount, int& promptCount);
    void backward(std::vector<float>& expectedH);
    void backward(std::vector<float>& expectedH, int& blockCount);
    void backward(std::vector<std::vector<float>>& expectedH);
    void backward(std::vector<std::vector<float>>& expectedH, int& blockCount);
    void train(int& promptCount, int& currentTokenCount, int& blockCount, bool& isSelf, std::vector<float>& expected);
    void train(int& promptCount, int& currentTokenCount, int& blockCount, bool& isSelf, std::vector<std::vector<float>>& expected);
    void train(std::vector<std::vector<float>>& sentence);
    void train(std::vector<std::vector<std::vector<float>>>& sentences);
    void train(std::vector<std::vector<float>>& prompt, std::vector<std::vector<float>>& response);
    void train(std::vector<std::vector<std::vector<float>>>& prompts, std::vector<std::vector<std::vector<float>>>& responses);
    void computeOutput(std::vector<float>& output, std::vector<std::vector<float>>& embeddings, int& voc, int& index);
    void run();

#ifdef USE_CUDA     // cuda implementation
    void cuComputeKdotQs(int& promptCount, int& currentTokenCount, int& blockCount, bool& isSelf);
    void cuForward();
    void cuBackward(std::vector<float>& expectedH);
    void cuBackward(std::vector<float>& expectedH, int& blockCount);
    void cuBackward(std::vector<std::vector<float>>& expectedH);
    void cuBackward(std::vector<std::vector<float>>& expectedH, int& blockCount);
    void cuBackward(std::vector<std::vector<std::vector<float>>>& expectedH);
    void cuBackward(std::vector<std::vector<std::vector<float>>>& expectedH, int& blockCount);
    void cuTrain(int& promptCount, int& currentTokenCount, int& blockCount, bool& isSelf, std::vector<float>& expected);
    void cuTrain(int& promptCount, int& currentTokenCount, int& blockCount, bool& isSelf, std::vector<std::vector<float>>& expected);
    void cuTrain(std::vector<std::vector<float>>& sentence);
    void cuTrain(std::vector<std::vector<std::vector<float>>>& sentences);
    void cuTrain(std::vector<std::vector<float>>& prompt, std::vector<std::vector<float>>& response);
    void cuTrain(std::vector<std::vector<std::vector<float>>>& prompts, std::vector<std::vector<std::vector<float>>>& responses);
    void cuComputeOutput(std::vector<float>& output, std::vector<float>& prediction, int voc);
    void cuRun();
#elif USE_OPENCL    // opencl implementation
    void clComputeKdotQs(int& promptCount, int& currentTokenCount, int& blockCount, bool& isSelf);
    void clForward();
    void clBackward(std::vector<float>& expectedH);
    void clBackward(std::vector<float>& expectedH, int& blockCount);
    void clBackward(std::vector<std::vector<float>>& expectedH);
    void clBackward(std::vector<std::vector<float>>& expectedH, int& blockCount);
    void clBackward(std::vector<std::vector<std::vector<float>>>& expectedH);
    void clBackward(std::vector<std::vector<std::vector<float>>>& expectedH, int& blockCount);
    void clTrain(int& promptCount, int& currentTokenCount, int& blockCount, bool& isSelf, std::vector<float>& expected);
    void clTrain(int& promptCount, int& currentTokenCount, int& blockCount, bool& isSelf, std::vector<std::vector<float>>& expected);
    void clTrain(std::vector<std::vector<float>>& sentence);
    void clTrain(std::vector<std::vector<std::vector<float>>>& sentences);
    void clTrain(std::vector<std::vector<float>>& prompt, std::vector<std::vector<float>>& response);
    void clTrain(std::vector<std::vector<std::vector<float>>>& prompts, std::vector<std::vector<std::vector<float>>>& responses);
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
    void cuComputeKeys(std::vector<float>& t, std::vector<std::vector<float>>& m, std::vector<float>& k);
    void cuComputeQuerys(std::vector<float>& t, std::vector<std::vector<float>>& m, std::vector<float>& q);
#elif USE_OPENCL
    // opencl implementation
    void clComputeDot(std::vector<float>& T, std::vector<std::vector<float>>& M, float& dot);
    void clComputeDot(std::vector<float>& T1, std::vector<float>& T2, std::vector<std::vector<float>>& M, float& dot);
    void clComputeDot(std::vector<float>& Ti, mat M, std::vector<float>& Tj, float& dot);
    void clComputeKdotQ(std::vector<std::vector<float>>& KdotQ, std::vector<std::vector<float>>& tokenEmbed, mat& M, int& currentTokenCount, 
        int& promptCount, bool& attentionType);
    void clComputeKdotQ(std::vector<std::vector<float>>& KdotQ, std::vector<std::vector<float>>& K, std::vector<std::vector<float>>& Q, 
        int& currentTokenCount, int& promptCount, bool& attentionType);
    void clComputeKeys(std::vector<float>& t, std::vector<std::vector<float>>& m, std::vector<float>& k);
    void clComputeQuerys(std::vector<float>& t, std::vector<std::vector<float>>& m, std::vector<float>& q);
#endif

#endif

/*
// i dont know why i it is needed, ai suggested it
// Singleton pattern implementation
    static transformer& getInstance() {
        if (instance == nullptr) {
            instance = new transformer();
        }
        return *instance;
    }
*/
