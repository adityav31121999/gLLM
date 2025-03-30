
// transformer.hpp: header source for transformer class
#ifndef TRANSFORMER_HPP
#define TRANSFORMER_HPP 1

#include "attention.hpp"
#include "block.hpp"
#include <string>
#include <cmath>
#include <fstream>
#include <vector>
#include <iostream>

#define NUMBER_OF_PA 8              // number of Partial Attentions in one Block
#define NUMBER_OF_HEADS 32          // number of heads in each layer (partial attention)
#define NUMBER_OF_BLOCKS 8          // number of blocks in transformer

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
    bool isTerminate;       // 

    // default constructor
    transformer() = default;
    transformer(int x, int y, int n, int d, int h, int l, int vocab);
    transformer(int m, int x, int y, int n, int d, int h, int l, int vocab);

    void setLearning(float learning);       // set learning rate for MLPs
    void setEpochs(int epochs);             // set epochs for MLPs
    void setReps(int reps);                 // set repetitions for conversation
    void setAttention(bool attentionType);      // set self attention or cross attention
    void setPrompts(std::string prompts);   // set prompts file

// training
    void forward();
    void backward();
    void train();           // train with feedforward()
    void instruct();        // instruct the transformer to do something
    void computeOutput(std::vector<float>& output, std::vector<float>& prediction, int voc);    // compute output

#ifdef USE_CUDA
    // cuda implementation
#elif USE_OPENCL
    // opencl implementation
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

/**
 * to make and train model, initiate memory for the model first which is not
 * RAM or VRAM, use secondary memory in different drive so that a good amount
 * memory is available. This helps in two-way process where all FFN data can
 * be stored and then backprop can be done to alter as per changes needed. All
 * the process will run on RAM and VRAM parallely alongwith operations and as 
 * one incomplete attention is completed shift data directly to the model file.
 */
