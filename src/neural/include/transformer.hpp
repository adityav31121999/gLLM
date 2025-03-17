
/**
 * @file Transformer class for token/chunk prediction and context retention
 * and grammatical restriction. This can be single block or multi-block.
 * For training purpose, the model will have many-block transformer, but 
 * for prediction purpose, the model will have single-block transformer.
 * This will help in making the model more efficient and effective and also
 * reduce memory usage while prediction. Having multi-block transformer will
 * help in increasing the context size and also help in making the model more
 * responsive for large data.
 */

#ifndef TRANSFORMER_HPP
#define TRANSFORMER_HPP 1

#include "attention.hpp"
#include "block.hpp"
#include <string>

#define CONTEXT_SIZE 8192       // Attention head dimension: 2^13
#define TOKEN_IMIN 1            // token input
#define TOKEN_IMAX 16384        // 2^14
#define TOKEN_OMIN 1            // token output
#define TOKEN_OMAX 2097152      // 8192*256 = 2^21
#define BLOCK_MIN 1             // number of blocks in transformer
#define BLOCK_MAX 256           // 2^7
// #define TERMINATE "@#O"         // end of conversation

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
    int n;          // total tokens for each attention head
    int d;          // token dimension
    int h;          // height of MQ, MK and columns of MV, MH
    int l;          // layers of mlp
    int totalParams;        // total parameters of transformer
    int blockCount;         // which block is working
    int tokenCount;         // how many tokens have been generated
    int totalTokens;        // total tokens generated
    double learning;        // learning rate for MLPs
    int epochs;             // number of epochs for MLPs
    int reps;               // repetitions for conversation when totalTokens is reahced and more needed till TERMINATE
    std::vector<block> b;   // attention block (1 or many)
    std::vector<std::string> tinput;    // token input
    std::vector<std::string> expected;  // expected token output
    std::vector<std::string> toutput;   // predicted token output
    std::vector<std::string> token;     // Hold all input, generated or predicted tokens till TERMINATOR MEETS
    std::vector<std::vector<double>> tokenEmbed;        // token embedding
    std::vector<std::vector<std::vector<double>>> holdEVs;      // hold all EVs for backpropagation
    std::vector<std::vector<std::vector<double>>> holddVs;      // hold all dVs for backpropagation
    std::vector<std::vector<std::vector<double>>> changeVs;     // change in dVs for backpropagation
    std::vector<std::vector<std::vector<double>>> errMLP;       // error of all MLPs
    std::vector<std::vector<std::vector<double>>> errAtt;       // error of all attentions

    // default constructor
    transformer() = default;
    transformer(int x, int y, int n, int d, int h, int l);
    transformer(int m, int x, int y, int n, int d, int h, int l);
// training
    // forward propagation for single-block and multi-block
    void forward(std::vector<std::vector<double>>& tokenEmbed, std::vector<std::vector<double>>& KdotQ, std::vector<std::vector<double>>& K,
                    std::vector<std::vector<double>>& Q, std::vector<std::vector<std::vector<double>>>& dv, std::vector<std::vector<std::vector<double>>>& EV, 
                    std::vector<std::vector<std::vector<double>>>& changeV, int& in, int& tokenCount, int& layers);
    void forward(std::vector<std::vector<double>>& tokenEmbed, std::vector<std::vector<double>>& KdotQ, std::vector<std::vector<double>>& K,
                    std::vector<std::vector<double>>& Q, std::vector<std::vector<std::vector<double>>>& dv, std::vector<std::vector<std::vector<double>>>& EVp, 
                    std::vector<std::vector<std::vector<double>>>& EVc, std::vector<std::vector<std::vector<double>>>& changeV, int& in, int& tokenCount, 
                    int& layers, int& blockCount);
    // fine tune
    void fineTune();        // fine-tune transformer => altering certain properties while training
    // backpropagations
    void feedBack();        // self-correcting feedback loop
    void spoonfeed();       // spoonfeeding => first to last
    void backfeed();        // backfeeding => last to first
    // train functions
    void train();           // train with feedforward()
    void train4feedback();  // train with feedback()
    void train4spoonfeed(); // train with spoonfeed()
    void train4backfeed();  // train with backfeed()
    // instruct model
    void instruct();        // instruct the transformer to do something
// set properties for transformer
    void setLearning(double learning);      // set learning rate for MLPs

    // default destructor
    ~transformer() = default;
};

#endif

/**
 * to make and train model, initiate memory for the model first which is not
 * RAM or VRAM, use secondary memory in different drive so that a good amount
 * memory is available. This helps in two-way process where all FFN data can
 * be stored and then backprop can be done to alter as per changes needed. All
 * the process will run on RAM and VRAM parallely alongwith operations and as 
 * one incomplete attention is completed shift data directly to the model file.
 */
