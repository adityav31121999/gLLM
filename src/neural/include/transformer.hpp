
// transformer.hpp: header source for transformer class
#ifndef TRANSFORMER_HPP
#define TRANSFORMER_HPP 1

#include "attention.hpp"
#include "block.hpp"
#include <string>
#include <cmath>

// token embedding
#define EMBED_MIN 64
#define EMBED_MAX (64*64*64)
// context window for each block
#define WINDOW_MIN 4096
#define WINDOW_MAX 8388608          // 4096*2048
// Weight matrix heights for MQ, MK, MV, MH
#define MATHEIGHT_MIN std::pow(EMBED_MIN, 2)
#define MATHEIGHT_MAX std::pow(EMBED_MAX, 2)
// token input and output for each block
#define TOKEN_IMIN 1
#define TOKEN_IMAX (WINDOW_MAX/2)
// number of blocks in transformer
#define BLOCK_MIN 1                 // number of blocks in transformer
#define BLOCK_MAX (WINDOW_MAX/WINDOW_MIN)
// token output for transformer
#define TOKEN_OMIN 1
#define TOKEN_OMAX (WINDOW_MAX*BLOCK_MAX)
// layers per block
#define BLAYER_MIN (8*BLOCK_MIN)
#define BLAYER_MAX (16*BLOCK_MAX)
// number of incomplete attentions in each partial attention
#define ATTENTION_MIN (4*LAYER_MIN)
#define ATTENTION_MAX (8*LAYER_MAX)
// number of layers in mlp
#define MLAYER_MIN (BLAYER_MIN*2)
#define MLAYER_MAX (BLAYER_MAX*8)

// properties for LLM
#define EMBEDDING 64        // token embedding
#define MATHEIGHTS 4096     // weight matrix heights
#define LAYERS_MLP 16       // layers of mlp
#define LEARNING 0.01       // learning rate for MLPs
#define EPOCHS 10           // number of epochs for MLPs


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
    int reps;               // repetitions for conversation when totalTokens is reahced and more needed till TERMINATE is met
    std::vector<block> b;   // attention block (1 or many)
    std::vector<std::string> tinput;    // token input
    std::vector<std::string> expected;  // expected token output
    std::vector<std::string> toutput;   // predicted token output
    std::vector<std::string> token;     // Hold all input, generated or predicted tokens till TERMINATOR MEETS
    std::vector<std::vector<float>> tokenEmbed;        // token embedding
    std::vector<std::vector<std::vector<float>>> holdEVs;      // hold all EVs for backpropagation
    std::vector<std::vector<std::vector<float>>> holddVs;      // hold all dVs for backpropagation
    std::vector<std::vector<std::vector<float>>> changeVs;     // change in dVs for backpropagation
    std::vector<std::vector<std::vector<float>>> errMLP;       // error of all MLPs
    int totalParams;        // total parameters of transformer
    int blockCount;         // which block is working
    int tokenCount;         // how many tokens have been generated
    int totalTokens;        // total tokens generated
    float error;            // error for transformer

    // default constructor
    transformer() = default;
    transformer(int x, int y, int n, int d, int h, int l);
    transformer(int m, int x, int y, int n, int d, int h, int l);

// training
    void forward(std::vector<std::vector<float>>& tokenEmbed, std::vector<std::vector<float>>& KdotQ, std::vector<std::vector<float>>& K,
                    std::vector<std::vector<float>>& Q, std::vector<std::vector<std::vector<float>>>& dv, std::vector<std::vector<std::vector<float>>>& EV, 
                    std::vector<std::vector<std::vector<float>>>& changeV, int& in, int& tokenCount, int& layers);
    void forward(std::vector<std::vector<float>>& tokenEmbed, std::vector<std::vector<float>>& KdotQ, std::vector<std::vector<float>>& K,
                    std::vector<std::vector<float>>& Q, std::vector<std::vector<std::vector<float>>>& dv, std::vector<std::vector<std::vector<float>>>& EVp, 
                    std::vector<std::vector<std::vector<float>>>& EVc, std::vector<std::vector<std::vector<float>>>& changeV, int& in, int& tokenCount, 
                    int& layers, int& blockCount);
    void fineTune();        // fine-tune transformer => altering certain properties while training
    void backward(std::vector<float>& expected, std::vector<std::vector<std::vector<float>>>& changeV, std::vector<std::vector<std::vector<float>>>& dv, 
                    std::vector<std::vector<std::vector<float>>>& EV, int& in, int& layers);
    void train();           // train with feedforward()
    void instruct();        // instruct the transformer to do something
    void computeOutput(std::vector<float>& output, std::vector<float>& prediction, int voc);    // compute output
// set properties for transformer
    void setLearning(float learning);      // set learning rate for MLPs
// run transformer
    void run();             // run the transformer

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
