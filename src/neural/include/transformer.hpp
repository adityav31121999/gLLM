
// transformer.hpp: transformer body and its functions
#ifndef TRANSFORMER_HPP
#define TRANSFORMER_HPP 1

#include "attention.hpp"
#include <string>

#define TOKEN_IMIN 1            // token input
#define TOKEN_IMAX 16384        // 2^14
#define TOKEN_OMIN 1            // token output
#define TOKEN_OMAX 2097152      // 2^20 => 2093
#define BLOCK_MIN 1             // number of blocks in transformer
#define BLOCK_MAX 256           // 2^7

/**
 * @brief Common Transformer class for token/chunk prediction and context 
 * retention and grammatical restriction. This can be single block or multi-block
 * architecture.
 */
class transformer {
public:
    int m;          // number of blocks
    int total;      // total tokenLimit -> m * n
    int x;          // number of incomplete attentions in each partial attention
    int y;          // number of layers of partial attention for complete attention block
    int n;          // total tokens for each attention head
    int d;          // token dimension
    int h;          // height of MQ, MK and columns of MV, MH
    int l;          // layers of mlp
    int totalParams;        // total parameters of transformer
    std::string tinput;     // token input
    std::string toutput;    // token output
    std::vector<std::vector<double>> stringToken;   // sentence property input
    std::vector<block> attblock;    // attention block (1 or many)

    // default constructor
    transformer() = default;
    transformer(int m, int x, int y, int n, int d, int h, int l);

    void runTransformer();  // run transformer
    void forward();         // forward propagation
    void fineTune();        // fine-tune transformer
    void feedBack();        // feed back from user
    void instruct();        // instruct the transformer
    void spoonfeed();       // spoonfeeding
    void backfeed();        // backfeeding
    void train();           // train the transformer

    ~transformer();         // default destructor
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
