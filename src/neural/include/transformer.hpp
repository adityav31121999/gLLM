
// transformer.hpp: transformer body and its functions
#ifndef TRANSFORMER_HPP
#define TRANSFORMER_HPP 1

#include "attention.hpp"
#include <string>

#define TOKEN_IMIN 1            // token input
#define TOKEN_IMAX 8192         // 2^13
#define TOKEN_OMIN 1            // token output
#define TOKEN_OMAX 1048576      // 2^20
#define BLOCK_MIN 1             // number of blocks in transformer
#define BLOCK_MAX 128           // 2^7

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
    void countParams();     // count parameters

    ~transformer();         // default destructor
};

#endif
