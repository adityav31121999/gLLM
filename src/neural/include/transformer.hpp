
// transformer.hpp: transformer body and its functions
#ifndef TRANSFORMER_HPP
#define TRANSFORMER_HPP 1

#include "attention.hpp"
#include <string>

#define TOKEN_IMIN 1
#define TOKEN_IMAX 8192         // 2^13
#define TOKEN_OMIN 1
#define TOKEN_OMAX 1048576      // 2^20
 
/**
 * @brief Common Transformer class for token/chunk prediction and context 
 * retention and grammatical restriction. This can be single block or multi-block
 * architecture.
 */
class transformer {
public:
    int total;      // total tokenLimit
    int m;          // number of blocks
    int x;          // number of incomplete attentions in each partial attention
    int y;          // number of layers of partial attention for complete attention block
    int n;          // total tokens for each attention head
    int d;          // token dimension
    int h;          // height of MQ, MK and columns of MV, MH
    int totalParams;        // total parameters of transformer
    std::string tinput;     // token input
    std::string toutput;    // token output
    std::vector<std::vector<double>> sinput;    // sentence property input
    std::vector<std::vector<double>> soutput;   // sentence property output
    std::vector<block> attblock;    // attention block (1 or many)

    transformer() = default;        // default constructor
    transformer(int m, int x, int y, int n, int d, int h);

    void runTransformer();  // run transformer
    void fineTune();        // fine-tune transformer
    void feedBack();        // feed back from user
    void instruct();        // instruct the transformer
    void spoonfeed();       // spoonfeeding
    void backfeed();        // backfeeding
    void countParams();     // count parameters

    ~transformer();         // default destructor
};

#endif
