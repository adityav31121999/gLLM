
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
 * retention and grammatical restriction
 */
class transformer {
public:
    std::string tinput;     // token input
    std::string toutput;    // token output
    std::vector<std::vector<double>> sinput;         // sentence property input
    std::vector<std::vector<double>> soutput;        // sentence property output
    
    block attblock;         // attention block

    transformer();         // default constructor

    void runTransformer();  // run transformer
    void fineTune();        // fine-tune transformer
    void feedBack();        // feed back from user
    void instruct();        // instruct the transformer

    ~transformer();         // default destructor
};


#endif
