
#include "include/attention.hpp"
#include "include/block.hpp"

/**
 * @brief partial attention forward propagation
 */
void block::partialforprop(std::vector<std::vector<double>> tokens, int tokenCount, int i) {
    // for one partial attention
    for(int j = 0; j < b[0].size(); j++) {
        b[i][j].forprop(tokens, tokenCount);      // incomplete attention forprop
        holdEVs[i][j] = holdEVs[i][j] + b[i][j].EV;
        if(j == b[0].size() - 1) 
            break;
        b[i][j + 1].EH = b[i][j].EH;
    }
}

/**
 * @brief forward propagation
 */
void block::forprop(std::vector<std::vector<double>> tokens, int tokenCount) {
    // y partial attention in x layers => x parallel processes
    while(1) {
        for(int i = 0; i < b.size(); i++) {
            partialforprop(tokens, tokenCount, i);
        }
        // @#0 = and its over
        // check for end of tokens and break
        if(str == "@#0")
            break;
        tokenCount++;
        if(tokenCount == b[0][0].head.size()) {
            break;
        }
    }
}
