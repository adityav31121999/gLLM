
#include "include/attention.hpp"
#include "include/block.hpp"

/**
 * @brief partial attention forward propagation (layer of block)
 * @param in embedding dimensions
 * @param tokenCount current token count
 * @param i ith layer of block
 * @param layers number of layers of MLP
 */
void block::partialforprop(int& in, int& tokenCount, int i, int& layers)
{
    // for one partial attention
    for(int j = 0; j < y; j++) {
        b[i][j].forprop(in, layers, tokenCount);      // incomplete attention forprop
        if(j == (y - 1)) 
            break;
        b[i][j + 1].EH = b[i][j].EH;
    }
}


/**
 * @brief partial attention forward propagation
 * @param EVp EVs of previous blocks ith layer
 * @param in embedding dimensions
 * @param tokenCount current token count
 * @param blockCount current block number
 * @param i ith layer of current block
 * @param layers number of layers of MLP
 * @param n number of tokens for each head (context window)
 */
void block::partialforprop(std::vector<std::vector<std::vector<float>>>& EVp, int& in, int& tokenCount, int& blockCount, int i, int& layers, int& n)
{
    // for one partial attention
    for(int j = 0; j < y; j++) {
        b[i][j].forprop(EVp[i], in, layers, tokenCount, blockCount, n);      // incomplete attention forprop
        if(j == (y - 1)) 
            break;
        b[i][j + 1].EH = b[i][j].EH;
    }
}


/**
 * @brief forward propagation for complete attention
 * @param tokenEmbed token embeddings
 * @param dv
 * @param EV
 * @param changeV
 * @param in
 * @param tokenCount
 * @param layers
 */
void block::forprop(int& in, int& tokenCount, int& layers)
{
    // y partial attention in x layers => x parallel processes
    while(1) {
        for(int i = 0; i < x; i++) {
            partialforprop(in, tokenCount, i, layers);
        }
        // dembed the EH and check for "@#O" token
        // @#0 = and its over
        // check for end of tokens and break
        if(str == TERMINATE)
            break;
        // push EH to tokenEmbed vector, dembed the EH and push to tokens vector of transformer
        tokenCount++;
        if(tokenCount == CONTEXT_WIN) {
            break;
        }
    }
}


/**
 * @brief forward propagation for complete attention
 * @param tokenEmbed token embeddings
 * @param dv
 * @param EV
 * @param changeV
 * @param in
 * @param tokenCount
 * @param layers
 * @param blockCount
 */
void block::forprop(std::vector<std::vector<std::vector<std::vector<float>>>>& EVp, int& in, int& tokenCount, int& blockCount, int& layers, int& n)
{
    // y partial attention in x layers => x parallel processes
    while(1) {
        for(int i = 0; i < b.size(); i++) {
            partialforprop(EVp[i], in, tokenCount, blockCount, i, layers, n);
        }
        // dembed the EH and check for "@#O" token
        // @#0 = and its over
        // check for end of tokens and break
        if(str == TERMINATE)
            break;
        // push EH to tokenEmbed vector, dembed the EH and push to tokens vector of transformer
        tokenCount++;
        if(tokenCount == CONTEXT_WIN) {
            break;
        }
    }
}
