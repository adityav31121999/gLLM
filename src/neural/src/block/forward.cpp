
#include "include/attention.hpp"
#include "include/block.hpp"

/**
 * @brief forward propagation for partial attention (layer of block)
 * @param in embedding dimensions
 * @param tokenCount current token count
 * @param i ith layer of block
 * @param layers number of layers of MLP
 */
void block::partialforprop(int& in, int& tokenCount, int i, int& layers) 
{
    // initialize the horizontal embedding vector to 0
    for (int j = 0; j < y; j++) {
        b[i][j].EH = std::vector<float>(in, 0.0f);
    }

    // for one partial attention
    for(int j = 0; j < y; j++) {
        b[i][j].forprop(in, layers, tokenCount);      // incomplete attention forprop
        // break when last head forprop is done
        if(j == (y - 1))
            break;
        b[i][j + 1].EH = b[i][j].EH;
    }
}


/**
 * @brief forward propagation for partial attention (ith layer of kth block)
 * @param EVp EVs of previous blocks ith layer
 * @param in embedding dimensions
 * @param tokenCount current token count
 * @param k current block number
 * @param i ith layer of current block
 * @param layers number of layers of MLP
 * @param n number of tokens for each head (context window)
 */
void block::partialforprop(std::vector<std::vector<std::vector<float>>>& EVp, int& in, int& tokenCount, int& k, int i,
    int& layers, int& n)
{
    // initialize the horizontal embedding vector to 0
    for (int j = 0; j < y; j++) {
        b[i][j].EH = std::vector<float>(in, 0.0f);
    }

    // for one partial attention
    for(int j = 0; j < y; j++) {
        b[i][j].forprop(EVp[i], in, layers, tokenCount, k, n);      // incomplete attention forprop
        // break when last head forprop is done
        if(j == (y - 1))
            break;
        b[i][j + 1].EH = b[i][j].EH;
    }
}


/**
 * @brief forward propagation for complete attention (for 1st block only)
 * @param in dimension of embedding
 * @param tokenCount current token count
 * @param layers layers of mlp
 */
void block::forprop(int& in, int& tokenCount, int& layers) 
{
    // y partial attention in x layers => x parallel processes
    for(int i = 0; i < x; i++) {
        // for all layers
        partialforprop(in, tokenCount, i, layers);
    }
}


/**
 * @brief forward propagation for complete attention (for kth block)
 * @param EVp previous block context retention vectors
 * @param in dimension of embedding
 * @param currentTokenCount current token count
 * @param layers layers of mlp
 * @param k current block count
 * @param n context window
 */
void block::forprop(std::vector<std::vector<std::vector<std::vector<float>>>>& EVp, int& in, int& currentTokenCount, int& k, 
    int& layers, int& n)
{
    // y partial attention in x layers => x parallel processes
    int tokenCount = std::abs(currentTokenCount - (n * k));
    // forward propagation for all layers
    for(int i = 0; i < x; i++) {
        partialforprop(EVp[i], in, tokenCount, k, i, layers, n);
    }
}
