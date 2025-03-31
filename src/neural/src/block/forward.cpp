
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
    // for one partial attention
    for(int j = 0; j < y; j++) {
        b[i][j].forprop(in, layers, tokenCount);      // incomplete attention forprop
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
    // for one partial attention
    for(int j = 0; j < y; j++) {
        b[i][j].forprop(EVp[i], in, layers, tokenCount, k, n);      // incomplete attention forprop
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
    while(1) {
        // forward propagation for all layers
        for(int i = 0; i < x; i++) {
            partialforprop(in, tokenCount, i, layers);
        }
        // dembed the EH and check for "@#O" token
        // @#0 = and its over
        // check for end of tokens and break
        if(str == TERMINATE)
            break;
        // push EH to tokenEmbed vector, dembed the EH and push to tokens vector of transformer
        for(int i = 0; i < x; i++) {
            for(int j = 0; j < y; j++) {
                // diagonal element
                b[i][j].KdotQ[tokenCount + 1][tokenCount + 1] = std::inner_product(b[i][j].K[tokenCount+1].begin(), 
                            b[i][j].K[tokenCount+1].end(), b[i][j].Q[tokenCount+1].begin(), 0.0f);
                // rows and columns
                for(int k = 0; k < tokenCount; k++) {
                    // row
                    b[i][j].KdotQ[tokenCount + 1][k] = std::inner_product(b[i][j].K[tokenCount+1].begin(), 
                            b[i][j].K[tokenCount+1].end(), b[i][j].Q[j].begin(), 0.0f);
                    // column
                    b[i][j].KdotQ[k][tokenCount + 1] = std::inner_product(b[i][j].K[j].begin(), b[i][j].K[j].end(), 
                            b[i][j].Q[tokenCount+1].begin(), 0.0f);
                }
            }
        }
        tokenCount++;
        // end the block forward propagation
        if(tokenCount == CONTEXT_WIN) {
            break;
        }
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
    int count = 0;
    while(1) {
        // forward propagation for all layers
        for(int i = 0; i < b.size(); i++) {
            partialforprop(EVp[i], in, tokenCount, k, i, layers, n);
        }
        // take sum of all EHs and then calculate the prediction for next token
        // dembed the EH and check for "@#O" token
        // @#0 = and its over
        // check for end of tokens and break
        if(str == TERMINATE)
            break;
        // push EH to tokenEmbed vector, dembed the EH and push to tokens vector of transformer
        for(int i = 0; i < x; i++) {
            for(int j = 0; j < y; j++) {
                // diagonal
                b[i][j].KdotQ[tokenCount + 1][tokenCount + 1] = std::inner_product(b[i][j].K[tokenCount+1].begin(), 
                            b[i][j].K[tokenCount+1].end(), b[i][j].Q[tokenCount+1].begin(), 0.0f);
                for(int k = 0; k < tokenCount; k++) {
                    // row
                    b[i][j].KdotQ[tokenCount + 1][k] = std::inner_product(b[i][j].K[tokenCount+1].begin(), 
                            b[i][j].K[tokenCount+1].end(), b[i][j].Q[j].begin(), 0.0f);
                    // column
                    b[i][j].KdotQ[k][tokenCount + 1] = std::inner_product(b[i][j].K[j].begin(), b[i][j].K[j].end(), 
                            b[i][j].Q[tokenCount+1].begin(), 0.0f);
                }
            }
        }
        tokenCount++;
        count++;
        // end block forward propagation
        if(tokenCount == CONTEXT_WIN) {
            break;
        }
    }
    currentTokenCount += count;
}
