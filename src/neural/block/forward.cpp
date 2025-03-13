
#include "include/attention.hpp"
#include "include/block.hpp"

/**
 * @brief partial attention forward propagation
 * @param tokenEmbed token embeddings
 * @param dv 
 * @param EV
 * @param changeV
 * @param in
 * @param tokenCount
 * @param i
 * @param blockCount
 */
void block::partialforprop(std::vector<std::vector<double>>& tokenEmbed, std::vector<std::vector<double>>& dv, 
    std::vector<std::vector<double>>& EV, std::vector<std::vector<double>>& changeV, int& in, int& tokenCount, 
    int& i, int& blockCount)
{
    if(blockCount == 1) {
        // for one partial attention
        for(int j = 0; j < b[0].size(); j++) {
            b[i][j].forprop(tokenEmbed, dv[i], EV[i], changeV[i], in, tokenCount);      // incomplete attention forprop
            if(j == b[0].size() - 1) 
                break;
            b[i][j + 1].EH = b[i][j].EH;
        }
    }
    else {
        // for multiple partial attentions
        for(int j = 0; j < b[0].size(); j++) {
            b[i][j].forprop(tokenEmbed, dv[i], EV[i], changeV[i], in, tokenCount);      // incomplete attention forprop
            if(j == b[0].size() - 1) 
                break;
            b[i][j + 1].EH = b[i][j].EH;
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
 * @param blockCount
 */
void block::forprop(std::vector<std::vector<double>>& tokenEmbed, std::vector<std::vector<std::vector<double>>>& dv, 
    std::vector<std::vector<std::vector<double>>>& EV, std::vector<std::vector<std::vector<double>>>& changeV, int& in, 
    int& tokenCount, int& blockCount)
{
    // y partial attention in x layers => x parallel processes
    while(1) {
        for(int i = 0; i < b.size(); i++) {
            partialforprop(tokenEmbed, dv[i], EV[i], changeV[i], in, tokenCount, i, blockCount);
        }
        // dembed the EH and check for "@#O" token
        // @#0 = and its over
        // check for end of tokens and break
        if(str == TERMINATE)
            break;
        // push EH to tokenEmbed vector, dembed the EH and push to tokens vector of transformer
        tokenCount++;
        if(tokenCount == b[0][0].head.size()) {
            break;
        }
    }
}
