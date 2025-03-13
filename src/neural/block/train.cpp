
#include "include/attention.hpp"
#include "include/block.hpp"

/**
 * @brief block training
 * @param tokenEmbed token embeddings
 * @param dv all the dvs
 * @param EV all the EVs
 * @param changeV all the changeVs
 * @param in input token count
 * @param tokenCount token count
 * @param blockCount block count
 * @param learning learning rate
 */
void block::train(std::vector<std::vector<double>>& tokenEmbed, std::vector<std::vector<std::vector<double>>>& dv, 
    std::vector<std::vector<std::vector<double>>>& EV, std::vector<std::vector<std::vector<double>>>& changeV, int& in, 
    int& tokenCount, int& blockCount)
{
    while(1) {
        forprop(tokenEmbed, dv, EV, changeV, in, tokenCount, blockCount);
        backward(tokenEmbed[tokenCount], changeV, dv, EV);
        if(str == TERMINATE)
            break;
        tokenCount++;
        if(tokenCount == b[0][0].head.size()) {
            break;
        }
    }
}
