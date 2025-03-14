
#include "include/attention.hpp"
#include "include/block.hpp"

/**
 * @brief block training
 * @param tokenEmbed token embeddings
 * @param expected expected token embedding
 * @param dv all the dvs
 * @param EV all the EVs
 * @param changeV all the changeVs
 * @param in input token count
 * @param tokenCount token count
 * @param layers layers of mlp
 */
void block::train(std::vector<std::vector<double>>& tokenEmbed, std::vector<double> expected, std::vector<std::vector<std::vector<double>>>& dv, 
    std::vector<std::vector<std::vector<double>>>& EV, std::vector<std::vector<std::vector<double>>>& changeV, int& in, int& tokenCount, int& layers)
{
    while(1) {
        forprop(tokenEmbed, dv, EV, changeV, in, tokenCount, layers);
        backward(expected, changeV, dv, EV, in, layers);
        if(str == TERMINATE)
            break;
        tokenCount++;
        if(tokenCount == b[0][0].head.size()) {
            break;
        }
    }
}


/**
 * @brief block training
 * @param tokenEmbed token embeddings
 * @param expected expected token embedding
 * @param dv all the dvs
 * @param EV all the EVs
 * @param changeV all the changeVs
 * @param in input token count
 * @param tokenCount token count
 * @param layers layers of mlp
 * @param blockCount block count
 */
void block::train(std::vector<std::vector<double>>& tokenEmbed, std::vector<double> expected, std::vector<std::vector<std::vector<double>>>& dv, 
    std::vector<std::vector<std::vector<double>>>& EV, std::vector<std::vector<std::vector<double>>>& changeV, int& in, int& tokenCount, int& layers, 
    int blockCount)
{
    while(1) {
        forprop(tokenEmbed, dv, EV, changeV, in, tokenCount, layers, blockCount);
        backward(expected, changeV, dv, EV, in, layers);
        if(str == TERMINATE)
            break;
        tokenCount++;
        if(tokenCount == b[0][0].head.size()) {
            break;
        }
    }
}

