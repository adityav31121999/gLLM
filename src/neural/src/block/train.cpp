
#include "include/attention.hpp"
#include "include/block.hpp"

/**
 * @brief FIRST block training
 * @param tokenEmbed token embeddings
 * @param expected expected token embedding
 * @param dv all the dvs
 * @param EV all the EVs
 * @param changeV all the changeVs
 * @param in input token count
 * @param tokenCount token count
 * @param layers layers of mlp
 */
void block::train(std::vector<std::vector<float>>& tokenEmbed, std::vector<float>& expected, std::vector<std::vector<float>>& KdotQ, std::vector<std::vector<float>>& K,
    std::vector<std::vector<float>>& Q, std::vector<std::vector<std::vector<float>>>& dv, std::vector<std::vector<std::vector<float>>>& EV, 
    std::vector<std::vector<std::vector<float>>>& changeV, int& in, int& tokenCount, int& layers)
{
    while(1) {
        forprop(tokenEmbed, KdotQ, K, Q, dv, EV, changeV, in, tokenCount, layers);
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
 * @brief INTERMEDIATE and END block training
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
void block::train(std::vector<std::vector<float>>& tokenEmbed, std::vector<float>& expected, std::vector<std::vector<float>>& KdotQ, std::vector<std::vector<float>>& K,
    std::vector<std::vector<float>>& Q, std::vector<std::vector<std::vector<float>>>& dv, std::vector<std::vector<std::vector<float>>>& EVp, 
    std::vector<std::vector<std::vector<float>>>& EVc, std::vector<std::vector<std::vector<float>>>& changeV, int& in, int& tokenCount, 
    int& layers, int& blockCount)
{
    while(1) {
        forprop(tokenEmbed, KdotQ, K, Q, dv, EVp, changeV, in, tokenCount, layers, blockCount);
        backward(expected, changeV, dv, EVp, in, layers);
        if(str == TERMINATE)
            break;
        tokenCount++;
        if(tokenCount == b[0][0].head.size()) {
            break;
        }
    }
}
