
#include "include/attention.hpp"
#include "include/block.hpp"
#include "include/transformer.hpp"

/**
 * @brief 
 */
void transformer::forward(std::vector<std::vector<double>>& tokenEmbed, std::vector<std::vector<double>>& KdotQ, std::vector<std::vector<double>>& K,
    std::vector<std::vector<double>>& Q, std::vector<std::vector<std::vector<double>>>& dv, std::vector<std::vector<std::vector<double>>>& EV, 
    std::vector<std::vector<std::vector<double>>>& changeV, int& in, int& tokenCount, int& layers)
{
    if(m == 1) {
        // for single block-model/short-context chat
        b[0].forprop(tokenEmbed, KdotQ, K, Q, dv, EV, changeV, in, tokenCount, layers);
    }
    else {
        // for multi-block model/long-context chat
        for(int i = 0; i < m; i++) {
            b[i].forprop(tokenEmbed, KdotQ, K, Q, dv, EV, changeV, in, tokenCount, layers, i);
        }
    }
}


/**
 * @brief 
 */
void transformer::forward(std::vector<std::vector<double>>& tokenEmbed, std::vector<std::vector<double>>& KdotQ, std::vector<std::vector<double>>& K,
    std::vector<std::vector<double>>& Q, std::vector<std::vector<std::vector<double>>>& dv, std::vector<std::vector<std::vector<double>>>& EVp, 
    std::vector<std::vector<std::vector<double>>>& EVc, std::vector<std::vector<std::vector<double>>>& changeV, int& in, int& tokenCount, 
    int& layers, int& blockCount)
{
    b[blockCount].forprop(tokenEmbed, KdotQ, K, Q, dv, EVp, changeV, in, tokenCount, layers, blockCount);
}
