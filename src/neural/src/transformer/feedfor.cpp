
#include "include/attention.hpp"
#include "include/block.hpp"
#include "include/transformer.hpp"

/**
 * @brief 
 */
void transformer::forward(std::vector<std::vector<float>>& tokenEmbed, std::vector<std::vector<float>>& KdotQ, std::vector<std::vector<float>>& K,
    std::vector<std::vector<float>>& Q, std::vector<std::vector<std::vector<float>>>& dv, std::vector<std::vector<std::vector<float>>>& EV, 
    std::vector<std::vector<std::vector<float>>>& changeV, int& in, int& tokenCount, int& layers)
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
void transformer::forward(std::vector<std::vector<float>>& tokenEmbed, std::vector<std::vector<float>>& KdotQ, std::vector<std::vector<float>>& K,
    std::vector<std::vector<float>>& Q, std::vector<std::vector<std::vector<float>>>& dv, std::vector<std::vector<std::vector<float>>>& EVp, 
    std::vector<std::vector<std::vector<float>>>& EVc, std::vector<std::vector<std::vector<float>>>& changeV, int& in, int& tokenCount, 
    int& layers, int& blockCount)
{
    b[blockCount].forprop(tokenEmbed, KdotQ, K, Q, dv, EVp, changeV, in, tokenCount, layers, blockCount);
}
