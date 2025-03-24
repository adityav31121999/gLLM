
#include "include/attention.hpp"
#include "include/block.hpp"
#include "include/transformer.hpp"

/**
 * @brief forward propagation for transformer (continuos training)
 * @param tokenEmbed token embedding
 * @param KdotQ KdotQ dot product for attention head
 * @param K keys => Tokens x MK
 * @param Q queries => Tokens x MQ
 * @param dv delta values
 * @param EV expected values
 * @param changeV change in values
 * @param in number of input tokens
 * @param tokenCount token count
 * @param layers layers in mlp
 * @param blockCount block count
 */
void transformer::forward(std::vector<std::vector<float>>& tokenEmbed, std::vector<std::vector<float>>& KdotQ, std::vector<std::vector<float>>& K,
    std::vector<std::vector<float>>& Q, std::vector<std::vector<std::vector<float>>>& dv, std::vector<std::vector<std::vector<float>>>& EV, 
    std::vector<std::vector<std::vector<float>>>& changeV, int& in, int& tokenCount, int& layers)
{
    if(m == 1) {
        // for single block-model/short-context training
        b[0].forprop(tokenEmbed, KdotQ, K, Q, dv, EV, changeV, in, tokenCount, layers);
    }
    else {
        // for multi-block model/long-context training
        for(int i = 0; i < m; i++) {
            b[i].forprop(tokenEmbed, KdotQ, K, Q, dv, EV, changeV, in, tokenCount, layers, i);
        }
    }
}


/**
 * @brief forward propagation for transformer (discrete training)
 * @param tokenEmbed token embedding
 * @param KdotQ KdotQ dot product for attention head
 * @param K keys => Tokens x MK
 * @param Q queries => Tokens x MQ
 * @param dv delta values
 * @param EVp previous block values
 * @param EVc current block values
 * @param changeV change in values
 * @param in input tokens
 * @param tokenCount token count
 * @param layers layers in mlp
 * @param blockCount block count
 */
void transformer::forward(std::vector<std::vector<float>>& tokenEmbed, std::vector<std::vector<float>>& KdotQ, std::vector<std::vector<float>>& K,
    std::vector<std::vector<float>>& Q, std::vector<std::vector<std::vector<float>>>& dv, std::vector<std::vector<std::vector<float>>>& EVp, 
    std::vector<std::vector<std::vector<float>>>& EVc, std::vector<std::vector<std::vector<float>>>& changeV, int& in, int& tokenCount, 
    int& layers, int& blockCount)
{
    b[blockCount].forprop(tokenEmbed, KdotQ, K, Q, dv, EVp, changeV, in, tokenCount, layers, blockCount);
}
