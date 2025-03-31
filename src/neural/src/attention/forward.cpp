
// This file contains the forward propagation for the attention class
#include <numeric>
#include <algorithm>
#include "include/attention.hpp"
#include <maths.hpp>

/**
 * IMP: keep keys and queries so that multiple times can be used from same place, 
 * and less calculation is needed, if not used then multiple times the keys and 
 * queries need to be calculated. Also, memory used is low. Initiate all the 
 * memory for this keys and queries so that continuous resizing and pushback can 
 * be ignored for fast operations.
 */

/**
 * @brief forward propagation for first block's attention class (incomplete attention)
 * @param in embedding dimension
 * @param layers layers of hidden weights in mlp
 * @param tokenCount token count for each attention head (how many tokens have been generated or taken as input)
 */
void attention::forprop(int& in, int& layers, int& tokenCount)
{
    // probability distribution
    int k, l;
    std::vector<std::vector<float>> head = std::vector<std::vector<float>>(tokenCount, std::vector<float>(tokenCount, 0.0f));
    head = LOTA(KdotQ, tokenCount);
    for(int i = 0; i < tokenCount; i++) {
        k = 0;
        l = 0;
        for(int j = 0; j < tokenCount; j++) {    
            k += head[i][j];
            l += head[j][i];
        }
        dh = dh + (k * K[i]);     // Ki.MV, dh = weighted sums horizontal
        dv = dv + (l * Q[i]);     // Qj.MH, dv = weighted sums vertical
    }
    dh = dot(dh, MH);
    dv = dot(dv, MV);
    // get the required change from MLPs
    hor.input = EH + dh;
    hor.forward(in, layers);
    // AND gate for the final output
    EH = EH + ReLUv(hor.output);
    // set for current token count
    EV[tokenCount] = EV[tokenCount] + ReLUv(ver.output);
}


/**
 * @brief forward propagation for a specific block's attention class (incomplete attention)
 * @param EVp EV vector from previous block
 * @param in input token count
 * @param tokenCount which token is being processed
 * @param blockCount which block is being processed
 * @param n number of tokens for each attention head (context window)
 */
void attention::forprop(std::vector<std::vector<float>> EVp, int& in, int& layers, int& tokenCount, int& blockCount, int& n)
{
    // KdotQ calculation by inner product of KEYS and QUERYS
    if(blockCount == 0) {
        // for first block's attention class
        forprop(in, layers, tokenCount);
        return;
    }
    // number of tokens in context window of this block
    int count = std::abs(tokenCount - n * blockCount);
    // probability distribution
    int k, l;
    std::vector<std::vector<float>> head = std::vector<std::vector<float>>(tokenCount, std::vector<float>(tokenCount, 0.0f));
    head = LOTA(KdotQ, tokenCount);
    for(int i = 0; i < count; i++) {
        k = 0;
        l = 0;
        for(int j = 0; j < count; j++) {
            k += head[i][j];
            l += head[j][i];
        }
        dh = dh + (k * K[i]);     // Ki.MV, dh = weighted sums horizontal
        dv = dv + (l * Q[i]);     // Qj.MH, dv = weighted sums vertical
    }
    dh = dot(dh, MH);       // dh = weighted sums by row * MH
    dv = dot(dv, MV);       // dv = weighted sums by column * MH
    // get the required change from MLPs
    hor.input = EH + dh;
    hor.forward(in, layers);
    // AND gate for the final output
    EH = EH + ReLUv(hor.output);
    // set for token count with respect to this attention
    EV[count] = EV[count] + ReLUv(ver.output);
}
