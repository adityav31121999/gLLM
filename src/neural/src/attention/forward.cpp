
/**
 * IMP: keep keys and queries so that multiple times can be used from same place, 
 * and less calculation is needed, if not saved then multiple times the keys and 
 * queries need to be calculated. Also, memory used is low. Initiate all the 
 * memory based on the deimensions for this keys and queries so that continuous
 * resizing and pushback can be ignored for fast operations.
 * Only thing that takes large memory is KdotQ 2D vector which can take upto 99%
 * of memory in attention class.
 */

#ifdef USE_CPU

#include <numeric>
#include <algorithm>
#include "include/attention.hpp"
#include <maths.hpp>

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
    std::vector<std::vector<float>> head(tokenCount, std::vector<float>(tokenCount, 0.0f));
    // self attention (ReLU masking) or cross attention
    head = LOTA(KdotQ, tokenCount, isSelfAttention);
    // get weighted sums
    for(int i = 0; i < tokenCount; i++) {
        k = 0;
        l = 0;
        for(int j = 0; j < (isSelfAttention ? i : tokenCount); j++) {    
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
    ver.input = EV[tokenCount] + dv;
    hor.forward(in, layers);
    ver.forward(in, layers);
    // AND gate for the final output
    EH = EH + ReLU(hor.output);
    EV[tokenCount] = EV[tokenCount] + ReLU(ver.output);
}


/**
 * @brief forward propagation for a 2nd to last block's attention class (incomplete attention)
 * @param EVp EV vector from previous block
 * @param in input token count
 * @param layers layers of MLPs
 * @param tokenCount number of tokens in full context
 * @param blockCount which block is being processed in full context
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
    int count = std::abs(tokenCount - n * (blockCount-1));
    // probability distribution
    int k, l;
    std::vector<std::vector<float>> head(count, std::vector<float>(count, 0.0f));
    // self attention (ReLU masking) or cross attention
    head = LOTA(KdotQ, tokenCount, isSelfAttention);
    // get weighted sums
    for(int i = 0; i < count; i++) {
        k = 0;
        l = 0;
        for(int j = 0; j < (isSelfAttention ? i : count); j++) {
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
    ver.input = EV[count] + dv;
    hor.forward(in, layers);
    ver.forward(in, layers);
    // AND gate for the final output
    EH = EH + ReLU(hor.output);
    EV[count] = EV[count] + ReLU(ver.output);
}

#endif
