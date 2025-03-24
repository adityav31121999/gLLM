
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
 * @param tokens token embeddings
 * @param holddv dv vector for next block
 * @param holdEV EV vector from this block
 * @param changeV vertical change vector for next block
 * @param in input token count
 * @param tokenCount token count for each attention head (how many tokens have been generated or taken as input)
 */
void attention::forprop(std::vector<std::vector<float>>& tokens, std::vector<std::vector<float>>& KdotQ, std::vector<std::vector<float>>& K,
    std::vector<std::vector<float>>& Q, std::vector<float>& dv, std::vector<float>& EV, std::vector<float>& changeV, int& in, int& layers, 
    int& tokenCount)
{
    // head calculation by inner product of KEYS and QUERYS
    if(tokenCount == 1) {
        // for single token input like "Hi", "Hello", "Hey", "How", "What", etc.
        K[0] = dot(tokens[0], MK);
        Q[0] = dot(tokens[0], MQ);
        KdotQ[0][0] = std::inner_product(K[0].begin(), K[0].end(), Q[0].begin(), 0.0)/SCALING;
    }
    else if(tokenCount == in) {
        // for a sentence and long prompt
        for(int i = 0; i < tokenCount; i++) {
            K[i] = dot(tokens[i], MK);
            Q[i] = dot(tokens[i], MQ);
        }
        for(int i = 0; i < tokenCount; i++) {
            for(int j = 0; j < tokenCount; j++) {
                KdotQ[i][j] = std::inner_product(K[i].begin(), K[i].end(), Q[j].begin(), 0.0)/SCALING;
            }
        }
    }
    else if(tokenCount > in) {
        // when new token is predicted or generated
        if(tokenCount - in == 1 || tokenCount - tokens.size() == 1) {
            K[tokenCount] = dot(tokens[0], MK);
            Q[tokenCount] = dot(tokens[0], MQ);
            KdotQ[tokenCount][tokenCount] = std::inner_product(K[tokenCount].begin(), K[tokenCount].end(), Q[tokenCount].begin(), 0.0)/SCALING;
            for(int j = 0; j < tokenCount-1; j++) {
                // head calculation
                KdotQ[tokenCount][j] = std::inner_product(K[tokenCount].begin(), K[tokenCount].end(), Q[j].begin(), 0.0)/SCALING;
                KdotQ[j][tokenCount] = std::inner_product(K[j].begin(), K[j].end(), Q[tokenCount].begin(), 0.0)/SCALING;
            }
        }
        else if(tokenCount - tokens.size() > 1) {
            // for next prediction
            int diff = tokenCount - tokens.size();
            K[tokenCount] = dot(tokens[tokenCount-1], MK);
            Q[tokenCount] = dot(tokens[tokenCount-1], MQ);
            KdotQ[tokenCount][tokenCount] = std::inner_product(K[tokenCount].begin(), K[tokenCount].end(), Q[tokenCount].begin(), 0.0)/SCALING;
            for(int i = tokens.size(); i < tokens.size() + diff - 1; i++) {
                for(int j = tokens.size(); j < tokens.size() + diff -1; j++) {
                    // head calculation: row
                    KdotQ[i][j] = std::inner_product(K[tokenCount].begin(), K[tokenCount].end(), Q[j].begin(), 0.0)/SCALING;
                    // head calculation: column
                    KdotQ[j][tokenCount] = std::inner_product(K[j].begin(), K[j].end(), Q[tokenCount].begin(), 0.0)/SCALING;
                }
            }
        }
        else {
            // for next prediction
            int diff = tokenCount - in;
            K[tokenCount] = dot(tokens[tokenCount-1], MK);
            Q[tokenCount] = dot(tokens[tokenCount-1], MQ);
            KdotQ[tokenCount][tokenCount] = std::inner_product(K[tokenCount].begin(), K[tokenCount].end(), Q[tokenCount].begin(), 0.0)/SCALING;
            for(int j = 0; j < diff; j++) {
                for(int i = 0; j < in + diff -1; i++) {
                    // head calculation
                    KdotQ[tokenCount][j] = std::inner_product(K[tokenCount].begin(), K[tokenCount].end(), Q[j].begin(), 0.0)/SCALING;
                    KdotQ[j][tokenCount] = std::inner_product(K[j].begin(), K[j].end(), Q[tokenCount].begin(), 0.0)/SCALING;
                }
            }
        }
    }
    // probability distribution
    int k, l;
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
    ver.input = EV + dv;
    hor.forward(in, layers);
    ver.forward(in, layers);
    // AND gate for the final output
    EH = EH + ReLUv(hor.output);
    EV = EV + ReLUv(ver.output);
}


/**
 * @brief forward propagation for a specific block's attention class (incomplete attention)
 * @param tokens token embeddings
 * @param dv dv vector
 * @param EVp EV vector from previous block
 * @param EVc EV vector for current block
 * @param changeV vertical change vector for current block
 * @param in input token count
 * @param tokenCount which token is being processed
 * @param blockCount which block is being processed
 * @param n number of tokens for each attention head
 */
void attention::forprop(std::vector<std::vector<float>>& tokens, std::vector<std::vector<float>>& KdotQ, std::vector<std::vector<float>>& K, 
    std::vector<std::vector<float>>& Q, std::vector<std::vector<float>>& EVp, std::vector<float>& dv, std::vector<float>& EVc, 
    std::vector<float>& changeV, int& in, int& layers, int& tokenCount, int& blockCount, int& n)
{
    // KdotQ calculation by inner product of KEYS and QUERYS
    if(blockCount == 0) {
        // for first block's attention class
        forprop(tokens, KdotQ, K, Q, dv, EVc, changeV, in, layers, tokenCount);
    }
    int count = tokenCount - n * blockCount;        // tokenCount for this head if blockCount block
    if(count == 1) {
        // for predicting first token of new block
        K[0] = dot(tokens[0], MK);
        Q[0] = dot(EVp[0], MQ);
        KdotQ[0][0] = std::inner_product(K[0].begin(), K[0].end(), Q[0].begin(), 0.0)/SCALING;
    }
    else {
        // for next prediction
        K[count] = dot(tokens[n*blockCount + count], MK);
        Q[count] = dot(EVp[count], MQ);
        KdotQ[count][count] = std::inner_product(K[count].begin(), K[count].end(), Q[count].begin(), 0.0)/SCALING;
        for(int j = 0; j < count - 1; j++) {
            // head calculation
            KdotQ[count][j] = std::inner_product(K[count].begin(), K[count].end(), Q[j].begin(), 0.0)/SCALING;
            KdotQ[j][count] = std::inner_product(K[j].begin(), K[j].end(), Q[count].begin(), 0.0)/SCALING;
        }
    }
    // probability distribution
    int k, l;
    head = LOTA(KdotQ, count);
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
    dh = dot(dh, MH);       // dh = weighted sums * MH
    dv = dot(dv, MV);       // dv = weighted sums * MV
    // get the required change from MLPs
    hor.input = EH + dh;
    ver.input = EVc + dv;
    hor.forward(in, layers);
    ver.forward(in, layers);
    // AND gate for the final output
    EH = EH + ReLUv(hor.output);
    EVc = EVc + ReLUv(ver.output);
}
