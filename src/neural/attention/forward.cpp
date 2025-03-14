
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
 * @brief forward propagation for attention class (incomplete attention)
 * @param tokens token embeddings
 * @param holddv dv vector for next block
 * @param holdEV EV vector from this block
 * @param changeV vertical change vector for next block
 * @param in input token count
 * @param tokenCount token count for each attention head (hiw many tokens have been generated or taken as input)
 */
void attention::forprop(std::vector<std::vector<double>>& tokens, std::vector<double>& dv, std::vector<double>& EV,
    std::vector<double>& changeV, int& in, int& layers, int& tokenCount)
{
    // head calculation by inner product of KEYS and QUERYS
    if(tokenCount == 1) {
        K[0] = dot(tokens[0], MK);
        Q[0] = dot(tokens[0], MQ);
        KdotQ[0][0] = std::inner_product(K[0].begin(), K[0].end(), Q[0].begin(), 0.0)/SCALING;
    }
    else if(tokenCount == in) {
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
        K[tokenCount] = dot(tokens[0], MK);
        Q[tokenCount] = dot(tokens[0], MQ);
        KdotQ[tokenCount][tokenCount] = std::inner_product(K[tokenCount].begin(), K[tokenCount].end(), Q[tokenCount].begin(), 0.0)/SCALING;
        for(int j = 0; j < tokenCount-1; j++) {
            // head calculation
            KdotQ[tokenCount][j] = std::inner_product(K[tokenCount].begin(), K[tokenCount].end(), Q[j].begin(), 0.0)/SCALING;
            KdotQ[j][tokenCount] = std::inner_product(K[j].begin(), K[j].end(), Q[tokenCount].begin(), 0.0)/SCALING;
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
        dh = dh + (k * dot(tokens[i], MH));     // Ki.MV, dh = weighted sums horizontal
        dv = dv + (l * dot(tokens[i], MV));     // Qj.MH, dv = weighted sums vertical
    }
    dh = dot(dh, MH);       // dh = weighted sums * MH
    dv = dot(dv, MV);       // dv = weighted sums * MV
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
 * @brief forward propagation for attention class (incomplete attention)
 * @param tokens token embeddings
 * @param dv dv vector
 * @param EVp EV vector from previous block
 * @param EVc EV vector for current block
 * @param changeV vertical change vector for current block
 * @param in input token count
 * @param tokenCount token count for each attention head (hiw many tokens have been generated or taken as input)
 * @param blockCount block count for each attention head (how many blocks have been processed)
 * @param n number of tokens for each attention head
 */
void attention::forprop(std::vector<std::vector<double>>& tokens, std::vector<std::vector<double>>& EVp, std::vector<double>& dv,
    std::vector<double>& EVc, std::vector<double>& changeV, int& in, int& layers, int& tokenCount, int& blockCount, int& n)
{
    // KdotQ calculation by inner product of KEYS and QUERYS
    if(blockCount == 0) {
        forprop(tokens, dv, EVc, changeV, in, layers, tokenCount);
    }
    if((tokenCount - n*blockCount) == 1) {
        KdotQ[0][0] = std::inner_product(dot(tokens[0], MK).begin(), dot(tokens[0], MK).end(), dot(EVp[0], MQ).begin(), 0.0)/SCALING;
    }
    else {
        for(int i = 0; i < tokenCount - n*blockCount - 1; i++) {
            for(int j = 0; j < tokenCount - n*blockCount - 1; j++) {
                // head calculation
                KdotQ[i][j] = std::inner_product(dot(tokens[i], MK).begin(), dot(tokens[i], MK).end(), dot(tokens[i], MQ).begin(), 0.0)/SCALING;
            }
        }
        KdotQ[tokenCount][tokenCount] = std::inner_product(dot(tokens[tokenCount], MK).begin(), dot(tokens[tokenCount], MK).end(), dot(EVp[tokenCount], MQ).begin(), 0.0)/SCALING;
        for(int j = 0; j < tokenCount-1; j++) {
            // head calculation
            KdotQ[tokenCount][j] = std::inner_product(dot(tokens[tokenCount], MK).begin(), dot(tokens[tokenCount], MK).end(), dot(EVp[j], MQ).begin(), 0.0)/SCALING;
            KdotQ[j][tokenCount] = std::inner_product(dot(tokens[j], MK).begin(), dot(tokens[j], MK).end(), dot(EVp[tokenCount], MQ).begin(), 0.0)/SCALING;
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
        dh = dh + (k * dot(tokens[i], MH));     // Ki.MV, dh = weighted sums horizontal
        dv = dv + (l * dot(tokens[i], MV));     // Qj.MH, dv = weighted sums vertical
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


/**
 * @brief forward propagation for attention class (incomplete attention)
 * @param tokens token embeddings
 * @param holddv dv vector for next block
 * @param holdEV EV vector for next block
 * @param changeV vertical change vector for next block
 * @param in input token count
 * @param tokenCount token count for each attention head (hiw many tokens have been generated or taken as input)
void attention::forprop(std::vector<std::vector<double>>& tokens, std::vector<double>& dv, std::vector<double>& EV,
    std::vector<double>& changeV, int& in, int& tokenCount)
{
    // take total tokens available in the tokens embeddings and then make head
    double d = sqrt(head.size());
    // head calculation by inner product of KEYS and QUERYS
    if(tokenCount == 1) {
        KdotQ[0][0] = std::inner_product(dot(tokens[0], MK).begin(), dot(tokens[0], MK).end(), dot(tokens[0], MQ).begin(), 0.0)/d;
    }
    else if(tokenCount == in) {
        for(int i = 0; i < tokenCount; i++) {
            for(int j = 0; j < tokenCount; j++) {
                // head calculation
                KdotQ[i][j] = std::inner_product(dot(tokens[i], MK).begin(), dot(tokens[i], MK).end(), dot(tokens[i], MQ).begin(), 0.0)/d;
            }
        }
    }
    else if(tokenCount > in) {
        KdotQ[tokenCount][tokenCount] = std::inner_product(dot(tokens[tokenCount], MK).begin(), dot(tokens[tokenCount], MK).end(), dot(tokens[tokenCount], MQ).begin(), 0.0)/d;
        for(int j = 0; j < tokenCount-1; j++) {
            // head calculation
            KdotQ[tokenCount][j] = std::inner_product(dot(tokens[tokenCount], MK).begin(), dot(tokens[tokenCount], MK).end(), dot(tokens[j], MQ).begin(), 0.0)/d;
            KdotQ[j][tokenCount] = std::inner_product(dot(tokens[j], MK).begin(), dot(tokens[j], MK).end(), dot(tokens[tokenCount], MQ).begin(), 0.0)/d;
        }
    }
    // probability distribution
    int k, l;
    head = LOTA(KdotQ, tokenCount);
    for(int i = 0; i < tokenCount; i++) {
        k = 0;
        l = 0;
        for(int j = 0; j < tokenCount; j++) {    
            k = head[i][j];
            l = head[j][i];
        }
        dh = dh + (k * dot(tokens[i], MH));     // Ki.MV, dh = weighted sums horizontal
        dv = dv + (l * dot(tokens[i], MV));     // Qj.MH, dv = weighted sums vertical
    }
    // hold in change vectors, do not overuse the memory for this calculation
    // as it is not necessary calculate them every time
    changeH = dot(dh, MH);
    changeV = dot(dv, MV);
    // get the required change from MLPs
    hor.input = EH + dh;
    ver.input = EV + dv;
    hor.forward();
    ver.forward();
    // AND gate for the final output
    EH = EH + ReLUv(hor.output);
    EV = EV + ReLUv(ver.output);
}


 * @brief forward propagation for attention class (incomplete attention)
 * @param tokens token embeddings
 * @param dv dv vector
 * @param EVp EV vector from previous block
 * @param EVc EV vector for current block
 * @param changeV vertical change vector for current block
 * @param in input token count
 * @param tokenCount token count for each attention head (hiw many tokens have been generated or taken as input)
 * @param blockCount block count for each attention head (how many blocks have been processed)
 * @param n number of tokens for each attention head
void attention::forprop(std::vector<std::vector<double>>& tokens, std::vector<std::vector<double>>& EVp, std::vector<double>& dv,
    std::vector<double>& EVc, std::vector<double>& changeV, int& in, int& tokenCount, int& blockCount, int& n)
{
    // take total tokens available in the tokens embeddings and then make head
    double d = sqrt(KdotQ.size());
    // KdotQ calculation by inner product of KEYS and QUERYS
    if((tokenCount - n*blockCount) == 1) {
        KdotQ[0][0] = std::inner_product(dot(tokens[0], MK).begin(), dot(tokens[0], MK).end(), dot(EVp[0], MQ).begin(), 0.0)/d;
    }
    else {
        for(int i = 0; i < tokenCount - n*blockCount - 1; i++) {
            for(int j = 0; j < tokenCount - n*blockCount - 1; j++) {
                // head calculation
                KdotQ[i][j] = std::inner_product(dot(tokens[i], MK).begin(), dot(tokens[i], MK).end(), dot(tokens[i], MQ).begin(), 0.0)/d;
            }
        }
        KdotQ[tokenCount][tokenCount] = std::inner_product(dot(tokens[tokenCount], MK).begin(), dot(tokens[tokenCount], MK).end(), dot(EVp[tokenCount], MQ).begin(), 0.0)/d;
        for(int j = 0; j < tokenCount-1; j++) {
            // head calculation
            KdotQ[tokenCount][j] = std::inner_product(dot(tokens[tokenCount], MK).begin(), dot(tokens[tokenCount], MK).end(), dot(EVp[j], MQ).begin(), 0.0)/d;
            KdotQ[j][tokenCount] = std::inner_product(dot(tokens[j], MK).begin(), dot(tokens[j], MK).end(), dot(EVp[tokenCount], MQ).begin(), 0.0)/d;
        }
    }
    // probability distribution
    head = LOTA(KdotQ, tokenCount);
    for(int i = 0; i < tokenCount; i++) {
        // hold in change vectors, do not overuse the memory for this calculation
        // as it is not necessary calculate them every time
        changeH = dot(dot(tokens[i], MK), MH);
        changeV = dot(dot(tokens[i], MQ), MV);
        for(int j = 0; j < tokenCount; j++) {
            dh = dh + (KdotQ[i][j] * changeH);      // Ki.MV, dh = weighted sums horizontal
            dv = dv + (KdotQ[j][i] * changeV);      // Qj.MH, dv = weighted sums vertical
        }
    }
    // get the required change from MLPs
    hor.input = EH + dh;
    ver.input = EVc + dv;
    hor.forward();
    ver.forward();
    // AND gate for the final output
    EH = EH + ReLUv(hor.output);
    EVc = EVc + ReLUv(ver.output);
}
 */