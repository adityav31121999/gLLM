
#include "include/attention.hpp"
#include <maths.hpp>
#include <numeric>
#include <algorithm>

/**
 * @brief forward propagation for attention class (incomplete attention)
 * @param tokens token embeddings
 * @param holddv dv vector for next block
 * @param holdEV EV vector for next block
 * @param changeV vertical change vector for next block
 * @param in input token count
 * @param tokenCount token count for each attention head (hiw many tokens have been generated or taken as input)
 */
void attention::forprop(std::vector<std::vector<double>>& tokens, std::vector<double>& dv, std::vector<double>& EV,
    std::vector<double>& changeV, int& in, int& tokenCount)
{
    // take total tokens available in the tokens embeddings and then make head
    double d = sqrt(head.size());
    // head calculation by inner product of KEYS and QUERYS
    if(tokenCount == 1) {
        head[0][0] = std::inner_product(dot(tokens[0], MK).begin(), dot(tokens[0], MK).end(), dot(tokens[0], MQ).begin(), 0.0)/d;
    }
    else if(tokenCount == in) {
        for(int i = 0; i < tokenCount; i++) {
            for(int j = 0; j < tokenCount; j++) {
                // head calculation
                head[i][j] = std::inner_product(dot(tokens[i], MK).begin(), dot(tokens[i], MK).end(), dot(tokens[i], MQ).begin(), 0.0)/d;
            }
        }
    }
    else if(tokenCount > in) {
        head[tokenCount][tokenCount] = std::inner_product(dot(tokens[tokenCount], MK).begin(), dot(tokens[tokenCount], MK).end(), dot(tokens[tokenCount], MQ).begin(), 0.0)/d;
        for(int j = 0; j < tokenCount-1; j++) {
            // head calculation
            head[tokenCount][j] = std::inner_product(dot(tokens[tokenCount], MK).begin(), dot(tokens[tokenCount], MK).end(), dot(tokens[j], MQ).begin(), 0.0)/d;
            head[j][tokenCount] = std::inner_product(dot(tokens[j], MK).begin(), dot(tokens[j], MK).end(), dot(tokens[tokenCount], MQ).begin(), 0.0)/d;
        }
    }
    // probability distribution
    KdotQ = LOTA(head, tokenCount);
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
    ver.input = EV + dv;
    hor.forward();
    ver.forward();
    // AND gate for the final output
    EH = EH + ReLUv(hor.output);
    EV = EV + ReLUv(ver.output);
}


/**
 * @brief forward propagation for attention class (incomplete attention)
 * @param tokens token embeddings
 * @param dv dv vector
 * @param EVp EV vector for previous block
 * @param EVc EV vector for current block
 * @param changeV vertical change vector for current block
 * @param in input token count
 * @param tokenCount token count for each attention head (hiw many tokens have been generated or taken as input)
 * @param blockCount block count for each attention head (how many blocks have been processed)
 * @param n number of tokens for each attention head
 */
void attention::forprop(std::vector<std::vector<double>>& tokens, std::vector<std::vector<double>>& EVp, std::vector<double>& dv,
    std::vector<double>& EVc, std::vector<double>& changeV, int& in, int& tokenCount, int& blockCount, int& n)
{
    // take total tokens available in the tokens embeddings and then make head
    double d = sqrt(head.size());
    // head calculation by inner product of KEYS and QUERYS
    if((tokenCount - n*blockCount) == 1) {
        head[0][0] = std::inner_product(dot(tokens[0], MK).begin(), dot(tokens[0], MK).end(), dot(EVp[0], MQ).begin(), 0.0)/d;
    }
    else {
        for(int i = 0; i < tokenCount - n*blockCount - 1; i++) {
            for(int j = 0; j < tokenCount - n*blockCount - 1; j++) {
                // head calculation
                head[i][j] = std::inner_product(dot(tokens[i], MK).begin(), dot(tokens[i], MK).end(), dot(tokens[i], MQ).begin(), 0.0)/d;
            }
        }
        head[tokenCount][tokenCount] = std::inner_product(dot(tokens[tokenCount], MK).begin(), dot(tokens[tokenCount], MK).end(), dot(EVp[tokenCount], MQ).begin(), 0.0)/d;
        for(int j = 0; j < tokenCount-1; j++) {
            // head calculation
            head[tokenCount][j] = std::inner_product(dot(tokens[tokenCount], MK).begin(), dot(tokens[tokenCount], MK).end(), dot(EVp[j], MQ).begin(), 0.0)/d;
            head[j][tokenCount] = std::inner_product(dot(tokens[j], MK).begin(), dot(tokens[j], MK).end(), dot(EVp[tokenCount], MQ).begin(), 0.0)/d;
        }
    }
    // probability distribution
    KdotQ = LOTA(head, tokenCount);
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
