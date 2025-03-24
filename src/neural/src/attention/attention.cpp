
// constructor for incomplete attention
#include "include/attention.hpp"
#include <numeric>

/**
 * @brief Constructor for incomplete attention
 * @param n number of tokens for each attention head
 * @param d dimension of each token
 * @param h height of MQ, MK and columns of MV, MH
 * @param l layers of mlp
 */
attention::attention(int n, int d, int h, int l) {
    // scaled dot product and activated attention head
    K = std::vector<std::vector<float>>(n, std::vector<float>(h, 0));
    Q = std::vector<std::vector<float>>(n, std::vector<float>(h, 0));
    head = std::vector<std::vector<float>>(n, std::vector<float>(n, 0));
    MQ = mat(h, d);     // hxd
    MK = mat(h, d);     // hxd
    MV = mat(d, h);     // dxh
    MH = mat(d, h);     // dxh
    dh = std::vector<float>(d, 0);     // dh = sum(dH)
    EH = std::vector<float>(d, 0);     // EH = EH + dH
    hor = mlp(d, l, 10, LEARNING);      // MLP for FFN in horizontal
    ver = mlp(d, l, 10, LEARNING);      // MLP for New Block Attention in vertical
    changeH = std::vector<float>(d, 0);    // change obtained from final step
}


/**
 * @brief compute attention head using keys and queries
 * @param KdotQ dot product matrix
 * @param Keys Keys vector
 * @param Queries Queries vector
 * @param in number of tokens in first prompt
 * @param count number of embeddings in tokens
 * @param promptCount number of tokens in prompt
 */
void attention::computeAttention(std::vector<std::vector<float>>& KdotQ, std::vector<std::vector<float>>& Keys, std::vector<std::vector<float>>& Queries, 
        int in, int count, int promptCount)
{
    // for single token input like "Hi", "Hello", "Hey", "How", "What", etc.
    if (in == 1 && promptCount == 1 && count == 0) {
        KdotQ[0][0] = std::inner_product(Keys[0].begin(), Keys[0].end(), Queries[0].begin(), 0.0)/SCALING;
        count++;
    }
    // for a first and long prompt input
    else if (in > 1 && promptCount == in && count == 0) {    
        for(int i = 0; i < count; i++) {
            for(int j = 0; j < count; j++) {
                // KdotQ[i][j] = Keys[i].Queries[j];
                KdotQ[i][j] = std::inner_product(Keys[i].begin(), Keys[i].end(), Queries[i].begin(), 0.0);
            }
        }
        count = in;
    }
    // for single term promt after response or newly predicted term
    else if(count > in) {
        if (promptCount == 1) {
            // diagonal calculation
            KdotQ[count][count] = std::inner_product(K[count].begin(), K[count].end(), Q[count].begin(), 0.0)/SCALING;
            for(int j = 0; j < count-1; j++) {
                // head calculation
                KdotQ[count][j] = std::inner_product(K[count].begin(), K[count].end(), Q[j].begin(), 0.0)/SCALING;
                KdotQ[j][count] = std::inner_product(K[j].begin(), K[j].end(), Q[count].begin(), 0.0)/SCALING;
            }
            count++;
        }
        else if(promptCount > 1) {
            int diff = count - promptCount;
            for(int i = 0; i < diff; i++) {
                // diagonal calculation
                KdotQ[count + i][count + i] = std::inner_product(K[count + i].begin(), K[count + i].end(), Q[count + i].begin(), 0.0)/SCALING;
                for(int j = 0; j < count + i; j++) {
                    // head calculation
                    KdotQ[i][j] = std::inner_product(K[i].begin(), K[i].end(), Q[j].begin(), 0.0)/SCALING;
                    KdotQ[j][i] = std::inner_product(K[j].begin(), K[j].end(), Q[i].begin(), 0.0)/SCALING;
                }
            }
            count += diff;
        }
    }
}


/**
 * @brief forward propagation for a specific block's attention class (incomplete attention)
 * @param T token embedding
 * @param M matrix for attention head calculation (MQ x MK')
 * @param dot dot product of T x M x T'
 */
void calculateDot(std::vector<float>& T, std::vector<std::vector<float>>& M, float& dot) {
    std::vector<double> temp(T.size(), 0);
    for(int i = 0; i < T.size(); i++) {
        temp[i] = std::inner_product(T.begin(), T.end(), M[i].begin(), 0.0);
    }
    dot = std::inner_product(temp.begin(), temp.end(), T.begin(), 0.0);
}


/**
 * @brief head calculation (via T x M x T')
 * @param tokens token embeddings
 * @param KdotQ dot product matrix
 * @param M matrix for attention head calculation (MQ x MK')
 * @param terms number of terms for attention head
 */
void calculateHead(std::vector<std::vector<float>>& KdotQ, std::vector<std::vector<float>>& tokens, std::vector<std::vector<float>>& M, int terms) {
    // i for token
    for(int i = 0; i < terms; i++) {
        // j for column of M
        for(int j = 0; j < terms; j++) {
            calculateDot(tokens[i], M, KdotQ[i][j]);
        }
    }
}
