
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
    K = std::vector<std::vector<float>>(n, std::vector<float>(h, 0));   // KEYS
    Q = std::vector<std::vector<float>>(n, std::vector<float>(h, 0));   // QUERYS
    MQ = mat(h, d);     // hxd
    MK = mat(h, d);     // hxd
    MV = mat(d, h);     // dxh
    MH = mat(d, h);     // dxh
    dh = std::vector<float>(d, 0);      // dh = sum(head[ith row])xK[i]
    dv = std::vector<float>(d, 0);      // dv = sum(head[ith col])xQ[i]
    EH = std::vector<float>(d, 0);      // EH = EH + dH
    // vertical retention vectors
    EV = std::vector<std::vector<float>>(CONTEXT_WIN, std::vector<float>(d, 0));
    hor = mlp(d, l, 10, LEARNING);      // MLP for FFN in horizontal
    ver = mlp(d, l, 10, LEARNING);      // MLP for New Block Attention in vertical
    isSelfAttention = 0;                // default attention: Self
}


/**
 * @brief Constructor for incomplete attention
 * @param n number of tokens for each attention head
 * @param d dimension of each token
 * @param h height of MQ, MK and columns of MV, MH
 * @param l layers of mlp
 */
attention::attention(int n, int d, int h, int l, bool isSelf) {
    K = std::vector<std::vector<float>>(n, std::vector<float>(h, 0));   // KEYS
    Q = std::vector<std::vector<float>>(n, std::vector<float>(h, 0));   // QUERYS
    MQ = mat(h, d);     // hxd
    MK = mat(h, d);     // hxd
    MV = mat(d, h);     // dxh
    MH = mat(d, h);     // dxh
    dh = std::vector<float>(d, 0);      // dh = sum(head[ith row])xK[i]
    dv = std::vector<float>(d, 0);      // dv = sum(head[ith col])xQ[i]
    EH = std::vector<float>(d, 0);      // EH = EH + dH
    // vertical retention vectors
    EV = std::vector<std::vector<float>>(CONTEXT_WIN, std::vector<float>(d, 0));
    hor = mlp(d, l, 10, LEARNING);      // MLP for FFN in horizontal
    ver = mlp(d, l, 10, LEARNING);      // MLP for New Block Attention in vertical
    isSelfAttention = isSelf;           // default attention: Self
}


/**
 * @brief set attention type of model for cross and self attention
 * @param isSelforCross 1 for self attention and 0 for cross attention
 */
void attention::setAttentionType(bool isSelforCross) {
    isSelfAttention = isSelforCross;
}


/**
 * @brief compute KdotQ for training of model
 * @param Keys tokenEmbed x MK
 * @param Queries tokenEmbed x MQ
 * @param currentTokenCount current token count in full context
 * @param promptCount count of current prompt
 * @param blockCount current block in full context
 */
void attention::computeKdotQforTrain(std::vector<std::vector<float>>& Keys, std::vector<std::vector<float>>& Queries, int&currentTokenCount, 
    int& promptCount, int& blockCount)
{
    
}


/**
 * @brief compute KdotQ for training of model
 * @param tokenEmbed token embeddings
 * @param qkMat QK' cache
 * @param currentTokenCount current token count in full context
 * @param promptCount count of current prompt
 * @param blockCount current block in full context
 */
void attention::computeKdotQforUse(std::vector<std::vector<float>>& tokenEmbed, std::vector<std::vector<float>>& qkMat, int&currentTokenCount, 
    int& promptCount, int& blockCount) 
{
    
}
