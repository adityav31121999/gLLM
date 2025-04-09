
// constructor for incomplete attention
#include "include/attention.hpp"
#include <numeric>

/**
 * @brief Constructor for incomplete attention (default: self attention, training)
 * @param n number of tokens for each attention head
 * @param d dimension of each token
 * @param h height of MQ, MK and columns of MV, MH
 * @param l layers of mlp
 */
attention::attention(int n, int d, int h, int l) {
    KdotQ = std::vector<std::vector<float>>(n, std::vector<float>(n, 0));   // KEYS
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
    isSelfAttention = 1;                // default
    inTraining = 1;                     // default
}


/**
 * @brief Constructor for incomplete attention (default: training)
 * @param n number of tokens for each attention head
 * @param d dimension of each token
 * @param h height of MQ, MK and columns of MV, MH
 * @param l layers of mlp
 */
attention::attention(int n, int d, int h, int l, bool isSelf) {
    KdotQ = std::vector<std::vector<float>>(n, std::vector<float>(n, 0));   // KEYS
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
    EV = std::vector<std::vector<float>>(n, std::vector<float>(d, 0));
    hor = mlp(d, l, 10, LEARNING);      // MLP for FFN in horizontal
    ver = mlp(d, l, 10, LEARNING);      // MLP for New Block Attention in vertical
    isSelfAttention = isSelf;           // default attention: Self
    inTraining = 1;                     // default
}


/**
 * @brief Constructor for incomplete attention
 * @param n number of tokens for each attention head
 * @param d dimension of each token
 * @param h height of MQ, MK and columns of MV, MH
 * @param l layers of mlp
 */
attention::attention(int n, int d, int h, int l, bool isSelf, bool inTraining) {
    if(inTraining == 1) {
        // for training
        KdotQ = std::vector<std::vector<float>>(n, std::vector<float>(n, 0));   // KEYS
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
        EV = std::vector<std::vector<float>>(n, std::vector<float>(d, 0));
        hor = mlp(d, l, 10, LEARNING);      // MLP for FFN in horizontal
        ver = mlp(d, l, 10, LEARNING);      // MLP for New Block Attention in vertical
        isSelfAttention = isSelf;           // default attention: Self
        inTraining = this->inTraining;
    }
    // for use
    else {
        KdotQ = std::vector<std::vector<float>>(n, std::vector<float>(n, 0));   // KEYS
        qkCache = mat(d, d);     // dxd
        qhCache = mat(d, d);     // dxd
        kvCache = mat(d, d);     // dxd
        dh = std::vector<float>(d, 0);      // dh = sum(head[ith row])xK[i]
        dv = std::vector<float>(d, 0);      // dv = sum(head[ith col])xQ[i]
        EH = std::vector<float>(d, 0);      // EH = EH + dH
        // vertical retention vectors
        EV = std::vector<std::vector<float>>(n, std::vector<float>(d, 0));
        hor = mlp(d, l, 10, LEARNING);      // MLP for FFN in horizontal
        ver = mlp(d, l, 10, LEARNING);      // MLP for New Block Attention in vertical
        isSelfAttention = isSelf;           // default attention: Self
        inTraining = this->inTraining;
    }
}


/**
 * @brief set attention type of model for cross and self attention
 * @param isSelforCross 1 for self attention and 0 for cross attention
 */
void attention::setAttentionType(bool isSelforCross) {
    isSelfAttention = isSelforCross;
}
