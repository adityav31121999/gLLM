
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
    // head = std::vector<std::vector<float>>(n, std::vector<float>(n, 0));
    MQ = mat(h, d);     // hxd
    MK = mat(h, d);     // hxd
    MV = mat(d, h);     // dxh
    MH = mat(d, h);     // dxh
    dh = std::vector<float>(d, 0);      // dh = sum(sum(head[ith row])xK[i])
    dv = std::vector<std::vector<float>>(n, std::vector<float>(h, 0));      // dv[i] = sum(head[ith col])xQ[i]
    EH = std::vector<float>(d, 0);      // EH = EH + dH
    EV = std::vector<std::vector<float>>(n, std::vector<float>(h, 0));
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
    // scaled dot product and activated attention head
    // scaled dot product and activated attention head
    K = std::vector<std::vector<float>>(n, std::vector<float>(h, 0));
    Q = std::vector<std::vector<float>>(n, std::vector<float>(h, 0));
    // head = std::vector<std::vector<float>>(n, std::vector<float>(n, 0));
    MQ = mat(h, d);     // hxd
    MK = mat(h, d);     // hxd
    MV = mat(d, h);     // dxh
    MH = mat(d, h);     // dxh
    dh = std::vector<float>(d, 0);      // dh = sum(sum(head[ith row])xK[i])
    EH = std::vector<float>(d, 0);      // EH = EH + dH
    dv = std::vector<std::vector<float>>(n, std::vector<float>(h, 0));      // dv[i] = sum(head[ith col])xQ[i]
    EV = std::vector<std::vector<float>>(n, std::vector<float>(h, 0));
    hor = mlp(d, l, 10, LEARNING);      // MLP for FFN in horizontal
    ver = mlp(d, l, 10, LEARNING);      // MLP for New Block Attention in vertical
    isSelfAttention = isSelf;           // default attention: Self
}

