
// constructor for incomplete attention
#include "include/attention.hpp"
#include "attention.hpp"


/**
 * @brief Constructor for incomplete attention
 * @param n number of tokens for each attention head
 * @param d dimension of each token
 * @param h height of MQ, MK and columns of MV, MH
 * @param l layers of mlp
 */
attention::attention(int n, int d, int h, int l) {
    // scaled dot product and activated attention head
    head = std::vector<std::vector<double>>(n, std::vector<double>(n, 0));
    MQ = mat(h, d);     // hxd
    MK = mat(h, d);     // hxd
    MV = mat(d, h);     // dxh
    MH = mat(d, h);     // dxh
    dh = std::vector<double>(d, 0);     // dh = sum(dH)
    EH = std::vector<double>(d, 0);     // EH = EH + dH
    hor = mlp(d, l, 10, LEARNING);      // MLP for FFN in horizontal
    ver = mlp(d, l, 10, LEARNING);      // MLP for New Block Attention in vertical
    changeH = std::vector<double>(d, 0);    // change obtained from final step
    error = 0;          // set initial error to 0
}
