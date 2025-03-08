
#include "include/attention.hpp"

/**
 * @brief Constructor for complete attention block
 * @param x number of partial attentions in block
 * @param y number of attention in each partial attention
 * @param n number of tokens for each attention head
 * @param d dimension of each token
 * @param h height of MQ, MK and columns of MV, MH
 */
block::block(int x, int y, int n, int d, int h, int l) : n(n), d(d), h(h), l(l) {
    this->x = x;    // number of partial attentions in complete attention
    this->y = y;    // number of incomplete attention for each partial attention
    this->n = n;    // number of tokens for each attention head
    this->d = d;    // dimension of each token embedding
    this->h = h;    // height of MQ, MK and columns of MV, MH
    this->l = l;    // layers of mlp
    // tokens for attention head
    tokens = std::vector<std::vector<double>>(n, std::vector<double>(d, 0));
    // initialize attention block (complete attention)
    b = std::vector<std::vector<attention>>(x, std::vector<attention>(y, attention(n, d, h, l)));
    // initialize holdEVs to hold all inbetween tokens transfer for context transfer and retention
    holdEVs = std::vector<std::vector<std::vector<double>>>(x, std::vector<std::vector<double>>(y, std::vector<double>(d, 0)));
    // total parameters of block
    totalParams = ((4 * h * d) + (2 * d * d * l)) * x * y;
}
