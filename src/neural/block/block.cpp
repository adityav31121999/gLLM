
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
    this->x = x;
    this->y = y;
    this->n = n;
    this->d = d;
    this->h = h;
    this->l = l;
    b = std::vector<std::vector<attention>>(x, std::vector<attention>(y, attention(n, d, h, l)));
    holdEVs = std::vector<std::vector<std::vector<double>>>(x, std::vector<std::vector<double>>(y, std::vector<double>(d, 0)));
    totalParams = ((4 * h * d) + (2 * d * d * l)) * x * y;
}
