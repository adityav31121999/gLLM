
#include "include/transformer.hpp"

/**
 * @brief Constructor for transformer
 * @param m number of blocks in transformer
 * @param x number of partial attentions in block
 * @param y number of attention in each partial attention
 * @param n number of tokens for each attention head
 * @param d dimension of each token
 * @param h height of MQ, MK and columns of MV, MH
 * @param l layers of mlp
 */
transformer::transformer(int m, int x, int y, int n, int d, int h, int l) {
    this->m = m;
    this->x = x;
    this->y = y;
    this->n = n;
    this->d = d;
    this->h = h;
    this->l = l;
    attblock = std::vector<block>(m, block(x, y, n, d, h, l));
    stringToken = std::vector<std::vector<double>>(m, std::vector<double>(d, 0));
    total = ((4 * h * d) + (2 * d * d * l)) * x * y * m * n;
}
