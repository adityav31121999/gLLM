
#include "include/transformer.hpp"

/**
 * @brief Constructor for transformer
 * @param m number of blocks in transformer
 * @param x number of partial attentions in block
 * @param y number of attention in each partial attention
 * @param n number of tokens for each attention head
 * @param d dimension of each token
 * @param h height of MQ, MK and columns of MV, MH
 */
transformer::transformer(int m, int x, int y, int n, int d, int h, int l) {
    attblock = std::vector<block>(m, block(x, y, n, d, h, l));
    stringToken = std::vector<std::vector<double>>(m, std::vector<double>(d, 0));
    countParams();
}


/**
 * @brief count parameters in transformer
 */
void transformer::countParams() {
    totalParams = attblock[0].totalParams * m;
}
