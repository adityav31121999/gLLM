
#include "include/attention.hpp"
#include "include/block.hpp"
#include "include/transformer.hpp"

/**
 * @brief Constructor for single-block transformer for prediction
 * @param x number of partial attentions in block
 * @param y number of attention in each partial attention
 * @param n number of tokens for each attention head
 * @param d dimension of each token
 * @param h height of MQ, MK and columns of MV, MH
 * @param l layers of mlp
 */
transformer::transformer(int x, int y, int n, int d, int h, int l, int vocab):
    m(1), x(x), y(y), n(n), d(d), h(h), l(l) {
    t = std::vector<block>(1, block(x, y, n, d, h, l, vocab));
    // total permissible tokens = n
    tokenEmbed = std::vector<std::vector<float>>(n, std::vector<float>(d, 0));
    totalParams = ((2 * h) + (l * d)) * 2 * d * x * y * n;
    total = n;
}


/**
 * @brief Constructor for many-block transformer for training
 * @param m number of blocks in transformer
 * @param x number of partial attentions in block
 * @param y number of attention in each partial attention
 * @param n number of tokens for each attention head
 * @param d dimension of each token
 * @param h height of MQ, MK and columns of MV, MH
 * @param l layers of mlp
 */
transformer::transformer(int m, int x, int y, int n, int d, int h, int l, int vocab) :
    m(m), x(x), y(y), n(n), d(d), h(h), l(l) {
    t = std::vector<block>(m, block(x, y, n, d, h, l, vocab));
    // total permissible tokens = m * n
    tokenEmbed = std::vector<std::vector<float>>(n * m, std::vector<float>(d, 0));
    totalParams = ((2 * h) + (l * d)) * 2 * d * x * y * m * n;
    total = m * n;
}
