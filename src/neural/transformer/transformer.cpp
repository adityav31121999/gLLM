
#include "include/attention.hpp"
#include "include/block.hpp"
#include "include/transformer.hpp"
#include "transformer.hpp"

/**
 * @brief Constructor for single-block transformer for prediction
 * @param x number of partial attentions in block
 * @param y number of attention in each partial attention
 * @param n number of tokens for each attention head
 * @param d dimension of each token
 * @param h height of MQ, MK and columns of MV, MH
 * @param l layers of mlp
 */
transformer::transformer(int x, int y, int n, int d, int h, int l):
    m(1), x(x), y(y), n(n), d(d), h(h), l(l) {
    b = std::vector<block>(1, block(x, y, n, d, h, l));
    // total permissible tokens = n
    tokenEmbed = std::vector<std::vector<double>>(n, std::vector<double>(d, 0));
    totalParams = 2 * d * ((2 * h) + (l * d)) * x * y * n;
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
transformer::transformer(int m, int x, int y, int n, int d, int h, int l) :
    m(m), x(x), y(y), n(n), d(d), h(h), l(l) {
    b = std::vector<block>(m, block(x, y, n, d, h, l));
    // total permissible tokens = m * n
    tokenEmbed = std::vector<std::vector<double>>(n * m, std::vector<double>(d, 0));
    totalParams = 2 * d * ((2 * h) + (l * d)) * x * y * m * n;
    total = m * n;
}


/**
 * @brief set all the input tokens in the beginning of token vector
 */
void transformer::takeInput() {
    for(auto& i : tinput) {
        token.push_back(i);
        // tokenEmbed.push_back(tokenise(i));
    }
}
