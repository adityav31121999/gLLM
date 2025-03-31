
#include "include/attention.hpp"
#include "include/block.hpp"

/**
 * @brief Constructor for complete attention block
 * @param x number of partial attentions in block
 * @param y number of attention in each partial attention
 * @param n number of tokens for each attention head
 * @param d dimension of each token
 * @param h height of MQ, MK and columns of MV, MH
 * @param l layers of mlp
 * @param vocab vocabulary size
 */
block::block(int x, int y, int n, int d, int h, int l, int vocab) : x(x), y(y) {
    // initialize attention block (complete attention)
    b = std::vector<std::vector<attention>>(x, std::vector<attention>(y, attention(n, d, h, l)));
    // probabiltiy vector
    probability = std::vector<float>(vocab, 0.0f);
    // common embeddings for horizontal and vertical pass
    // EH = std::vector<float>(EMBEDDING, 0.0);
    // EV = std::vector<float>(EMBEDDING, 0.0);
    // EV = std::vector<std::vector<std::vector<float>>>(x, std::vector<std::vector<float>>(y, std::vector<float>(d, 0)));
    EV = std::vector<std::vector<std::vector<std::vector<float>>>>(x, std::vector<std::vector<std::vector<float>>>(y,\
        std::vector<std::vector<float>>(CONTEXT_WIN, std::vector<float>(d, 0))));
    // expected vectors for backprop in horizontal and vertical pass
    // expectedH = std::vector<float>(EMBEDDING, 0.0); // for continued backpropagation
    // for step-wise backpropagation
    // expectedV = std::vector<std::vector<std::vector<float>>>(x, std::vector<std::vector<float>>(y, std::vector<float>(EMBEDDING, 0)));
}
