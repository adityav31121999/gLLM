
#include "include/attention.hpp"

/**
 * @brief Constructor for attention
 * @param n number of tokens/embeddings
 * @param d dimension of each token
 */
attention::attention(int n, int d) : n(n), d(d) {
    head = std::vector<std::vector<double>>(n, std::vector<double>(n, 0));
    dH = std::vector<std::vector<double>>(n, std::vector<double>(d, 0));
    dV = std::vector<std::vector<double>>(n, std::vector<double>(d, 0));
    EH = std::vector<double>(d, 0);
    EV = std::vector<double>(d, 0);
    dh = std::vector<double>(d, 0);
    dv = std::vector<double>(d, 0);
    MQ = mat(n, n);
    MK = mat(n, n);
    MV = mat(n, n);
    MH = mat(n, n);
    // in = out = d
    // layers = epochs = 16
    // neurons = d*d
    h = mlp(d, 16, d, 16, 0.01);
    v = mlp(d, 16, d, 16, 0.01);
}
