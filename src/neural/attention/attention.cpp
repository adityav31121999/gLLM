
#include "include/attention.hpp"

/**
 * @brief Constructor for attention
 * @param tokenEmbed number of tokens/embeddings
 */
attention::attention(int n, int d) : n(n), d(d) {
    head = std::vector<std::vector<double>>(n, std::vector<double>(n, 0));
    MQ = mat(n, n);
    MK = mat(n, n);
    MV = mat(n, n);
    MH = mat(n, n);
    dH = std::vector<std::vector<double>>(n, std::vector<double>(d, 0));
    dV = std::vector<std::vector<double>>(n, std::vector<double>(d, 0));
    EH = std::vector<double>(d, 0);
    EV = std::vector<double>(d, 0);
    dh = std::vector<double>(d, 0);
    dv = std::vector<double>(d, 0);
    h = mlp(d, d, d, std::sqrt(d), 16, 0.01);
    v = mlp(d, d, d, std::sqrt(d), 16, 0.01);
}
