
// constructor
#include "include/attention.hpp"
#include "attention.hpp"

/**
 * @brief Constructor for incomplete attention
 * @param n number of tokens for each attention head
 * @param d dimension of each token
 * @param h height of MQ, MK and columns of MV, MH
 */
attention::attention(int n, int d, int h, int l) : n(n), d(d), h(h), l(l) {
    // nxn matrix to hold temporary scalar values
    tokens = std::vector<std::vector<double>>(n, std::vector<double>(d, 0.0));
    head = std::vector<std::vector<double>>(n, std::vector<double>(n, 0));
    next = std::vector<std::vector<double>>(n, std::vector<double>(d, 0));
    tokB = std::vector<std::vector<double>>(n, std::vector<double>(d, 0));
    MQ = mat(h, d);     // hxd
    MK = mat(h, d);     // hxd
    MV = mat(d, h);     // dxh
    MH = mat(d, h);     // dxh
    KEYS = std::vector<std::vector<double>>(n, std::vector<double>(h, 0));      // Ei.MK
    QUERYS = std::vector<std::vector<double>>(n, std::vector<double>(h, 0));    // Ei.MQ
    // Ki.MH
    dH = std::vector<std::vector<std::vector<double>>>(n, std::vector<std::vector<double>>(n, std::vector<double>(d, 0)));
    // Qi.MV
    dV = std::vector<std::vector<std::vector<double>>>(n, std::vector<std::vector<double>>(n, std::vector<double>(d, 0)));
    dh = std::vector<double>(d, 0);     // dh = sum(dH)
    dv = std::vector<double>(d, 0);     // dv = sum(dV)
    EH = std::vector<double>(d, 0);     // EH = EH + dH
    EV = std::vector<double>(d, 0);     // EV = EV + dV
    hor = mlp(d, l, 10, 0.01);      // MLP for FFN
    ver = mlp(d, l, 10, 0.01);      // MLP for New Block Attention
    changeH = std::vector<double>(d, 0);    // change obtained from final step
    changeV = std::vector<double>(d, 0);    // change obtained from final block
    totalParams = (4 * h * d) + (2 * d * d * l);
}

/**
 * @brief Constructor for partial attention
 * @param x number of incomplete attentions in partial attention
 * @param n number of tokens for each attention head
 * @param d dimension of each token
 * @param h height of MQ, MK and columns of MV, MH
 * @param l layers of mlp
 */
pattention::pattention(int x, int n, int d, int h, int l) {
    this->x = x;
    this->n = n;
    this->d = d;
    this->h = h;
    this->l = l;
    b = std::vector<attention>(x, attention(n, d, h, l));
    holdEVs = std::vector<std::vector<double>>(x, std::vector<double>(d, 0));
    totalParams = x * ((4 * h * d) + (2 * d * d * l));
}
