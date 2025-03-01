
#include "include/attention.hpp"

block::block(int x, int y, int n, int d, int h) : n(n), d(d), h(h) {
    b = std::vector<std::vector<attention>>(x, std::vector<attention>(y, attention(n, d, h)));
}

void block::countParams() {
    totalParams = b[0][0].totalParams * x * y;
}
