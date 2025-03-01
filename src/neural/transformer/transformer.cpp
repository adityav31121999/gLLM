
#include "include/transformer.hpp"

transformer::transformer(int m, int x, int y, int n, int d, int h) {
    attblock = std::vector<block>(m, block(x, y, n, d, h));
    countParams();
}

void transformer::countParams() {
    totalParams = attblock[0].totalParams * m;
}
