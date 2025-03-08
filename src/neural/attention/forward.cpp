
#include "include/attention.hpp"
#include <maths.hpp>
#include "attention.hpp"

/**
 * @brief default forward propagation for attention class
 */
void attention::forprop(std::vector<std::vector<double>> tokens, int tokenCount) {
    // take total tokens available in the tokens embeddings and then make head
    // key and query calculation
    for(int i = 0; i < tokenCount; i++) {
        KEYS[i] = dot(tokens[i], MK);
        QUERYS[i] = dot(tokens[i], MQ);
    }
    for(int i = 0; i < tokenCount; i++) {
        for(int j = 0; j < tokenCount; j++) {
            // head calculation
            head[i][j] = std::inner_product(KEYS[i].begin(), KEYS[i].end(), QUERYS[i].begin(), 0.0);
        }
    }
    head = LOTA(head, tokenCount);

    // Ki.MV, dh = weighted sums horizontal
    for(int i = 0; i < tokenCount; i++) {
        for(int j = 0; j < tokenCount; j++) {
            dH[i][j] = dot(KEYS[i], MH);
            dh = dh + (head[i][j] * dH[i][j]);
        }
    }

    // Qi.MH, dv = weighted sums vertically
    for(int i = 0; i < tokenCount; i++) {
        for(int j = 0; j < tokenCount; j++) {
            dV[j][i] = dot(QUERYS[j], MV);
            dv = dv + (head[j][i] * dV[j][i]);
        }
    }

    hor.input = EH + dh, ver.input = EV + dv;
    hor.forward();
    ver.forward();
    mh = ReLUv(hor.output), mv = ReLUv(ver.output);
    EH = EH + mh, EV = EV + mv;
}
