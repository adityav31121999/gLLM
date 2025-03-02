
#include "include/attention.hpp"
#include <maths.hpp>

/**
 * @brief Forward Propagation for FFN
 */
void attention::forward() {
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

    // Ki.MV, dh = weighted sums horizontal
    for(int i = 0; i < tokenCount; i++) {
        for(int j = 0; j < tokenCount; j++) {
            dH[i][j] = dot(KEYS[i], MH);
            dh = dh + (head[i][j] * dH[i][j]);
        }
        EH = EH + dh;
    }

    // Qi.MH, dv = weighted sums vertically
    for(int i = 0; i < tokenCount; i++) {
        for(int j = 0; j < tokenCount; j++) {
            dV[j][i] = dot(QUERYS[j], MV);
            dv = dv + (head[j][i] * dV[j][i]);
        }
        EV = EV + dv;
    }

    // mlp horizontal and vertical
    hor.input = EH;
    hor.forward();
    ver.input = EV;
    ver.forward();
}
