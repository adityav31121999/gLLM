
#include "include/attention.hpp"
#include <maths.hpp>
#include <numeric>
#include <algorithm>

/**
 * @brief forward propagation for attention class (incomplete attention)
 */
void attention::forprop(std::vector<std::vector<double>> tokens, int tokenCount) {
    // take total tokens available in the tokens embeddings and then make head
    for(int i = 0; i < tokenCount; i++) {
        // calculate the KEYS and QUERYS
        KEYS[i] = dot(tokens[i], MK) / sqrt(head.size());
        QUERYS[i] = dot(tokens[i], MQ) / sqrt(head.size());
    }
    // head calculation by inner product of KEYS and QUERYS
    for(int i = 0; i < tokenCount; i++) {
        for(int j = 0; j < tokenCount; j++) {
            // head calculation
            head[i][j] = std::inner_product(KEYS[i].begin(), KEYS[i].end(), QUERYS[i].begin(), 0.0);
        }
    }
    // probability distribution
    head = LOTA(head, tokenCount);
    for(int i = 0; i < tokenCount; i++) {
        // hold in change vectors, do not overuse the memory for this calculation
        // as it is not necessary calculate them every time
        changeH = dot(KEYS[i], MH);
        changeH = dot(QUERYS[i], MV);
        for(int j = 0; j < tokenCount; j++) {
            dh = dh + (head[i][j] * changeH);   // Ki.MV, dh = weighted sums horizontal
            dv = dv + (head[j][i] * changeV);   // Qj.MH, dv = weighted sums vertical
        }
    }
    // get the required change from MLPs
    hor.input = EH + dh, ver.input = EV + dv;
    hor.forward();
    ver.forward();
    // AND gate for the final output
    EH = EH + ReLUv(hor.output);
    EV = EV + ReLUv(ver.output);
}
