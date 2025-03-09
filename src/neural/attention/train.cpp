
#include "include/attention.hpp"

/**
 * @brief train the attention mechanism
 * @param tokens input tokens / prompt
 * @param tExp expected tokens for prediction
 */
void attention::train(std::vector<std::vector<double>> tokens, int tokenCount, double learning) {
    while(error > 0.01) {
        forprop(tokens, tokenCount);
        backward(tExp, learning);
    }
}
