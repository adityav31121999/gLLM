
#include "include/attention.hpp"

/**
 * @brief Forward Propagation for FFN
 */
void attention::train(std::vector<std::vector<double>> tokens, int tokenCount) {
    forprop(tokens, tokenCount);
    backward(tExp);
}
