
#include "include/attention.hpp"

/**
 * @brief train the attention mechanism
 * @param tokens token embeddings
 * @param holddv dv vector for next block
 * @param holdEV EV vector for next block
 * @param changeV vertical change vector for next block
 * @param in input token count
 * @param tokenCount token count for each attention head (how many tokens have been generated or taken as input)
 * @param learning learning rate for MLPs
 */
void attention::train(std::vector<std::vector<double>>& tokens, std::vector<double>& dv, std::vector<double>& EV, 
    std::vector<double>& changeV, int in, int tokenCount) 
{
    while(error > 0.01) {
        forprop(tokens, dv, EV, changeV, in, tokenCount);
        backward(tokens[tokenCount], changeV, dv, EV);
    }
}
