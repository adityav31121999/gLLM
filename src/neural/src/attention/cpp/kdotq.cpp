#ifdef USE_CPU
#include <vector>
#include <cmath> // For std::abs, std::min
#include "include/attention.hpp"


/**
 * @brief calculate key/query matrix using token embedding matrix and K/Q weight matrices
 * @param tokenOrEV token matrix or from prev block (context window x embedding dimension)
 * @param KQweights K/Q weight matrices (context window x embedding dimension)
 */
void attention::getKeyQuery(const mat& tokenOrEV, const mat& KQweights)
{
    K.mult_A_Bt(tokenOrEV, KQweights);
}

#endif