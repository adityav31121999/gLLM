
#include "include/attention.hpp"
#include "include/block.hpp"
#include "include/transformer.hpp"


/**
 * @brief backward propagation for last to first block
 */
void transformer::backward(std::vector<float>& expected) {
    int count = 0;
    while (count < m) {
        if(count == 0) {
            // start from last block where Expected EVs are not known
            t[m-1].backward(expected, d, l);
            count++;
        }
        else {
            // backward propagation
            t[m-count-1].backward(t[m-count+1].EV, d, l);
            count++;
        }
    }
}


/**
 * @brief backward propagation for kth to first block
 * @param k block number
 */
void transformer::backward(std::vector<float>& expected, int& k) {
    int count = 0;
    while (count < k) {
        if(count == 0) {
            // start from last block where Expected EVs are not known
            t[m-1].backward(expected, d, l);
            count++;
        }
        else {
            // backward propagation
            t[m-count-1].backward(t[m-count+1].EV, d, l);
            count++;
        }
    }
}
