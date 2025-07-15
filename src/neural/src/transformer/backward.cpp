#ifdef USE_CPU

// backward propagation for transformer
#include "include/attention.hpp"
#include "include/block.hpp"
#include "include/transformer.hpp"

/**
 * @brief backward propagation for last to first block
 *          (common expected EH for last block)
 * @param expected expected token embedding from horizontal pass
 */
void transformer::backward(std::vector<float>& expected) {
    int count = 0;
    while (count < m) {
        if(count == 0) {
            // start from last block where Expected EVs are not known
            t[m-1].backward(expected, d, l, m, learning);
            count++;    // = 1
        }
        else if(count >=1 || count < m-1) {
            // backward propagation when expected vectors are known
            // from 2nd last to 2nd block
            t[m-count-1].backward(t[m-count].EV, d, l, m-count, learning);
            count++;    // = 2 to m-1
        }
        else if(count == m-1) {
            // for first block
            t[0].backward1stBlock(t[1].EV, d, l, learning);
            break;
        }
    }
}


/**
 * @brief backward propagation for kth to first block, Expected EVs are not known
 *          (common expected EH for kth block)
 * @param expected expected token embedding from horizontal pass
 * @param k block number (1-based index)
 */
void transformer::backward(std::vector<float>& expected, int& k) {
    int count = 0;
    while (count < k) {
        if(count == 0) {
            // start from last block
            t[k-1].backward(expected, d, l, k, learning);
            count++;    // = 1
        }
        // backward propagation for 2nd last to 2nd block, with combined expected EH vector
        else if (count >=1 || count < k-1) {
            t[k-count-1].backward(t[m-count].EV, d, l, k-count, learning);
            count++;    // = 2 to k-1
        }
        else if(count == k-1) {
            // for first block
            t[0].backward1stBlock(t[1].EV, d, l, learning);
        }
    }
}


/**
 * @brief backward propagation for last to first block
 *          (distinct expected EH for last block)
 * @param expected expected token embeddings from horizontal pass
 */
void transformer::backward(std::vector<std::vector<float>>& expected) {
    int count = 0;
    while (count < m) {
        if(count == 0) {
            // start from last block where Expected EVs are not known
            t[m-1].backward(expected, d, l, m, learning);
            count++;    // = 1
        }
        // backward propagation when expected EV vectors are known
        else if(count >=1 || count < m-1) {
            // from 2nd last to 2nd block
            t[m-count-1].backward(t[m-count].EV, d, l, m-count, learning);
            count++;    // = 2 to m-1
        }
        else if(count == m-1) {
            // for first block
            t[0].backward1stBlock(t[1].EV, d, l, learning);
            break;
        }
    }
}


/**
 * @brief backward propagation for kth to first block, Expected EVs are not known
 *          (distinct expected EH for last block)
 * @param expected expected token embeddings from horizontal pass
 * @param k block number (1-based index)
 */
void transformer::backward(std::vector<std::vector<float>>& expected, int& k) {
    int count = 0;
    while (count < k) {
        if(count == 0) {
            // start from last block
            t[k-1].backward(expected, d, l, k, learning);
            count++;    // = 1
        }
        // backward propagation for 2nd last to 2nd block, with combined expected EH vector
        else if (count >=1 || count < k-1) {
            t[k-count-1].backward(t[m-count].EV, d, l, k-count, learning);
            count++;    // = 2 to k-1
        }
        else if(count == k-1) {
            // for first block
            t[0].backward1stBlock(t[1].EV, d, l, learning);
        }
    }
}

#endif
