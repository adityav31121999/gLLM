
// backward propagation for transformer
#include "include/attention.hpp"
#include "include/block.hpp"
#include "include/transformer.hpp"

#ifdef USE_CPU

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
            t[m-1].backward(expected, d, l);
            count++;
        }
        else {
            // backward propagation when expected vectors are known
            if(count >=1 || count < m-1) {
                // from 2nd last to 2nd block
                t[m-count-1].backward(t[m-count].EV, d, l);
                count++;
            }
            if(count == m-1) {
                // for first block
                t[0].backward1stBlock(t[1].EV, d, l);
            }
        }
    }
}


/**
 * @brief backward propagation for kth to first block, Expected EVs are not known
 *          (common expected EH for kth block)
 * @param expected expected token embedding from horizontal pass
 * @param k block number (= (index of block from t vector) + 1)
 */
void transformer::backward(std::vector<float>& expected, int& k) {
    int count = 0;
    while (count < k) {
        if(count == 0) {
            // start from last block
            t[k-1].backward(expected, d, l);
            count++;
        }
        else {
            // backward propagation for 2nd last to 2nd block, with combined expected EH vector
            if (count >=1 || count < k-1) {
                t[m-count-1].backward(t[m-count].EV, d, l);
                count++;
            }
            /**
            std::vector<std::vector<float>> ex(x, std::vector<float>(EMBEDDING, 0));
            for(int i = 0; i < x; i++) {
                ex[i] = t[m-count-1].b[k][i].EH;
            }
            t[m-count-1].backward(t[m-count-1].EV, ex, d, l);
             */
            if(count == k-1) {
                // for first block
                t[0].backward1stBlock(t[1].EV, d, l);
            }
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
            t[m-1].backward(expected, d, l);
            count++;
        }
        else {
            // backward propagation when expected vectors are known
            if(count >=1 || count < m-1) {
                // from 2nd last to 2nd block
                t[m-count-1].backward(t[m-count].EV, d, l);
                count++;
            }
            if(count == m-1) {
                // for first block
                t[0].backward1stBlock(t[1].EV, d, l);
            }
        }
    }
}


/**
 * @brief backward propagation for kth to first block, Expected EVs are not known
 *          (distinct expected EH for last block)
 * @param expected expected token embeddings from horizontal pass
 * @param k block number
 */
void transformer::backward(std::vector<std::vector<float>>& expected, int& k) {
    int count = 0;
    while (count < k) {
        if(count == 0) {
            // start from last block
            t[k-1].backward(expected, d, l);
            count++;
        }
        else {
            // backward propagation for 2nd last to 2nd block, with combined expected EH vector
            if (count >=1 || count < k-1) {
                t[m-count-1].backward(t[m-count].EV, d, l);
                count++;
            }
            if(count == k-1) {
                // for first block
                t[0].backward1stBlock(t[1].EV, d, l);
            }
        }
    }
}

#endif
