
#include "include/attention.hpp"
#include "include/block.hpp"
#include "include/transformer.hpp"

/**
 * @brief forward propagation for transformer
 */
void transformer::forward() {
    // set transfer to 0, repetitions to 0
    toNextBlock = 0;
    int i;
    i = 0;
    while(i < m) {
        if(i == 0) {
            // forward propagation
            t[0].forprop(d, tokenCount, l);
            // condition for next block
            if((tokenCount % CONTEXT_WIN) == 0) {
                toNextBlock = 1;
            }
            // terminate loop when @#0 is reached
            if(t[0].str == TERMINATE) {
                break;
            }
        }
        else {
            if (toNextBlock == 1) {
                // forward propagation
                t[i].forprop(t[i-1].EV, d, tokenCount, i, l, n);
                // condition for next block
                if((tokenCount % CONTEXT_WIN) == 0) {
                    toNextBlock = 1;
                }
                // terminate loop when @#0 is reached
                if(t[i].str == TERMINATE) {
                    break;
                }
            }
        }
        // end while loop when token limit is reached
        if(tokenCount == m * CONTEXT_WIN)
            break;
        i += 1;
    }
}
