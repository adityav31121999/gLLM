
#include "include/attention.hpp"

/**
 * @brief Forward Propagation for FFN
 */
void attention::train() {
    forward();
    backward();
}
