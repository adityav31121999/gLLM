
#include "include/attention.hpp"

/**
 * @brief Forward Propagation for FFN
 */
void attention::backward() {
    // changeH and changeV are to hold the expected vectors
    hor.expected = changeH;
    ver.expected = changeV;
    ver.backprop();
    hor.backprop();
    // backprop the input obtained for ver and hor mlp
    // backprop the MH and MV
    // backprop the MQ and MK
}
