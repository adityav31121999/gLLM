
#include "include/attention.hpp"

/**
 * @brief Forward Propagation for attention
 */
void attention::backward(std::vector<double> expected) {
    // get required changes
    for(int i = 0; i < d; i++) {
        changeH[i] = EH[i] - expected[i];
        changeV[i] = EV[i] - expected[i];
    }
    hor.expected = changeH;
    ver.expected = changeV;
    ver.backprop();
    hor.backprop();
    // backpropagate to input of MLPs (EH-dh and EV-dv)
    for(int i = 0; i < d; i++) {
        changeH[i] = EH[i] - dh[i];
        changeV[i] = EV[i] - dv[i];
    }
    // backpropagate the error to EH and EV
    for(int j = 0; j < d; j++) {
        // change should be provided with respect to each gradient from first layer only
        changeH[j] = changeV[j] = 0;
        for(int i = 0; i < d; i++) {
            changeH[j] += EH[j] - learning * hor.gweights[0][j][i];
            changeV[j] += EV[j] - learning * ver.gweights[0][j][i];
        }
    }

    // backpropagate the error to the MH and MV
    for(int i = 0; i < h; i++) {
        for(int j = 0; j < d; j++) {
            MV.a[i][j] = MH.a[i][j] * (1 - learning * changeV[i]);
            MH.a[i][j] = MK.a[i][j] * (1 - learning * changeH[i]);
        }
    }

    // backpropagate the error to the MQ and MK
    for(int i = 0; i < d; i++) {
        for(int j = 0; j < h; j++) {
            MQ.a[i][j] = MH.a[i][j] * (1 - learning * changeV[i]);
            MK.a[i][j] = MK.a[i][j] * (1 - learning * changeH[i]);
        }
    }
}
