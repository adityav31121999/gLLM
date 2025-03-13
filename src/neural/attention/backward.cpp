
#include "include/attention.hpp"

/**
 * @brief Forward Propagation for attention
 * @param expected expected vector
 * @param changeV 
 */
void attention::backward(std::vector<double>& expected, std::vector<double>& changeV, std::vector<double>& dv, 
    std::vector<double>& EV) 
{
    // get required changes
    hor.expected = expected;
    ver.expected = expected;
    hor.backprop2in();
    ver.backprop2in();
    // backpropagate to input of MLPs (EH-dh and EV-dv)
    dh = dh - hor.input;
    // dv = dv - ver.input;
    // backpropagate the error to EH and EV
    for(int j = 0; j < dh.size(); j++) {
        // change should be provided with respect to each gradient from first layer only
        changeH[j] = changeV[j] = 0;
        for(int i = 0; i < dh.size(); i++) {
            changeH[j] += dh[j] - hor.learning * hor.gweights[0][j][i];
            changeV[j] += dv[j] - ver.learning * ver.gweights[0][j][i];
        }
        EH[j] = EH[j] - changeH[j];
        EV[j] = EV[j] - changeV[j];
    }

    // backpropagate the error to the MH and MV
    for(int i = 0; i < MV.a.size(); i++) {
        for(int j = 0; j < MV.a[0].size(); j++) {
            MV.a[i][j] = MH.a[i][j] * (1 - ver.learning * changeV[i]);
            MH.a[i][j] = MK.a[i][j] * (1 - hor.learning * changeH[i]);
        }
    }

    // backpropagate the error to the MQ and MK
    for(int i = 0; i < MQ.a.size(); i++) {
        for(int j = 0; j < MQ.a[0].size(); j++) {
            MQ.a[i][j] = MH.a[i][j] * (1 - hor.learning * changeV[i]);
            MK.a[i][j] = MK.a[i][j] * (1 - ver.learning * changeH[i]);
        }
    }
}
