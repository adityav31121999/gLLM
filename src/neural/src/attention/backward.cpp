
#include "include/attention.hpp"

/**
 * @brief Backward Propagation
 * @param expected expected vector
 * @param dh horizontal weighted sum vector
 * @param dv vertical weighted sum vector
 * @param EH horizontal embedding vector
 * @param EV vertical embedding vector
 * @param in input tokens
 * @param layers layers in mlp
 */
void attention::backward(std::vector<float>& expected, std::vector<double>& dh, std::vector<float>& dv, std::vector<float>& EV, 
    std::vector<double>& EH, int& in, int& layers)
{
    // get required changes
    hor.expected = expected;
    ver.expected = expected;
    hor.backward(in, layers, LEARNING);
    ver.backward(in, layers, LEARNING);

    // set gradients
    // std::vector<std::vector<float>> gradH = hor.gweights[0];
    // std::vector<std::vector<float>> gradV = ver.gweights[0];
    std::vector<float> gH = std::vector<float>(EMBEDDING, 0.0f);
    std::vector<float> gV = std::vector<float>(EMBEDDING, 0.0f);

    // error for EH and EV
    std::vector<float> changeH = std::vector<float>(EMBEDDING, 0.0f);
    std::vector<float> changeV = std::vector<float>(EMBEDDING, 0.0f);

    // add gradients and calculate errors
    for(int i = 0; i < EMBEDDING; i++) {
        changeH[i] = EH[i] - expected[i];
        changeV[i] = EV[i] - expected[i];
        gH[i] = std::accumulate(hor.gweights[0][i].begin(), hor.gweights[0][i].end(), 0.0f);
        gV[i] = std::accumulate(ver.gweights[0][i].begin(), ver.gweights[0][i].end(), 0.0f);
    }

    // change should be reflected to matrix (matheights x embedding) via gradient of MLP
    for (int i = 0; i < MATHEIGHTS; i++) {
        for (int j = 0; j < EMBEDDING; j++) {
            MH.a[i][j] += (gH[j] * LEARNING);       // Update MH using the gradient of MLP hor
            MV.a[i][j] += (gV[j] * LEARNING);       // update MV using the gradient of MLP ver
        }
    }

    // change should be reflected to MQ and MK (embedding x matheights) via gradient of MLP hor and vor
    for (int i = 0; i < EMBEDDING; i++) {
        for (int j = 0; j < MATHEIGHTS; j++) {
            MK.a[i][j] += (gH[i] * MH.a[j][i] * LEARNING);       // Update MH using the gradient of MLP hor
            MQ.a[i][j] += (gV[i] * MV.a[j][i] * LEARNING);       // update MV using the gradient of MLP ver
        }
    }
}
