
// backward propagation
#include "include/attention.hpp"

/**
 * @brief Backward Propagation for the attention class using gradients from expected Horizontal output.
 *      Use for First block only.
 * @param expected Expected output vector (target embedding for next token prediction)
 * @param EV Vertical embedding vector (context transfer to next head)
 * @param in Input size (embedding dimension)
 * @param layers Number of layers in the MLPs
 */
void attention::backward(std::vector<float> expected, int& in, int& count, int& layers) 
{
    // Step 1: Compute loss gradient w.r.t. EH (for token prediction) and EV (for context)
    std::vector<float> grad_EH(EMBEDDING, 0.0f);
    std::vector<float> grad_EV(EMBEDDING, 0.0f);
    // EH is primarily responsible for token prediction
    for (int i = 0; i < EMBEDDING; i++) {
        grad_EH[i] = 2.0f * (EH[i] - expected[i]); // MSE gradient for EH
        grad_EV[i] = grad_EH[i] * 0.1f; // EV gets a smaller portion of gradient (context preservation)
    }

    // Step 2: Backprop through MLPs (hor for EH, ver for EV)
    // hor.expected = expected; // Commented out as per your version
    // ver.expected = expected; // Commented out as per your version

    // Pass gradients through ReLU derivatives
    std::vector<float> grad_hor_input(EMBEDDING, 0.0f);
    std::vector<float> grad_ver_input(EMBEDDING, 0.0f);
    for (int i = 0; i < EMBEDDING; i++) {
        grad_hor_input[i] = grad_EH[i] * (hor.output[i] > 0 ? 1.0f : 0.0f);
        grad_ver_input[i] = grad_EV[i] * (ver.output[i] > 0 ? 1.0f : 0.0f);
    }

    // Set MLP inputs for backprop
    hor.output = grad_hor_input;
    ver.output = grad_ver_input;
    hor.backward(in, layers, LEARNING);
    ver.backward(in, layers, LEARNING);

    // Step 3: Compute gradients w.r.t. dh and dv
    std::vector<float> grad_dh(EMBEDDING, 0.0f);
    std::vector<float> grad_dv(EMBEDDING, 0.0f);
    for (int i = 0; i < EMBEDDING; i++) {
        grad_dh[i] = hor.gweights[0][i][0]; // Gradient from first layer of hor MLP
        grad_dv[i] = ver.gweights[0][i][0]; // Gradient from first layer of ver MLP
    }

    // Step 4: Compute gradients w.r.t. MH and MV
    std::vector<std::vector<float>> grad_MH(MATHEIGHTS, std::vector<float>(EMBEDDING, 0.0f));
    std::vector<std::vector<float>> grad_MV(MATHEIGHTS, std::vector<float>(EMBEDDING, 0.0f));
    std::vector<float> pre_MH(MATHEIGHTS, 0.0f);
    std::vector<float> pre_MV(MATHEIGHTS, 0.0f);

    std::vector<std::vector<float>> head = std::vector<std::vector<float>>(tokenCount, std::vector<float>(tokenCount, 0.0f));
    head = LOTA(KdotQ, tokenCount);

    int tokenCount = K.size();
    for (int i = 0; i < tokenCount; i++) {
        float sum_head_row = 0.0f; // sum(head[i][j]) over j
        float sum_head_col = 0.0f; // sum(head[j][i]) over j
        for (int j = 0; j < tokenCount; j++) {
            sum_head_row += head[i][j];
            sum_head_col += head[j][i];
        }
        for (int h = 0; h < MATHEIGHTS; h++) {
            pre_MH[h] += sum_head_row * K[i][h];
            pre_MV[h] += sum_head_col * Q[i][h];
        }
    }

    for (int h = 0; h < MATHEIGHTS; h++) {
        for (int d = 0; d < EMBEDDING; d++) {
            grad_MH[h][d] = pre_MH[h] * grad_dh[d];
            grad_MV[h][d] = pre_MV[h] * grad_dv[d];
        }
    }

    // Step 5: Compute gradients w.r.t. head
    std::vector<std::vector<float>> grad_head(tokenCount, std::vector<float>(tokenCount, 0.0f));
    for (int i = 0; i < tokenCount; i++) {
        for (int j = 0; j < tokenCount; j++) {
            float grad_dh_sum = 0.0f;
            float grad_dv_sum = 0.0f;
            for (int d = 0; d < EMBEDDING; d++) {
                for (int h = 0; h < MATHEIGHTS; h++) {
                    grad_dh_sum += K[i][h] * MH.a[h][d] * grad_dh[d];
                    grad_dv_sum += Q[j][h] * MV.a[h][d] * grad_dv[d];
                }
            }
            grad_head[i][j] = grad_dh_sum + grad_dv_sum;
        }
    }

    // Step 6: Backprop through LOTA
    std::vector<std::vector<float>> grad_KdotQ(tokenCount, std::vector<float>(tokenCount, 0.0f));
    std::vector<std::vector<float>> lota_output = LOTA(KdotQ, tokenCount);
    float sum_lota = 0.0f;
    for (int i = 0; i < tokenCount; i++) {
        for (int j = 0; j < tokenCount; j++) {
            sum_lota += lota_output[i][j];
        }
    }

    for (int i = 0; i < tokenCount; i++) {
        for (int j = 0; j < tokenCount; j++) {
            float dLOTA_dKdotQ = (sum_lota - lota_output[i][j]) / (sum_lota * sum_lota);
            grad_KdotQ[i][j] = grad_head[i][j] * dLOTA_dKdotQ / SCALING;
        }
    }

    // Step 7: Compute gradients w.r.t. K and Q
    std::vector<std::vector<float>> grad_K(tokenCount, std::vector<float>(MATHEIGHTS, 0.0f));
    std::vector<std::vector<float>> grad_Q(tokenCount, std::vector<float>(MATHEIGHTS, 0.0f));
    for (int i = 0; i < tokenCount; i++) {
        for (int j = 0; j < tokenCount; j++) {
            for (int h = 0; h < MATHEIGHTS; h++) {
                grad_K[i][h] += grad_KdotQ[i][j] * Q[j][h];
                grad_Q[j][h] += grad_KdotQ[i][j] * K[i][h];
            }
        }
    }

    // Step 8: Compute gradients w.r.t. MQ and MK
    std::vector<std::vector<float>> grad_MQ(MATHEIGHTS, std::vector<float>(EMBEDDING, 0.0f));
    std::vector<std::vector<float>> grad_MK(MATHEIGHTS, std::vector<float>(EMBEDDING, 0.0f));

    for (int i = 0; i < tokenCount; i++) {
        for (int h = 0; h < MATHEIGHTS; h++) {
            for (int d = 0; d < EMBEDDING; d++) {
                grad_MK[h][d] += grad_K[i][h] * K[i][d]; // Simplified
                grad_MQ[h][d] += grad_Q[i][h] * Q[i][d]; // Simplified
            }
        }
    }

    // Step 9: Update weights MH, MV, MQ, MK
    for (int i = 0; i < MATHEIGHTS; i++) {
        for (int j = 0; j < EMBEDDING; j++) {
            MH.a[j][i] -= LEARNING * grad_MH[j][i];
            MV.a[j][i] -= LEARNING * grad_MV[j][i];
            MQ.a[i][j] -= LEARNING * grad_MQ[i][j];
            MK.a[i][j] -= LEARNING * grad_MK[i][j];
        }
    }

    // Step 10: Update EH and EV using gradients
    for (int i = 0; i < EMBEDDING; i++) {
        EH[i] -= LEARNING * grad_EH[i]; // Update EH with its gradient
        EV[count][i] -= LEARNING * grad_EV[i]; // Update EV with its gradient
    }
}


/**
 * @brief Backward Propagation for the attention class using gradients from expected Horizontal nd Vertical output.
 *      Use for Second to Last Block and repetitions.
 * @param expectedH Expected horizontal output vector (target embedding for next token prediction)
 * @param expectedV Expected vertical output vector (target retention of context for new block)
 * @param dh Horizontal weighted sum vector (computed in forward pass)
 * @param dv Vertical weighted sum vector (computed in forward pass)
 * @param EV Vertical embedding vector (context transfer to next head)
 * @param EH Horizontal embedding vector (next token prediction)
 * @param in Input size (number of tokens)
 * @param layers Number of layers in the MLPs
 */
void attention::backward(std::vector<float> expectedH, std::vector<float> expectedV, int& in, int& count, int& layers) 
{
    // Step 1: Compute loss gradient w.r.t. EH (for token prediction) and EV (for context)
    std::vector<float> grad_EH(EMBEDDING, 0.0f);
    std::vector<float> grad_EV(EMBEDDING, 0.0f);

    // EH is primarily responsible for token prediction
    for (int i = 0; i < EMBEDDING; i++) {
        grad_EH[i] = 2.0f * (EH[i] - expectedH[i]); // MSE gradient for EH
        grad_EV[i] = 2.0f * (EV[count][i] - expectedV[i]); // MSE gradient for EV
    }

    // Step 2: Backprop through MLPs (hor for EH, ver for EV)
    // hor.expected = expected; // Commented out as per your version
    // ver.expected = expected; // Commented out as per your version

    // Pass gradients through ReLU derivatives
    std::vector<float> grad_hor_input(EMBEDDING, 0.0f);
    std::vector<float> grad_ver_input(EMBEDDING, 0.0f);
    for (int i = 0; i < EMBEDDING; i++) {
        grad_hor_input[i] = grad_EH[i] * (hor.output[i] > 0 ? 1.0f : 0.0f);
        grad_ver_input[i] = grad_EV[i] * (ver.output[i] > 0 ? 1.0f : 0.0f);
    }

    // Set MLP inputs for backprop
    hor.output = grad_hor_input;
    ver.output = grad_ver_input;
    hor.backward(in, layers, LEARNING);
    ver.backward(in, layers, LEARNING);

    // Step 3: Compute gradients w.r.t. dh and dv
    std::vector<float> grad_dh(EMBEDDING, 0.0f);
    std::vector<float> grad_dv(EMBEDDING, 0.0f);
    for (int i = 0; i < EMBEDDING; i++) {
        grad_dh[i] = hor.gweights[0][i][0]; // Gradient from first layer of hor MLP
        grad_dv[i] = ver.gweights[0][i][0]; // Gradient from first layer of ver MLP
    }

    // Step 4: Compute gradients w.r.t. MH and MV
    std::vector<std::vector<float>> grad_MH(MATHEIGHTS, std::vector<float>(EMBEDDING, 0.0f));
    std::vector<std::vector<float>> grad_MV(MATHEIGHTS, std::vector<float>(EMBEDDING, 0.0f));
    std::vector<float> pre_MH(MATHEIGHTS, 0.0f);
    std::vector<float> pre_MV(MATHEIGHTS, 0.0f);

    std::vector<std::vector<float>> head = std::vector<std::vector<float>>(tokenCount, std::vector<float>(tokenCount, 0.0f));
    head = LOTA(KdotQ, tokenCount);

    int tokenCount = K.size();
    for (int i = 0; i < tokenCount; i++) {
        float sum_head_row = 0.0f; // sum(head[i][j]) over j
        float sum_head_col = 0.0f; // sum(head[j][i]) over j
        for (int j = 0; j < tokenCount; j++) {
            sum_head_row += head[i][j];
            sum_head_col += head[j][i];
        }
        for (int h = 0; h < MATHEIGHTS; h++) {
            pre_MH[h] += sum_head_row * K[i][h];
            pre_MV[h] += sum_head_col * Q[i][h];
        }
    }

    for (int h = 0; h < MATHEIGHTS; h++) {
        for (int d = 0; d < EMBEDDING; d++) {
            grad_MH[h][d] = pre_MH[h] * grad_dh[d];
            grad_MV[h][d] = pre_MV[h] * grad_dv[d];
        }
    }

    // Step 5: Compute gradients w.r.t. head
    std::vector<std::vector<float>> grad_head(tokenCount, std::vector<float>(tokenCount, 0.0f));
    for (int i = 0; i < tokenCount; i++) {
        for (int j = 0; j < tokenCount; j++) {
            float grad_dh_sum = 0.0f;
            float grad_dv_sum = 0.0f;
            for (int d = 0; d < EMBEDDING; d++) {
                for (int h = 0; h < MATHEIGHTS; h++) {
                    grad_dh_sum += K[i][h] * MH.a[h][d] * grad_dh[d];
                    grad_dv_sum += Q[j][h] * MV.a[h][d] * grad_dv[d];
                }
            }
            grad_head[i][j] = grad_dh_sum + grad_dv_sum;
        }
    }

    // Step 6: Backprop through LOTA
    std::vector<std::vector<float>> grad_KdotQ(tokenCount, std::vector<float>(tokenCount, 0.0f));
    std::vector<std::vector<float>> lota_output = LOTA(KdotQ, tokenCount);
    float sum_lota = 0.0f;
    for (int i = 0; i < tokenCount; i++) {
        for (int j = 0; j < tokenCount; j++) {
            sum_lota += lota_output[i][j];
        }
    }

    for (int i = 0; i < tokenCount; i++) {
        for (int j = 0; j < tokenCount; j++) {
            float dLOTA_dKdotQ = (sum_lota - lota_output[i][j]) / (sum_lota * sum_lota);
            grad_KdotQ[i][j] = grad_head[i][j] * dLOTA_dKdotQ / SCALING;
        }
    }

    // Step 7: Compute gradients w.r.t. K and Q
    std::vector<std::vector<float>> grad_K(tokenCount, std::vector<float>(MATHEIGHTS, 0.0f));
    std::vector<std::vector<float>> grad_Q(tokenCount, std::vector<float>(MATHEIGHTS, 0.0f));
    for (int i = 0; i < tokenCount; i++) {
        for (int j = 0; j < tokenCount; j++) {
            for (int h = 0; h < MATHEIGHTS; h++) {
                grad_K[i][h] += grad_KdotQ[i][j] * Q[j][h];
                grad_Q[j][h] += grad_KdotQ[i][j] * K[i][h];
            }
        }
    }

    // Step 8: Compute gradients w.r.t. MQ and MK
    std::vector<std::vector<float>> grad_MQ(MATHEIGHTS, std::vector<float>(EMBEDDING, 0.0f));
    std::vector<std::vector<float>> grad_MK(MATHEIGHTS, std::vector<float>(EMBEDDING, 0.0f));

    for (int i = 0; i < tokenCount; i++) {
        for (int h = 0; h < MATHEIGHTS; h++) {
            for (int d = 0; d < EMBEDDING; d++) {
                grad_MK[h][d] += grad_K[i][h] * K[i][d]; // Simplified
                grad_MQ[h][d] += grad_Q[i][h] * Q[i][d]; // Simplified
            }
        }
    }

    // Step 9: Update weights MH, MV, MQ, MK
    for (int i = 0; i < MATHEIGHTS; i++) {
        for (int j = 0; j < EMBEDDING; j++) {
            MH.a[j][i] -= LEARNING * grad_MH[j][i];
            MV.a[j][i] -= LEARNING * grad_MV[j][i];
            MQ.a[i][j] -= LEARNING * grad_MQ[i][j];
            MK.a[i][j] -= LEARNING * grad_MK[i][j];
        }
    }

    // Step 10: Update EH and EV using gradients
    for (int i = 0; i < EMBEDDING; i++) {
        EH[i] -= LEARNING * grad_EH[i]; // Update EH with its gradient
        EV[count][i] -= LEARNING * grad_EV[i]; // Update EV with its gradient
    }
}
