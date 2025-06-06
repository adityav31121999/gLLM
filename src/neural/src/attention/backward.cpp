
// backward propagation for attention class
#include "include/attention.hpp"

#ifdef USE_CPU

/**
 * @brief Backward propagation for subsequent heads in blocks.
 * @param in Input size (embedding dimension)
 * @param layers Number of layers in the MLPs
 */
void attention::backward(std::vector<float>& expected, int& in, int& layers) 
{
    // Ensure tokenCount is valid and matrices are mapped
    if (this->tokenCount <= 0 || K.mapped_data == nullptr || Q.mapped_data == nullptr || KdotQ.mapped_data == nullptr || EV.mapped_data == nullptr || MH.mapped_data == nullptr || MV.mapped_data == nullptr || MQ.mapped_data == nullptr || MK.mapped_data == nullptr) {
        throw std::runtime_error("Invalid tokenCount or unmapped matrix in backward (H)");
    }
    
    // Step 1: Compute loss gradient w.r.t. EH (for token prediction) and EV (for context)
    std::vector<float> grad_EH(EMBEDDING, 0.0f);
    std::vector<float> grad_EV(EMBEDDING, 0.0f);
    for (int i = 0; i < EMBEDDING; i++) {
        grad_EH[i] = 2.0f * (EH[i] - expected[i]); // MSE gradient for EH
        grad_EV[i] = grad_EH[i] * 0.1f; // EV gets a smaller portion of gradient (context preservation)
    }

    // Step 2: Backprop through MLPs (hor for EH, ver for EV)
    std::vector<float> grad_hor_input(EMBEDDING, 0.0f);
    std::vector<float> grad_ver_input(EMBEDDING, 0.0f);
    for (int i = 0; i < EMBEDDING; i++) {
        grad_hor_input[i] = grad_EH[i] * (hor.output[i] > 0 ? 1.0f : 0.0f);
        grad_ver_input[i] = grad_EV[i] * (ver.output[i] > 0 ? 1.0f : 0.0f);
    }

    // Set MLP inputs for backprop
    hor.expected = grad_hor_input;
    ver.expected = grad_ver_input;
    hor.backprop(in, layers, LEARNING);
    ver.backward(in, layers, LEARNING);

    if (hor.gweights.empty() || hor.gweights[0].mapped_data == nullptr || ver.gweights.empty() || ver.gweights[0].mapped_data == nullptr) {
        throw std::runtime_error("MLP gweights not initialized in backward (H)");
    }

    // Step 3: Compute gradients w.r.t. dh and dv
    std::vector<float> grad_dh(EMBEDDING, 0.0f);
    std::vector<float> grad_dv(EMBEDDING, 0.0f);
    for (int i = 0; i < EMBEDDING; i++) {
        for(int j = 0; j < EMBEDDING; j++) {
            grad_dh[i] += hor.gweights[0](j, i);
            grad_dv[i] += ver.gweights[0](j, i);
        }
    }

    // Step 4: Compute gradients w.r.t. MH and MV
    mat grad_MH(MATHEIGHTS, EMBEDDING);
    mat grad_MV(MATHEIGHTS, EMBEDDING);
    std::fill_n(grad_MH.mapped_data, grad_MH.row * grad_MH.col, 0.0f);
    std::fill_n(grad_MV.mapped_data, grad_MV.row * grad_MV.col, 0.0f);
    std::vector<float> pre_MH(MATHEIGHTS, 0.0f);
    std::vector<float> pre_MV(MATHEIGHTS, 0.0f);

    mat head = LOTA(KdotQ, tokenCount, isSelfAttention);

    for (int i = 0; i < this->tokenCount; i++) {
        float sum_head_row = 0.0f;
        float sum_head_col = 0.0f;
        int limit_j = this->tokenCount;
        limit_j = std::min({limit_j, head.row, head.col});
        for (int j = 0; j < limit_j; j++) {
            sum_head_row += head(i, j);
            sum_head_col += head(j, i);
        }
        for (int h = 0; h < MATHEIGHTS; h++) {
            if (i < K.row) pre_MH[h] += sum_head_row * K(i, h);
            if (i < Q.row) pre_MV[h] += sum_head_col * Q(i, h);
        }
    }

    for (int h = 0; h < MATHEIGHTS; h++) {
        for (int d = 0; d < EMBEDDING; d++) {
            grad_MH(h, d) = pre_MH[h] * grad_dh[d];
            grad_MV(h, d) = pre_MV[h] * grad_dv[d];
        }
    }

    // Step 5: Compute gradients w.r.t. head
    mat grad_head(this->tokenCount, this->tokenCount);
    std::fill_n(grad_head.mapped_data, grad_head.row * grad_head.col, 0.0f);

    for (int i = 0; i < this->tokenCount; i++) {
        int limit_j = this->tokenCount;
        limit_j = std::min(limit_j, grad_head.col);
        for (int j = 0; j < limit_j; j++) {
            if (i >= K.row || j >= Q.row) continue;
            float grad_dh_sum = 0.0f;
            float grad_dv_sum = 0.0f;
            for (int d = 0; d < EMBEDDING; d++) {
                for (int h = 0; h < MATHEIGHTS; h++) {
                    grad_dh_sum += K(i, h) * MH(h, d) * grad_dh[d];
                    grad_dv_sum += Q(j, h) * MV(h, d) * grad_dv[d];
                }
            }
            grad_head(i, j) = grad_dh_sum;
        }
    }

    // Step 6: Backprop through LOTA
    mat grad_KdotQ(this->tokenCount, this->tokenCount);
    mat lota_derivative = LOTAder(KdotQ, this->tokenCount, isSelfAttention);

    if (grad_head.row != lota_derivative.row || grad_head.col != lota_derivative.col ||
        grad_head.row != grad_KdotQ.row || grad_head.col != grad_KdotQ.col) {
        throw std::runtime_error("Dimension mismatch for LOTA backprop in backward (H)");
    }

    float inv_scaling = 1.0f / SCALING;
    for (int i = 0; i < this->tokenCount; i++) {
        int limit_j = this->tokenCount;
        limit_j = std::min(limit_j, grad_KdotQ.col);
        for (int j = 0; j < limit_j; j++) {
            grad_KdotQ(i, j) = grad_head(i, j) * lota_derivative(i, j) * inv_scaling;
        }
    }

    // Step 7: Compute gradients w.r.t. K and Q
    // grad_K = grad_KdotQ * Q^T
    // grad_Q = grad_KdotQ^T * K
    mat grad_K(this->tokenCount, MATHEIGHTS);
    mat grad_Q(this->tokenCount, MATHEIGHTS);
    std::fill_n(grad_K.mapped_data, grad_K.row * grad_K.col, 0.0f);
    std::fill_n(grad_Q.mapped_data, grad_Q.row * grad_Q.col, 0.0f);

    for (int i = 0; i < this->tokenCount; i++) {
        int limit_j = this->tokenCount;
        limit_j = std::min({limit_j, grad_KdotQ.row, grad_KdotQ.col, K.row, Q.row});
        for (int j = 0; j < limit_j; j++) {
            for (int h = 0; h < MATHEIGHTS; h++) {
                grad_K(i, h) += grad_KdotQ(i, j) * Q(j, h);
                grad_Q(i, h) += grad_KdotQ(j, i) * K(j, h);
            }
        }
    }

    // Step 8: Compute gradients w.r.t. MQ and MK
    mat grad_MQ(MATHEIGHTS, EMBEDDING);
    mat grad_MK(MATHEIGHTS, EMBEDDING);
    std::fill_n(grad_MQ.mapped_data, grad_MQ.row * grad_MQ.col, 0.0f);
    std::fill_n(grad_MK.mapped_data, grad_MK.row * grad_MK.col, 0.0f);

    for (int i = 0; i < this->tokenCount; i++) {
        if (i >= K.row || i >= Q.row || i >= grad_K.row || i >= grad_Q.row) continue;
        for (int h = 0; h < MATHEIGHTS; h++) {
            for (int d = 0; d < EMBEDDING; d++) {
                grad_MK(h, d) += grad_K(i, h) * K(i, d);
                grad_MQ(h, d) += grad_Q(i, h) * Q(i, d);
            }
        }
    }

    // Step 9: Update weights MH, MV, MQ, MK
    for (int i = 0; i < MATHEIGHTS; i++) {
        for (int j = 0; j < EMBEDDING; j++) {
            if (i < MH.row && j < MH.col) MH(i, j) -= LEARNING * grad_MH(i, j);
            if (i < MV.row && j < MV.col) MV(i, j) -= LEARNING * grad_MV(i, j);
            if (i < MQ.row && j < MQ.col) MQ(i, j) -= LEARNING * grad_MQ(i, j);
            if (i < MK.row && j < MK.col) MK(i, j) -= LEARNING * grad_MK(i, j);
        }
    }

    // Step 10: Update EH and EV using gradients
    for (int i = 0; i < EMBEDDING; i++) {
        EH[i] -= LEARNING * grad_EH[i];
    }
    for(int i = 0; i < CONTEXT_WIN; i++) {
        if (i >= EV.row) 
            break; // Check EV row bounds
        for(int j = 0; j < EMBEDDING; j++) {
            if (j < EV.col) EV(i, j) -= LEARNING * grad_EV[j];
        }
    }
}


/**
 * @brief Backward Propagation for the attention class using gradients from expected Vertical output only.
 *      Used for all Blocks and repetitions (applicabel when there is continuation from  previous blocks 
 *      and repetitions). This is to adjust MQ and MV so that there is no loss for context retention and 
 *      matrices for Horizontal pass remain un-affected from these changes i.e., no change in MK and MH
 *      (for blocks between first and last or kth block).
 * @param expectedV Vertical retention vector (target context)
 * @param layers Number of layers in the MLPs
 */
void attention::backward(std::vector<std::vector<float>>& expectedV, int& layers) 
{
    if (this->tokenCount <= 0 || K.mapped_data == nullptr || Q.mapped_data == nullptr || 
        KdotQ.mapped_data == nullptr || EV.mapped_data == nullptr || MH.mapped_data == nullptr || 
        MV.mapped_data == nullptr || MQ.mapped_data == nullptr || MK.mapped_data == nullptr) 
    {
        throw std::runtime_error("Invalid tokenCount or unmapped matrix in backward (V)");
    }
    if (EV.row != CONTEXT_WIN || EV.col != EMBEDDING || expectedV.size() != CONTEXT_WIN || 
        (!expectedV.empty() && expectedV[0].size() != EMBEDDING)) 
    {
        throw std::runtime_error("Dimension mismatch for expectedV/EV in backward (V)");
    }

    // Step 1: Compute loss gradient w.r.t. EV (element-wise) and average for MLP input
    mat grad_EV_mat(CONTEXT_WIN, EMBEDDING);
    std::vector<float> grad_EV_summed(EMBEDDING, 0.0f);

    for(int j = 0; j < CONTEXT_WIN; j++) {
        if (j >= EV.row || j >= expectedV.size()) continue;
        for(int i = 0; i < EMBEDDING; i++) {
            if (i >= EV.col || i >= expectedV[j].size()) 
                continue;
            grad_EV_mat(j, i) = 2.0f * (EV(j, i) - expectedV[j][i]);
            grad_EV_summed[i] += grad_EV_mat(j, i);
        }
    }
    for(int i = 0; i < EMBEDDING; i++){
        grad_EV_summed[i] /= CONTEXT_WIN;
    }

    // Step 2: Backprop through MLPs (hor for EH, ver for EV)
    std::vector<float> grad_ver_input(EMBEDDING, 0.0f);
    for (int i = 0; i < EMBEDDING; i++) {
        grad_ver_input[i] = grad_EV_summed[i] * (ver.output[i] > 0 ? 1.0f : 0.0f);
    }

    ver.expected = grad_ver_input;
    ver.backward(layers, EMBEDDING, LEARNING); 

    // Ensure MLP gradients are available
    if (ver.gweights.empty() || ver.gweights[0].mapped_data == nullptr) {
        throw std::runtime_error("MLP gweights not initialized in backward (V)");
    }

    // Step 3: Compute gradients w.r.t. dh and dv
    std::vector<float> grad_dv(EMBEDDING, 0.0f);
    for (int i = 0; i < EMBEDDING; i++) {
        for(int j = 0; j < EMBEDDING; ++j) {
            grad_dv[i] += ver.gweights[0](j, i);
        }
    }

    // Step 4: Compute gradients w.r.t. MH and MV
    mat grad_MV(MATHEIGHTS, EMBEDDING);
    std::fill_n(grad_MV.mapped_data, grad_MV.row * grad_MV.col, 0.0f);
    std::vector<float> pre_MV(MATHEIGHTS, 0.0f);

    mat head = LOTA(KdotQ, this->tokenCount, isSelfAttention);

    for (int i = 0; i < this->tokenCount; i++) {
        float sum_head_col = 0.0f;
        int limit_j = this->tokenCount;
        limit_j = std::min({limit_j, head.row, head.col});
        for (int j = 0; j < limit_j; j++) {
            sum_head_col += head(j, i);
        }
        for (int h = 0; h < MATHEIGHTS; h++) {
            if (i < Q.row) pre_MV[h] += sum_head_col * Q(i, h);
        }
    }

    for (int h = 0; h < MATHEIGHTS; h++) {
        for (int d = 0; d < EMBEDDING; d++) {
            grad_MV(h, d) = pre_MV[h] * grad_dv[d];
        }
    }

    // Step 5: Compute gradients w.r.t. head
    mat grad_head(this->tokenCount, this->tokenCount);
    std::fill_n(grad_head.mapped_data, grad_head.row * grad_head.col, 0.0f);

    for (int i = 0; i < this->tokenCount; i++) {
        int limit_j = this->tokenCount;
        limit_j = std::min(limit_j, grad_head.col);
        for (int j = 0; j < limit_j; j++) {
            if (j >= Q.row) continue;
            float grad_dv_sum = 0.0f;
            for (int d = 0; d < EMBEDDING; d++) {
                for (int h = 0; h < MATHEIGHTS; h++) {
                    grad_dv_sum += Q(j, h) * MV(h, d) * grad_dv[d];
                }
            }
            if (i < grad_head.row) grad_head(i, j) = grad_dv_sum;
        }
    }

    // Step 6: Backprop through LOTA
    mat grad_KdotQ(this->tokenCount, this->tokenCount);
    mat lota_derivative = LOTAder(KdotQ, this->tokenCount, isSelfAttention);
    if (grad_head.row != lota_derivative.row || grad_head.col != lota_derivative.col ||
        grad_head.row != grad_KdotQ.row || grad_head.col != grad_KdotQ.col) {
        throw std::runtime_error("Dimension mismatch for LOTA backprop in backward (V)");
    }
    float inv_scaling = 1.0f / SCALING;
    for (int i = 0; i < this->tokenCount; i++) {
        int limit_j = this->tokenCount;
        limit_j = std::min(limit_j, grad_KdotQ.col);
        for (int j = 0; j < limit_j; j++) {
            grad_KdotQ(i, j) = grad_head(i, j) * lota_derivative(i, j) * inv_scaling;
        }
    }

    // Step 7: Compute gradients w.r.t. K and Q
    mat grad_Q(this->tokenCount, MATHEIGHTS);
    std::fill_n(grad_Q.mapped_data, grad_Q.row * grad_Q.col, 0.0f);
    for (int i = 0; i < this->tokenCount; i++) {
        int limit_j = this->tokenCount;
        limit_j = std::min({limit_j, grad_KdotQ.row, grad_KdotQ.col, K.row});
        for (int j = 0; j < limit_j; j++) {
            for (int h = 0; h < MATHEIGHTS; h++) {
                grad_Q(i, h) += grad_KdotQ(j, i) * K(j, h);
            }
        }
    }

    // Step 7.5: Removed redundant/incorrect grad_KdotQ recalculation
    // Step 8: Compute gradients w.r.t. MQ and MK (more sophisticated)
    mat grad_MQ(MATHEIGHTS, EMBEDDING);
    mat grad_MK_correction(MATHEIGHTS, EMBEDDING);
    std::fill_n(grad_MQ.mapped_data, grad_MQ.row * grad_MQ.col, 0.0f);
    std::fill_n(grad_MK_correction.mapped_data, grad_MK_correction.row * grad_MK_correction.col, 0.0f);

    // Calculate grad_MQ first (using Q as proxy for T)
    for (int i = 0; i < this->tokenCount; i++) {
        if (i >= Q.row || i >= grad_Q.row) continue;
        for (int h = 0; h < MATHEIGHTS; h++) {
            for (int d = 0; d < EMBEDDING; d++) {
                grad_MQ(h, d) += grad_Q(i, h) * Q(i, d);
            }
        }
    }

    // Calculate grad_MK_correction using the final grad_MQ
    for (int i = 0; i < this->tokenCount; i++) {
        if (i >= K.row) continue;
        for (int j = 0; j < this->tokenCount; j++) {
            if (j >= Q.row) continue;
            for (int h = 0; h < MATHEIGHTS; h++) {
                for (int d = 0; d < EMBEDDING; d++) {
                    grad_MK_correction(h, d) += -grad_MQ(h, d) * Q(j, h) * K(i, h);
                }
            }
        }
    }

    // Step 9: Update weights MH, MV, MQ, MK
    if (MV.row != grad_MV.row || MV.col != grad_MV.col ||
        MQ.row != grad_MQ.row || MQ.col != grad_MQ.col ||
        MK.row != grad_MK_correction.row || MK.col != grad_MK_correction.col) 
    {
        throw std::runtime_error("Weight and gradient dimension mismatch in backward (V)");
    }
    for (int i = 0; i < MATHEIGHTS; i++) {
        for (int j = 0; j < EMBEDDING; j++) {
            if (i < MV.row && j < MV.col) MV(i, j) -= LEARNING * grad_MV(i, j);
            if (i < MQ.row && j < MQ.col) MQ(i, j) -= LEARNING * grad_MQ(i, j);
            if (i < MK.row && j < MK.col) MK(i, j) -= LEARNING * grad_MK_correction(i, j);
        }
    }

    // Step 10: Update EV using element-wise gradients
    for(int i = 0; i < CONTEXT_WIN; i++) {
        if (i >= EV.row || i >= grad_EV_mat.row) 
            break;
        for(int j = 0; j < EMBEDDING; j++) {
            if (j >= EV.col || j >= grad_EV_mat.col) 
                break;
            EV(i, j) -= LEARNING * grad_EV_mat(i, j);
        }
    }
}

#endif
