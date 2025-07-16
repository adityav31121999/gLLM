#ifdef USE_CPU
#include "include/attention.hpp"
#include <vector>
#include <cmath>
#include <stdexcept>
#include <maths.hpp>
#include <numeric>


/**
 * @brief Backward propagation for heads of first block. Use in training of first block.
 * @param expected Expected output vector (target embedding for next token prediction).
 * @param in Input size (embedding dimension)
 * @param layers Number of layers in the MLPs
 * @param headnumber 1-based index of position of head in local context
 * @param learning Current learning rate
 */
void attention::backward1stHead(std::vector<float>& expected, int& in, int& layers, int headnumber, float& learning)
{
    // Check dimensions before starting
    if (expected.size() != EMBEDDING || EH.size() != EMBEDDING) {
        throw std::runtime_error("Dimension mismatch for expected/EH in backward1stHead (H)");
    }
    if (this->tokenCount <= 0 || this->tokenCount > K.row || this->tokenCount > Q.row || this->tokenCount > KdotQ.row || this->tokenCount > KdotQ.col) {
        throw std::runtime_error("Invalid or inconsistent tokenCount in backward1stHead (H)");
    }

    // Step 1: Compute loss gradient w.r.t. EH (for token prediction) and EV (for context)
    std::vector<float> grad_EH(EMBEDDING, 0.0f);
    std::vector<float> grad_EV(EMBEDDING, 0.0f);
    for (int i = 0; i < EMBEDDING; i++) {
        grad_EH[i] = 2.0f * (EH[i] - expected[i]);
        grad_EV[i] = grad_EH[i] * 0.1f; // EV gets a smaller portion of gradient (context preservation);
    }

    // Step 2: Backprop through MLPs (hor for EH, ver for EV)
    // Pass gradients through ReLU derivatives
    std::vector<float> grad_hor_input(EMBEDDING, 0.0f);
    std::vector<float> grad_ver_input(EMBEDDING, 0.0f);
    for (int i = 0; i < EMBEDDING; i++) {
        grad_hor_input[i] = grad_EH[i] * (hor.output[i] > 0 ? 1.0f : 0.0f);
        grad_ver_input[i] = grad_EV[i] * (ver.output[i] > 0 ? 1.0f : 0.0f);
    }

    // Set MLP for backprop
    hor.expected = grad_hor_input;
    // Note: The original code assigns to ver.output, which is unusual for 'expected' values in backprop.
    // Assuming ver.output here is meant to be ver.expected based on common backprop patterns.
    ver.expected = grad_ver_input;
    hor.backwithElasticNet(in, layers, learning);
    ver.backwithElasticNet(in, layers, learning);

    // Step 3: Compute gradients w.r.t. dh and dv
    std::vector<float> grad_dh(EMBEDDING, 0.0f);
    std::vector<float> grad_dv(EMBEDDING, 0.0f);
    for (int i = 0; i < EMBEDDING; i++) {
        if (hor.gweights.empty() || hor.gweights[0].mapped_data == nullptr ||
            ver.gweights.empty() || ver.gweights[0].mapped_data == nullptr) {
            throw std::runtime_error("MLP gweights not initialized in backward1stHead (H)");
        }
        for(int j = 0; j < EMBEDDING; j++) {
            grad_dh[i] += hor.gweights[0](j, i); // Gradient from first layer of hor MLP
            grad_dv[i] += ver.gweights[0](j, i); // Gradient from first layer of ver MLP
        }
    }

    // Step 4: Compute gradients w.r.t. MH and MV
    mat grad_MH(MATHEIGHTS, EMBEDDING);
    mat grad_MV(MATHEIGHTS, EMBEDDING);
    std::fill_n(grad_MH.mapped_data, grad_MH.row * grad_MH.col, 0.0f);
    std::fill_n(grad_MV.mapped_data, grad_MV.row * grad_MV.col, 0.0f);

    // Declare and initialize pre_MH and pre_MV
    std::vector<float> pre_MH(MATHEIGHTS, 0.0f);
    std::vector<float> pre_MV(MATHEIGHTS, 0.0f);
    mat head = LOTA(KdotQ, this->tokenCount, isSelfAttention);

    for (int i = 0; i < this->tokenCount; i++) {
        float sum_head_row = 0.0f;
        float sum_head_col = 0.0f;
        int limit_j = isSelfAttention ? (i + 1) : this->tokenCount;
        limit_j = std::min(limit_j, head.col);
        for (int j = 0; j < limit_j; j++) {
            sum_head_row += head(i, j);
            if (j < head.row && i < head.col) { // Original condition for sum_head_col
                sum_head_col += head(j, i);
            }
        }
        std::vector<float> K_row_i = getRow(K, i);
        std::vector<float> Q_row_i = getRow(Q, i);

        for (int h = 0; h < MATHEIGHTS; h++) {
            if (h < K_row_i.size())
                pre_MH[h] += sum_head_row * K_row_i[h];
            if (h < Q_row_i.size())
                pre_MV[h] += sum_head_col * Q_row_i[h];
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
        std::vector<float> K_row_i = getRow(K, i);
        int limit_j = isSelfAttention ? (i + 1) : this->tokenCount;
        limit_j = std::min(limit_j, grad_head.col);
        for (int j = 0; j < limit_j; j++) {
            if (j >= Q.row) continue;
            std::vector<float> Q_row_j = getRow(Q, j);
            float grad_dh_sum = 0.0f;
            float grad_dv_sum = 0.0f;
            for (int d = 0; d < EMBEDDING; d++) {
                float mh_grad_dh_d_term = 0.0f;
                float mv_grad_dv_d_term = 0.0f;
                for (int h_idx = 0; h_idx < MATHEIGHTS; h_idx++) { // Renamed h to h_idx to avoid conflict
                    if (d < MH.col) mh_grad_dh_d_term += MH(h_idx, d) * grad_dh[d];
                    if (d < MV.col) mv_grad_dv_d_term += MV(h_idx, d) * grad_dv[d];
                }
                if (d < K_row_i.size()) grad_dh_sum += K_row_i[d] * mh_grad_dh_d_term;
                if (d < Q_row_j.size()) grad_dv_sum += Q_row_j[d] * mv_grad_dv_d_term;
            }
            if (i < grad_head.row && j < grad_head.col) {
                grad_head(i, j) = grad_dh_sum + grad_dv_sum;
            }
        }
    }

    // Step 6: Backprop through LOTA
    mat grad_KdotQ(this->tokenCount, this->tokenCount);
    mat lota_derivative = LOTAder(KdotQ, this->tokenCount, isSelfAttention);

    if (grad_head.row != lota_derivative.row || grad_head.col != lota_derivative.col ||
        grad_head.row != grad_KdotQ.row || grad_head.col != grad_KdotQ.col) {
        throw std::runtime_error("Dimension mismatch for LOTA backprop in backward1stHead (H)");
    }

    float inv_scaling = 1.0f / SCALING;
    for (int i = 0; i < this->tokenCount; i++) {
        int limit_j = isSelfAttention ? (i + 1) : this->tokenCount;
        limit_j = std::min(limit_j, grad_KdotQ.col);
        for (int j = 0; j < limit_j; j++)
        {
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
        int limit_j_k = this->tokenCount;
        limit_j_k = std::min({limit_j_k, grad_KdotQ.col, Q.row});
        for (int j = 0; j < limit_j_k; j++) {
            std::vector<float> Q_row_j = getRow(Q, j);
            float grad_kq_ij = (i < grad_KdotQ.row && j < grad_KdotQ.col) ? grad_KdotQ(i, j) : 0.0f;
            for (int h = 0; h < MATHEIGHTS; h++) {
                if (h < Q_row_j.size())
                {
                    grad_K(i, h) += grad_kq_ij * Q_row_j[h];
                }
            }
        }
        int limit_j_q = this->tokenCount;
        limit_j_q = std::min({limit_j_q, grad_KdotQ.row, K.row});
        for (int j = 0; j < limit_j_q; ++j) {
            std::vector<float> K_row_j = getRow(K, j);
            float grad_kq_ji = (j < grad_KdotQ.row && i < grad_KdotQ.col) ? grad_KdotQ(j, i) : 0.0f; // Transposed access
            for (int h = 0; h < MATHEIGHTS; ++h)
            {
                if (h < K_row_j.size()) {
                    grad_Q(i, h) += grad_kq_ji * K_row_j[h];
                }
            }
        }
    }

    // Step 8: Compute gradients w.r.t. MQ and MK
    // grad_MK = InputEmbed^T * grad_K (simplified logic from original code)
    // grad_MQ = InputEmbed^T * grad_Q (simplified logic from original code)
    mat grad_MQ(MATHEIGHTS, EMBEDDING);
    mat grad_MK(MATHEIGHTS, EMBEDDING);
    std::fill_n(grad_MQ.mapped_data, grad_MQ.row * grad_MQ.col, 0.0f);
    std::fill_n(grad_MK.mapped_data, grad_MK.row * grad_MK.col, 0.0f);

    for (int i = 0; i < this->tokenCount; i++) {
        std::vector<float> grad_K_row_i = getRow(grad_K, i);
        std::vector<float> grad_Q_row_i = getRow(grad_Q, i);
        std::vector<float> K_row_i = getRow(K, i); // Proxy for input tokens for MK
        std::vector<float> Q_row_i = getRow(Q, i); // Proxy for input tokens for MQ

        for (int h = 0; h < MATHEIGHTS; h++) {
            for (int d = 0; d < EMBEDDING; d++) {
                float K_proxy_d = (d < K_row_i.size()) ? K_row_i[d] : 0.0f;
                float Q_proxy_d = (d < Q_row_i.size()) ? Q_row_i[d] : 0.0f;
                if (h < grad_K_row_i.size())
                    grad_MK(h, d) += grad_K_row_i[h] * K_proxy_d;
                if (h < grad_Q_row_i.size())
                    grad_MQ(h, d) += grad_Q_row_i[h] * Q_proxy_d;
            }
        }
    }

    // Ensure dimensions match before update
    if (MH.row != grad_MH.row || MH.col != grad_MH.col ||
        MV.row != grad_MV.row || MV.col != grad_MV.col ||
        MQ.row != grad_MQ.row || MQ.col != grad_MQ.col ||
        MK.row != grad_MK.row || MK.col != grad_MK.col)
    {
        throw std::runtime_error("Weight and gradient dimension mismatch in backward1stHead (H) during update.");
    }

    float learning_rate_val = learning; // Renamed to avoid shadowing parameter
    for (int i = 0; i < MATHEIGHTS; i++) {
        for (int j = 0; j < EMBEDDING; j++) {
            // MH update with Elastic Net
            if (i < MH.row && j < MH.col) {
                float sgn_MH = (MH(i, j) > 0 ? 1.0f : (MH(i, j) < 0 ? -1.0f : 0.0f));
                float l1_reg_term_MH = this->lambda_L1 * sgn_MH;
                float l2_reg_term_MH = 2.0f * this->lambda_L2 * MH(i, j);
                MH(i, j) -= learning_rate_val * (grad_MH(i, j) + l1_reg_term_MH + l2_reg_term_MH);
            }
            // MV update with Elastic Net
            if (i < MV.row && j < MV.col) {
                float sgn_MV = (MV(i, j) > 0 ? 1.0f : (MV(i, j) < 0 ? -1.0f : 0.0f));
                float l1_reg_term_MV = this->lambda_L1 * sgn_MV;
                float l2_reg_term_MV = 2.0f * this->lambda_L2 * MV(i, j);
                MV(i, j) -= learning_rate_val * (grad_MV(i, j) + l1_reg_term_MV + l2_reg_term_MV);
            }
            // MQ update with Elastic Net
            if (i < MQ.row && j < MQ.col) {
                float sgn_MQ = (MQ(i, j) > 0 ? 1.0f : (MQ(i, j) < 0 ? -1.0f : 0.0f));
                float l1_reg_term_MQ = this->lambda_L1 * sgn_MQ;
                float l2_reg_term_MQ = 2.0f * this->lambda_L2 * MQ(i, j);
                MQ(i, j) -= learning_rate_val * (grad_MQ(i, j) + l1_reg_term_MQ + l2_reg_term_MQ);
            }
            // MK update with Elastic Net
            if (i < MK.row && j < MK.col) {
                float sgn_MK = (MK(i, j) > 0 ? 1.0f : (MK(i, j) < 0 ? -1.0f : 0.0f));
                float l1_reg_term_MK = this->lambda_L1 * sgn_MK;
                float l2_reg_term_MK = 2.0f * this->lambda_L2 * MK(i, j);
                MK(i, j) -= learning_rate_val * (grad_MK(i, j) + l1_reg_term_MK + l2_reg_term_MK);
            }
        }
    }

    // Step 10: Update EH and EV using gradients
    if(headnumber > 1) { // Original code only updates EH if headnumber > 1
        for (int i = 0; i < EMBEDDING; i++) {
            EH[i] -= learning * grad_EH[i];
        }
    }
    // The commented out section for EV update was consistent with the condition,
    // but typically EV is also updated. Depending on your architecture's intent,
    // you might want to uncomment it or keep it as is.
    /*for(int i = 0; i < CONTEXT_WIN; i++) {
        if (i >= EV.row) break; // Check EV row bounds
        for(int j = 0; j < EMBEDDING; j++) {
            if (j < EV.col) EV(i, j) -= learning * grad_EV[j];
        }
    }*/
}


/**
 * @brief Backward Propagation (for first head) for the attention class using gradients from expected
 *      Vertical output only. Used for all Blocks and repetitions (applicable when there is continuation
 *      from  previous blocks and repetitions). This is to adjust MQ and MV so that there is no loss for
 *      context retention and matrices for Horizontal pass remain un-affected from these changes i.e.,
 *      no change in MK and MH (first head of each block, except first block).
 * @param expectedV vertical retention vector
 * @param in Input size (number of tokens)
 * @param layers Number of layers in the MLPs
 * @param learning Current learning rate
 */
void attention::backward1stHead(std::vector<std::vector<float>>& expectedV, int& in, int& layers, float& learning)
{
    // Ensure tokenCount is valid
    if (this->tokenCount <= 0 || K.mapped_data == nullptr || Q.mapped_data == nullptr || KdotQ.mapped_data == nullptr || this->tokenCount > K.row || this->tokenCount > Q.row || this->tokenCount > KdotQ.row || this->tokenCount > KdotQ.col) {
        throw std::runtime_error("Invalid or inconsistent tokenCount in backward1stHead (V)");
    }
    // Corrected EV dimension checks: EV is MATHEIGHTS x EMBEDDING, not EMBEDDING x CONTEXT_WIN as implied by original comments/code
    if (EV.mapped_data == nullptr || EV.row != MATHEIGHTS || EV.col != EMBEDDING || expectedV.size() != MATHEIGHTS || (!expectedV.empty() && expectedV[0].size() != EMBEDDING)) {
        throw std::runtime_error("Dimension mismatch for expectedV/EV in backward1stHead (V)");
    }

    // Step 1: Compute loss gradient w.r.t. EV (for context)
    mat grad_EV_mat(MATHEIGHTS, EMBEDDING); // Corrected dimensions to match EV
    std::fill_n(grad_EV_mat.mapped_data, grad_EV_mat.row * grad_EV_mat.col, 0.0f);
    std::vector<float> grad_EV_summed_for_mlp(EMBEDDING, 0.0f); // Summed for MLP input

    for(int i = 0; i < MATHEIGHTS; i++) { // Iterate over rows
        for(int j = 0; j < EMBEDDING; j++) { // Iterate over columns
            if (i < EV.row && j < EV.col && i < expectedV.size() && j < expectedV[i].size()) {
                grad_EV_mat(i, j) = 2.0f * (EV(i, j) - expectedV[i][j]);
                grad_EV_summed_for_mlp[j] += grad_EV_mat(i, j); // Sum columns for MLP input
            }
        }
    }
    for(int i = 0; i < EMBEDDING; i++){
        grad_EV_summed_for_mlp[i] /= MATHEIGHTS; // Average over MATHEIGHTS
    }

    // Step 2: Backprop through MLPs (ver for EV)
    std::vector<float> grad_ver_input(EMBEDDING, 0.0f);
    for (int i = 0; i < EMBEDDING; i++) {
        grad_ver_input[i] = grad_EV_summed_for_mlp[i] * (ver.output[i] > 0 ? 1.0f : 0.0f);
    }

    // Set MLP inputs for backprop
    // Note: Original code assigned to ver.output, corrected to ver.expected
    ver.expected = grad_ver_input;
    ver.backwithElasticNet(in, layers, learning); // 'in' refers to input size, which is EMBEDDING for this MLP

    // Step 3: Compute gradients w.r.t. dv
    std::vector<float> grad_dv(EMBEDDING, 0.0f);
    if (ver.gweights.empty() || ver.gweights[0].mapped_data == nullptr) {
        throw std::runtime_error("MLP gweights not initialized in backward1stHead (V)");
    }
    for (int i = 0; i < EMBEDDING; i++) {
        for(int j = 0; j < EMBEDDING; ++j) {
            grad_dv[i] += ver.gweights[0](j, i);
        }
    }

    // Step 4: Compute gradients w.r.t. MV
    mat grad_MV(MATHEIGHTS, EMBEDDING);
    std::fill_n(grad_MV.mapped_data, grad_MV.row * grad_MV.col, 0.0f);
    std::vector<float> pre_MV(MATHEIGHTS, 0.0f);

    mat head = LOTA(KdotQ, this->tokenCount, isSelfAttention);

    for (int i = 0; i < this->tokenCount; i++) {
        float sum_head_col = 0.0f;
        int limit_j = this->tokenCount;
        limit_j = std::min(limit_j, head.row);
        for (int j = 0; j < limit_j; j++) {
            if (i < head.col) { // Check column index for head
                sum_head_col += head(j, i);
            }
        }
        std::vector<float> Q_row_i = getRow(Q, i);

        for (int h = 0; h < MATHEIGHTS; h++) {
            if (h < Q_row_i.size()) pre_MV[h] += sum_head_col * Q_row_i[h];
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
            std::vector<float> Q_row_j = getRow(Q, j);
            float grad_dv_sum = 0.0f;
            for (int d = 0; d < EMBEDDING; d++) {
                float mv_grad_dv_d_term = 0.0f;
                for (int h_idx = 0; h_idx < MATHEIGHTS; h_idx++) { // Renamed h to h_idx
                    if (d < MV.col) mv_grad_dv_d_term += MV(h_idx, d) * grad_dv[d];
                }
                // Accumulate dot product parts
                if (d < Q_row_j.size()) grad_dv_sum += Q_row_j[d] * mv_grad_dv_d_term;
            }
            if (i < grad_head.row && j < grad_head.col) {
                grad_head(i, j) = grad_dv_sum;
            }
        }
    }

    // Step 6: Backprop through LOTA
    mat grad_KdotQ(this->tokenCount, this->tokenCount);
    mat lota_derivative = LOTAder(KdotQ, this->tokenCount, isSelfAttention);

    if (grad_head.row != lota_derivative.row || grad_head.col != lota_derivative.col ||
        grad_head.row != grad_KdotQ.row || grad_head.col != grad_KdotQ.col) {
        throw std::runtime_error("Dimension mismatch for LOTA backprop in backward1stHead (V)");
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
    // grad_Q = grad_KdotQ^T * K
    mat grad_Q(this->tokenCount, MATHEIGHTS); // Corrected to MATHEIGHTS as per usage below
    std::fill_n(grad_Q.mapped_data, grad_Q.row * grad_Q.col, 0.0f);

    for (int i = 0; i < this->tokenCount; i++) {
        int limit_j_q = this->tokenCount;
        limit_j_q = std::min({limit_j_q, grad_KdotQ.row, K.row});
        for (int j = 0; j < limit_j_q; ++j) {
             std::vector<float> K_row_j = getRow(K, j);
             float grad_kq_ji = (j < grad_KdotQ.row && i < grad_KdotQ.col) ? grad_KdotQ(j, i) : 0.0f; // Transposed access
            for (int h = 0; h < MATHEIGHTS; h++) {
                 if (h < K_row_j.size()) {
                    grad_Q(i, h) += grad_kq_ji * K_row_j[h];
                 }
             }
        }
    }

    // Step 8: Compute gradients w.r.t. MQ and MK correction
    // Corrected dimensions for grad_MQ to MATHEIGHTS x EMBEDDING
    mat grad_MQ(MATHEIGHTS, EMBEDDING);
    std::fill_n(grad_MQ.mapped_data, grad_MQ.row * grad_MQ.col, 0.0f);
    for (int i = 0; i < this->tokenCount; i++) {
        std::vector<float> grad_Q_row_i = getRow(grad_Q, i);
        std::vector<float> Q_row_i = getRow(Q, i);

        for (int h = 0; h < MATHEIGHTS; h++) { // Loop over rows of grad_MQ (MATHEIGHTS)
            for (int d = 0; d < EMBEDDING; d++) { // Loop over columns of grad_MQ (EMBEDDING)
                float Q_proxy_d = (d < Q_row_i.size()) ? Q_row_i[d] : 0.0f; // Assuming Q_row_i represents embedding dim
                if (h < grad_Q_row_i.size()) { // Assuming grad_Q_row_i is MATHEIGHTS long
                     grad_MQ(h, d) += grad_Q_row_i[h] * Q_proxy_d;
                }
            }
        }
    }

    mat grad_MK_correction(MATHEIGHTS, EMBEDDING);
    std::fill_n(grad_MK_correction.mapped_data, grad_MK_correction.row * grad_MK_correction.col, 0.0f);
    for (int i = 0; i < this->tokenCount; ++i) {
        std::vector<float> K_row_i = getRow(K, i);
        for (int j = 0; j < this->tokenCount; ++j) {
            std::vector<float> Q_row_j = getRow(Q, j);
            for (int h = 0; h < MATHEIGHTS; h++) {
                for (int d = 0; d < EMBEDDING; d++) {
                    float Q_proxy_h = (h < Q_row_j.size()) ? Q_row_j[h] : 0.0f;
                    float K_proxy_h = (h < K_row_i.size()) ? K_row_i[h] : 0.0f;
                    // This term assumes MQ(h,d) is involved in how MK gets its gradient.
                    // This specific interaction might need mathematical re-verification depending on exact architecture.
                    if (h < grad_MQ.row && d < grad_MQ.col)
                        grad_MK_correction(h, d) += -grad_MQ(h, d) * Q_proxy_h * K_proxy_h;
                }
            }
        }
    }


    // Step 9: Update weights MV, MQ, MK with Elastic Net
    if (MV.row != grad_MV.row || MV.col != grad_MV.col ||
        MQ.row != grad_MQ.row || MQ.col != grad_MQ.col ||
        MK.row != grad_MK_correction.row || MK.col != grad_MK_correction.col) {
         throw std::runtime_error("Weight and gradient dimension mismatch in backward1stHead (V) during update.");
    }

    float learning_rate_val = learning; // Renamed to avoid shadowing parameter
    for (int i = 0; i < MATHEIGHTS; i++) { // Corrected loop order for MV, MQ, MK
        for (int j = 0; j < EMBEDDING; j++) {
            // MK update with Elastic Net
            if (i < MK.row && j < MK.col) {
                float sgn_MK = (MK(i, j) > 0 ? 1.0f : (MK(i, j) < 0 ? -1.0f : 0.0f));
                float l1_reg_term_MK = this->lambda_L1 * sgn_MK;
                float l2_reg_term_MK = 2.0f * this->lambda_L2 * MK(i, j);
                MK(i, j) -= learning_rate_val * (grad_MK_correction(i, j) + l1_reg_term_MK + l2_reg_term_MK);
            }
            // MV update with Elastic Net
            if (i < MV.row && j < MV.col) {
                float sgn_MV = (MV(i, j) > 0 ? 1.0f : (MV(i, j) < 0 ? -1.0f : 0.0f));
                float l1_reg_term_MV = this->lambda_L1 * sgn_MV;
                float l2_reg_term_MV = 2.0f * this->lambda_L2 * MV(i, j);
                MV(i, j) -= learning_rate_val * (grad_MV(i, j) + l1_reg_term_MV + l2_reg_term_MV);
            }
            // MQ update with Elastic Net
            if (i < MQ.row && j < MQ.col) {
                float sgn_MQ = (MQ(i, j) > 0 ? 1.0f : (MQ(i, j) < 0 ? -1.0f : 0.0f));
                float l1_reg_term_MQ = this->lambda_L1 * sgn_MQ;
                float l2_reg_term_MQ = 2.0f * this->lambda_L2 * MQ(i, j);
                MQ(i, j) -= learning_rate_val * (grad_MQ(i, j) + l1_reg_term_MQ + l2_reg_term_MQ);
            }
        }
    }
    // No MH update in this function as per comment "no change in MK and MH" (though MK is updated)


    // Step 10: Update EV using element-wise gradients
    // This function specifically for first head of each block, meaning blocknumber might always be 1 or handled differently.
    // If blocknumber != 1 means subsequent blocks, then the EV update here might be for all blocks.
    // Assuming you want EV updated based on grad_EV_mat for the current block.
    for(int i = 0; i < MATHEIGHTS; i++) { // Iterate over rows of EV
        for(int j = 0; j < EMBEDDING; j++) { // Iterate over columns of EV
            if (i < EV.row && j < EV.col && i < grad_EV_mat.row && j < grad_EV_mat.col) {
                EV(i, j) -= learning * grad_EV_mat(i, j);
            }
        }
    }
}

#endif
