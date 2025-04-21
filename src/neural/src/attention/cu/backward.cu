
// Include necessary headers from the project
#include <maths.hpp>
#include "include/attention.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cfloat>
#include <crt/device_functions.h>
#include <vector>
#include <stdexcept>
#include <iostream>

/**
 * @brief Backward Propagation for the attention class using gradients from expected Horizontal output.
 *      Use for first (when sentence ends in first block itself) and last block only.
 * @param expected Expected output vector (target embedding for next token prediction)
 * @param in Input size (embedding dimension) - Corresponds to EMBEDDING
 * @param layers Number of layers in the MLPs
 */
void attention::cuBackward(std::vector<float>& expected, int& in, int& layers)
{
    const int embedding_dim = EMBEDDING;
    const int mat_heights = MATHEIGHTS;
    const int context_win = CONTEXT_WIN;
    const float learning_rate = LEARNING;
    const float scaling_factor = SCALING;
    const int head_size = this->tokenCount * this->tokenCount;
    const int mh_mv_mq_mk_size = mat_heights * embedding_dim;
    const int k_q_size = this->tokenCount * mat_heights;
    const int ev_size = context_win * embedding_dim;

    // --- Allocate Temporary Device Memory ---
    float *d_expected_h = nullptr, *d_grad_EH = nullptr, *d_grad_EV_scaled = nullptr;
    float *d_grad_hor_input = nullptr, *d_grad_ver_input = nullptr;
    float *d_grad_dh = nullptr, *d_grad_dv = nullptr;
    float *d_pre_MH = nullptr, *d_pre_MV = nullptr;
    float *d_grad_MH = nullptr, *d_grad_MV = nullptr;
    float *d_head = nullptr, *d_grad_head = nullptr;
    float *d_global_sum_head = nullptr, *d_lota_deriv_simple = nullptr;
    float *d_grad_KdotQ = nullptr;
    float *d_grad_K = nullptr, *d_grad_Q = nullptr;
    float *d_grad_MQ = nullptr, *d_grad_MK = nullptr;

    try {
        // calculate error from expected token embedding to output token
        // backpropagate mlp hor and ver
        // backpropagate with gradients of mlps and update for horizontal and vertical pass
    }
    catch (const std::exception& e) {
        //
    }
}

/**
 * @brief Backward Propagation for the attention class using gradients from expected Vertical output only.
 *      Used for intermediate blocks. Adjusts MQ, MV, and MK (correction).
 *      Corresponds to C++: attention::backward(std::vector<std::vector<float>>& expectedV, ...)
 * @param expectedV vertical retention vector (host)
 * @param in Input size (number of tokens) - Corresponds to tokenCount
 * @param layers Number of layers in the MLPs
 */
void attention::cuBackward(std::vector<std::vector<float>>& expectedV, int& in, int& layers)
{
    const int embedding_dim = EMBEDDING;
    const int mat_heights = MATHEIGHTS;
    const int context_win = CONTEXT_WIN;
    const float learning_rate = LEARNING;
    const float scaling_factor = SCALING; // Needed for complex LOTA derivative kernel
    const int head_size = this->tokenCount * this->tokenCount;
    const int mh_mv_mq_mk_size = mat_heights * embedding_dim;
    const int k_q_size = this->tokenCount * mat_heights;
    const int ev_size = context_win * embedding_dim;

    float *d_expected_v = nullptr, *d_grad_EV_full = nullptr, *d_grad_EV_summed = nullptr, *d_grad_EV_scaled = nullptr; // Renamed
    float *d_grad_ver_input = nullptr;
    float *d_grad_dv = nullptr; // Points to MLP gradient
    float *d_pre_MV = nullptr;
    float *d_grad_MV = nullptr;
    float *d_head = nullptr, *d_grad_head = nullptr;
    float *d_lota_deriv_complex = nullptr;
    float *d_grad_KdotQ = nullptr;
    float *d_grad_Q = nullptr;
    float *d_grad_MQ = nullptr, *d_grad_MK_correction = nullptr;

    try {
        //
    }
    catch (const std::exception& e) {
        //
    }
}


/**
 * @brief Backward Propagation (for first head) using gradients from expected Horizontal output.
 *      Almost identical to cuBackward(expected, ...), but only updates EH conditionally.
 * @param expected Expected output vector (target embedding for next token prediction)
 * @param in Input size (embedding dimension)
 * @param layers Number of layers in the MLPs
 * @param first Boolean flag (passed by value for CUDA context) - Determines if EH is updated.
 */
void attention::cuBackward1stHead(std::vector<float>& expected, int& in, int& layers) {
    // --- Setup ---
    const int embedding_dim = EMBEDDING;
    const int mat_heights = MATHEIGHTS;
    const int context_win = CONTEXT_WIN; // Needed for size constants
    const float learning_rate = LEARNING;
    const float scaling_factor = SCALING;
    const int head_size = this->tokenCount * this->tokenCount;
    const int mh_mv_mq_mk_size = mat_heights * embedding_dim;
    const int k_q_size = this->tokenCount * mat_heights;

    float *d_expected_h = nullptr, *d_grad_EH = nullptr, *d_grad_EV_for_mlp = nullptr; // Grad EV only needed for MLP
    float *d_grad_hor_input = nullptr, *d_grad_ver_input = nullptr;
    float *d_grad_dh = nullptr, *d_grad_dv = nullptr;
    float *d_pre_MH = nullptr, *d_pre_MV = nullptr;
    float *d_grad_MH = nullptr, *d_grad_MV = nullptr;
    float *d_head = nullptr, *d_grad_head = nullptr;
    float *d_global_sum_head = nullptr, *d_lota_deriv_simple = nullptr;
    float *d_grad_KdotQ = nullptr;
    float *d_grad_K = nullptr, *d_grad_Q = nullptr;
    float *d_grad_MQ = nullptr, *d_grad_MK = nullptr;

    try {
        //
    }
    catch (const std::exception& e) {
        //
    }
}


/**
 * @brief Backward Propagation (for first head) using gradients from expected Vertical output only.
 *      Adjusts MQ, MV, and MK (correction). No update to EH/EV.
 * @param expectedV vertical retention vector (host)
 * @param in Input size (number of tokens)
 * @param layers Number of layers in the MLPs
 */
void attention::cuBackward1stHead(std::vector<std::vector<float>>& expectedV, int& in, int& layers) 
{
    const int embedding_dim = EMBEDDING;
    const int mat_heights = MATHEIGHTS;
    const int context_win = CONTEXT_WIN;
    const float learning_rate = LEARNING;
    const float scaling_factor = SCALING;
    const int head_size = this->tokenCount * this->tokenCount;
    const int mh_mv_mq_mk_size = mat_heights * embedding_dim;
    const int k_q_size = this->tokenCount * mat_heights;
    const int ev_size = context_win * embedding_dim;

    float *d_expected_v = nullptr, *d_grad_EV_full = nullptr, *d_grad_EV_summed = nullptr, *d_grad_EV_scaled = nullptr;
    float *d_grad_ver_input = nullptr;
    float *d_grad_dv = nullptr;
    float *d_pre_MV = nullptr;
    float *d_grad_MV = nullptr;
    float *d_head = nullptr, *d_grad_head = nullptr;
    float *d_lota_deriv_complex = nullptr;
    float *d_grad_KdotQ = nullptr;
    float *d_grad_Q = nullptr;
    float *d_grad_MQ = nullptr, *d_grad_MK_correction = nullptr;

    try {
        //
    }
    catch (const std::exception& e) {
        //
    }
}


/**
 * @brief Backward Propagation (for first head) using gradients from both Horizontal and Vertical outputs.
 *      Updates MH, MV, MQ, MK. No update to EH/EV.
 * @param expectedH Horizontal embedding vector (next token prediction) (host)
 * @param expectedV Vertical retention vector (host)
 * @param in Input size (number of tokens)
 * @param layers Number of layers in the MLPs
 */
void attention::cuBackward1stHead(std::vector<float>& expectedH, std::vector<std::vector<float>>& expectedV, int& in, int& layers) 
{
    const int embedding_dim = EMBEDDING;
    const int mat_heights = MATHEIGHTS;
    const int context_win = CONTEXT_WIN;
    const float learning_rate = LEARNING;
    const float scaling_factor = SCALING;
    const int head_size = this->tokenCount * this->tokenCount;
    const int mh_mv_mq_mk_size = mat_heights * embedding_dim;
    const int k_q_size = this->tokenCount * mat_heights;
    const int ev_size = context_win * embedding_dim;

    float *d_expected_h = nullptr, *d_expected_v = nullptr;
    float *d_grad_EH = nullptr, *d_grad_EV_full = nullptr, *d_grad_EV_summed = nullptr, *d_grad_EV_scaled = nullptr; // d_grad_EV is scaled summed
    float *d_grad_hor_input = nullptr, *d_grad_ver_input = nullptr;
    float *d_grad_dh = nullptr, *d_grad_dv = nullptr;
    float *d_pre_MH = nullptr, *d_pre_MV = nullptr;
    float *d_grad_MH = nullptr, *d_grad_MV = nullptr;
    float *d_head = nullptr, *d_grad_head = nullptr;
    float *d_global_sum_head = nullptr, *d_lota_deriv_simple = nullptr; // Using simple LOTA deriv based on C++
    float *d_grad_KdotQ = nullptr;
    float *d_grad_K = nullptr, *d_grad_Q = nullptr;
    float *d_grad_MQ = nullptr, *d_grad_MK = nullptr;

    try {
        //
    }
    catch (const std::exception& e) {
        //
    }
}
