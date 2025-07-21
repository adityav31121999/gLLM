// Helper macro for indexing flattened matrix (assuming row-major)
#define IDX(row, col, dim) ((row) * (dim) + (col))

// Enable extensions for atomics and potentially double precision (which might include float atomics)
// #pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
// #pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable
// #pragma OPENCL EXTENSION cl_khr_fp64 : enable // For double support
// #pragma OPENCL EXTENSION cl_khr_float_atomics : enable // Not supported on target, using manual implementation

__kernel void kernelUpdateWeights_EH_EV(__global float* mh_a, __global float* mv_a, __global float* mq_a, __global float* mk_a,
                                        __global float* eh, __global float* ev,
                                        __global const float* grad_mh, __global const float* grad_mv,
                                        __global const float* grad_mq, __global const float* grad_mk,
                                        __global const float* grad_eh, __global const float* grad_ev_scaled,
                                        float learning_rate, int update_eh, int update_ev,
                                        int mat_heights, int embedding_dim, int context_win)
{
    int idx = get_global_id(0);
    int ev_size = context_win * embedding_dim; // Define ev_size
    int matrix_size = mat_heights * embedding_dim;
    if (idx < matrix_size) {
        mh_a[idx] -= learning_rate * grad_mh[idx];
        mv_a[idx] -= learning_rate * grad_mv[idx];
        mq_a[idx] -= learning_rate * grad_mq[idx];
        mk_a[idx] -= learning_rate * grad_mk[idx];
    }
    if (update_eh != 0 && idx < embedding_dim) {
        eh[idx] -= learning_rate * grad_eh[idx];
    }
    if (update_ev != 0) {
        if (idx < ev_size) {
            int embed_idx = idx % embedding_dim;
            ev[idx] -= learning_rate * grad_ev_scaled[embed_idx];
        }
    }
}

__kernel void kernelUpdateWeights_1stHead_H(__global float* mh_a, __global float* mv_a, __global float* mq_a, __global float* mk_a,
                                            __global float* eh,
                                            __global const float* grad_mh, __global const float* grad_mv,
                                            __global const float* grad_mq, __global const float* grad_mk,
                                            __global const float* grad_eh,
                                            float learning_rate, int update_eh,
                                            int mat_heights, int embedding_dim)
{
    int idx = get_global_id(0);
    int matrix_size = mat_heights * embedding_dim;
    if (idx < matrix_size) {
        if(grad_mh != NULL) mh_a[idx] -= learning_rate * grad_mh[idx];
        if(grad_mv != NULL) mv_a[idx] -= learning_rate * grad_mv[idx];
        if(grad_mq != NULL) mq_a[idx] -= learning_rate * grad_mq[idx];
        if(grad_mk != NULL) mk_a[idx] -= learning_rate * grad_mk[idx];
    }
    // only update EH when updateEH is true, this for all heads of blocks,except first column
    if (update_eh != 0 && idx < embedding_dim) {
        if(grad_eh != NULL) eh[idx] -= learning_rate * grad_eh[idx];
    }
}

__kernel void kernelUpdateWeights_1stHead_V(__global float* mv_a, __global float* mq_a, __global float* mk_a,
                                            __global const float* grad_mv, __global const float* grad_mq,
                                            __global const float* grad_mk_correction,
                                            float learning_rate, int mat_heights, int embedding_dim)
{
    int idx = get_global_id(0);
    int matrix_size = mat_heights * embedding_dim;
    if (idx < matrix_size) {
        if(grad_mv != NULL) mv_a[idx] -= learning_rate * grad_mv[idx];
        if(grad_mq != NULL) mq_a[idx] -= learning_rate * grad_mq[idx];
        if(grad_mk_correction != NULL) mk_a[idx] -= learning_rate * grad_mk_correction[idx];
    }
}

__kernel void kernelUpdateWeights_1stHead_HV(__global float* mh_a, __global float* mv_a, __global float* mq_a, __global float* mk_a,
                                             __global const float* grad_mh, __global const float* grad_mv,
                                             __global const float* grad_mq, __global const float* grad_mk,
                                             float learning_rate, int mat_heights, int embedding_dim)
{
    int idx = get_global_id(0);
    int matrix_size = mat_heights * embedding_dim;
    if (idx < matrix_size) {
        if(grad_mh != NULL) mh_a[idx] -= learning_rate * grad_mh[idx];
        if(grad_mv != NULL) mv_a[idx] -= learning_rate * grad_mv[idx];
        if(grad_mq != NULL) mq_a[idx] -= learning_rate * grad_mq[idx];
        if(grad_mk != NULL) mk_a[idx] -= learning_rate * grad_mk[idx];
    }
}

__kernel void kernelUpdateWeights_EV_V(__global float* mv_a, __global float* mq_a, __global float* mk_a, __global float* ev,
                                       __global const float* grad_mv, __global const float* grad_mq, // grad_mv, grad_mq are mat_heights x embedding_dim
                                       __global const float* grad_mk_correction, // grad_mk_correction is mat_heights x embedding_dim
                                       __global const float* grad_ev_full,
                                       float learning_rate,
                                       int update_ev, int mat_heights, int embedding_dim, int context_win)
{
    int idx = get_global_id(0);
    int matrix_size = mat_heights * embedding_dim;
    if (idx < matrix_size) {
        mv_a[idx] -= learning_rate * grad_mv[idx];
        mq_a[idx] -= learning_rate * grad_mq[idx];
        mk_a[idx] -= learning_rate * grad_mk_correction[idx];
    }
    // update for all blocks, except first block
    if (update_ev != 0) {
        int ev_size = context_win * embedding_dim;
        if (idx < ev_size) {
            ev[idx] -= learning_rate * grad_ev_full[idx];
        }
    }
}

// 1. L1-only variant for kernelUpdateWeights_EV_V
__kernel void kernelUpdateWeights_EV_V_L1(__global float* mv_a, __global float* mq_a, __global float* mk_a, __global float* ev,
                                       __global const float* grad_mv, __global const float* grad_mq,
                                       __global const float* grad_mk_correction,
                                       __global const float* grad_ev_full,
                                       float learning_rate,
                                       float lambda_l1, // L1 parameter only
                                       int update_ev, int mat_heights, int embedding_dim, int context_win)
{
    int idx = get_global_id(0);
    int matrix_size = mat_heights * embedding_dim;
    if (idx < matrix_size) {
        // MV update with L1
        if(grad_mv != NULL) {
            float l1_reg_term_MV = lambda_l1 * sign_f(mv_a[idx]);
            mv_a[idx] -= learning_rate * (grad_mv[idx] + l1_reg_term_MV);
        }
        // MQ update with L1
        if(grad_mq != NULL) {
            float l1_reg_term_MQ = lambda_l1 * sign_f(mq_a[idx]);
            mq_a[idx] -= learning_rate * (grad_mq[idx] + l1_reg_term_MQ);
        }
        // MK update with L1
        if(grad_mk_correction != NULL) {
            float l1_reg_term_MK = lambda_l1 * sign_f(mk_a[idx]);
            mk_a[idx] -= learning_rate * (grad_mk_correction[idx] + l1_reg_term_MK);
        }
    }
    // EV update (not a weight matrix, no L1 regularization here)
    if (update_ev != 0) {
        int ev_size = context_win * embedding_dim;
        if (idx < ev_size) {
            if(grad_ev_full != NULL) ev[idx] -= learning_rate * grad_ev_full[idx];
        }
    }
}

// 2. L1-only variant for kernelUpdateWeights_EH_EV
__kernel void kernelUpdateWeights_EH_EV_L1(__global float* mh_a, __global float* mv_a, __global float* mq_a, __global float* mk_a,
                                        __global float* eh, __global float* ev,
                                        __global const float* grad_mh, __global const float* grad_mv,
                                        __global const float* grad_mq, __global const float* grad_mk,
                                        __global const float* grad_eh, __global const float* grad_ev_scaled,
                                        float learning_rate, int update_eh, int update_ev,
                                        float lambda_l1, // L1 parameter only
                                        int mat_heights, int embedding_dim, int context_win)
{
    int idx = get_global_id(0);
    int ev_size = context_win * embedding_dim;
    int matrix_size = mat_heights * embedding_dim;
    if (idx < matrix_size) {
        // MH update with L1
        if (grad_mh != NULL) {
            float l1_reg_term_MH = lambda_l1 * sign_f(mh_a[idx]);
            mh_a[idx] -= learning_rate * (grad_mh[idx] + l1_reg_term_MH);
        }
        // MV update with L1
        if (grad_mv != NULL) {
            float l1_reg_term_MV = lambda_l1 * sign_f(mv_a[idx]);
            mv_a[idx] -= learning_rate * (grad_mv[idx] + l1_reg_term_MV);
        }
        // MQ update with L1
        if (grad_mq != NULL) {
            float l1_reg_term_MQ = lambda_l1 * sign_f(mq_a[idx]);
            mq_a[idx] -= learning_rate * (grad_mq[idx] + l1_reg_term_MQ);
        }
        // MK update with L1
        if (grad_mk != NULL) {
            float l1_reg_term_MK = lambda_l1 * sign_f(mk_a[idx]);
            mk_a[idx] -= learning_rate * (grad_mk[idx] + l1_reg_term_MK);
        }
    }
    // EH and EV updates (not weight matrices, no L1 regularization here)
    if (update_eh != 0 && idx < embedding_dim) {
        if(grad_eh != NULL) eh[idx] -= learning_rate * grad_eh[idx];
    }
    if (update_ev != 0) {
        if (idx < ev_size) {
            int embed_idx = idx % embedding_dim;
            if(grad_ev_scaled != NULL) ev[idx] -= learning_rate * grad_ev_scaled[embed_idx];
        }
    }
}

// 3. L1-only variant for kernelUpdateWeights_1stHead_H
__kernel void kernelUpdateWeights_1stHead_H_L1(__global float* mh_a, __global float* mv_a, __global float* mq_a, __global float* mk_a,
                                            __global float* eh,
                                            __global const float* grad_mh, __global const float* grad_mv,
                                            __global const float* grad_mq, __global const float* grad_mk,
                                            __global const float* grad_eh,
                                            float learning_rate, int update_eh,
                                            float lambda_l1, // L1 parameter only
                                            int mat_heights, int embedding_dim)
{
    int idx = get_global_id(0);
    int matrix_size = mat_heights * embedding_dim;
    if (idx < matrix_size) {
        // MH update with L1
        if(grad_mh != NULL) {
            float l1_reg_term_MH = lambda_l1 * sign_f(mh_a[idx]);
            mh_a[idx] -= learning_rate * (grad_mh[idx] + l1_reg_term_MH);
        }
        // MV update with L1
        if(grad_mv != NULL) {
            float l1_reg_term_MV = lambda_l1 * sign_f(mv_a[idx]);
            mv_a[idx] -= learning_rate * (grad_mv[idx] + l1_reg_term_MV);
        }
        // MQ update with L1
        if(grad_mq != NULL) {
            float l1_reg_term_MQ = lambda_l1 * sign_f(mq_a[idx]);
            mq_a[idx] -= learning_rate * (grad_mq[idx] + l1_reg_term_MQ);
        }
        // MK update with L1
        if(grad_mk != NULL) {
            float l1_reg_term_MK = lambda_l1 * sign_f(mk_a[idx]);
            mk_a[idx] -= learning_rate * (grad_mk[idx] + l1_reg_term_MK);
        }
    }
    // EH update (not a weight matrix, no L1 regularization here)
    if (update_eh != 0 && idx < embedding_dim) {
        if(grad_eh != NULL) eh[idx] -= learning_rate * grad_eh[idx];
    }
}

// 4. L1-only variant for kernelUpdateWeights_1stHead_V
__kernel void kernelUpdateWeights_1stHead_V_L1(__global float* mv_a, __global float* mq_a, __global float* mk_a,
                                            __global const float* grad_mv, __global const float* grad_mq,
                                            __global const float* grad_mk_correction,
                                            float learning_rate,
                                            float lambda_l1, // L1 parameter only
                                            int mat_heights, int embedding_dim)
{
    int idx = get_global_id(0);
    int matrix_size = mat_heights * embedding_dim;
    if (idx < matrix_size) {
        // MV update with L1
        if(grad_mv != NULL) {
            float l1_reg_term_MV = lambda_l1 * sign_f(mv_a[idx]);
            mv_a[idx] -= learning_rate * (grad_mv[idx] + l1_reg_term_MV);
        }
        // MQ update with L1
        if(grad_mq != NULL) {
            float l1_reg_term_MQ = lambda_l1 * sign_f(mq_a[idx]);
            mq_a[idx] -= learning_rate * (grad_mq[idx] + l1_reg_term_MQ);
        }
        // MK update with L1
        if(grad_mk_correction != NULL) {
            float l1_reg_term_MK = lambda_l1 * sign_f(mk_a[idx]);
            mk_a[idx] -= learning_rate * (grad_mk_correction[idx] + l1_reg_term_MK);
        }
    }
}

// 5. L1-only variant for kernelUpdateWeights_1stHead_HV
__kernel void kernelUpdateWeights_1stHead_HV_L1(__global float* mh_a, __global float* mv_a, __global float* mq_a, __global float* mk_a,
                                             __global const float* grad_mh, __global const float* grad_mv,
                                             __global const float* grad_mq, __global const float* grad_mk,
                                             float learning_rate,
                                             float lambda_l1, // L1 parameter only
                                             int mat_heights, int embedding_dim)
{
    int idx = get_global_id(0);
    int matrix_size = mat_heights * embedding_dim;
    if (idx < matrix_size) {
        // MH update with L1
        if(grad_mh != NULL) {
            float l1_reg_term_MH = lambda_l1 * sign_f(mh_a[idx]);
            mh_a[idx] -= learning_rate * (grad_mh[idx] + l1_reg_term_MH);
        }
        // MV update with L1
        if(grad_mv != NULL) {
            float l1_reg_term_MV = lambda_l1 * sign_f(mv_a[idx]);
            mv_a[idx] -= learning_rate * (grad_mv[idx] + l1_reg_term_MV);
        }
        // MQ update with L1
        if(grad_mq != NULL) {
            float l1_reg_term_MQ = lambda_l1 * sign_f(mq_a[idx]);
            mq_a[idx] -= learning_rate * (grad_mq[idx] + l1_reg_term_MQ);
        }
        // MK update with L1
        if(grad_mk != NULL) {
            float l1_reg_term_MK = lambda_l1 * sign_f(mk_a[idx]);
            mk_a[idx] -= learning_rate * (grad_mk[idx] + l1_reg_term_MK);
        }
    }
}

// 1. L2-only variant for kernelUpdateWeights_EV_V
__kernel void kernelUpdateWeights_EV_V_L2(__global float* mv_a, __global float* mq_a, __global float* mk_a, __global float* ev,
                                       __global const float* grad_mv, __global const float* grad_mq,
                                       __global const float* grad_mk_correction,
                                       __global const float* grad_ev_full,
                                       float learning_rate,
                                       float lambda_l2, // L2 parameter only
                                       int update_ev, int mat_heights, int embedding_dim, int context_win)
{
    int idx = get_global_id(0);
    int matrix_size = mat_heights * embedding_dim;
    if (idx < matrix_size) {
        // MV update with L2
        if(grad_mv != NULL) {
            float l2_reg_term_MV = 2.0f * lambda_l2 * mv_a[idx];
            mv_a[idx] -= learning_rate * (grad_mv[idx] + l2_reg_term_MV);
        }
        // MQ update with L2
        if(grad_mq != NULL) {
            float l2_reg_term_MQ = 2.0f * lambda_l2 * mq_a[idx];
            mq_a[idx] -= learning_rate * (grad_mq[idx] + l2_reg_term_MQ);
        }
        // MK update with L2
        if(grad_mk_correction != NULL) {
            float l2_reg_term_MK = 2.0f * lambda_l2 * mk_a[idx];
            mk_a[idx] -= learning_rate * (grad_mk_correction[idx] + l2_reg_term_MK);
        }
    }
    // EV update (not a weight matrix, no L2 regularization here)
    if (update_ev != 0) {
        int ev_size = context_win * embedding_dim;
        if (idx < ev_size) {
            if(grad_ev_full != NULL) ev[idx] -= learning_rate * grad_ev_full[idx];
        }
    }
}

// 2. L2-only variant for kernelUpdateWeights_EH_EV
__kernel void kernelUpdateWeights_EH_EV_L2(__global float* mh_a, __global float* mv_a, __global float* mq_a, __global float* mk_a,
                                        __global float* eh, __global float* ev,
                                        __global const float* grad_mh, __global const float* grad_mv,
                                        __global const float* grad_mq, __global const float* grad_mk,
                                        __global const float* grad_eh, __global const float* grad_ev_scaled,
                                        float learning_rate, int update_eh, int update_ev,
                                        float lambda_l2, // L2 parameter only
                                        int mat_heights, int embedding_dim, int context_win)
{
    int idx = get_global_id(0);
    int ev_size = context_win * embedding_dim;
    int matrix_size = mat_heights * embedding_dim;
    if (idx < matrix_size) {
        // MH update with L2
        if (grad_mh != NULL) {
            float l2_reg_term_MH = 2.0f * lambda_l2 * mh_a[idx];
            mh_a[idx] -= learning_rate * (grad_mh[idx] + l2_reg_term_MH);
        }
        // MV update with L2
        if (grad_mv != NULL) {
            float l2_reg_term_MV = 2.0f * lambda_l2 * mv_a[idx];
            mv_a[idx] -= learning_rate * (grad_mv[idx] + l2_reg_term_MV);
        }
        // MQ update with L2
        if (grad_mq != NULL) {
            float l2_reg_term_MQ = 2.0f * lambda_l2 * mq_a[idx];
            mq_a[idx] -= learning_rate * (grad_mq[idx] + l2_reg_term_MQ);
        }
        // MK update with L2
        if (grad_mk != NULL) {
            float l2_reg_term_MK = 2.0f * lambda_l2 * mk_a[idx];
            mk_a[idx] -= learning_rate * (grad_mk[idx] + l2_reg_term_MK);
        }
    }
    // EH and EV updates (not weight matrices, no L2 regularization here)
    if (update_eh != 0 && idx < embedding_dim) {
        if(grad_eh != NULL) eh[idx] -= learning_rate * grad_eh[idx];
    }
    if (update_ev != 0) {
        if (idx < ev_size) {
            int embed_idx = idx % embedding_dim;
            if(grad_ev_scaled != NULL) ev[idx] -= learning_rate * grad_ev_scaled[embed_idx];
        }
    }
}

// 3. L2-only variant for kernelUpdateWeights_1stHead_H
__kernel void kernelUpdateWeights_1stHead_H_L2(__global float* mh_a, __global float* mv_a, __global float* mq_a, __global float* mk_a,
                                            __global float* eh,
                                            __global const float* grad_mh, __global const float* grad_mv,
                                            __global const float* grad_mq, __global const float* grad_mk,
                                            __global const float* grad_eh,
                                            float learning_rate, int update_eh,
                                            float lambda_l2, // L2 parameter only
                                            int mat_heights, int embedding_dim)
{
    int idx = get_global_id(0);
    int matrix_size = mat_heights * embedding_dim;
    if (idx < matrix_size) {
        // MH update with L2
        if(grad_mh != NULL) {
            float l2_reg_term_MH = 2.0f * lambda_l2 * mh_a[idx];
            mh_a[idx] -= learning_rate * (grad_mh[idx] + l2_reg_term_MH);
        }
        // MV update with L2
        if(grad_mv != NULL) {
            float l2_reg_term_MV = 2.0f * lambda_l2 * mv_a[idx];
            mv_a[idx] -= learning_rate * (grad_mv[idx] + l2_reg_term_MV);
        }
        // MQ update with L2
        if(grad_mq != NULL) {
            float l2_reg_term_MQ = 2.0f * lambda_l2 * mq_a[idx];
            mq_a[idx] -= learning_rate * (grad_mq[idx] + l2_reg_term_MQ);
        }
        // MK update with L2
        if(grad_mk != NULL) {
            float l2_reg_term_MK = 2.0f * lambda_l2 * mk_a[idx];
            mk_a[idx] -= learning_rate * (grad_mk[idx] + l2_reg_term_MK);
        }
    }
    // EH update (not a weight matrix, no L2 regularization here)
    if (update_eh != 0 && idx < embedding_dim) {
        if(grad_eh != NULL) eh[idx] -= learning_rate * grad_eh[idx];
    }
}

// 4. L2-only variant for kernelUpdateWeights_1stHead_V
__kernel void kernelUpdateWeights_1stHead_V_L2(__global float* mv_a, __global float* mq_a, __global float* mk_a,
                                            __global const float* grad_mv, __global const float* grad_mq,
                                            __global const float* grad_mk_correction,
                                            float learning_rate,
                                            float lambda_l2, // L2 parameter only
                                            int mat_heights, int embedding_dim)
{
    int idx = get_global_id(0);
    int matrix_size = mat_heights * embedding_dim;
    if (idx < matrix_size) {
        // MV update with L2
        if(grad_mv != NULL) {
            float l2_reg_term_MV = 2.0f * lambda_l2 * mv_a[idx];
            mv_a[idx] -= learning_rate * (grad_mv[idx] + l2_reg_term_MV);
        }
        // MQ update with L2
        if(grad_mq != NULL) {
            float l2_reg_term_MQ = 2.0f * lambda_l2 * mq_a[idx];
            mq_a[idx] -= learning_rate * (grad_mq[idx] + l2_reg_term_MQ);
        }
        // MK update with L2
        if(grad_mk_correction != NULL) {
            float l2_reg_term_MK = 2.0f * lambda_l2 * mk_a[idx];
            mk_a[idx] -= learning_rate * (grad_mk_correction[idx] + l2_reg_term_MK);
        }
    }
}

// 5. L2-only variant for kernelUpdateWeights_1stHead_HV
__kernel void kernelUpdateWeights_1stHead_HV_L2(__global float* mh_a, __global float* mv_a, __global float* mq_a, __global float* mk_a,
                                             __global const float* grad_mh, __global const float* grad_mv,
                                             __global const float* grad_mq, __global const float* grad_mk,
                                             float learning_rate,
                                             float lambda_l2, // L2 parameter only
                                             int mat_heights, int embedding_dim)
{
    int idx = get_global_id(0);
    int matrix_size = mat_heights * embedding_dim;
    if (idx < matrix_size) {
        // MH update with L2
        if(grad_mh != NULL) {
            float l2_reg_term_MH = 2.0f * lambda_l2 * mh_a[idx];
            mh_a[idx] -= learning_rate * (grad_mh[idx] + l2_reg_term_MH);
        }
        // MV update with L2
        if(grad_mv != NULL) {
            float l2_reg_term_MV = 2.0f * lambda_l2 * mv_a[idx];
            mv_a[idx] -= learning_rate * (grad_mv[idx] + l2_reg_term_MV);
        }
        // MQ update with L2
        if(grad_mq != NULL) {
            float l2_reg_term_MQ = 2.0f * lambda_l2 * mq_a[idx];
            mq_a[idx] -= learning_rate * (grad_mq[idx] + l2_reg_term_MQ);
        }
        // MK update with L2
        if(grad_mk != NULL) {
            float l2_reg_term_MK = 2.0f * lambda_l2 * mk_a[idx];
            mk_a[idx] -= learning_rate * (grad_mk[idx] + l2_reg_term_MK);
        }
    }
}

__kernel void kernelUpdateWeights_EH_EV_ElasticNet(__global float* mh_a, __global float* mv_a, __global float* mq_a, __global float* mk_a,
                                        __global float* eh, __global float* ev,
                                        __global const float* grad_mh, __global const float* grad_mv,
                                        __global const float* grad_mq, __global const float* grad_mk,
                                        __global const float* grad_eh, __global const float* grad_ev_scaled,
                                        float learning_rate, int update_eh, int update_ev,
                                        float lambda_l1, float lambda_l2, // ADDED: Elastic Net regularization parameters
                                        int mat_heights, int embedding_dim, int context_win)
{
    int idx = get_global_id(0);
    int ev_size = context_win * embedding_dim;
    int matrix_size = mat_heights * embedding_dim;

    if (idx < matrix_size) {
        // MH update with Elastic Net
        if (grad_mh != NULL) {
            float sgn_MH = sign_f(mh_a[idx]);
            float l1_reg_term_MH = lambda_l1 * sgn_MH;
            float l2_reg_term_MH = 2.0f * lambda_l2 * mh_a[idx];
            mh_a[idx] -= learning_rate * (grad_mh[idx] + l1_reg_term_MH + l2_reg_term_MH);
        }

        // MV update with Elastic Net
        if (grad_mv != NULL) {
            float sgn_MV = sign_f(mv_a[idx]);
            float l1_reg_term_MV = lambda_l1 * sgn_MV;
            float l2_reg_term_MV = 2.0f * lambda_l2 * mv_a[idx];
            mv_a[idx] -= learning_rate * (grad_mv[idx] + l1_reg_term_MV + l2_reg_term_MV);
        }

        // MQ update with Elastic Net
        if (grad_mq != NULL) {
            float sgn_MQ = sign_f(mq_a[idx]);
            float l1_reg_term_MQ = lambda_l1 * sgn_MQ;
            float l2_reg_term_MQ = 2.0f * lambda_l2 * mq_a[idx];
            mq_a[idx] -= learning_rate * (grad_mq[idx] + l1_reg_term_MQ + l2_reg_term_MQ);
        }

        // MK update with Elastic Net
        if (grad_mk != NULL) {
            float sgn_MK = sign_f(mk_a[idx]);
            float l1_reg_term_MK = lambda_l1 * sgn_MK;
            float l2_reg_term_MK = 2.0f * lambda_l2 * mk_a[idx];
            mk_a[idx] -= learning_rate * (grad_mk[idx] + l1_reg_term_MK + l2_reg_term_MK);
        }
    }
    // EH and EV updates (not weight matrices, so no Elastic Net here)
    if (update_eh != 0 && idx < embedding_dim) {
        if(grad_eh != NULL) eh[idx] -= learning_rate * grad_eh[idx];
    }
    if (update_ev != 0) {
        if (idx < ev_size) {
            int embed_idx = idx % embedding_dim;
            if(grad_ev_scaled != NULL) ev[idx] -= learning_rate * grad_ev_scaled[embed_idx];
        }
    }
}

__kernel void kernelUpdateWeights_1stHead_H_ElasticNet(__global float* mh_a, __global float* mv_a, __global float* mq_a, __global float* mk_a,
                                            __global float* eh,
                                            __global const float* grad_mh, __global const float* grad_mv,
                                            __global const float* grad_mq, __global const float* grad_mk,
                                            __global const float* grad_eh,
                                            float learning_rate, int update_eh,
                                            float lambda_l1, float lambda_l2, // ADDED: Elastic Net regularization parameters
                                            int mat_heights, int embedding_dim)
{
    int idx = get_global_id(0);
    int matrix_size = mat_heights * embedding_dim;

    if (idx < matrix_size) {
        // MH update with Elastic Net
        if(grad_mh != NULL) {
            float sgn_MH = sign_f(mh_a[idx]);
            float l1_reg_term_MH = lambda_l1 * sgn_MH;
            float l2_reg_term_MH = 2.0f * lambda_l2 * mh_a[idx];
            mh_a[idx] -= learning_rate * (grad_mh[idx] + l1_reg_term_MH + l2_reg_term_MH);
        }
        // MV update with Elastic Net
        if(grad_mv != NULL) {
            float sgn_MV = sign_f(mv_a[idx]);
            float l1_reg_term_MV = lambda_l1 * sgn_MV;
            float l2_reg_term_MV = 2.0f * lambda_l2 * mv_a[idx];
            mv_a[idx] -= learning_rate * (grad_mv[idx] + l1_reg_term_MV + l2_reg_term_MV);
        }
        // MQ update with Elastic Net
        if(grad_mq != NULL) {
            float sgn_MQ = sign_f(mq_a[idx]);
            float l1_reg_term_MQ = lambda_l1 * sgn_MQ;
            float l2_reg_term_MQ = 2.0f * lambda_l2 * mq_a[idx];
            mq_a[idx] -= learning_rate * (grad_mq[idx] + l1_reg_term_MQ + l2_reg_term_MQ);
        }
        // MK update with Elastic Net
        if(grad_mk != NULL) {
            float sgn_MK = sign_f(mk_a[idx]);
            float l1_reg_term_MK = lambda_l1 * sgn_MK;
            float l2_reg_term_MK = 2.0f * lambda_l2 * mk_a[idx];
            mk_a[idx] -= learning_rate * (grad_mk[idx] + l1_reg_term_MK + l2_reg_term_MK);
        }
    }
    // EH update (not a weight matrix, no Elastic Net here)
    if (update_eh != 0 && idx < embedding_dim) {
        if(grad_eh != NULL) eh[idx] -= learning_rate * grad_eh[idx];
    }
}

__kernel void kernelUpdateWeights_1stHead_V_ElasticNet(__global float* mv_a, __global float* mq_a, __global float* mk_a,
                                            __global const float* grad_mv, __global const float* grad_mq,
                                            __global const float* grad_mk_correction,
                                            float learning_rate,
                                            float lambda_l1, float lambda_l2, // ADDED: Elastic Net regularization parameters
                                            int mat_heights, int embedding_dim)
{
    int idx = get_global_id(0);
    int matrix_size = mat_heights * embedding_dim;

    if (idx < matrix_size) {
        // MV update with Elastic Net
        if(grad_mv != NULL) {
            float sgn_MV = sign_f(mv_a[idx]);
            float l1_reg_term_MV = lambda_l1 * sgn_MV;
            float l2_reg_term_MV = 2.0f * lambda_l2 * mv_a[idx];
            mv_a[idx] -= learning_rate * (grad_mv[idx] + l1_reg_term_MV + l2_reg_term_MV);
        }
        // MQ update with Elastic Net
        if(grad_mq != NULL) {
            float sgn_MQ = sign_f(mq_a[idx]);
            float l1_reg_term_MQ = lambda_l1 * sgn_MQ;
            float l2_reg_term_MQ = 2.0f * lambda_l2 * mq_a[idx];
            mq_a[idx] -= learning_rate * (grad_mq[idx] + l1_reg_term_MQ + l2_reg_term_MQ);
        }
        // MK update with Elastic Net
        if(grad_mk_correction != NULL) {
            float sgn_MK = sign_f(mk_a[idx]);
            float l1_reg_term_MK = lambda_l1 * sgn_MK;
            float l2_reg_term_MK = 2.0f * lambda_l2 * mk_a[idx];
            mk_a[idx] -= learning_rate * (grad_mk_correction[idx] + l1_reg_term_MK + l2_reg_term_MK);
        }
    }
}

__kernel void kernelUpdateWeights_EV_V_ElasticNet(__global float* mv_a, __global float* mq_a, __global float* mk_a, __global float* ev,
                                       __global const float* grad_mv, __global const float* grad_mq,
                                       __global const float* grad_mk_correction,
                                       __global const float* grad_ev_full,
                                       float learning_rate,
                                       float lambda_l1, float lambda_l2, // ADDED: Elastic Net regularization parameters
                                       int update_ev, int mat_heights, int embedding_dim, int context_win)
{
    int idx = get_global_id(0);
    int matrix_size = mat_heights * embedding_dim;

    if (idx < matrix_size) {
        // MV update with Elastic Net
        if(grad_mv != NULL) {
            float sgn_MV = sign_f(mv_a[idx]);
            float l1_reg_term_MV = lambda_l1 * sgn_MV;
            float l2_reg_term_MV = 2.0f * lambda_l2 * mv_a[idx];
            mv_a[idx] -= learning_rate * (grad_mv[idx] + l1_reg_term_MV + l2_reg_term_MV);
        }
        // MQ update with Elastic Net
        if(grad_mq != NULL) {
            float sgn_MQ = sign_f(mq_a[idx]);
            float l1_reg_term_MQ = lambda_l1 * sgn_MQ;
            float l2_reg_term_MQ = 2.0f * lambda_l2 * mq_a[idx];
            mq_a[idx] -= learning_rate * (grad_mq[idx] + l1_reg_term_MQ + l2_reg_term_MQ);
        }
        // MK update with Elastic Net
        if(grad_mk_correction != NULL) {
            float sgn_MK = sign_f(mk_a[idx]);
            float l1_reg_term_MK = lambda_l1 * sgn_MK;
            float l2_reg_term_MK = 2.0f * lambda_l2 * mk_a[idx];
            mk_a[idx] -= learning_rate * (grad_mk_correction[idx] + l1_reg_term_MK + l2_reg_term_MK);
        }
    }
    // EV update (not a weight matrix, no Elastic Net here)
    if (update_ev != 0) {
        int ev_size = context_win * embedding_dim;
        if (idx < ev_size) {
            if(grad_ev_full != NULL) ev[idx] -= learning_rate * grad_ev_full[idx];
        }
    }
}

__kernel void kernelUpdateWeights_1stHead_HV_ElasticNet(__global float* mh_a, __global float* mv_a, __global float* mq_a, __global float* mk_a,
                                             __global const float* grad_mh, __global const float* grad_mv,
                                             __global const float* grad_mq, __global const float* grad_mk,
                                             float learning_rate,
                                             float lambda_l1, float lambda_l2, // ADDED: Elastic Net regularization parameters
                                             int mat_heights, int embedding_dim)
{
    int idx = get_global_id(0);
    int matrix_size = mat_heights * embedding_dim;
    if (idx < matrix_size) {
        // MH update with Elastic Net
        if(grad_mh != NULL) {
            float sgn_MH = sign_f(mh_a[idx]);
            float l1_reg_term_MH = lambda_l1 * sgn_MH;
            float l2_reg_term_MH = 2.0f * lambda_l2 * mh_a[idx];
            mh_a[idx] -= learning_rate * (grad_mh[idx] + l1_reg_term_MH + l2_reg_term_MH);
        }
        // MV update with Elastic Net
        if(grad_mv != NULL) {
            float sgn_MV = sign_f(mv_a[idx]);
            float l1_reg_term_MV = lambda_l1 * sgn_MV;
            float l2_reg_term_MV = 2.0f * lambda_l2 * mv_a[idx];
            mv_a[idx] -= learning_rate * (grad_mv[idx] + l1_reg_term_MV + l2_reg_term_MV);
        }
        // MQ update with Elastic Net
        if(grad_mq != NULL) {
            float sgn_MQ = sign_f(mq_a[idx]);
            float l1_reg_term_MQ = lambda_l1 * sgn_MQ;
            float l2_reg_term_MQ = 2.0f * lambda_l2 * mq_a[idx];
            mq_a[idx] -= learning_rate * (grad_mq[idx] + l1_reg_term_MQ + l2_reg_term_MQ);
        }
        // MK update with Elastic Net
        if(grad_mk != NULL) {
            float sgn_MK = sign_f(mk_a[idx]);
            float l1_reg_term_MK = lambda_l1 * sgn_MK;
            float l2_reg_term_MK = 2.0f * lambda_l2 * mk_a[idx];
            mk_a[idx] -= learning_rate * (grad_mk[idx] + l1_reg_term_MK + l2_reg_term_MK);
        }
    }
}