
/**
 * @file CLass BLOCK for complete attention using 2d array of attention class. This
 * helps in maintining continuity for context in next blcok keeps producing tokens
 * till context limit for each block is reached. After all blocks are processed, the 
 * last of EVs are then used to continue the context for next iteration on command.
 *                                                                Complete Attention
 *                                                                        |
 * ----------------------------------------------------------------------\/------------
 * (Attention Head - Attention Head ----- - Attention Head -> E') --> Partial attention
 * (Attention Head - Attention Head ----- - Attention Head -> E') --> Partial attention
 * (Attention Head - Attention Head ----- - Attention Head -> E') --> Partial attention
 *      |                   |                   |             |             |
 *      |                   |                   |             |             |
 *      |                   |                   |             |             |
 * (Attention Head - Attention Head ----- - Attention Head -> E') <-- Partial attention
 * -----/\-----------------------------------------------------------------------------
 *      |
 *  Parallel
 */

#ifndef BLOCK_HPP
#define BLOCK_HPP 1

#include <vector>
#include <maths.hpp>
#include <map>
#include "mlp.hpp"
#include "attention.hpp"


/**
 * @brief block for complete attention (local context)
 */
class block {
public:
    int x, y;               // x layers with y heads in each layer
    float error;            // error for block, mean of all Attention Heads
    bool isSelfAttention;   // if its self (1) or cross (0) attention
    bool inTraining;        // = 1 for training, = 0 for in use
    std::string str;        // to check whether new token is "@#O" or part of conversation
    std::vector<float> EH;  // common horizontal retention for token prediction (summed or concatanted)
    // hold all vertical retention vector EVs from each attention heads
    std::vector<std::vector<std::vector<std::vector<float>>>> EV;
    std::vector<std::vector<float>> tokForBlock;        // tokens for block
    std::vector<std::vector<attention>> b;      // block complete attention

    // default constructor
    block() = default;
    block(int x, int y, int n, int d, int h, int l, int vocab);
    block(int x, int y, int n, int d, int h, int l, int vocab, bool attentionType);
    block(int x, int y, int n, int d, int h, int l, int vocab, bool attentionType, bool inTraining);

    // KdotQ for parallels
    void parallelKdotQ(int& columnNumber, int& tokenCount);
    // partial attention forprop
    void partialforprop(int& in, int& tokenCount, int i, int& layers);
    void partialforprop(std::vector<std::vector<std::vector<float>>>& EVp, int& in, int& tokenCount, int& blockCount, int i, int& layers, int& n);
    // parallel partialforprop(i)
    void forprop(int& in, int& tokenCount, int& layers);
    void forprop(std::vector<std::vector<std::vector<std::vector<float>>>>& EVp, int& in, int& tokenCount, int& blockCount, int& layers, int& n);
    // set retention vectors for vertical pass
    void setVerticalRetention(std::vector<std::vector<std::vector<std::vector<float>>>>& EV);
    // partial attention backward
    void partialbackward1stBlock(std::vector<float>& expectedH, int& in, int& layers, int layno);
    void partialbackward1stBlock(std::vector<std::vector<std::vector<float>>>& expectedV, int& in, int& layers, int layno);
    void partialbackward(std::vector<float>& expectedH, int& in, int& layers, int layno);
    void partialbackward(std::vector<std::vector<std::vector<float>>>& expectedV, int& in, int& layers, int layno);
    // parallel partialbackward(i)
    void backward1stBlock(std::vector<float>& expectedH, int& in, int& layers);
    void backward1stBlock(std::vector<std::vector<float>>& expectedH, int& in, int& layers);
    void backward1stBlock(std::vector<std::vector<std::vector<std::vector<float>>>>& expectedV, int& in, int& layers);
    void backward(std::vector<float>& expectedH, int& in, int& layers);
    void backward(std::vector<std::vector<float>>& expectedH, int& in, int& layers);
    void backward(std::vector<std::vector<std::vector<std::vector<float>>>>& expectedV, int& in, int& layers);

    #ifdef USE_CUDA
    // cuda equivalent functions for block (for parallels in loop)
        void cuParallelKdotQ(int& columnNumber, int& blockNumber, int& promptCount, int& tokenCount, bool isSelfAttention);
        void cuParallelUseKdotQ(const std::vector<std::vector<float>>& tokenEmbed, int& columnNumber, int& tokenCount, int& promptCount, bool isSelfAttention);
        void cuParallelUseKdotQ(const std::vector<std::vector<std::vector<float>>>& EVp, int& columnNumber, int& blockNumber, int& tokenCount, int& promptCount, bool isSelfAttention);
        // for single parallel
        void cu1parallelForprop(int& in, int& tokenCount, int i, int& layers);
        void cu1ParallelForprop(std::vector<std::vector<std::vector<float>>>& EVp, int& in, int& tokenCount, int& blockCount, int i, int& layers, int& n);
        // for complete block
        void cuForprop(int& in, int& tokenCount, int& layers);
        void cuForprop(std::vector<std::vector<std::vector<std::vector<float>>>>& EVp, int& in, int& tokenCount, int& blockCount, int& layers, int& n);
        // for single parallel
        void cu1ParallelBackward1stBlock(std::vector<float>& expectedH, int& in, int& layers, int layno);
        void cu1ParallelBackward1stBlock(std::vector<std::vector<std::vector<float>>>& expectedV, int& in, int& layers, int layno);
        void cu1ParallelBackward(std::vector<float>& expectedH, int& in, int& layers, int layno);
        void cu1ParallelBackward(std::vector<std::vector<std::vector<float>>>& expectedV, int& in, int& layers, int layno);
        // for complete block
        void cuBackward1stBlock(std::vector<float>& expectedH, int& in, int& layers);
        void cuBackward1stBlock(std::vector<std::vector<float>>& expectedH, int& in, int& layers);
        void cuBackward1stBlock(std::vector<std::vector<std::vector<std::vector<float>>>>& expectedV, int& in, int& layers);
        void cuBackward(std::vector<float>& expectedH, int& in, int& layers);
        void cuBackward(std::vector<std::vector<float>>& expectedH, int& in, int& layers);
        void cuBackward(std::vector<std::vector<std::vector<std::vector<float>>>>& expectedV, int& in, int& layers);
    #elif USE_OPENCL
    #include <CL/cl.hpp>
        cl::Context context;
        cl::CommandQueue queue;
        cl::Program program; // Holds the compiled program from clcompute.cl and others
        std::map<std::string, cl::Kernel> kernels; // Map to store kernel objects by name
        cl::Device default_device; // Store the device being used
        // opencl equivalent functions for block (for parallels in loop)
        void clParallelKdotQ(int& columnNumber, int& blockNumber, int& promptCount, int& tokenCount, bool isSelfAttention);
        void clParallelUseKdotQ(const std::vector<std::vector<float>>& tokenEmbed, int& columnNumber, int& tokenCount, int& promptCount, bool isSelfAttention);
        void clParallelUseKdotQ(const std::vector<std::vector<std::vector<float>>>& EVp, int& columnNumber, int& blockNumber, int& tokenCount, int& promptCount, bool isSelfAttention);
        // for single parallel
        void cl1parallelForprop(int& in, int& tokenCount, int i, int& layers);
        void cl1ParallelForprop(std::vector<std::vector<std::vector<float>>>& EVp, int& in, int& tokenCount, int& blockCount, int i, int& layers, int& n);
        // for complete block
        void clForprop(int& in, int& tokenCount, int& layers);
        void clForprop(std::vector<std::vector<std::vector<std::vector<float>>>>& EVp, int& in, int& tokenCount, int& blockCount, int& layers, int& n);
        // for single parallel
        void cl1ParallelBackward1stBlock(std::vector<float>& expectedH, int& in, int& layers, int layno);
        void cl1ParallelBackward1stBlock(std::vector<std::vector<std::vector<float>>>& expectedV, int& in, int& layers, int layno);
        void cl1ParallelBackward(std::vector<float>& expectedH, int& in, int& layers, int layno);
        void cl1ParallelBackward(std::vector<std::vector<std::vector<float>>>& expectedV, int& in, int& layers, int layno);
        // for complete block
        void clBackward1stBlock(std::vector<float>& expectedH, int& in, int& layers);
        void clBackward1stBlock(std::vector<std::vector<float>>& expectedH, int& in, int& layers);
        void clBackward1stBlock(std::vector<std::vector<std::vector<std::vector<float>>>>& expectedV, int& in, int& layers);
        void clBackward(std::vector<float>& expectedH, int& in, int& layers);
        void clBackward(std::vector<std::vector<float>>& expectedH, int& in, int& layers);
        void clBackward(std::vector<std::vector<std::vector<std::vector<float>>>>& expectedV, int& in, int& layers);
        cl_mem cl_create_buffer(cl_context context, cl_mem_flags flags, size_t size, void* host_ptr, cl_int& err);
        void cl_write_buffer(cl_command_queue queue, cl_mem buffer, size_t size, const void* ptr, cl_bool blocking = CL_TRUE);
        void cl_read_buffer(cl_command_queue queue, cl_mem buffer, size_t size, void* ptr, cl_bool blocking = CL_TRUE);
        void cl_fill_buffer(cl_command_queue queue, cl_mem buffer, const void* pattern, size_t pattern_size, size_t offset, size_t size);
        void cl_set_kernel_arg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void* arg_value);
        void cl_enqueue_nd_range_kernel(cl_command_queue queue, cl_kernel kernel, cl_uint work_dim, const size_t* global_work_offset, const size_t* global_work_size, const size_t* local_work_size);
        void cl_finish(cl_command_queue queue);
        void cl_release_mem_object(cl_mem memobj);
        #define CL_CHECK(err) CL_CHECK(err, __FILE__, __LINE__)
    #endif

    // default destructor
    ~block() = default;
};

#endif
