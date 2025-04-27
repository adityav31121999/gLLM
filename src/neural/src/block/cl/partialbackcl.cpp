// Add this to a new file, e.g., block/cl/partialbackcl.cpp
#ifdef USE_OPENCL

#include "include/block.hpp"     // Provides block class declaration and attention.hpp
#include <vector>                // For std::vector
#include <stdexcept>             // For std::out_of_range, std::runtime_error
#include <string>                // For std::to_string in error messages
#include <map>                   // For kernel map
#include <CL/cl.hpp>             // Or <CL/cl.h>

/**
 * @brief OpenCL backward propagation for a single column in the FIRST block,
 *        driven by horizontal error (EH). Processes heads b[row][layno].
 *        Calls attention::clbackward1stHead for each head.
 *        Matches signature: void block::cl1ParallelBackward1stBlock(cl_context context, cl_command_queue queue, std::map<std::string, cl_kernel>& kernels, std::vector<float>& expectedH, int& in, int& layers, int layno);
 *
 * @param context OpenCL context.
 * @param queue OpenCL command queue.
 * @param kernels Map of compiled OpenCL kernels.
 * @param expectedH Expected horizontal embedding for the output of this column's heads.
 * @param in Embedding dimension.
 * @param layers Number of MLP layers.
 * @param layno The column index within the block (0 to y-1).
 */
void block::cl1ParallelBackward1stBlock(std::vector<float>& expectedH, int& in, int& layers, int layno)
{
    // Validate column number
    if (layno < 0 || layno >= y) {
        throw std::out_of_range("cl1ParallelBackward1stBlock(H): Column index 'layno' (" + std::to_string(layno) +
            ") is out of range [0, " + std::to_string(y - 1) + "].");
    }
    if (expectedH.size() != EMBEDDING) {
        throw std::runtime_error("ExpectedH vector size mismatch in cl1ParallelBackward1stBlock(H). Expected " +
            std::to_string(EMBEDDING) + ", got " + std::to_string(expectedH.size()));
    }
    if (EMBEDDING != in) {
        throw std::runtime_error("Embedding dimension mismatch: EMBEDDING vs in");
    }

    // Iterate backwards through the rows (parallels/layers) for the specified column
    // This matches the CUDA implementation's loop order for the H-driven case.
    for (int i = x - 1; i >= 0; --i) {
        attention& head_obj = b[i][layno]; // Reference to the current head object
        bool is_first_head = (i == 0 && layno == 0); // Check if it's the very first head

        try {
            if (is_first_head) {
                // Call the specific overload for the very first head (updates EH)
                head_obj.clbackward1stHead(context, queue, kernels, expectedH, in, layers);
            } else {
                // Call the overload for other heads in the first block (doesn't update EH)
                // This requires an overload in attention::clbackward1stHead that takes a dummy expectedV
                // or a specific overload designed for this case.
                // Assuming an overload exists like:
                // void attention::clbackward1stHead(cl_context, cl_command_queue, kernels, expectedH, dummyV, in, layers);
                // OR, if the logic is combined, we might need to adjust the call.
                // Let's assume the combined overload exists:
                // void attention::clbackward1stHead(cl_context, cl_command_queue, kernels, expectedH, expectedV, in, layers);
                // We need a dummy expectedV.
                std::vector<std::vector<float>> dummyExpectedV; // Empty vector
                head_obj.clbackward1stHead(context, queue, kernels, expectedH, dummyExpectedV, in, layers);
            }
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Exception during cl1ParallelBackward1stBlock(H) for head ["
                                     + std::to_string(i) + "][" + std::to_string(layno) + "]: " + e.what());
        }
    }
}


/**
 * @brief OpenCL backward propagation for a single column in the FIRST block,
 *        driven by vertical error (EV). Processes heads b[row][layno].
 *        Calls attention::clbackward1stHead for each head.
 *        Matches signature: void block::cl1ParallelBackward1stBlock(cl_context context, cl_command_queue queue, std::map<std::string, cl_kernel>& kernels, std::vector<std::vector<std::vector<float>>>& expectedV, int& in, int& layers, int layno);
 *
 * @param context OpenCL context.
 * @param queue OpenCL command queue.
 * @param kernels Map of compiled OpenCL kernels.
 * @param expectedV Expected vertical embeddings for all heads in this column. Shape: [x][CONTEXT_WIN][EMBEDDING].
 * @param in Embedding dimension.
 * @param layers Number of MLP layers.
 * @param layno The column index within the block (0 to y-1).
 */
void block::cl1ParallelBackward1stBlock(std::vector<std::vector<std::vector<float>>>& expectedV, int& in, int& layers, int layno)
{
    // Validate column number and input shape
    if (layno < 0 || layno >= y) {
        throw std::out_of_range("cl1ParallelBackward1stBlock(V): Column index 'layno' (" + std::to_string(layno) + ") is out of range [0, " + std::to_string(y - 1) + "].");
    }
    if (expectedV.size() != static_cast<size_t>(x)) {
        throw std::runtime_error("ExpectedV size mismatch (number of rows) in cl1ParallelBackward1stBlock(V). Expected " + std::to_string(x) + ", got " + std::to_string(expectedV.size()));
    }
    // Further dimension checks can be added if needed, but attention::clbackward1stHead should also validate.
    if (EMBEDDING != in) {
        throw std::runtime_error("Embedding dimension mismatch: EMBEDDING vs in");
    }

    // Iterate through the rows (parallels/layers) in the specified column
    // This matches the CUDA implementation's loop order for the V-driven case.
    for (int i = 0; i < x; ++i) {
        attention& head_obj = b[i][layno]; // Reference to the current head object
        std::vector<std::vector<float>>& expectedV_head = expectedV[i]; // Expected output for this head

        try {
            // Call the overload of clbackward1stHead that takes only expectedV
            head_obj.clbackward1stHead(context, queue, kernels, expectedV_head, in, layers);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Exception during cl1ParallelBackward1stBlock(V) for head ["
                                     + std::to_string(i) + "][" + std::to_string(layno) + "]: " + e.what());
        }
    }
}


/**
 * @brief OpenCL backward propagation for a single column in a NON-FIRST block,
 *        driven by horizontal error (EH). Processes heads b[row][layno].
 *        Calls attention::clbackward for each head.
 *        Matches signature: void block::cl1ParallelBackward(cl_context context, cl_command_queue queue, std::map<std::string, cl_kernel>& kernels, std::vector<float>& expectedH, int& in, int& layers, int layno);
 *
 * @param context OpenCL context.
 * @param queue OpenCL command queue.
 * @param kernels Map of compiled OpenCL kernels.
 * @param expectedH Expected horizontal embedding for the output of this column's heads.
 * @param in Embedding dimension.
 * @param layers Number of MLP layers.
 * @param layno The column index within the block (0 to y-1).
 */
void block::cl1ParallelBackward(std::vector<float>& expectedH, int& in, int& layers, int layno)
{
    // Validate column number
    if (layno < 0 || layno >= y) {
        throw std::out_of_range("cl1ParallelBackward(H): Column index 'layno' (" + std::to_string(layno) + ") is out of range [0, " + std::to_string(y - 1) + "].");
    }
    if (expectedH.size() != EMBEDDING) {
        throw std::runtime_error("ExpectedH vector size mismatch in cl1ParallelBackward(H). Expected " +
            std::to_string(EMBEDDING) + ", got " + std::to_string(expectedH.size()));
    }
    if (EMBEDDING != in) {
        throw std::runtime_error("Embedding dimension mismatch: EMBEDDING vs in");
    }

    // Iterate backwards through the rows (parallels/layers) for the specified column
    // Matches CUDA implementation loop order.
    for (int i = x - 1; i >= 0; --i) {
        attention& head_obj = b[i][layno]; // Reference to the current head object

        try {
            // Call the general backward function for non-first-block heads (updates EH and EV)
            head_obj.clbackward(context, queue, kernels, expectedH, in, layers);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Exception during cl1ParallelBackward(H) for head ["
                                     + std::to_string(i) + "][" + std::to_string(layno) + "]: " + e.what());
        }
    }
}


/**
 * @brief OpenCL backward propagation for a single column in a NON-FIRST block,
 *        driven by vertical error (EV). Processes heads b[row][layno].
 *        Calls attention::clbackward for each head.
 *        Matches signature: void block::cl1ParallelBackward(cl_context context, cl_command_queue queue, std::map<std::string, cl_kernel>& kernels, std::vector<std::vector<std::vector<float>>>& expectedV, int& in, int& layers, int layno);
 *
 * @param context OpenCL context.
 * @param queue OpenCL command queue.
 * @param kernels Map of compiled OpenCL kernels.
 * @param expectedV Expected vertical embeddings for all heads in this column. Shape: [x][CONTEXT_WIN][EMBEDDING].
 * @param in Embedding dimension.
 * @param layers Number of MLP layers.
 * @param layno The column index within the block (0 to y-1).
 */
void block::cl1ParallelBackward(std::vector<std::vector<std::vector<float>>>& expectedV, int& in, int& layers, int layno)
{
    // Validate column number and input shape
    if (layno < 0 || layno >= y) {
        throw std::out_of_range("cl1ParallelBackward(V): Column index 'layno' (" + std::to_string(layno) + ") is out of range [0, " + std::to_string(y - 1) + "].");
    }
    if (expectedV.size() != static_cast<size_t>(x)) {
        throw std::runtime_error("ExpectedV size mismatch (number of rows) in cl1ParallelBackward(V). Expected " + std::to_string(x) + ", got " + std::to_string(expectedV.size()));
    }
    // Further dimension checks can be added if needed, but attention::clbackward should also validate.
    if (EMBEDDING != in) {
        throw std::runtime_error("Embedding dimension mismatch: EMBEDDING vs in");
    }

    // Iterate through the rows (parallels/layers) in the specified column
    // Matches CUDA implementation loop order.
    for (int i = 0; i < x; ++i) {
        attention& head_obj = b[i][layno]; // Reference to the current head object
        std::vector<std::vector<float>>& expectedV_head = expectedV[i]; // Expected output for this head

        try {
            // Call the general backward function for non-first-block heads (updates MV, MQ, MK_corr, EV)
            head_obj.clbackward(context, queue, kernels, expectedV_head, in, layers);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Exception during cl1ParallelBackward(V) for head ["
                                     + std::to_string(i) + "][" + std::to_string(layno) + "]: " + e.what());
        }
    }
}

#endif // USE_OPENCL
