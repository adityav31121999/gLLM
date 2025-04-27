// Add this to a new file, e.g., block/cl/forwardcl.cpp
#ifdef USE_OPENCL

#include "include/block.hpp"     // Provides block class declaration and attention.hpp
#include <vector>                // For std::vector
#include <stdexcept>             // For std::out_of_range, std::runtime_error
#include <string>                // For std::to_string in error messages
#include <map>                   // For kernel map
#include <CL/cl.hpp>             // Or <CL/cl.h>


/**
 * @brief OpenCL forward propagation on single ith column of the FIRST block.
 *        Calls attention::clforprop for each head in the specified column.
 *        Matches signature: void block::cl1parallelForprop(cl_context context, cl_command_queue queue, std::map<std::string, cl_kernel>& kernels, int& in, int& tokenCount, int i, int& layers);
 *
 * @param context OpenCL context.
 * @param queue OpenCL command queue.
 * @param kernels Map of compiled OpenCL kernels.
 * @param in dimension size (maps to d_embedding in attention::clforprop)
 * @param tokenCount number of tokens (maps to currentTokenCount in attention::clforprop)
 * @param i column index (0-based)
 * @param layers number of layers in mlp (maps to layers_mlp in attention::clforprop)
 */
void block::cl1parallelForprop(int& in, int& tokenCount, int i, int& layers)
{
    if (i < 0 || i >= this->y) {
        throw std::out_of_range("cl1parallelForprop (first block): column index 'i' (" + std::to_string(i) + ") is out of range [0, " + std::to_string(this->y - 1) + "].");
    }
    // Iterate through the layers (rows) of attention heads in the specified column 'i'
    for (int layer_idx = 0; layer_idx < this->x; ++layer_idx) {
        attention& head = this->b[layer_idx][i];
        try {
            // Call the first overload of attention::clforprop (for first block)
            head.clforprop(context, queue, kernels, in, layers, tokenCount);
        }
        catch (const std::exception& e) {
            // Add context to the exception before re-throwing
            throw std::runtime_error("Exception in attention::clforprop (first block overload) for head ["
                                     + std::to_string(layer_idx) + "][" + std::to_string(i) + "]: " + e.what());
        }
    }
}


/**
 * @brief OpenCL forward propagation on single ith column of a SUBSEQUENT block (blockCount > 0).
 *        Calls attention::clforprop for each head in the specified column, passing the relevant EVp slice.
 *        Matches signature: void block::cl1ParallelForprop(cl_context context, cl_command_queue queue, std::map<std::string, cl_kernel>& kernels, std::vector<std::vector<std::vector<float>>>& EVp, int& in, int& tokenCount, int& blockCount, int i, int& layers, int& n);
 *
 * @param context OpenCL context.
 * @param queue OpenCL command queue.
 * @param kernels Map of compiled OpenCL kernels.
 * @param EVp vertical retention vectors from previous block for THIS COLUMN (shape [layer][token][embedding]).
 * @param in dimension size (maps to d_embedding in attention::clforprop)
 * @param tokenCount number of tokens in full context (maps to totalTokenCount in attention::clforprop)
 * @param blockCount position of block in full context (1-based, maps to blockIdx in attention::clforprop)
 * @param i column index (0-based)
 * @param layers number of layers in mlp (maps to layers_mlp in attention::clforprop)
 * @param n context window size (maps to contextWindowSize in attention::clforprop)
 */
void block::cl1ParallelForprop(std::vector<std::vector<std::vector<float>>>& EVp, int& in, int& tokenCount, int& blockCount,
                               int i, int& layers, int& n)
{
    if (i < 0 || i >= this->y) {
        throw std::out_of_range("cl1ParallelForprop (subsequent block): column index 'i' (" + std::to_string(i) + ") is out of range [0, " + std::to_string(this->y - 1) + "].");
    }

    // Validate the incoming EVp for this column
    if (EVp.size() != static_cast<size_t>(this->x)) {
         throw std::runtime_error("cl1ParallelForprop (subsequent block): EVp layer dimension mismatch for column " + std::to_string(i)
                                  + ". Expected " + std::to_string(this->x) + " layers, got " + std::to_string(EVp.size()) + ".");
    }

    // Iterate through the layers (rows) of attention heads in the specified column 'i'
    for (int layer_idx = 0; layer_idx < this->x; ++layer_idx) {
        attention& head = this->b[layer_idx][i];
        // EVp[layer_idx] contains the [token][embedding] data for this specific head from the previous block
        std::vector<std::vector<float>>& EVp_layer = EVp[layer_idx];
        try {
            // Call the second overload of attention::clforprop (for subsequent blocks)
            head.clforprop(context, queue, kernels, EVp_layer, in, layers, tokenCount, blockCount, n);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Exception in attention::clforprop (subsequent block overload) for head ["
                                     + std::to_string(layer_idx) + "][" + std::to_string(i) + "]: " + e.what());
        }
    }
}


/**
 * @brief OpenCL forward propagation on the FIRST block.
 *        Iterates through all columns (parallels) and calls cl1parallelForprop for each.
 *        Matches signature: void block::clForprop(cl_context context, cl_command_queue queue, std::map<std::string, cl_kernel>& kernels, int& in, int& tokenCount, int& layers);
 *
 * @param context OpenCL context.
 * @param queue OpenCL command queue.
 * @param kernels Map of compiled OpenCL kernels.
 * @param in dimension size (maps to d_embedding in attention::clforprop)
 * @param tokenCount number of tokens (maps to currentTokenCount in attention::clforprop)
 * @param layers number of layers in mlp (maps to layers_mlp in attention::clforprop)
 */
void block::clForprop(int& in, int& tokenCount, int& layers)
{
    // Iterate through all columns (parallels)
    for (int j = 0; j < this->y; ++j) {
        try {
            // Call the first block version for the column j
            this->cl1parallelForprop(context, queue, kernels, in, tokenCount, j, layers);
        }
        catch (const std::exception& e) {
            // Add context to the exception before re-throwing
            throw std::runtime_error("Exception in cl1parallelForprop (first block) for column ["
                                     + std::to_string(j) + "]: " + e.what());
        }
    }
}


/**
 * @brief OpenCL forward propagation of a SUBSEQUENT block (blockCount > 0).
 *        Iterates through all columns (parallels) and calls cl1ParallelForprop for each,
 *        passing the relevant EVp slice for that column.
 *        Matches signature: void block::clForprop(cl_context context, cl_command_queue queue, std::map<std::string, cl_kernel>& kernels, std::vector<std::vector<std::vector<std::vector<float>>>>& EVp, int& in, int& tokenCount, int& blockCount, int& layers, int& n);
 *
 * @param context OpenCL context.
 * @param queue OpenCL command queue.
 * @param kernels Map of compiled OpenCL kernels.
 * @param EVp vertical retention vectors from previous blocks (shape [x][y][token][embedding]).
 * @param in dimension size (maps to d_embedding in attention::clforprop)
 * @param tokenCount number of tokens in full context (maps to totalTokenCount in attention::clforprop)
 * @param blockCount position of block in full context (1-based, maps to blockIdx in attention::clforprop)
 * @param layers number of layers in mlp (maps to layers_mlp in attention::clforprop)
 * @param n context window size (maps to contextWindowSize in attention::clforprop)
 */
void block::clForprop(std::vector<std::vector<std::vector<std::vector<float>>>>& EVp, int& in, int& tokenCount,
                      int& blockCount, int& layers, int& n)
{
    // Validate the overall structure of EVp
    if (EVp.size() != static_cast<size_t>(this->x)) {
        throw std::runtime_error("clForprop (subsequent block): EVp layer dimension mismatch. Expected "
                                 + std::to_string(this->x) + " layers, got " + std::to_string(EVp.size()) + ".");
    }
    if (!EVp.empty() && EVp[0].size() != static_cast<size_t>(this->y)) {
        throw std::runtime_error("clForprop (subsequent block): EVp column dimension mismatch. Expected "
                                 + std::to_string(this->y) + " columns, got " + std::to_string(EVp[0].size()) + ".");
    }

    // Iterate through all columns (parallels)
    for (int j = 0; j < this->y; ++j) {
        // Create the slice of EVp specific to this column j
        // EVp_col_j will have shape [x][token][embedding]
        std::vector<std::vector<std::vector<float>>> EVp_col_j(this->x);
        for (int layer_idx = 0; layer_idx < this->x; ++layer_idx) {
            EVp_col_j[layer_idx] = EVp[layer_idx][j];
        }

        try {
            // Call the subsequent block version for the column j, passing the column-specific EVp
            this->cl1ParallelForprop(context, queue, kernels, EVp_col_j, in, tokenCount, blockCount, j, layers, n);
        }
        catch (const std::exception& e) {
            // Add context to the exception before re-throwing
            throw std::runtime_error("Exception in cl1ParallelForprop (subsequent block) for column ["
                                     + std::to_string(j) + "]: " + e.what());
        }
    }
}

#endif // USE_OPENCL
