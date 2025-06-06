
#ifdef USE_OPENCL

#include "include/block.hpp"
#include <maths.hpp>
#include <vector>
#include <stdexcept>
#include <string>                // For std::to_string in error messages
#include <map>                   // For kernel map
#include <CL/cl.hpp>             // Or <CL/cl.h>

/**
 * @brief OpenCL backward propagation for the FIRST block, driven by a single horizontal error vector (EH).
 *        Applies the same expectedH to all columns. Iterates columns in reverse.
 * @param context OpenCL context.
 * @param queue OpenCL command queue.
 * @param kernels Map of compiled OpenCL kernels.
 * @param expectedH Expected horizontal embedding (common for all columns).
 * @param in Embedding dimension.
 * @param layers Number of MLP layers.
 */
void block::clBackward1stBlock(std::vector<float>& expectedH, int& in, int& layers)
{
    // Validate input size
    if (expectedH.size() != EMBEDDING) {
        throw std::runtime_error("clBackward1stBlock(vector<float>): ExpectedH vector size mismatch. Expected "
                                + std::to_string(EMBEDDING) + ", got " + std::to_string(expectedH.size()));
    }
    if (EMBEDDING != in) {
        throw std::runtime_error("Embedding dimension mismatch: EMBEDDING vs in");
    }
    // Iterate through all columns (parallels) in REVERSE order
    for (int j = this->y - 1; j >= 0; --j) { // j is the column index (layno)
        try {
            // Call the partial backward function for the current column j
            // This function handles the logic for the very first head vs others internally.
            if(j == this->y-1) {
                this->cl1ParallelBackward1stBlock(expectedH, in, layers, j);
            }
            else if(j > 0 && j < this->y-1) {
                //
            }
            else if(j == 0) {
                //
            }
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Exception in cl1ParallelBackward1stBlock(H) for column ["
                                    + std::to_string(j) + "]: " + e.what());
        }
    }
    // serialise(blockFilePath);
}


/**
 * @brief OpenCL backward propagation for the FIRST block, driven by multiple horizontal error vectors (EH).
 *        Applies expectedH[j] to column j. Iterates columns in reverse.
 * @param context OpenCL context.
 * @param queue OpenCL command queue.
 * @param kernels Map of compiled OpenCL kernels.
 * @param expectedH Vector of expected horizontal embeddings (one per column). Shape: [y][EMBEDDING].
 * @param in Embedding dimension.
 * @param layers Number of MLP layers.
 */
void block::clBackward1stBlock(std::vector<std::vector<float>>& expectedH, int& in, int& layers)
{
    std::cout << "First Block Backprop ";
    // Validate input size - should have one vector per column
    if (expectedH.size() != static_cast<size_t>(this->y)) {
        throw std::runtime_error("clBackward1stBlock(vector<vector<float>>): ExpectedH outer dimension mismatch. Expected "
                                + std::to_string(this->y) + " columns, got " + std::to_string(expectedH.size()));
    }
    if (!expectedH.empty() && expectedH[0].size() != EMBEDDING) {
        throw std::runtime_error("clBackward1stBlock(vector<vector<float>>): ExpectedH inner dimension mismatch. Expected "
                                + std::to_string(EMBEDDING) + ", got " + std::to_string(expectedH[0].size()));
    }
    if (EMBEDDING != in) {
        throw std::runtime_error("Embedding dimension mismatch: EMBEDDING vs in");
    }
    
    // Iterate through all columns (parallels) in REVERSE order
    for (int j = this->y - 1; j >= 0; --j) { // j is the column index (layno)
        try {
            // Call the partial backward function for the current column j, passing the specific expectedH[j]
            this->cl1ParallelBackward1stBlock(expectedH[j], in, layers, j);
            
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Exception in cl1ParallelBackward1stBlock(H) for column ["
                                    + std::to_string(j) + "]: " + e.what());
        }
    }
    // serialise(blockFilePath);
}


/**
 * @brief OpenCL backward propagation for the FIRST block, driven by vertical error vectors (EV).
 *        Iterates columns in reverse.
 * @param context OpenCL context.
 * @param queue OpenCL command queue.
 * @param kernels Map of compiled OpenCL kernels.
 * @param expectedV Expected vertical embeddings for all heads. Shape: [x][y][CONTEXT_WIN][EMBEDDING].
 * @param in Embedding dimension.
 * @param layers Number of MLP layers.
 */
void block::clBackward1stBlock(std::vector<std::vector<std::vector<std::vector<float>>>>& expectedV, int& in, int& layers)
{
    // Validate input dimensions
    if (expectedV.size() != static_cast<size_t>(this->x)) {
        throw std::runtime_error("clBackward1stBlock(V): ExpectedV outer dimension (rows) mismatch. Expected "
                                + std::to_string(this->x) + ", got " + std::to_string(expectedV.size()));
    }
    if (!expectedV.empty() && expectedV[0].size() != static_cast<size_t>(this->y)) {
        throw std::runtime_error("clBackward1stBlock(V): ExpectedV second dimension (columns) mismatch. Expected "
                                + std::to_string(this->y) + ", got " + std::to_string(expectedV[0].size()));
    }
    // Deeper validation happens within cl1ParallelBackward1stBlock
    if (EMBEDDING != in) {
        throw std::runtime_error("Embedding dimension mismatch: EMBEDDING vs in");
    }
    
    // Iterate through all columns (parallels) in REVERSE order
    for (int j = this->y - 1; j >= 0; --j) {
        // j is the column index (layno)
        // Prepare the expectedV slice for the current column j
        std::vector<std::vector<std::vector<float>>> expectedV_col_j(this->x);
        for (int i = 0; i < this->x; ++i) { // i is the row index
            // Ensure expectedV[i] has enough columns before accessing expectedV[i][j]
            if (expectedV[i].size() <= static_cast<size_t>(j)) {
                throw std::runtime_error("clBackward1stBlock(V): Column index " + std::to_string(j) +
                                        " out of bounds for expectedV row " + std::to_string(i) +
                                        " (size: " + std::to_string(expectedV[i].size()) + ")");
            }
            expectedV_col_j[i] = expectedV[i][j];
        }

        try {
            // Call the partial backward function for the current column j
            this->cl1ParallelBackward1stBlock(expectedV_col_j, in, layers, j);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Exception in cl1ParallelBackward1stBlock(V) for column ["
                                    + std::to_string(j) + "]: " + e.what());
        }
    }
    // serialise(blockFilePath);
}


/**
 * @brief OpenCL backward propagation for a NON-FIRST block, driven by a single horizontal error vector (EH).
 *        Applies the same expectedH to all columns. Iterates columns in reverse.
 * @param context OpenCL context.
 * @param queue OpenCL command queue.
 * @param kernels Map of compiled OpenCL kernels.
 * @param expectedH Expected horizontal embedding (common for all columns).
 * @param in Embedding dimension.
 * @param layers Number of MLP layers.
 */
void block::clBackward(std::vector<float>& expectedH, int& blockCount, int& in, int& layers)
{
    // Validate input size
    if (expectedH.size() != EMBEDDING) {
        throw std::runtime_error("clBackward(vector<float>): ExpectedH vector size mismatch. Expected "
                                + std::to_string(EMBEDDING) + ", got " + std::to_string(expectedH.size()));
    }
    if (EMBEDDING != in) {
        throw std::runtime_error("Embedding dimension mismatch: EMBEDDING vs in");
    }
    
    // Iterate through all columns (parallels) in REVERSE order
    for (int j = this->y - 1; j >= 0; --j) { // j is the column index (layno)
        try {
            // Call the partial backward function for non-first blocks
            this->cl1ParallelBackward(expectedH, in, layers, j);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Exception in cl1ParallelBackward(H) for column ["
                                    + std::to_string(j) + "]: " + e.what());
        }
    }
}


/**
 * @brief OpenCL backward propagation for a NON-FIRST block, driven by multiple horizontal error vectors (EH).
 *        Applies expectedH[j] to column j. Iterates columns in reverse.
 *        Matches signature: void block::clBackward(cl_context context, cl_command_queue queue, std::map<std::string, cl_kernel>& kernels, std::vector<std::vector<float>>& expectedH, int& in, int& layers);
 *
 * @param context OpenCL context.
 * @param queue OpenCL command queue.
 * @param kernels Map of compiled OpenCL kernels.
 * @param expectedH Vector of expected horizontal embeddings (one per column). Shape: [y][EMBEDDING].
 * @param in Embedding dimension.
 * @param layers Number of MLP layers.
 */
void block::clBackward(std::vector<std::vector<float>>& expectedH, int& blockCount, int& in, int& layers)
{
    // Validate input size - should have one vector per column
    if (expectedH.size() != static_cast<size_t>(this->y)) {
        throw std::runtime_error("clBackward(vector<vector<float>>): ExpectedH outer dimension mismatch. Expected "
                                + std::to_string(this->y) + " columns, got " + std::to_string(expectedH.size()));
    }
    if (!expectedH.empty() && expectedH[0].size() != EMBEDDING) {
        throw std::runtime_error("clBackward(vector<vector<float>>): ExpectedH inner dimension mismatch. Expected "
                                + std::to_string(EMBEDDING) + ", got " + std::to_string(expectedH[0].size()));
    }
    if (EMBEDDING != in) {
        throw std::runtime_error("Embedding dimension mismatch: EMBEDDING vs in");
    }
    
    // Iterate through all columns (parallels) in REVERSE order
    for (int j = this->y - 1; j >= 0; --j) { // j is the column index (layno)
        try {
            // Call the partial backward function for non-first blocks, passing the specific expectedH[j]
            this->cl1ParallelBackward(expectedH[j], in, layers, j);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Exception in cl1ParallelBackward(H) for column ["
                                    + std::to_string(j) + "]: " + e.what());
        }
    }
}


/**
 * @brief OpenCL backward propagation for a NON-FIRST block, driven by vertical error vectors (EV).
 *        Iterates columns in reverse.
 *        Matches signature: void block::clBackward(cl_context context, cl_command_queue queue, std::map<std::string, cl_kernel>& kernels, std::vector<std::vector<std::vector<std::vector<float>>>>& expectedV, int& in, int& layers);
 *
 * @param context OpenCL context.
 * @param queue OpenCL command queue.
 * @param kernels Map of compiled OpenCL kernels.
 * @param expectedV Expected vertical embeddings for all heads. Shape: [x][y][CONTEXT_WIN][EMBEDDING].
 * @param in Embedding dimension.
 * @param layers Number of MLP layers.
 */
void block::clBackward(std::vector<std::vector<std::vector<std::vector<float>>>>& expectedV, int& blockCount, int& in, int& layers)
{
    // Validate input dimensions
    if (expectedV.size() != static_cast<size_t>(this->x)) {
        throw std::runtime_error("clBackward(V): ExpectedV outer dimension (rows) mismatch. Expected "
                                + std::to_string(this->x) + ", got " + std::to_string(expectedV.size()));
    }
    if (!expectedV.empty() && expectedV[0].size() != static_cast<size_t>(this->y)) {
        throw std::runtime_error("clBackward(V): ExpectedV second dimension (columns) mismatch. Expected "
                                + std::to_string(this->y) + ", got " + std::to_string(expectedV[0].size()));
    }
    // Deeper validation happens within cl1ParallelBackward
    if (EMBEDDING != in) {
        throw std::runtime_error("Embedding dimension mismatch: EMBEDDING vs in");
    }
    
    // Iterate through all columns (parallels) in REVERSE order
    for (int j = this->y - 1; j >= 0; --j) { // j is the column index (layno)
        // Prepare the expectedV slice for the current column j
        std::vector<std::vector<std::vector<float>>> expectedV_col_j(this->x);
        for (int i = 0; i < this->x; ++i) { // i is the row index
            // Ensure expectedV[i] has enough columns before accessing expectedV[i][j]
            if (expectedV[i].size() <= static_cast<size_t>(j)) {
                throw std::runtime_error("clBackward(V): Column index " + std::to_string(j) +
                                        " out of bounds for expectedV row " + std::to_string(i) +
                                        " (size: " + std::to_string(expectedV[i].size()) + ")");
            }
            expectedV_col_j[i] = expectedV[i][j];
        }

        try {
            // Call the partial backward function for non-first blocks
            this->cl1ParallelBackward(expectedV_col_j, in, layers, j);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Exception in cl1ParallelBackward(V) for column ["
                                    + std::to_string(j) + "]: " + e.what());
        }
    }
}

#endif // USE_OPENCL
