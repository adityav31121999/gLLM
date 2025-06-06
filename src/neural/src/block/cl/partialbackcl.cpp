
#ifdef USE_OPENCL

#include <maths.hpp>
#include "include/block.hpp"
#include <vector>
#include <stdexcept>
#include <string>
#include <map>
#include <CL/cl.hpp>

/**
 * @brief OpenCL backward propagation for a single column in the FIRST block,
 *        driven by horizontal error (EH). Processes heads b[row][layno].
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

    for (int i = 0; i < x; i++) {
        attention& head_obj = b[i][layno]; // Reference to the current head object
        head_obj.tokenCount = this->tokenCount; // Set token count for the head
        try {
            if (layno > 0) {
                head_obj.clbackward(expectedH, in, layers);
            } 
            else {
                head_obj.clbackward1stHead(expectedH, in, layers);
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
    for (int i = 0; i < x; ++i) {
        attention& head_obj = b[i][layno]; // Reference to the current head object
        head_obj.tokenCount = this->tokenCount;
        std::vector<std::vector<float>>& expectedV_head = expectedV[i]; // Expected output for this head

        try {
            // Call the overload of clbackward1stHead that takes only expectedV
            head_obj.clbackward1stHead(expectedV_head, in, layers);
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
 * @param expectedH Expected horizontal embedding for the output of this column's heads.
 * @param in Embedding dimension.
 * @param layers Number of MLP layers.
 * @param layno The column index within the block (0 to y-1).
 */
void block::cl1ParallelBackward(std::vector<float>& expectedH, int& in, int& layers, int layno)
{
    std::cout << "Done this" << std::endl;
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
    for (int i = x - 1; i >= 0; --i) {
        attention& head_obj = b[i][layno]; // Reference to the current head object
        head_obj.tokenCount = this->tokenCount;
        try {
            // Call the general backward function for non-first-block heads (updates EH and EV)
            head_obj.clbackward(expectedH, in, layers);
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
    for (int i = 0; i < x; ++i) {
        attention& head_obj = b[i][layno]; // Reference to the current head object
        head_obj.tokenCount = this->tokenCount;
        std::vector<std::vector<float>>& expectedV_head = expectedV[i]; // Expected output for this head

        try {
            // Call the general backward function for non-first-block heads (updates MV, MQ, MK_corr, EV)
            head_obj.clbackward(expectedV_head, in, layers);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Exception during cl1ParallelBackward(V) for head ["
                                     + std::to_string(i) + "][" + std::to_string(layno) + "]: " + e.what());
        }
    }
}

#endif // USE_OPENCL
