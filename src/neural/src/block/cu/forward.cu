
#include "include/attention.hpp" // Provides attention class and cuforprop declarations
#include "include/block.hpp"     // Provides block class declaration
#include <cuda_runtime.h>        // Standard CUDA includes
#include <stdexcept>             // For std::out_of_range, std::runtime_error
#include <vector>                // For std::vector
#include <string>                // For std::to_string in error messages

/**
 * @brief CUDA forward propagation on single ith column of the FIRST block.
 *        Calls attention::cuforprop for each head in the specified column.
 * @param in dimension size (maps to d_embedding in attention::cuforprop)
 * @param tokenCount number of tokens (maps to currentTokenCount in attention::cuforprop)
 * @param i column index (0-based)
 * @param layers number of layers in mlp (maps to layers_mlp in attention::cuforprop)
 */
void block::cu1parallelForprop(int& in, int& tokenCount, int i, int& layers)
{
    if (i < 0 || i >= this->y) {
        throw std::out_of_range("cu1parallelForprop (first block): column index 'i' (" + std::to_string(i) + ") is out of range [0, " + std::to_string(this->y - 1) + "].");
    }
    for (int layer_idx = 0; layer_idx < this->x; ++layer_idx) {
        attention& head = this->b[layer_idx][i];
        try {
            head.cuforprop(in, layers, tokenCount);
        }
        catch (const std::exception& e) {
            // Add context to the exception before re-throwing
            throw std::runtime_error("Exception in attention::cuforprop (first block overload) for head ["
                                     + std::to_string(layer_idx) + "][" + std::to_string(i) + "]: " + e.what());
        }
    }
}


/**
 * @brief CUDA forward propagation on single ith column of a SUBSEQUENT block (blockCount > 0).
 *        Calls attention::cuforprop for each head in the specified column, passing the relevant EVp slice.
 * @param EVp vertical retention vectors from previous blocks (shape [layer][token][embedding] for this column).
 * @param in dimension size (maps to d_embedding in attention::cuforprop)
 * @param tokenCount number of tokens in full context (maps to totalTokenCount in attention::cuforprop)
 * @param blockCount position of block in full context (1-based, maps to blockIdx in attention::cuforprop)
 * @param i column index (0-based)
 * @param layers number of layers in mlp (maps to layers_mlp in attention::cuforprop)
 * @param n context window size (maps to contextWindowSize in attention::cuforprop)
 */
void block::cu1ParallelForprop(std::vector<std::vector<std::vector<float>>>& EVp, int& in, int& tokenCount, int& blockCount,
    int i, int& layers, int& n)
{
    if (i < 0 || i >= this->y) {
        throw std::out_of_range("cu1ParallelForprop (subsequent block): column index 'i' (" + std::to_string(i) + ") is out of range [0, " + std::to_string(this->y - 1) + "].");
    }

    if (EVp.size() != static_cast<size_t>(this->x)) {
         throw std::runtime_error("cu1ParallelForprop (subsequent block): EVp layer dimension mismatch. Expected "
                                  + std::to_string(this->x) + " layers, got " + std::to_string(EVp.size()) + ".");
    }

    for (int layer_idx = 0; layer_idx < this->x; ++layer_idx) {
        attention& head = this->b[layer_idx][i];
        std::vector<std::vector<float>>& EVp_layer = EVp[layer_idx];
        try {
            head.cuforprop(EVp_layer, in, layers, tokenCount, blockCount, n);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Exception in attention::cuforprop (subsequent block overload) for head ["
                                     + std::to_string(layer_idx) + "][" + std::to_string(i) + "]: " + e.what());
        }
    }
}


/**
 * @brief CUDA forward propagation on the FIRST block.
 *        Iterates through all columns (parallels) and calls cu1parallelForprop for each.
 * @param in dimension size (maps to d_embedding in attention::cuforprop)
 * @param tokenCount number of tokens (maps to currentTokenCount in attention::cuforprop)
 * @param layers number of layers in mlp (maps to layers_mlp in attention::cuforprop)
 */
void block::cuForprop(int& in, int& tokenCount, int& layers)
{
    for (int j = 0; j < this->y; ++j) {
        try {
            this->cu1parallelForprop(in, tokenCount, j, layers);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Exception in cu1parallelForprop (first block) for column ["
                                     + std::to_string(j) + "]: " + e.what());
        }
    }
}


/**
 * @brief CUDA forward propagation of a SUBSEQUENT block (blockCount > 0).
 *        Iterates through all columns (parallels) and calls cu1ParallelForprop for each,
 *        passing the relevant EVp slice for that column.
 * @param EVp vertical retention vectors from previous blocks (shape [x][y][token][embedding]).
 * @param in dimension size (maps to d_embedding in attention::cuforprop)
 * @param tokenCount number of tokens in full context (maps to totalTokenCount in attention::cuforprop)
 * @param blockCount position of block in full context (1-based, maps to blockIdx in attention::cuforprop)
 * @param layers number of layers in mlp (maps to layers_mlp in attention::cuforprop)
 * @param n context window size (maps to contextWindowSize in attention::cuforprop)
 */
void block::cuForprop(std::vector<std::vector<std::vector<std::vector<float>>>>& EVp, int& in, int& tokenCount,
    int& blockCount, int& layers, int& n)
{
    if (EVp.size() != static_cast<size_t>(this->x)) {
        throw std::runtime_error("cuForprop (subsequent block): EVp layer dimension mismatch. Expected "
                                 + std::to_string(this->x) + " layers, got " + std::to_string(EVp.size()) + ".");
    }
    if (!EVp.empty() && EVp[0].size() != static_cast<size_t>(this->y)) {
        throw std::runtime_error("cuForprop (subsequent block): EVp column dimension mismatch. Expected "
                                 + std::to_string(this->y) + " columns, got " + std::to_string(EVp[0].size()) + ".");
    }

    for (int j = 0; j < this->y; ++j) {
        std::vector<std::vector<std::vector<float>>> EVp_col_j(this->x);
        for (int layer_idx = 0; layer_idx < this->x; ++layer_idx) {
            EVp_col_j[layer_idx] = EVp[layer_idx][j];
        }

        try {
            this->cu1ParallelForprop(EVp_col_j, in, tokenCount, blockCount, j, layers, n);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Exception in cu1ParallelForprop (subsequent block) for column ["
                                     + std::to_string(j) + "]: " + e.what());
        }
    }
}
