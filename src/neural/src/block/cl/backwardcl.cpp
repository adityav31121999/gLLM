#ifdef USE_OPENCL
#if defined(_WIN64) 
    #include <CL/cl.hpp>
#elif defined(__linux__)
    #include <CL/opencl.hpp>
#endif
#include <maths.hpp>
#include "include/mlp.hpp"
#include "include/attention.hpp"
#include "include/block.hpp"
#include <vector>
#include <stdexcept> // For runtime_error, out_of_range
#include <iostream> // For error logging
#include <string>   // For std::to_string in error messages


/**
 * @brief CUDA backward propagation for the FIRST block, driven by a single horizontal error vector (EH).
 *        Applies the same expectedH to all columns. Iterates columns in reverse.
 * @param expectedH Expected horizontal embedding (common for all columns).
 * @param in Embedding dimension.
 * @param layers Number of MLP layers.
 * @param learning Learning rate.
 * @param lambda_l1 L1 regularization parameter.
 * @param lambda_l2 L2 regularization parameter.
 * @param clip_norm Maximum L2 norm for gradient clipping (new parameter).
 */
void block::clbackward1stBlock(std::vector<float>& expectedH, int& in, int& layers, float& learning, float& lambda_l1, float& lambda_l2, float& clip_norm) {
    // Validate input size
    if (expectedH.size() != EMBEDDING) {
        throw std::runtime_error("cuBackward1stBlock(vector<float>): ExpectedH vector size mismatch. Expected "
                                + std::to_string(EMBEDDING) + ", got " + std::to_string(expectedH.size()));
    }
    // serialise(blockFilePath);
    // Iterate through all columns (parallels) in REVERSE order
    // The internal cu1ParallelBackward1stBlock handles the backward row iteration.
    for (int j = this->y - 1; j >= 0; j--) { // j is the column index (layno)
        try {
            for(int i = 0; i < x; i++) {
                b[i][j].tokenCount = this->tokenCount;
            }
            // Call the partial backward function for the current column j
            if(j == this->y-1) {
                // for last column
                clpartialbackward1stBlock(expectedH, in, layers, j, learning, lambda_l1, lambda_l2, clip_norm);
            }
            else if(j >= 0 && j < this->y-1) {
                // for columns inbetween
                // Note: The previous implementation used expectedH (single vector) for all columns.
                // If it needs to pass block EH to the next partial, this needs adjustment.
                // Assuming clpartialbackward1stBlock signature for vector<float> expectedH means it's for this single EH.
                clpartialbackward1stBlock(expectedH, in, layers, j, learning, lambda_l1, lambda_l2, clip_norm);
            }
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Exception in clbackward1stBlock(H) for column ["
                                    + std::to_string(j) + "]: " + e.what());
        }
    }
}


/**
 * @brief CUDA backward propagation for the FIRST block, driven by multiple horizontal error vectors (EH).
 *        Applies expectedH[j] to column j. Iterates columns in reverse.
 * @param expectedH Vector of expected horizontal embeddings (one per head per column if `clpartialbackward1stBlock` takes a 2D vector for EH).
 *                  If this `expectedH` is meant to be passed directly to `clpartialbackward1stBlock`'s `std::vector<std::vector<float>>` overload,
 *                  then its outer dimension should be `x` (number of heads), and it represents the error for the *entire column*.
 *                  The current context of the usage below suggests `expectedH` represents the EH for the *current column's heads*.
 * @param in Embedding dimension.
 * @param layers Number of MLP layers.
 * @param learning Learning rate.
 * @param lambda_l1 L1 regularization parameter.
 * @param lambda_l2 L2 regularization parameter.
 * @param clip_norm Maximum L2 norm for gradient clipping (new parameter).
 */
void block::clbackward1stBlock(std::vector<std::vector<float>>& expectedH, int& in, int& layers, float& learning, float& lambda_l1, float& lambda_l2, float& clip_norm) {
    // Validate input size - should have one vector per column
    // This overload's expectedH represents the error for ALL HEADS IN THE CURRENT COLUMN.
    // So expectedH.size() should be x (number of heads), not y (number of columns).
    // The calling context of this function usually provides `expectedH` as a slice for the current column.
    if (expectedH.size() != static_cast<size_t>(this->x)) { // Changed validation to x
        throw std::runtime_error("clbackward1stBlock(vector<vector<float>>): ExpectedH outer dimension mismatch. Expected "
                                + std::to_string(this->x) + " heads, got " + std::to_string(expectedH.size()));
    }
    if (!expectedH.empty() && expectedH[0].size() != EMBEDDING) {
        throw std::runtime_error("clbackward1stBlock(vector<vector<float>>): ExpectedH inner dimension mismatch. Expected "
                                + std::to_string(EMBEDDING) + ", got " + std::to_string(expectedH[0].size()));
    }
    // serialise(blockFilePath);
    // Iterate through all columns (parallels) in REVERSE order
    for (int j = this->y - 1; j >= 0; --j) { // j is the column index (layno)
        try {
            for(int i = 0; i < x; i++) {
                b[i][j].tokenCount = this->tokenCount;
            }
            // Call the partial backward function for the current column j
            if(j == this->y-1) {
                // for last column, use the provided expectedH
                clpartialbackward1stBlock(expectedH, in, layers, j, learning, lambda_l1, lambda_l2, clip_norm);
            }
            else if(j >= 0 && j < this->y-1) {
                // for columns inbetween, compute expectedH from the next column's EH
                std::vector<std::vector<float>> exp2h(x, std::vector<float>(EMBEDDING, 0.0));
                for(int i = 0; i < this->x; i++) {
                    exp2h[i] = b[i][j+1].EH;
                }
                clpartialbackward1stBlock(exp2h, in, layers, j, learning, lambda_l1, lambda_l2, clip_norm);
            }
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Exception in clbackward1stBlock(H_2D) for column ["
                                    + std::to_string(j) + "]: " + e.what());
        }
    }
}


/**
 * @brief CUDA backward propagation for a general block (not necessarily the first), driven by a single horizontal error vector (EH).
 *        Applies the same expectedH to all columns. Iterates columns in reverse.
 * @param expectedH Expected horizontal embedding (common for all columns).
 * @param in Embedding dimension.
 * @param layers Number of MLP layers.
 * @param blockCount Current block index (1-based).
 * @param learning Learning rate.
 * @param lambda_l1 L1 regularization parameter.
 * @param lambda_l2 L2 regularization parameter.
 * @param clip_norm Maximum L2 norm for gradient clipping (new parameter).
 */
void block::clbackward(std::vector<float>& expectedH, int& in, int& layers, int& blockCount, float& learning, float& lambda_l1, float& lambda_l2, float& clip_norm) {
    // Validate input size
    if (expectedH.size() != EMBEDDING) {
        throw std::runtime_error("clbackward(vector<float>): ExpectedH vector size mismatch. Expected "
                                + std::to_string(EMBEDDING) + ", got " + std::to_string(expectedH.size()));
    }
    // serialise(blockFilePath);
    // Iterate through all columns (parallels) in REVERSE order
    for (int j = this->y - 1; j >= 0; --j) { // j is the column index (layno)
        try {
            for(int i = 0; i < x; i++) {
                b[i][j].tokenCount = this->tokenCount;
            }
            // Call the partial backward function for the current column j
            if(j == this->y-1) {
                // for last column
                clpartialbackward(expectedH, in, layers, j, learning, lambda_l1, lambda_l2, clip_norm);
            }
            else if(j >= 0 && j < this->y-1) {
                // for columns inbetween
                // Note: The previous implementation used expectedH (single vector) for all columns.
                // If it needs to pass block EH to the next partial, this needs adjustment.
                // Assuming clpartialbackward signature for vector<float> expectedH means it's for this single EH.
                clpartialbackward(expectedH, in, layers, j, learning, lambda_l1, lambda_l2, clip_norm);
            }
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Exception in clbackward(H) for column ["
                                    + std::to_string(j) + "]: " + e.what());
        }
    }
}


/**
 * @brief CUDA backward propagation for a general block (not necessarily the first), driven by multiple horizontal error vectors (EH).
 *        Applies expectedH[j] to column j. Iterates columns in reverse.
 * @param expectedH Vector of expected horizontal embeddings (one per head per column if `clpartialbackward` takes a 2D vector for EH).
 *                  If this `expectedH` is meant to be passed directly to `clpartialbackward`'s `std::vector<std::vector<float>>` overload,
 *                  then its outer dimension should be `x` (number of heads), and it represents the error for the *entire column*.
 * @param in Embedding dimension.
 * @param layers Number of MLP layers.
 * @param blockCount Current block index (1-based).
 * @param learning Learning rate.
 * @param lambda_l1 L1 regularization parameter.
 * @param lambda_l2 L2 regularization parameter.
 * @param clip_norm Maximum L2 norm for gradient clipping (new parameter).
 */
void block::clbackward(std::vector<std::vector<float>>& expectedH, int& in, int& layers, int& blockCount, float& learning, float& lambda_l1, float& lambda_l2, float& clip_norm) {
    // Validate input size - should have one vector per column
    // This overload's expectedH represents the error for ALL HEADS IN THE CURRENT COLUMN.
    // So expectedH.size() should be x (number of heads), not y (number of columns).
    // The calling context of this function usually provides `expectedH` as a slice for the current column.
    if (expectedH.size() != static_cast<size_t>(this->x)) { // Changed validation to x
        throw std::runtime_error("clbackward(vector<vector<float>>): ExpectedH outer dimension mismatch. Expected "
                                + std::to_string(this->x) + " heads, got " + std::to_string(expectedH.size()));
    }
    if (!expectedH.empty() && expectedH[0].size() != EMBEDDING) {
        throw std::runtime_error("clbackward(vector<vector<float>>): ExpectedH inner dimension mismatch. Expected "
                                + std::to_string(EMBEDDING) + ", got " + std::to_string(expectedH[0].size()));
    }
    // serialise(blockFilePath);
    // Iterate through all columns (parallels) in REVERSE order
    for (int j = this->y - 1; j >= 0; --j) { // j is the column index (layno)
        try {
            for(int i = 0; i < x; i++) {
                b[i][j].tokenCount = this->tokenCount;
            }
            // Call the partial backward function for the current column j
            if(j == this->y-1) {
                // for last column, use the provided expectedH
                clpartialbackward(expectedH, in, layers, j, learning, lambda_l1, lambda_l2, clip_norm);
            }
            else if(j >= 0 && j < this->y-1) {
                // for columns inbetween, compute expectedH from the next column's EH
                std::vector<std::vector<float>> exp2h(x, std::vector<float>(EMBEDDING, 0.0));
                for(int i = 0; i < this->x; i++) {
                    exp2h[i] = b[i][j+1].EH;
                }
                clpartialbackward(exp2h, in, layers, j, learning, lambda_l1, lambda_l2, clip_norm);
            }
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Exception in clbackward(H_2D) for column ["
                                    + std::to_string(j) + "]: " + e.what());
        }
    }
}


/**
 * @brief CUDA backward propagation for the FIRST block, driven by vertical error vectors (EV).
 *        Iterates columns in reverse.
 * @param expectedV Expected vertical embeddings for all heads in all columns. Shape: [x][y][CONTEXT_WIN][EMBEDDING].
 * @param in Embedding dimension.
 * @param layers Number of MLP layers.
 * @param learning Learning rate.
 * @param lambda_l1 L1 regularization parameter.
 * @param lambda_l2 L2 regularization parameter.
 * @param clip_norm Maximum L2 norm for gradient clipping (new parameter).
 */
void block::clbackward1stBlock(std::vector<std::vector<std::vector<std::vector<float>>>>& expectedV, int& in, int& layers, float& learning, float& lambda_l1, float& lambda_l2, float& clip_norm) {
    // Validate input dimensions
    if (expectedV.size() != static_cast<size_t>(this->x)) {
        throw std::runtime_error("cuBackward1stBlock(V): ExpectedV outer dimension (rows) mismatch. Expected "
                                + std::to_string(this->x) + ", got " + std::to_string(expectedV.size()));
    }
    if (!expectedV.empty() && expectedV[0].size() != static_cast<size_t>(this->y)) {
        throw std::runtime_error("cuBackward1stBlock(V): ExpectedV second dimension (columns) mismatch. Expected "
                                + std::to_string(this->y) + ", got " + std::to_string(expectedV[0].size()));
    }
    // Deeper validation happens within clpartialbackward1stBlock
    // serialise(blockFilePath);
    // Iterate through all columns (parallels) in REVERSE order
    for (int j = this->y - 1; j >= 0; --j) {
        // j is the column index (layno)
        // Prepare the expectedV slice for the current column j
        std::vector<std::vector<std::vector<float>>> expectedV_col_j(this->x);

        try {
            for(int i = 0; i < x; i++) {
                b[i][j].tokenCount = this->tokenCount;
            }

            if(j == y - 1) {
                for (int i = 0; i < this->x; ++i) { // i is the row index
                    // Ensure expectedV[i] has enough columns before accessing expectedV[i][j]
                    if (expectedV[i].size() <= static_cast<size_t>(j)) {
                        throw std::runtime_error("cuBackward(V): Column index " + std::to_string(j) +
                                                " out of bounds for expectedV row " + std::to_string(i) +
                                                " (size: " + std::to_string(expectedV[i].size()) + ")");
                    }
                    expectedV_col_j[i] = expectedV[i][j];
                }
            }
            else if(j < y - 1 && j >= 0) {
                for (int i = 0; i < this->x; ++i) { // i is the row index
                    // Ensure expectedV[i] has enough columns before accessing expectedV[i][j]
                    if (expectedV[i].size() <= static_cast<size_t>(j)) { // This check might be redundant if the block structure guarantees existence
                        throw std::runtime_error("cuBackward(V): Column index " + std::to_string(j) +
                                                " out of bounds for expectedV row " + std::to_string(i) +
                                                " (size: " + std::to_string(expectedV[i].size()) + ")");
                    }
                    // This line is crucial for passing the EV from the next block
                    expectedV_col_j[i] = b[i][j+1].EV.make2dVector(b[i][j+1].EV, CONTEXT_WIN, EMBEDDING);
                }
            }

            // Call the partial backward function for the current column j
            clpartialbackward1stBlock(expectedV_col_j, in, layers, j, learning, lambda_l1, lambda_l2, clip_norm);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Exception in clbackward1stBlock(V) for column ["
                                    + std::to_string(j) + "]: " + e.what());
        }
    }
}


/**
 * @brief CUDA backward propagation for a general block (not necessarily the first), driven by vertical error vectors (EV).
 *        Iterates columns in reverse.
 * @param expectedV Expected vertical embeddings for all heads. Shape: [x][y][CONTEXT_WIN][EMBEDDING].
 * @param in Embedding dimension.
 * @param layers Number of MLP layers.
 * @param blockCount Current block index (1-based).
 * @param learning Learning rate.
 * @param lambda_l1 L1 regularization parameter.
 * @param lambda_l2 L2 regularization parameter.
 * @param clip_norm Maximum L2 norm for gradient clipping (new parameter).
 */
void block::clbackward(std::vector<std::vector<std::vector<std::vector<float>>>>& expectedV, int& in, int& layers, int& blockCount, float& learning, float& lambda_l1, float& lambda_l2, float& clip_norm) {
    // Validate input dimensions
    if (expectedV.size() != static_cast<size_t>(this->x)) {
        throw std::runtime_error("clbackward(V): ExpectedV outer dimension (rows) mismatch. Expected "
                                + std::to_string(this->x) + ", got " + std::to_string(expectedV.size()));
    }
    if (!expectedV.empty() && expectedV[0].size() != static_cast<size_t>(this->y)) {
        throw std::runtime_error("clbackward(V): ExpectedV second dimension (columns) mismatch. Expected "
                                + std::to_string(this->y) + ", got " + std::to_string(expectedV[0].size()));
    }
    // Deeper validation happens within clpartialbackward
    // deserialise(blockFilePath);
    // Iterate through all columns (parallels) in REVERSE order
    for (int j = this->y - 1; j >= 0; --j) { // j is the column index (layno)
        // Prepare the expectedV slice for the current column j
        std::vector<std::vector<std::vector<float>>> expectedV_col_j(this->x);
        try {
            for(int i = 0; i < x; i++) {
                b[i][j].tokenCount = this->tokenCount;
            }

            if(j == y - 1) {
                for (int i = 0; i < this->x; ++i) { // i is the row index
                    // Ensure expectedV[i] has enough columns before accessing expectedV[i][j]
                    if (expectedV[i].size() <= static_cast<size_t>(j)) {
                        throw std::runtime_error("clbackward(V): Column index " + std::to_string(j) +
                                                " out of bounds for expectedV row " + std::to_string(i) +
                                                " (size: " + std::to_string(expectedV[i].size()) + ")");
                    }
                    expectedV_col_j[i] = expectedV[i][j];
                }
            }
            else if(j < y - 1 && j >= 0) {
                for (int i = 0; i < this->x; ++i) { // i is the row index
                    // Ensure expectedV[i] has enough columns before accessing expectedV[i][j]
                    if (expectedV[i].size() <= static_cast<size_t>(j)) {
                        throw std::runtime_error("clbackward(V): Column index " + std::to_string(j) +
                                                " out of bounds for expectedV row " + std::to_string(i) +
                                                " (size: " + std::to_string(expectedV[i].size()) + ")");
                    }
                    // This line is crucial for passing the EV from the next block
                    expectedV_col_j[i] = b[i][j+1].EV.make2dVector(b[i][j+1].EV, CONTEXT_WIN, EMBEDDING);
                }
            }

            // Call the partial backward function for the current column j
            clpartialbackward(expectedV_col_j, in, layers, j, blockCount, learning, lambda_l1, lambda_l2, clip_norm);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Exception in clbackward(V) for column ["
                                    + std::to_string(j) + "]: " + e.what());
        }
    }
}

#endif