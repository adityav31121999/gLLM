
#include <iostream>
#include <fstream>      // For file stream operations
#include <stdexcept>
#include <vector>       // For std::vector
#include <filesystem>   // For path joining (C++17)
#include <string>       // For std::string
#include "include/model.hpp"

/**
 * @brief Deserialise a mat object (matrix or cache) from a specific offset within a binary file.
 *        Assumes dimensions are stored at the calculated offset.
 * @param a The mat object to load data into.
 * @param blockCount Index 'i' of the block (for offset calculation).
 * @param rows Index 'j' of the block (for offset calculation).
 * @param cols Index 'k' of the block (for offset calculation).
 * @param offset Number of float values in a single instance of this component.
 * @param locationOfBin The path to the input binary file.
 */
void model::deserialise(mat& a, int blockCount, int rows, int cols, int offset, const std::string& locationOfBin) {
    std::ifstream inFile(locationOfBin, std::ios::binary | std::ios::in);

    // Calculate the starting block index and byte offset
    // Using long long for byte_offset to prevent potential integer overflow
    long long blockCountndex = static_cast<long long>((blockCount+1) * x * y) + static_cast<long long>(rows * y) + static_cast<long long>(cols);
    long long byte_offset = blockCountndex * offset * sizeof(float);

    if (!inFile.is_open()) {
        throw std::runtime_error("Error: Could not open file for reading mat: " + locationOfBin);
    }

    int file_rows, file_cols;
    // Read dimensions from the file
    inFile.seekg(byte_offset, std::ios::beg); // Seek to the calculated position
    if (inFile.fail()) {
        inFile.close();
        throw std::runtime_error("Error: Failed to seek to offset " + std::to_string(byte_offset) + " in file: " + locationOfBin);
    }

    inFile.read(reinterpret_cast<char*>(&file_rows), sizeof(int));
    if (inFile.fail()) { inFile.close(); throw std::runtime_error("Error: Failed to read rows from offset " + std::to_string(byte_offset) + " in file: " + locationOfBin); }
    inFile.read(reinterpret_cast<char*>(&file_cols), sizeof(int));
    if (inFile.fail()) {
         inFile.close();
         throw std::runtime_error("Error: Failed to read cols from offset " + std::to_string(byte_offset) + " in file: " + locationOfBin);
    }

    // Resize the matrix object
    // Note: This assumes the matrix 'a' passed in might not have the correct size yet.
    // If 'a' is guaranteed to be correctly sized beforehand, you could verify dimensions instead:
    // if (file_rows != a.row || file_cols != a.col) { /* throw error */ }
    a.resize(file_rows, file_cols); // Use resize method from mat.hpp

    // Read matrix data row by row
    for (int i = 0; i < file_rows; ++i) {
        inFile.read(reinterpret_cast<char*>(a.a[i].data()), file_cols * sizeof(float));
        if (inFile.fail()) {
            inFile.close();
            throw std::runtime_error("Error: Failed to read data for row " + std::to_string(i) + " from offset " + std::to_string(byte_offset) + " in file: " + locationOfBin);
        }
    }
    // inFile closes automatically when it goes out of scope
}

/**
 * @brief Deserialise an mlp object from a specific offset within a binary file.
 *        Assumes dimensions and weights are stored starting at the calculated offset.
 * @param network The mlp object to load data into.
 * @param blockCount Index 'i' of the block (for offset calculation).
 * @param rows Index 'j' of the block (for offset calculation).
 * @param cols Index 'k' of the block (for offset calculation).
 * @param offset Number of float values in a single instance of this component's weights.
 * @param locationOfBin The path to the input binary file.
 */
void model::deserialise(mlp& network, int blockCount, int rows, int cols, int offset, const std::string& locationOfBin) {
    std::ifstream inFile(locationOfBin, std::ios::binary | std::ios::in);

    // Calculate the starting block index and byte offset
    // Using long long for byte_offset to prevent potential integer overflow
    long long blockCountndex = static_cast<long long>((blockCount+1) * x * y) + static_cast<long long>(rows * y) + static_cast<long long>(cols);
    long long byte_offset = blockCountndex * offset * sizeof(float);

    if (!inFile.is_open()) {
        throw std::runtime_error("Error: Could not open file for reading mlp: " + locationOfBin);
    }

    // 1. Read the number of weight matrices (outer dimension)
    size_t num_matrices;
    inFile.seekg(byte_offset, std::ios::beg); // Seek to the calculated position
    if (inFile.fail()) {
        inFile.close();
        throw std::runtime_error("Error: Failed to seek to offset " + std::to_string(byte_offset) + " in file: " + locationOfBin);
    }

    inFile.read(reinterpret_cast<char*>(&num_matrices), sizeof(size_t));
    if (inFile.fail()) { inFile.close(); throw std::runtime_error("Error reading num_matrices from offset " + std::to_string(byte_offset) + " in file: " + locationOfBin); }

    network.weights.resize(num_matrices); // Resize the outer vector

    // 2. Iterate through each weight matrix
    for (size_t i = 0; i < num_matrices; ++i) {
        // 2a. Read the number of rows for this matrix (middle dimension)
        size_t rows;
        inFile.read(reinterpret_cast<char*>(&rows), sizeof(size_t));
        if (inFile.fail()) { inFile.close(); throw std::runtime_error("Error reading rows for matrix " + std::to_string(i) + " from offset " + std::to_string(byte_offset) + " in file: " + locationOfBin); }

        network.weights[i].resize(rows); // Resize the middle vector

        // 2b. Read the number of columns for this matrix
        size_t cols;
        inFile.read(reinterpret_cast<char*>(&cols), sizeof(size_t));
        if (inFile.fail()) { inFile.close(); throw std::runtime_error("Error reading cols for matrix " + std::to_string(i) + " from offset " + std::to_string(byte_offset) + " in file: " + locationOfBin); }

        // 2c. Read the data for each row
        for (size_t j = 0; j < rows; ++j) {
            network.weights[i][j].resize(cols); // Resize the inner vector (row)
            inFile.read(reinterpret_cast<char*>(network.weights[i][j].data()), cols * sizeof(float));
            if (inFile.fail()) {
                inFile.close();
                throw std::runtime_error("Error reading data for matrix " + std::to_string(i) + ", row " + std::to_string(j) + " from offset " + std::to_string(byte_offset) + " in file: " + locationOfBin);
            }
        }
    }
    // inFile closes automatically
}

/**
 * @brief Deserialise whole model (matrices, caches, mlps) - Placeholder.
 */
void model::deserialise() {
    std::string defaultPath = "./model_data"; // Define the default path relative to the executable
    try {
        std::filesystem::create_directories(defaultPath); // Ensure the directory exists (Requires C++17)
        deserialise(defaultPath); // Call the overload that takes a path
    } 
    catch (const std::filesystem::filesystem_error& e) {
        throw std::runtime_error("Error creating default directory '" + defaultPath + "': " + e.what());
    }
}

/**
 * @brief Deserialise the entire model's state (matrices, caches, MLPs) from binary files
 *        within the specified directory.
 * @param locationOfModel The directory path where the .bin files are located.
 */
void model::deserialise(std::string& locationOfModel) {
    std::filesystem::path basePath = locationOfModel; // Requires C++17

    // Determine expected dimensions based on model parameters (d, h, l)
    // These are no longer strictly needed here if dimensions are read from the file within helpers,
    // but can be useful for understanding the expected structure.
    // int mat_qk_rows = h; // Assuming h is height of MQ/MK
    // int mat_qk_cols = d; // Assuming d is embedding dimension / width
    // int mat_vh_rows = d; // Assuming d is embedding dimension / width
    // int mat_vh_cols = h; // Assuming h is height of MV/MH
    int cache_dim = d;   // Assuming caches are d x d

    // Loop through the blocks/layers as defined in the model structure
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < x; ++j) {
            for (int k = 0; k < y; ++k) {
                // Deserialise Matrices for block (i, j, k)
                // Pass block indices (i, j, k) and the size of one matrix component (MATOFFSET)
                // The helper function will calculate the byte offset and seek.

                // Assuming files are overwritten (current state):
                deserialise(T.t[i].b[j][k].MQ, i, j, k, MATOFFSET, (basePath / "MQ.bin").string());
                deserialise(T.t[i].b[j][k].MK, i, j, k, MATOFFSET, (basePath / "MK.bin").string());
                deserialise(T.t[i].b[j][k].MV, i, j, k, MATOFFSET, (basePath / "MV.bin").string());
                deserialise(T.t[i].b[j][k].MH, i, j, k, MATOFFSET, (basePath / "MH.bin").string());

                // Deserialise Caches for block (i, j, k)
                deserialise(T.t[i].b[j][k].qkCache, i, j, k, CACHEOFFSET, (basePath / "QK.bin").string());
                deserialise(T.t[i].b[j][k].khCache, i, j, k, CACHEOFFSET, (basePath / "KH.bin").string());
                deserialise(T.t[i].b[j][k].qvCache, i, j, k, CACHEOFFSET, (basePath / "QV.bin").string());

                // Deserialise MLPs for block (i, j, k)
                // Pass block indices (i, j, k) and the size of one MLP component (MLPOFFSET)
                deserialise(T.t[i].b[j][k].hor, i, j, k, MLPOFFSET, (basePath / "hor.bin").string());
                deserialise(T.t[i].b[j][k].ver, i, j, k, MLPOFFSET, (basePath / "ver.bin").string());
            }
        }
    }
}
