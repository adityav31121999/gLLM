
#include "include/model.hpp"
#include <iostream>
#include <fstream> // Include fstream for file operations
#include <stdexcept>
#include <vector> // Include vector for std::vector
#include <filesystem> // Include for path joining

/**
 * @brief Serialise a mat object to a binary file.
 * @param a The mat object to serialise.
 * @param locationOfBin The path to the output binary file.
 */
void model::serialise(mat& a, const std::string& locationOfBin) { // <-- Change to const reference
    // Open the file in binary write mode
    std::ofstream outFile(locationOfBin, std::ios::binary | std::ios::out);

    if (!outFile.is_open()) {
        throw std::runtime_error("Error: Could not open file for writing mat: " + locationOfBin);
    }

    // Write dimensions first (optional but recommended for deserialisation)
    outFile.write(reinterpret_cast<const char*>(&a.row), sizeof(int)); // Assuming row/col are int
    // outFile.write(reinterpret_cast<const char*>(&a.col), sizeof(int));

    for(int i = 0; i < a.row; i++) {
        // Write each row's data directly
        outFile.write(reinterpret_cast<const char*>(a.a[i].data()), a.col * sizeof(float));
    }
}

/**
 * @brief Serialise the weights of an mlp object to a binary file.
 * @param network The mlp object whose weights are to be serialised.
 * @param locationOfBin The path to the output binary file.
 */
void model::serialise(mlp& network, const std::string& locationOfBin) { // <-- Change to const reference
    std::ofstream outFile(locationOfBin, std::ios::binary | std::ios::out);

    if (!outFile.is_open()) {
        throw std::runtime_error("Error: Could not open file for writing mlp: " + locationOfBin);
    }

    // Get the weights vector reference for convenience
    const auto& weights = network.weights;

    // 1. Write the number of weight matrices (outer dimension)
    size_t num_matrices = weights.size();
    outFile.write(reinterpret_cast<const char*>(&num_matrices), sizeof(size_t));

    // 2. Iterate through each weight matrix
    for (size_t i = 0; i < num_matrices; ++i) {
        const auto& matrix = weights[i]; // Reference to the current 2D matrix
        // 2a. Write the number of rows for this matrix (middle dimension)
        size_t rows = matrix.size();
        outFile.write(reinterpret_cast<const char*>(&rows), sizeof(size_t));

        if (rows > 0) {
            // Assuming rectangular matrix, get cols from the first row
            const auto& first_row = matrix[0];
            size_t cols = first_row.size(); // Inner dimension
            // 2b. Write the number of columns for this matrix
            outFile.write(reinterpret_cast<const char*>(&cols), sizeof(size_t));

            // 2c. Write the data for each row
            for (size_t j = 0; j < rows; ++j) {
                // Optional: Add a check for rectangular consistency
                if (matrix[j].size() != cols) {
                     outFile.close(); // Ensure file is closed before throwing
                     throw std::runtime_error("Error: Weight matrix " + std::to_string(i) + " is not rectangular.");
                }
                outFile.write(reinterpret_cast<const char*>(matrix[j].data()), cols * sizeof(float));
            }
        } else {
            // If rows is 0, write 0 for columns dimension
            size_t cols = 0;
            outFile.write(reinterpret_cast<const char*>(&cols), sizeof(size_t));
        }
    }
}

/**
 * @brief Serialise the entire model's state to a default directory ("./model_data")
 *        within the current project directory.
 */
void model::serialise() {
    std::string defaultPath = "./model_data"; // Define the default path relative to the executable
    try {
        std::filesystem::create_directories(defaultPath); // Ensure the directory exists (Requires C++17)
        serialise(defaultPath); // Call the overload that takes a path
    } 
    catch (const std::filesystem::filesystem_error& e) {
        throw std::runtime_error("Error creating default directory '" + defaultPath + "': " + e.what());
    }
}

/**
 * @brief Serialise the entire model's state (matrices, caches, MLPs) to binary files
 *        within the specified directory.
 *      with these bin files:
 *          1. Matrices: MQ.bin, MK.bin, MH.bin, MV.bin
 *          2. MLPs: hor.bin, ver.bin
 *          3. Caches: QK.bin, KH.bin, QV.bin
 * @param locationOfModel The directory path where the .bin files will be saved.
 */
void model::serialise(std::string& locationOfModel) {
    // Ensure the target directory exists (optional, but good practice)
    // std::filesystem::create_directories(locationOfModel); // Requires C++17

    // Construct full paths using std::filesystem::path for better cross-platform compatibility
    std::filesystem::path basePath = locationOfModel; // Requires C++17

    // --- WARNING ---
    // The current loop structure overwrites the same files (MQ.bin, MK.bin, etc.)
    // in each iteration. You likely want to either:
    // 1. Create unique filenames for each block/layer (e.g., MQ_i_j_k.bin)
    // 2. Append data to the files (requires careful handling during deserialisation).
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < x; j++) {
            for(int k = 0; k < y; k++) {
                // serialise mats: MQ, MK, MV, MH
                serialise(T.t[i].b[j][k].MQ, (basePath / "MQ.bin").string());
                serialise(T.t[i].b[j][k].MK, (basePath / "MK.bin").string());
                serialise(T.t[i].b[j][k].MV, (basePath / "MV.bin").string());
                serialise(T.t[i].b[j][k].MH, (basePath / "MH.bin").string());
                // serialise caches: qkCache, qvCache, khCache
                serialise(T.t[i].b[j][k].qkCache, (basePath / "QK.bin").string());
                serialise(T.t[i].b[j][k].khCache, (basePath / "KH.bin").string());
                serialise(T.t[i].b[j][k].qvCache, (basePath / "QV.bin").string());
                // serialise mlps: hor, ver
                serialise(T.t[i].b[j][k].hor, (basePath / "hor.bin").string());
                serialise(T.t[i].b[j][k].ver, (basePath / "ver.bin").string());
            }
        }
    }    
}
