
// get values from .bin files for cache and mlp
#include "include/transformer.hpp" // Includes block.hpp -> attention.hpp -> mlp.hpp, maths.hpp (for mat)
#include <fstream>                // Required for file operations
#include <stdexcept>              // Required for exception handling
#include <string>                 // Required for string manipulation (path joining)
#include <vector>                 // Required for vector operations
#include <iostream>               // For potential debugging output
#include "transformer.hpp"

// Helper function to construct full path (assuming it's defined elsewhere or here)
inline std::string join_path(const std::string& folder, const std::string& filename) {
#ifdef _WIN32
    const char separator = '\\';
#else
    const char separator = '/';
#endif
    if (folder.empty()) {
        return filename;
    }
    // Avoid double separators if folder already ends with one
    if (folder.back() == separator) {
        return folder + filename;
    }
    return folder + separator + filename;
}

/**
 * @brief Loads all cache matrices and MLP weights for a specific block from binary files
 * @param blockCount The 1-based index of the block whose data needs to be loaded.
 * @param path2folderOfAllBins The path to the directory containing the binary files
 *      (QK.bin, QV.bin, KH.bin, hor.bin, ver.bin).
 */
void transformer::getAllValues(int blockCount, std::string path2folderOfAllBins, bool& inTraining)
{
    // --- Basic Validation ---
    if (blockCount <= 0) {
        throw std::runtime_error("getAllValues: blockCount must be greater than 0.");
    }
    if (t.empty()) {
         throw std::runtime_error("getAllValues: Transformer block vector 't' is empty. Initialize it first.");
    }
    // Ensure the target block structure within t[0] is initialized
    if (t[0].b.empty() || t[0].b[0].empty()) {
         throw std::runtime_error("getAllValues: Transformer block structure t[0].b not initialized (vector<vector<attention>> is empty).");
    }
    if (x <= 0 || y <= 0 || d <= 0 || h <= 0 || l <= 0) {
        throw std::runtime_error("getAllValues: Transformer dimensions (x, y, d, h, l) must be positive.");
    }
    // Check consistency between transformer dimensions and actual block structure size
    if (static_cast<size_t>(x) != t[0].b.size() || static_cast<size_t>(y) != t[0].b[0].size()) {
        std::cerr << "Warning: getAllValues: Transformer dimensions (x,y) = (" << x << "," << y
                  << ") do not match allocated block structure size (" << t[0].b.size() << ","
                  << (t[0].b.empty() ? 0 : t[0].b[0].size()) // Avoid accessing b[0] if b is empty
                  << "). Ensure consistency." << std::endl;
    }

    if(inTraining == 1) {
        // --- Calculate Sizes and Offsets ---
        // matrices - Assuming d rows, h columns for MQ and MK and h row, d columns for MH and MV
        // (as per attention.hpp members MQ, MK, MH, MV)
        size_t cache_element_count = static_cast<size_t>(d) * h; // Based on mat dimensions d x h
        size_t cache_size_bytes = cache_element_count * sizeof(float);
        size_t caches_per_row_in_block = y; // Number of attention heads per row (partial attention)
        size_t caches_per_block = static_cast<size_t>(x) * caches_per_row_in_block;
        size_t block_cache_offset_bytes = static_cast<size_t>(blockCount - 1) * caches_per_block * cache_size_bytes;
        // --- Load Cache Matrices (QK, QV, KH) using Pointer-to-Member of 'attention' class ---
        const char* mat_files[] = {"MQ.bin", "MK.bin", "MV.bin", "MH.bin"};
        // Correctly point to 'mat' members within the 'attention' class
        using CacheMemberPtr = mat(attention::*);
        CacheMemberPtr cache_members[] = {&attention::qkCache, &attention::qvCache, &attention::khCache};
        // MQ and MK
        for (int c_idx = 0; c_idx < 2; ++c_idx) {
            std::string file_path = join_path(path2folderOfAllBins, mat_files[c_idx]);
            std::ifstream file(file_path, std::ios::binary | std::ios::in);
            if (!file.is_open()) {
                throw std::runtime_error("getAllValues: Could not open cache file: " + file_path);
            }

            // Seek to the beginning of the data for the specified block
            file.seekg(block_cache_offset_bytes, std::ios::beg);
            if (!file) {
                file.close();
                throw std::runtime_error("getAllValues: Failed to seek in cache file: " + file_path + " to offset " + std::to_string(block_cache_offset_bytes));
            }

            for (int i = 0; i < x; ++i) {
                for (int j = 0; j < y; ++j) {
                    mat& current_cache = t[0].b[i][j].*(cache_members[c_idx]);
                    current_cache.resize(d, h);

                    // Read data row by row
                    for (int row_idx = 0; row_idx < d; ++row_idx) {
                        // Access the underlying vector 'a' from the 'mat' object
                        file.read(reinterpret_cast<char*>(current_cache.a[row_idx].data()), h * sizeof(float));
                        if (!file) {
                            file.close();
                            throw std::runtime_error("getAllValues: Error reading data for cache " + std::string(mat_files[c_idx]) +
                                                    " at block " + std::to_string(blockCount) + ", pos (" + std::to_string(i) +
                                                    ", " + std::to_string(j) + "), row " + std::to_string(row_idx) + " from " + file_path);
                        }
                    }
                }
            }
            file.close(); // Close file after processing all i,j for this cache type
        }
        cache_element_count = static_cast<size_t>(h) * d; // Based on mat dimensions d x h
        cache_size_bytes = cache_element_count * sizeof(float);
        caches_per_row_in_block = y; // Number of attention heads per row (partial attention)
        caches_per_block = static_cast<size_t>(x) * caches_per_row_in_block;
        block_cache_offset_bytes = static_cast<size_t>(blockCount - 1) * caches_per_block * cache_size_bytes;
        // MH and MV
        for (int c_idx = 0; c_idx < 2; ++c_idx) {
            std::string file_path = join_path(path2folderOfAllBins, mat_files[c_idx+2]);
            std::ifstream file(file_path, std::ios::binary | std::ios::in);
            if (!file.is_open()) {
                throw std::runtime_error("getAllValues: Could not open cache file: " + file_path);
            }

            // Seek to the beginning of the data for the specified block
            file.seekg(block_cache_offset_bytes, std::ios::beg);
            if (!file) {
                file.close();
                throw std::runtime_error("getAllValues: Failed to seek in cache file: " + file_path + " to offset " + std::to_string(block_cache_offset_bytes));
            }

            for (int i = 0; i < x; ++i) {
                for (int j = 0; j < y; ++j) {
                    mat& current_cache = t[0].b[i][j].*(cache_members[c_idx]);
                    current_cache.resize(d, h);

                    // Read data row by row
                    for (int row_idx = 0; row_idx < h; ++row_idx) {
                        // Access the underlying vector 'a' from the 'mat' object
                        file.read(reinterpret_cast<char*>(current_cache.a[row_idx].data()), d * sizeof(float));
                        if (!file) {
                            file.close();
                            throw std::runtime_error("getAllValues: Error reading data for cache " + std::string(mat_files[c_idx+2]) +
                                                    " at block " + std::to_string(blockCount) + ", pos (" + std::to_string(i) +
                                                    ", " + std::to_string(j) + "), row " + std::to_string(row_idx) + " from " + file_path);
                        }
                    }
                }
            }
            file.close(); // Close file after processing all i,j for this cache type
        }
    }

    // --- Calculate Sizes and Offsets ---
    // Cache (QK, QV, KH) - Assuming d rows, d columns (as per attention.hpp members qkCache, qvCache, khCache dimensions)
    size_t cache_element_count = static_cast<size_t>(d) * d; // Based on mat dimensions d x d
    size_t cache_size_bytes = cache_element_count * sizeof(float);
    size_t caches_per_row_in_block = y; // Number of attention heads per row (partial attention)
    size_t caches_per_block = static_cast<size_t>(x) * caches_per_row_in_block;
    size_t block_cache_offset_bytes = static_cast<size_t>(blockCount - 1) * caches_per_block * cache_size_bytes;

    // MLP weights (hor, ver) - Assuming l layers, d x d weights per layer (as per mlp.hpp structure)
    size_t mlp_layer_element_count = static_cast<size_t>(d) * d;
    size_t mlp_total_element_count = static_cast<size_t>(l) * mlp_layer_element_count;
    size_t mlp_size_bytes = mlp_total_element_count * sizeof(float);
    size_t mlps_per_row_in_block = y; // Number of attention heads per row
    size_t mlps_per_block = static_cast<size_t>(x) * mlps_per_row_in_block;
    size_t block_mlp_offset_bytes = static_cast<size_t>(blockCount - 1) * mlps_per_block * mlp_size_bytes;

    // --- Load Cache Matrices (QK, QV, KH) using Pointer-to-Member of 'attention' class ---
    const char* cache_files[] = {"QK.bin", "QV.bin", "KH.bin"};
    // Correctly point to 'mat' members within the 'attention' class
    using CacheMemberPtr = mat(attention::*);
    CacheMemberPtr cache_members[] = {&attention::qkCache, &attention::qvCache, &attention::khCache};

    for (int c_idx = 0; c_idx < 3; ++c_idx) {
        std::string file_path = join_path(path2folderOfAllBins, cache_files[c_idx]);
        std::ifstream file(file_path, std::ios::binary | std::ios::in);
        if (!file.is_open()) {
            throw std::runtime_error("getAllValues: Could not open cache file: " + file_path);
        }

        // Seek to the beginning of the data for the specified block
        file.seekg(block_cache_offset_bytes, std::ios::beg);
        if (!file) {
             file.close();
             throw std::runtime_error("getAllValues: Failed to seek in cache file: " + file_path + " to offset " + std::to_string(block_cache_offset_bytes));
        }

        for (int i = 0; i < x; ++i) {
            for (int j = 0; j < y; ++j) {
                mat& current_cache = t[0].b[i][j].*(cache_members[c_idx]);
                current_cache.resize(d, d);

                // Read data row by row
                for (int row_idx = 0; row_idx < d; ++row_idx) {
                    // Access the underlying vector 'a' from the 'mat' object
                    file.read(reinterpret_cast<char*>(current_cache.a[row_idx].data()), d * sizeof(float));
                    if (!file) {
                        file.close();
                        throw std::runtime_error("getAllValues: Error reading data for cache " + std::string(cache_files[c_idx]) +
                                                 " at block " + std::to_string(blockCount) + ", pos (" + std::to_string(i) +
                                                 ", " + std::to_string(j) + "), row " + std::to_string(row_idx) + " from " + file_path);
                    }
                }
            }
        }
        file.close(); // Close file after processing all i,j for this cache type
    }

    // --- Load MLP Weights (hor, ver) using Pointer-to-Member of 'attention' class ---
    const char* mlp_files[] = {"hor.bin", "ver.bin"};
    // Correctly point to 'mlp' members within the 'attention' class
    using MlpMemberPtr = mlp(attention::*);
    MlpMemberPtr mlp_members[] = {&attention::hor, &attention::ver};

    for (int m_idx = 0; m_idx < 2; ++m_idx) {
        std::string file_path = join_path(path2folderOfAllBins, mlp_files[m_idx]);
        std::ifstream file(file_path, std::ios::binary | std::ios::in);
        if (!file.is_open()) {
            throw std::runtime_error("getAllValues: Could not open MLP file: " + file_path);
        }

        // Seek to the beginning of the data for the specified block
        file.seekg(block_mlp_offset_bytes, std::ios::beg);
         if (!file) {
             file.close();
             throw std::runtime_error("getAllValues: Failed to seek in MLP file: " + file_path + " to offset " + std::to_string(block_mlp_offset_bytes));
        }

        for (int i = 0; i < x; ++i) { // Iterate through rows of the block
            for (int j = 0; j < y; ++j) { // Iterate through attention heads within the row
                // Get the specific MLP member using the pointer-to-member
                // t[0] is the block, b[i][j] is the attention object at (i, j)
                mlp& current_mlp = t[0].b[i][j].*(mlp_members[m_idx]);

                // Resize the weights structure if necessary (important!)
                // Access the 'weights' member of the 'mlp' class
                current_mlp.weights.resize(l, std::vector<std::vector<float>>(d, std::vector<float>(d)));

                // Read data layer by layer, row by row
                for (int layer_idx = 0; layer_idx < l; ++layer_idx) {
                    for (int row_idx = 0; row_idx < d; ++row_idx) {
                        // Access the underlying vector 'weights' from the 'mlp' object
                        file.read(reinterpret_cast<char*>(current_mlp.weights[layer_idx][row_idx].data()), d * sizeof(float));
                        if (!file) {
                            file.close();
                            throw std::runtime_error("getAllValues: Error reading data for MLP " + std::string(mlp_files[m_idx]) +
                                                     " at block " + std::to_string(blockCount) + ", pos (" + std::to_string(i) +
                                                     ", " + std::to_string(j) + "), layer " + std::to_string(layer_idx) +
                                                     ", row " + std::to_string(row_idx) + " from " + file_path);
                        }
                    }
                }
            }
        }
        file.close(); // Close file after processing all i,j for this mlp type
    }
}


/**
 * @brief Gets a specific cache matrix from a binary file for a given block and position (attention head).
 * @param blockCount The 1-based index of the block.
 * @param i The 0-based row index within the block's attention grid (0 to x-1).
 * @param j The 0-based column index within the block's attention grid (0 to y-1).
 * @param q The output matrix (std::vector<std::vector<float>>&) to store the cache data. Passed by reference.
 * @param path2file The full path to the specific binary cache file (e.g., "path/to/QK.bin").
 */
void transformer::getcache(int blockCount, int i, int j, std::vector<std::vector<float>>& q, std::string path2file)
{
    // --- Validation ---
    if (blockCount <= 0) {
        throw std::out_of_range("getcache: blockCount must be greater than 0.");
    }
     if (i < 0 || i >= x || j < 0 || j >= y) {
         throw std::out_of_range("getcache: Index (i, j) = (" + std::to_string(i) + ", " + std::to_string(j) +
                                ") is out of bounds for grid dimensions (x, y) = (" + std::to_string(x) + ", " + std::to_string(y) + ").");
    }
    if (d <= 0 || h <= 0) {
        throw std::runtime_error("getcache: Transformer dimensions d and h must be positive.");
    }

    // --- Calculate Offset ---
    // Cache matrix size: d rows, h columns
    size_t cache_element_count = static_cast<size_t>(d) * d;
    size_t cache_size_bytes = cache_element_count * sizeof(float);
    size_t caches_per_row_in_block = y;
    size_t caches_per_block = static_cast<size_t>(x) * caches_per_row_in_block;

    // Calculate the index of the target cache within the flattened sequence of all caches
    size_t target_cache_index = (static_cast<size_t>(blockCount - 1) * caches_per_block) // Caches in previous blocks
                               + (static_cast<size_t>(i) * caches_per_row_in_block)     // Caches in previous rows of current block
                               + static_cast<size_t>(j);                                // Caches in current row before target

    size_t offset_bytes = target_cache_index * cache_size_bytes;

    // --- File Operations ---
    std::ifstream file(path2file, std::ios::binary | std::ios::in);
    if (!file.is_open()) {
        throw std::runtime_error("getcache: Could not open cache file: " + path2file);
    }

    // Seek to position
    file.seekg(offset_bytes, std::ios::beg);
    if (!file) {
        file.close();
        throw std::runtime_error("getcache: Failed to seek in cache file: " + path2file + " to offset " + std::to_string(offset_bytes));
    }

    // Resize output vector (passed by reference)
    q.resize(d, std::vector<float>(d));

    // Read data row by row
    for (int row_idx = 0; row_idx < d; ++row_idx) {
        file.read(reinterpret_cast<char*>(q[row_idx].data()), d * sizeof(float));
        if (!file) {
            file.close();
            throw std::runtime_error("getcache: Error reading data at block " + std::to_string(blockCount) +
                                     ", pos (" + std::to_string(i) + ", " + std::to_string(j) +
                                     "), row " + std::to_string(row_idx) + " from " + path2file);
        }
    }
    file.close();
}


/**
 * @brief Gets the weights for a specific MLP from a binary file for a given block and position (attention head).
 * @param blockCount The 1-based index of the block.
 * @param i The 0-based row index within the block's attention grid (0 to x-1).
 * @param j The 0-based column index within the block's attention grid (0 to y-1).
 * @param weights The output 3D vector (std::vector<std::vector<std::vector<float>>>&) to store the MLP weights [layer][row][col]. Passed by reference.
 * @param path2file The full path to the specific binary MLP weights file (e.g., "path/to/hor.bin").
 */
void transformer::getmlp(int blockCount, int i, int j, std::vector<std::vector<std::vector<float>>>& weights, std::string path2file)
{
    // --- Validation ---
     if (blockCount <= 0) {
        throw std::out_of_range("getmlp: blockCount must be greater than 0.");
    }
     if (i < 0 || i >= x || j < 0 || j >= y) {
         throw std::out_of_range("getmlp: Index (i, j) = (" + std::to_string(i) + ", " + std::to_string(j) +
                                ") is out of bounds for grid dimensions (x, y) = (" + std::to_string(x) + ", " + std::to_string(y) + ").");
    }
    if (d <= 0 || l <= 0) {
        throw std::runtime_error("getmlp: Transformer dimensions d and l must be positive.");
    }

    // --- Calculate Offset ---
    // MLP size: l layers, d x d weights per layer
    size_t mlp_layer_element_count = static_cast<size_t>(d) * d;
    size_t mlp_total_element_count = static_cast<size_t>(l) * mlp_layer_element_count;
    size_t mlp_size_bytes = mlp_total_element_count * sizeof(float);
    size_t mlps_per_row_in_block = y;
    size_t mlps_per_block = static_cast<size_t>(x) * mlps_per_row_in_block;

    // Calculate the index of the target MLP within the flattened sequence of all MLPs
    size_t target_mlp_index = (static_cast<size_t>(blockCount - 1) * mlps_per_block) // MLPs in previous blocks
                             + (static_cast<size_t>(i) * mlps_per_row_in_block)     // MLPs in previous rows of current block
                             + static_cast<size_t>(j);                              // MLPs in current row before target

    size_t offset_bytes = target_mlp_index * mlp_size_bytes;

    // --- File Operations ---
    std::ifstream file(path2file, std::ios::binary | std::ios::in);
    if (!file.is_open()) {
        throw std::runtime_error("getmlp: Could not open MLP file: " + path2file);
    }

    // Seek to position
    file.seekg(offset_bytes, std::ios::beg);
     if (!file) {
        file.close();
        throw std::runtime_error("getmlp: Failed to seek in MLP file: " + path2file + " to offset " + std::to_string(offset_bytes));
    }

    // Resize output vector (passed by reference)
    weights.resize(l, std::vector<std::vector<float>>(d, std::vector<float>(d)));

    // Read data layer by layer, row by row
    for (int layer_idx = 0; layer_idx < l; ++layer_idx) {
        for (int row_idx = 0; row_idx < d; ++row_idx) {
            // Read one full row (d floats) for the current layer
            file.read(reinterpret_cast<char*>(weights[layer_idx][row_idx].data()), d * sizeof(float));
            if (!file) {
                file.close();
                throw std::runtime_error("getmlp: Error reading data at block " + std::to_string(blockCount) +
                                         ", pos (" + std::to_string(i) + ", " + std::to_string(j) +
                                         "), layer " + std::to_string(layer_idx) + ", row " + std::to_string(row_idx) +
                                         " from " + path2file);
            }
        }
    }
    file.close();
}


/**
 * @brief Gets a specific cache matrix from a binary file for a given block and position (attention head).
 * @param blockCount The 1-based index of the block.
 * @param i The 0-based row index within the block's attention grid (0 to x-1).
 * @param j The 0-based column index within the block's attention grid (0 to y-1).
 * @param q The output matrix (std::vector<std::vector<float>>&) to store the cache data. Passed by reference.
 * @param path2file The full path to the specific binary cache file (e.g., "path/to/QK.bin").
 */
void transformer::getmat(int blockCount, int i, int j, std::vector<std::vector<float>>& q, std::string path2file, int& row, int& column)
{
    // --- Validation ---
    if (blockCount <= 0) {
        throw std::out_of_range("getcache: blockCount must be greater than 0.");
    }
     if (i < 0 || i >= x || j < 0 || j >= y) {
         throw std::out_of_range("getcache: Index (i, j) = (" + std::to_string(i) + ", " + std::to_string(j) +
                                ") is out of bounds for grid dimensions (x, y) = (" + std::to_string(x) + ", " + std::to_string(y) + ").");
    }
    if (row <= 0 || column <= 0) {
        throw std::runtime_error("getcache: Transformer dimensions d and h must be positive.");
    }

    // --- Calculate Offset ---
    // Cache matrix size: d rows, h columns
    size_t cache_element_count = static_cast<size_t>(row) * column;
    size_t cache_size_bytes = cache_element_count * sizeof(float);
    size_t caches_per_row_in_block = y;
    size_t caches_per_block = static_cast<size_t>(x) * caches_per_row_in_block;

    // Calculate the index of the target cache within the flattened sequence of all caches
    size_t target_cache_index = (static_cast<size_t>(blockCount - 1) * caches_per_block) // Caches in previous blocks
                               + (static_cast<size_t>(i) * caches_per_row_in_block)     // Caches in previous rows of current block
                               + static_cast<size_t>(j);                                // Caches in current row before target

    size_t offset_bytes = target_cache_index * cache_size_bytes;

    // --- File Operations ---
    std::ifstream file(path2file, std::ios::binary | std::ios::in);
    if (!file.is_open()) {
        throw std::runtime_error("getcache: Could not open cache file: " + path2file);
    }

    // Seek to position
    file.seekg(offset_bytes, std::ios::beg);
    if (!file) {
        file.close();
        throw std::runtime_error("getcache: Failed to seek in cache file: " + path2file + " to offset " + std::to_string(offset_bytes));
    }

    // Resize output vector (passed by reference)
    q.resize(row, std::vector<float>(column));

    // Read data row by row
    for (int row_idx = 0; row_idx < row; ++row_idx) {
        file.read(reinterpret_cast<char*>(q[row_idx].data()), column * sizeof(float));
        if (!file) {
            file.close();
            throw std::runtime_error("getcache: Error reading data at block " + std::to_string(blockCount) +
                                     ", pos (" + std::to_string(i) + ", " + std::to_string(j) +
                                     "), row " + std::to_string(row_idx) + " from " + path2file);
        }
    }
    file.close();
}
