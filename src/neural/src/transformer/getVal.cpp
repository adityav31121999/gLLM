
// get values from .bin files for cache and mlp
#include "include/transformer.hpp" // Includes block.hpp -> attention.hpp -> mlp.hpp, maths.hpp (for mat)
#include <fstream>                // Required for file operations
#include <stdexcept>              // Required for exception handling
#include <string>                 // Required for string manipulation (path joining)
#include <vector>                 // Required for vector operations
#include <iostream>               // For potential debugging output

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
    if (t[0].b.empty() || t[0].b[0].empty()) {
         throw std::runtime_error("getAllValues: Transformer block structure t[0].b not initialized (vector<vector<attention>> is empty).");
    }
    if (this->x <= 0 || this->y <= 0 || this->d <= 0 || this->h <= 0 || this->l <= 0) {
        throw std::runtime_error("getAllValues: Transformer dimensions (x, y, d, h, l) must be positive.");
    }
    if (static_cast<size_t>(this->x) != t[0].b.size() || static_cast<size_t>(this->y) != t[0].b[0].size()) {
        std::cerr << "Warning: getAllValues: Transformer dimensions (x,y) = (" << this->x << "," << this->y
                  << ") do not match allocated block structure size (" << t[0].b.size() << ","
                  << (t[0].b.empty() ? 0 : t[0].b[0].size()) // Avoid accessing b[0] if b is empty
                  << "). Ensure consistency." << std::endl;
    }

    if(inTraining == 1) {
        const char* training_mat_files[] = {"MQ.bin", "MK.bin", "MV.bin", "MH.bin"};
        using TrainingMatMemberPtr = mat(attention::*);
        TrainingMatMemberPtr training_members[] = {&attention::MQ, &attention::MK, &attention::MV, &attention::MH};

        int current_rows, current_cols;

        for (int tm_idx = 0; tm_idx < 4; ++tm_idx) {
            std::string file_path = join_path(path2folderOfAllBins, training_mat_files[tm_idx]);
            std::ifstream file(file_path, std::ios::binary | std::ios::in);
            if (!file.is_open()) {
                throw std::runtime_error("getAllValues: Could not open training matrix file: " + file_path);
            }

            if (tm_idx < 2) {
                current_rows = this->d;
                current_cols = this->h;
            }
            else {
                current_rows = this->h;
                current_cols = this->d;
            }

            size_t mat_element_count = static_cast<size_t>(current_rows) * current_cols;
            size_t mat_size_bytes = mat_element_count * sizeof(float);
            size_t mats_per_block = static_cast<size_t>(this->x) * this->y;
            size_t block_offset_bytes = static_cast<size_t>(blockCount - 1) * mats_per_block * mat_size_bytes;

            file.seekg(block_offset_bytes, std::ios::beg);
            if (!file) {
                file.close();
                throw std::runtime_error("getAllValues: Failed to seek in training matrix file: " + file_path + " to offset " + std::to_string(block_offset_bytes));
            }

            for (int i = 0; i < this->x; ++i) {
                for (int j = 0; j < this->y; ++j) {
                    mat temp_mat(current_rows, current_cols);

                    for (int row_idx = 0; row_idx < current_rows; ++row_idx) {
                        file.read(reinterpret_cast<char*>(temp_mat.mapped_data + static_cast<size_t>(row_idx) * current_cols),
                                  static_cast<size_t>(current_cols) * sizeof(float));
                        if (!file) {
                            file.close();
                            throw std::runtime_error("getAllValues: Error reading data for training matrix " + std::string(training_mat_files[tm_idx]) +
                                                    " at block " + std::to_string(blockCount) + ", pos (" + std::to_string(i) +
                                                    ", " + std::to_string(j) + "), row " + std::to_string(row_idx) + " from " + file_path);
                        }
                    }
                    t[0].b[i][j].*(training_members[tm_idx]) = std::move(temp_mat);
                }
            }
            file.close();
        }
    }

    // --- Calculate Sizes and Offsets for regular caches (QK, QV, KH) ---
    size_t cache_element_count = static_cast<size_t>(this->d) * this->d;
    size_t cache_size_bytes = cache_element_count * sizeof(float);
    size_t caches_per_row_in_block = this->y;
    size_t caches_per_block = static_cast<size_t>(this->x) * caches_per_row_in_block;
    size_t block_cache_offset_bytes = static_cast<size_t>(blockCount - 1) * caches_per_block * cache_size_bytes;

    // --- Calculate Sizes and Offsets for MLP weights (hor, ver) ---
    size_t mlp_layer_element_count = static_cast<size_t>(this->d) * this->d;
    size_t mlp_total_element_count = static_cast<size_t>(this->l) * mlp_layer_element_count;
    size_t mlp_size_bytes = mlp_total_element_count * sizeof(float);
    size_t mlps_per_row_in_block = this->y;
    size_t mlps_per_block = static_cast<size_t>(this->x) * mlps_per_row_in_block;
    size_t block_mlp_offset_bytes = static_cast<size_t>(blockCount - 1) * mlps_per_block * mlp_size_bytes;

    // --- Load Regular Cache Matrices (QK, QV, KH) ---
    const char* cache_files[] = {"QK.bin", "QV.bin", "KH.bin"};
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

        for (int i = 0; i < this->x; ++i) {
            for (int j = 0; j < this->y; ++j) {
                // Create a new mat; this will handle memory mapping.
                mat temp_mat(this->d, this->d);

                for (int row_idx = 0; row_idx < this->d; ++row_idx) {
                    file.read(reinterpret_cast<char*>(temp_mat.mapped_data + static_cast<size_t>(row_idx) * this->d),
                              static_cast<size_t>(this->d) * sizeof(float));
                    if (!file) {
                        file.close();
                        throw std::runtime_error("getAllValues: Error reading data for cache " + std::string(cache_files[c_idx]) +
                                                 " at block " + std::to_string(blockCount) + ", pos (" + std::to_string(i) +
                                                 ", " + std::to_string(j) + "), row " + std::to_string(row_idx) + " from " + file_path);
                    }
                }
                t[0].b[i][j].*(cache_members[c_idx]) = std::move(temp_mat);
            }
        }
        file.close();
    }

    // --- Load MLP Weights (hor, ver) ---
    const char* mlp_files[] = {"hor.bin", "ver.bin"};
    using MlpMemberPtr = mlp(attention::*);
    MlpMemberPtr mlp_members[] = {&attention::hor, &attention::ver};

    for (int m_idx = 0; m_idx < 2; ++m_idx) {
        std::string file_path = join_path(path2folderOfAllBins, mlp_files[m_idx]);
        std::ifstream file(file_path, std::ios::binary | std::ios::in);
        if (!file.is_open()) {
            throw std::runtime_error("getAllValues: Could not open MLP file: " + file_path);
        }

        file.seekg(block_mlp_offset_bytes, std::ios::beg);
         if (!file) {
             file.close();
             throw std::runtime_error("getAllValues: Failed to seek in MLP file: " + file_path + " to offset " + std::to_string(block_mlp_offset_bytes));
        }

        for (int i = 0; i < this->x; ++i) {
            for (int j = 0; j < this->y; ++j) {
                mlp& current_mlp = t[0].b[i][j].*(mlp_members[m_idx]);
                current_mlp.weights.resize(this->l);

                for (int layer_idx = 0; layer_idx < this->l; ++layer_idx) {
                    mat temp_layer_mat(this->d, this->d);
                    for (int row_idx = 0; row_idx < this->d; ++row_idx) {
                        file.read(reinterpret_cast<char*>(temp_layer_mat.mapped_data + static_cast<size_t>(row_idx) * this->d),
                                  static_cast<size_t>(this->d) * sizeof(float));
                        if (!file) {
                            file.close();
                            throw std::runtime_error("getAllValues: Error reading data for MLP " + std::string(mlp_files[m_idx]) +
                                                     " at block " + std::to_string(blockCount) + ", pos (" + std::to_string(i) +
                                                     ", " + std::to_string(j) + "), layer " + std::to_string(layer_idx) +
                                                     ", row " + std::to_string(row_idx) + " from " + file_path);
                        }
                    }
                    current_mlp.weights[layer_idx] = std::move(temp_layer_mat);
                }
            }
        }
        file.close();
    }
}


/**
 * @brief Gets a specific cache matrix from a binary file for a given block and position (attention head).
 * @param blockCount The 1-based index of the block.
 * @param i The 0-based row index within the block's attention grid (0 to x-1).
 * @param j The 0-based column index within the block's attention grid (0 to y-1).
 * @param q The output matrix (mat&) to store the cache data. Passed by reference.
 * @param path2file The full path to the specific binary cache file (e.g., "path/to/QK.bin").
 */
void transformer::getcache(int blockCount, int i, int j, mat& q, std::string path2file)
{
    if (blockCount <= 0) {
        throw std::out_of_range("getcache: blockCount must be greater than 0.");
    }
     if (i < 0 || i >= this->x || j < 0 || j >= this->y) {
         throw std::out_of_range("getcache: Index (i, j) = (" + std::to_string(i) + ", " + std::to_string(j) +
                                ") is out of bounds for grid dimensions (x, y) = (" + std::to_string(this->x) + ", " + std::to_string(this->y) + ").");
    }
    if (this->d <= 0 || this->h <= 0) {
        throw std::runtime_error("getcache: Transformer dimensions d and h must be positive.");
    }

    // --- Calculate Offset ---
    size_t cache_element_count = static_cast<size_t>(this->d) * this->d;
    size_t cache_size_bytes = cache_element_count * sizeof(float);
    size_t caches_per_row_in_block = this->y;
    size_t caches_per_block = static_cast<size_t>(this->x) * caches_per_row_in_block;

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

    // Assign a new mat object to q. This creates a new backing file.
    // The old mat in q (if any) will be destructed.
    q = mat(this->d, this->d);

    // Read data directly into the mapped_data of q
    for (int row_idx = 0; row_idx < this->d; ++row_idx) {
        file.read(reinterpret_cast<char*>(q.mapped_data + static_cast<size_t>(row_idx) * this->d), static_cast<size_t>(this->d) * sizeof(float));
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
 * @param weights The output vector of mat objects (std::vector<mat>&) to store the MLP weights [layer]. Passed by reference.
 * @param path2file The full path to the specific binary MLP weights file (e.g., "path/to/hor.bin").
 */
void transformer::getmlp(int blockCount, int i, int j, std::vector<mat>& weights, std::string path2file)
{
    // --- Validation ---
     if (blockCount <= 0) {
        throw std::out_of_range("getmlp: blockCount must be greater than 0.");
    }
     if (i < 0 || i >= this->x || j < 0 || j >= this->y) {
         throw std::out_of_range("getmlp: Index (i, j) = (" + std::to_string(i) + ", " + std::to_string(j) +
                                ") is out of bounds for grid dimensions (x, y) = (" + std::to_string(this->x) + ", " + std::to_string(this->y) + ").");
    }
    if (this->d <= 0 || this->l <= 0) {
        throw std::runtime_error("getmlp: Transformer dimensions d and l must be positive.");
    }

    // --- Calculate Offset ---
    // MLP size: l layers, d x d weights per layer
    size_t mlp_layer_element_count = static_cast<size_t>(this->d) * this->d;
    size_t mlp_total_element_count = static_cast<size_t>(this->l) * mlp_layer_element_count;
    size_t mlp_size_bytes = mlp_total_element_count * sizeof(float);
    size_t mlps_per_row_in_block = this->y;
    size_t mlps_per_block = static_cast<size_t>(this->x) * mlps_per_row_in_block;

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

    // Resize the output vector of mat objects
    weights.resize(this->l);

    for (int layer_idx = 0; layer_idx < this->l; ++layer_idx) {
        // Assign a new mat object for the current layer.
        weights[layer_idx] = mat(this->d, this->d);

        // Read data for this layer directly into the mapped_data of weights[layer_idx]
        for (int row_idx = 0; row_idx < this->d; ++row_idx) {
            file.read(reinterpret_cast<char*>(weights[layer_idx].mapped_data + static_cast<size_t>(row_idx) * this->d),
                      static_cast<size_t>(this->d) * sizeof(float));
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
 * @param q The output matrix (mat&) to store the data. Passed by reference.
 * @param path2file The full path to the specific binary cache file (e.g., "path/to/QK.bin").
 */
void transformer::getmat(int blockCount, int i, int j, mat& q, std::string path2file, int& row, int& column)
{
    // --- Validation ---
    if (blockCount <= 0) {
        throw std::out_of_range("getcache: blockCount must be greater than 0.");
    }
     if (i < 0 || i >= this->x || j < 0 || j >= this->y) {
         throw std::out_of_range("getcache: Index (i, j) = (" + std::to_string(i) + ", " + std::to_string(j) +
                                ") is out of bounds for grid dimensions (x, y) = (" + std::to_string(this->x) + ", " + std::to_string(this->y) + ").");
    }
    if (row <= 0 || column <= 0) {
        throw std::runtime_error("getmat: Matrix dimensions 'row' and 'column' must be positive.");
    }

    // --- Calculate Offset ---
    // Cache matrix size: d rows, h columns
    size_t cache_element_count = static_cast<size_t>(row) * column;
    size_t cache_size_bytes = cache_element_count * sizeof(float);
    size_t caches_per_row_in_block = this->y;
    size_t caches_per_block = static_cast<size_t>(this->x) * caches_per_row_in_block;

    // Calculate the index of the target cache within the flattened sequence of all caches
    size_t target_cache_index = (static_cast<size_t>(blockCount - 1) * caches_per_block) // Caches in previous blocks
                               + (static_cast<size_t>(i) * caches_per_row_in_block)     // Caches in previous rows of current block
                               + static_cast<size_t>(j);                                // Caches in current row before target

    size_t offset_bytes = target_cache_index * cache_size_bytes;

    // --- File Operations ---
    std::ifstream file(path2file, std::ios::binary | std::ios::in);
    if (!file.is_open()) {
        throw std::runtime_error("getmat: Could not open file: " + path2file);
    }

    // Seek to position
    file.seekg(offset_bytes, std::ios::beg);
    if (!file) {
        file.close();
        throw std::runtime_error("getmat: Failed to seek in file: " + path2file + " to offset " + std::to_string(offset_bytes));
    }

    // Assign a new mat object to q.
    q = mat(row, column);

    // Read data directly into the mapped_data of q
    for (int row_idx = 0; row_idx < row; ++row_idx) {
        file.read(reinterpret_cast<char*>(q.mapped_data + static_cast<size_t>(row_idx) * column),
                  static_cast<size_t>(column) * sizeof(float));
        if (!file) {
            file.close();
            throw std::runtime_error("getcache: Error reading data at block " + std::to_string(blockCount) +
                                     ", pos (" + std::to_string(i) + ", " + std::to_string(j) +
                                     "), row " + std::to_string(row_idx) + " from " + path2file);
        }
    }
    file.close();
}
