
#include "include/mat.hpp"
#include <fstream>
#include <filesystem>

/**
 * FOR TRAINING:
 * serialise all the mats and mlps and vectors in this way:
 * MQ, MK, MV, MH
 * hor, ver,
 * K, Q, KdotQ
 * EH, EV
 */
void mat::serialise(unsigned long long offset, const std::string& locationofbinfile) {
    // Open the file for binary read and write, expecting it to exist and be pre-sized.
    std::fstream outFile(locationofbinfile, std::ios::binary | std::ios::in | std::ios::out);
    if (!outFile.is_open()) {
        throw std::runtime_error("Failed to open file for writing (it may not exist or permissions are wrong): " + locationofbinfile);
    }

    // Serialize matrix data
    size_t num_elements_in_mat_a = static_cast<size_t>(row) * static_cast<size_t>(col);

    if (num_elements_in_mat_a == 0 && (row > 0 || col > 0)) {
        // This case implies one dimension is zero, which is valid for an empty matrix.
        // No data to write for a 0-element matrix.
        outFile.close();
        return;
    }

    if (num_elements_in_mat_a > 0) {
        const float* data_ptr = is_shared_segment ? data_segment_start : mapped_data;
        if (data_ptr == nullptr && num_elements_in_mat_a > 0) { // Should not happen for a valid matrix
            outFile.close();
            throw std::runtime_error("Matrix has non-zero size but data pointer is null. File: " + locationofbinfile);
        }

        // Use the passed 'offset' (element offset) to calculate the byte offset.
        std::streamoff byte_offset_from_param = static_cast<std::streamoff>(static_cast<size_t>(offset) * sizeof(float));

        outFile.seekp(byte_offset_from_param, std::ios::beg);
        if (!outFile) { // Check for errors after seekp
            outFile.close();
            throw std::runtime_error("Error seeking in file: " + locationofbinfile + " to offset " + std::to_string(byte_offset_from_param));
        }

        size_t bytes_per_matrix = num_elements_in_mat_a * sizeof(float); // Size of the current matrix 'a'
        outFile.write(reinterpret_cast<const char*>(data_ptr), static_cast<std::streamsize>(bytes_per_matrix));
        
        if (!outFile) { // Check for errors after writing data
            outFile.close();
            throw std::runtime_error("Error writing matrix data to file: " + locationofbinfile);
        }
    }
    outFile.close();
    if (!outFile) { // Check for errors that might occur during close (e.g., buffer flush failure)
        throw std::runtime_error("Error occurred while closing file (e.g., flush error): " + locationofbinfile);
    }
}


/**
 * FOR TRAINING:
 * serialise all the mats and mlps and vectors in this way:
 * MQ, MK, MV, MH
 * hor, ver,
 * K, Q, KdotQ
 * EH, EV
 */
void mat::deserialise(unsigned long long offset, const std::string& locationofbinfile) {
    if (row == 0 || col == 0) {
        return;
    }
    float* data_ptr_to_fill = is_shared_segment ? data_segment_start : mapped_data;
    if (!data_ptr_to_fill) {
        throw std::runtime_error("Matrix data pointer is null before deserialising. File: " + locationofbinfile);
    }

    std::ifstream inFile(locationofbinfile, std::ios::binary | std::ios::ate);
    if (!inFile) {
        throw std::runtime_error("Failed to open monolithic training file for reading: " + locationofbinfile);
    }
    std::streamsize fileSize = inFile.tellg();
    inFile.seekg(0, std::ios::beg);

    // Use getOffset to find the absolute element offset in the monolithic file
    // For a mat, mlpCount in getOffset should be 1 (or 0 if it means "not an mlp")

    std::streamoff byte_offset = static_cast<std::streamoff>(static_cast<size_t>(offset) * sizeof(float));
    size_t bytes_to_read = static_cast<size_t>(row) * col * sizeof(float);

    if (bytes_to_read == 0) { // Should be caught by a.row/col check
        inFile.close();
        return;
    }

    if (byte_offset < 0 || byte_offset + static_cast<std::streamoff>(bytes_to_read) > fileSize) {
        inFile.close();
        throw std::runtime_error("Read operation exceeds monolithic file bounds. File: " + locationofbinfile +
                                 ", Offset(B): " + std::to_string(byte_offset) + ", Read(B): " + std::to_string(bytes_to_read) +
                                 ", Size(B): " + std::to_string(fileSize));
    }

    inFile.seekg(byte_offset, std::ios::beg);
    if (!inFile) {
        inFile.close();
        throw std::runtime_error("Error seeking in monolithic file: " + locationofbinfile + " to offset " + std::to_string(byte_offset));
    }

    inFile.read(reinterpret_cast<char*>(data_ptr_to_fill), static_cast<std::streamsize>(bytes_to_read));
    if (!inFile || inFile.gcount() != static_cast<std::streamsize>(bytes_to_read)) {
        inFile.close();
        throw std::runtime_error("Error reading matrix data from monolithic file: " + locationofbinfile);
    }
    inFile.close();
}


/**
 * @brief Creates a matrix with Xavier/Glorot initialization
 * @param row Number of rows
 * @param col Number of columns
 * @param use_gain Whether to use the gain parameter (default: false)
 * @param gain Scaling factor for the initialization (default: 1.0)
 * @return A new matrix with Xavier/Glorot initialized values
 */
mat mat::initXavier(int row, int col, bool use_gain, float gain) {
    // Xavier/Glorot initialization: scale = sqrt(2.0 / (fan_in + fan_out))
    // For linear layers, fan_in is the input dimension and fan_out is the output dimension
    float scale = 1.0f;
    if (row > 0 && col > 0) {
        scale = std::sqrt(2.0f / (row + col));
    }
    if (use_gain) {
        scale *= gain;
    }
    
    // Create a random matrix with values in [-scale, scale]
    mat result(row, col);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-scale, scale);
    
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            result(i, j) = dist(gen);
        }
    }
    
    return result;
}

/**
 * @brief Creates a matrix with He initialization
 * @param row Number of rows (output dimension for linear layers)
 * @param col Number of columns (input dimension for linear layers)
 * @param use_gain Whether to use the gain parameter (default: false)
 * @param gain Scaling factor for the initialization (default: 1.0)
 * @return A new matrix with He initialized values
 */
mat mat::initHe(int row, int col, bool use_gain, float gain) {
    // He initialization: scale = sqrt(2.0 / fan_in)
    // For linear layers, fan_in is the input dimension (col)
    float scale = 1.0f;
    if (col > 0) {
        scale = std::sqrt(2.0f / col);
    }
    if (use_gain) {
        scale *= gain;
    }
    
    // Create a random matrix with normal distribution N(0, scale^2)
    mat result(row, col);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, scale);
    
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            result(i, j) = dist(gen);
        }
    }
    
    return result;
}
