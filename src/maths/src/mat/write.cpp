
#include "include/mat.hpp"
#include <fstream>
#include <filesystem>

/**
 * serialise the the matrix to a bin after offset number of float values
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
 * serialise the the matrix to a bin by appending from last
 */
void mat::serialise(const std::string& locationofbinfile) {
    // Open the file for binary write in append mode.
    // If the file does not exist, it will be created.
    // If it exists, new data will be appended to the end.
    std::fstream outFile(locationofbinfile, std::ios::binary | std::ios::app);
    if (!outFile.is_open()) {
        throw std::runtime_error("Failed to open file for appending (check permissions or path): " + locationofbinfile);
    }

    // Serialize matrix data
    size_t num_elements_in_mat_a = static_cast<size_t>(row) * static_cast<size_t>(col);

    if (num_elements_in_mat_a == 0) {
        // No data to write for a 0-element matrix.
        outFile.close();
        // Check for errors during close, especially if buffers needed flushing
        if (!outFile) {
            throw std::runtime_error("Error occurred while closing file (e.g., flush error) for empty matrix: " + locationofbinfile);
        }
        return;
    }

    // A non-zero size implies we need a valid data pointer.
    const float* data_ptr = is_shared_segment ? data_segment_start : mapped_data;
    if (data_ptr == nullptr) { // Should not happen for a valid non-empty matrix
        outFile.close();
        throw std::runtime_error("Matrix has non-zero size but data pointer is null. File: " + locationofbinfile);
    }

    size_t bytes_to_write = num_elements_in_mat_a * sizeof(float);

    // In std::ios::app mode, the output pointer is automatically positioned at the end of the file
    // before each write operation. Therefore, no explicit seekp is needed for appending.
    outFile.write(reinterpret_cast<const char*>(data_ptr), static_cast<std::streamsize>(bytes_to_write));
    
    if (!outFile) { // Check for errors after writing data
        outFile.close();
        throw std::runtime_error("Error writing matrix data to file: " + locationofbinfile);
    }

    outFile.close();
    if (!outFile) { // Check for errors that might occur during close (e.g., buffer flush failure)
        throw std::runtime_error("Error occurred while closing file (e.g., flush error): " + locationofbinfile);
    }
}


/**
 * deserialise the the bin to a matrix from offset number of float values
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
