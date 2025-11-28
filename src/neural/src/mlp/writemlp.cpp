
#include "include/mlp.hpp"
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
void mlp::serialise(unsigned long long offset, const std::string& locationofbinfile) {
    std::fstream outFile(locationofbinfile, std::ios::binary | std::ios::in | std::ios::out);
    if (!outFile.is_open()) {
        throw std::runtime_error("Failed to open file for writing MLP (it may not exist or permissions are wrong): " + locationofbinfile);
    }

    size_t total_floats_per_mlp_structure = 0;
    if (num_layers > 1) {
        for (const auto& weight_matrix : weights) {
            total_floats_per_mlp_structure += static_cast<size_t>(weight_matrix.row) * weight_matrix.col;
        }
    }

    if (!gweights.empty()) {
        if (gweights.size() == weights.size()) {
            for (const auto& grad_matrix : gweights) {
                total_floats_per_mlp_structure += static_cast<size_t>(grad_matrix.row) * grad_matrix.col;
            }
        }
        else if (!weights.empty()) {
             outFile.close();
             throw std::runtime_error("MLP gweights size mismatch with weights. File: " + locationofbinfile);
        }
    }

    if (total_floats_per_mlp_structure == 0 && num_layers > 1) {
        outFile.close();
        return; 
    }
    if (total_floats_per_mlp_structure == 0) {
        outFile.close();
        return;
    }


    std::streamoff mlp_start_byte_offset_from_param = static_cast<std::streamoff>(static_cast<size_t>(offset) * sizeof(float));

    outFile.seekp(mlp_start_byte_offset_from_param, std::ios::beg);
    if (!outFile) {
        outFile.close();
        throw std::runtime_error("Error seeking in MLP file: " + locationofbinfile + " to offset " + std::to_string(mlp_start_byte_offset_from_param));
    }

    // Serialize weight matrices
    for (const auto& weight_matrix : weights) {
        const float* data_ptr = weight_matrix.is_shared_segment ? weight_matrix.data_segment_start : weight_matrix.mapped_data;
        size_t num_elements = static_cast<size_t>(weight_matrix.row) * weight_matrix.col;
        if (num_elements > 0) {
            if (data_ptr == nullptr) { outFile.close(); throw std::runtime_error("MLP weight matrix data pointer is null. File: " + locationofbinfile); }
            outFile.write(reinterpret_cast<const char*>(data_ptr), static_cast<std::streamsize>(num_elements * sizeof(float)));
            if (!outFile) { outFile.close(); throw std::runtime_error("Error writing MLP weight matrix data to file: " + locationofbinfile); }
        }
    }

    // Serialize gradient matrices if in training mode and they exist
    if (!gweights.empty()) {
        for (const auto& grad_matrix : gweights) {
            const float* data_ptr = grad_matrix.is_shared_segment ? grad_matrix.data_segment_start : grad_matrix.mapped_data;
            size_t num_elements = static_cast<size_t>(grad_matrix.row) * grad_matrix.col;
            if (num_elements > 0) {
                if (data_ptr == nullptr) { outFile.close(); throw std::runtime_error("MLP gradient matrix data pointer is null. File: " + locationofbinfile); }
                outFile.write(reinterpret_cast<const char*>(data_ptr), static_cast<std::streamsize>(num_elements * sizeof(float)));
                if (!outFile) { outFile.close(); throw std::runtime_error("Error writing MLP gradient matrix data to file: " + locationofbinfile); }
            }
        }
    }

    outFile.close();
    if (!outFile) {
        throw std::runtime_error("Error occurred while closing MLP file (e.g., flush error): " + locationofbinfile);
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
void mlp::deserialise(unsigned long long offset, const std::string& locationofbinfile) {
    if (weights.empty()) {
        return;
    }

    std::ifstream inFile(locationofbinfile, std::ios::binary | std::ios::ate);
    if (!inFile) {
        throw std::runtime_error("Failed to open monolithic training file for reading: " + locationofbinfile);
    }
    std::streamsize fileSize = inFile.tellg();
    inFile.seekg(0, std::ios::beg);

    std::streamoff current_byte_offset = static_cast<std::streamoff>(static_cast<size_t>(offset) * sizeof(float));

    for (size_t i = 0; i < weights.size(); ++i) {
        mat& w = weights[i];
        if (w.row == 0 || w.col == 0) continue;

        float* data_ptr_to_fill = w.is_shared_segment ? w.data_segment_start : w.mapped_data;
        if (!data_ptr_to_fill) {
            inFile.close();
            throw std::runtime_error("MLP weight matrix data pointer is null. File: " + locationofbinfile + ", W_idx: " + std::to_string(i));
        }

        size_t bytes_to_read = static_cast<size_t>(w.row) * w.col * sizeof(float);
        if (bytes_to_read == 0) continue;

        if (current_byte_offset < 0 || current_byte_offset + static_cast<std::streamoff>(bytes_to_read) > fileSize) {
            inFile.close();
            throw std::runtime_error("MLP weight read exceeds monolithic file bounds. File: " + locationofbinfile +
                                     ", W_idx: " + std::to_string(i) + ", Offset(B): " + std::to_string(current_byte_offset) +
                                     ", Read(B): " + std::to_string(bytes_to_read) + ", Size(B): " + std::to_string(fileSize));
        }

        inFile.seekg(current_byte_offset, std::ios::beg);
        if (!inFile) {
            inFile.close();
            throw std::runtime_error("Error seeking for MLP weight in monolithic file: " + locationofbinfile + " to offset " + std::to_string(current_byte_offset));
        }

        inFile.read(reinterpret_cast<char*>(data_ptr_to_fill), static_cast<std::streamsize>(bytes_to_read));
        if (!inFile || inFile.gcount() != static_cast<std::streamsize>(bytes_to_read)) {
            inFile.close();
            throw std::runtime_error("Error reading MLP weight data from monolithic file: " + locationofbinfile + ", W_idx: " + std::to_string(i));
        }
        current_byte_offset += static_cast<std::streamoff>(bytes_to_read);
    }
    inFile.close();
}
