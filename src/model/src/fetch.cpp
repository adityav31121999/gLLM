
#include <iostream>
#include <fstream>      // For file stream operations
#include <stdexcept>
#include <vector>       // For std::vector
#include <filesystem>   // For path joining (C++17)
#include <string>       // For std::string
#include "include/model.hpp"


// fetch matrix from .bin file with offsets calculated and assign to matrix a
// trainLocation is the FULL PATH to the specific .bin file (e.g., "baseDir/bin/MQ.bin")
// x_idx is 1-based PA layer index, y_idx is 1-based head index
void model::fetchmat(mat& a, int blockCount, int x_idx, int y_idx, std::string& specificComponentBinFile) {
    if (a.row == 0 || a.col == 0) return;

    float* data_ptr_to_fill = a.is_shared_segment ? a.data_segment_start : a.mapped_data;
    if (!data_ptr_to_fill) {
        throw std::runtime_error("Matrix data pointer is null before fetching. File: " + specificComponentBinFile);
    }

    std::ifstream inFile(specificComponentBinFile, std::ios::binary | std::ios::ate);
    if (!inFile) {
        throw std::runtime_error("Failed to open component file for reading: " + specificComponentBinFile);
    }
    std::streamsize fileSize = inFile.tellg();
    inFile.seekg(0, std::ios::beg);

    size_t elements_per_instance = static_cast<size_t>(a.row) * a.col;
    if (elements_per_instance == 0) { inFile.close(); return; }

    // Calculate 0-based index of this specific matrix instance within its component file
    size_t instance_idx = static_cast<size_t>((blockCount-1)*x*y*(a.row*a.col) +    // for previous blocks
                          (x_idx-1)*y*(a.row*a.col) +       // for previous rows
                          (y_idx-1)*(a.row*a.col));         // for previous matrices

    size_t element_offset_in_component_file = instance_idx * elements_per_instance;
    std::streamoff byte_offset = static_cast<std::streamoff>(element_offset_in_component_file * sizeof(float));
    size_t bytes_to_read = elements_per_instance * sizeof(float);

    if (byte_offset < 0 || byte_offset + static_cast<std::streamoff>(bytes_to_read) > fileSize) {
        inFile.close();
        throw std::runtime_error("Fetch mat operation exceeds component file bounds. File: " + specificComponentBinFile +
                                 ", Offset(B): " + std::to_string(byte_offset) + ", Read(B): " + std::to_string(bytes_to_read) +
                                 ", Size(B): " + std::to_string(fileSize));
    }

    inFile.seekg(byte_offset, std::ios::beg);
    if (!inFile) {
        inFile.close();
        throw std::runtime_error("Error seeking in component file: " + specificComponentBinFile + " to offset " + std::to_string(byte_offset));
    }

    inFile.read(reinterpret_cast<char*>(data_ptr_to_fill), static_cast<std::streamsize>(bytes_to_read));
    if (!inFile || inFile.gcount() != static_cast<std::streamsize>(bytes_to_read)) {
        inFile.close();
        throw std::runtime_error("Error reading matrix data from component file: " + specificComponentBinFile);
    }
    inFile.close();
}

// trainLocation is the FULL PATH to the specific .bin file (e.g., "baseDir/bin/hor.bin")
void model::fetchmlp(mlp& network, int blockCount, int x_idx, int y_idx, std::string& specificComponentBinFile) {
    if (network.weights.empty()) return;

    std::ifstream inFile(specificComponentBinFile, std::ios::binary | std::ios::ate);
    if (!inFile) {
        throw std::runtime_error("Failed to open component file for reading: " + specificComponentBinFile);
    }
    std::streamsize fileSize = inFile.tellg();
    inFile.seekg(0, std::ios::beg);

    size_t elements_per_mlp_instance = 0;
    for (const auto& w_mat : network.weights) {
        elements_per_mlp_instance += static_cast<size_t>(w_mat.row) * w_mat.col;
    }
    if (elements_per_mlp_instance == 0) { inFile.close(); return; }

    size_t instance_idx = static_cast<size_t>((blockCount-1)*x*y*(network.layer_sizes[0]*network.layer_sizes[0]*network.layer_sizes.size()) +    // for previous blocks
                          (x_idx-1)*y*(network.layer_sizes[0]*network.layer_sizes[0]*network.layer_sizes.size()) +       // for previous rows
                          (y_idx-1)*(network.layer_sizes[0]*network.layer_sizes[0]*network.layer_sizes.size()));         // for previous mlps;

    size_t element_offset_mlp_start_in_component_file = instance_idx * elements_per_mlp_instance;
    std::streamoff current_byte_offset = static_cast<std::streamoff>(element_offset_mlp_start_in_component_file * sizeof(float));

    for (size_t i = 0; i < network.weights.size(); ++i) {
        mat& w = network.weights[i];
        if (w.row == 0 || w.col == 0) continue;

        float* data_ptr_to_fill = w.is_shared_segment ? w.data_segment_start : w.mapped_data;
        if (!data_ptr_to_fill) {
            inFile.close();
            throw std::runtime_error("MLP weight matrix data pointer is null for fetch. File: " + specificComponentBinFile + ", W_idx: " + std::to_string(i));
        }

        size_t bytes_to_read = static_cast<size_t>(w.row) * w.col * sizeof(float);
        if (bytes_to_read == 0) continue;

        if (current_byte_offset < 0 || current_byte_offset + static_cast<std::streamoff>(bytes_to_read) > fileSize) {
            inFile.close();
            throw std::runtime_error("Fetch MLP weight read exceeds component file bounds. File: " + specificComponentBinFile +
                                     ", W_idx: " + std::to_string(i) + ", Offset(B): " + std::to_string(current_byte_offset) +
                                     ", Read(B): " + std::to_string(bytes_to_read) + ", Size(B): " + std::to_string(fileSize));
        }

        inFile.seekg(current_byte_offset, std::ios::beg);
        if (!inFile) {
            inFile.close();
            throw std::runtime_error("Error seeking for MLP weight in component file: " + specificComponentBinFile + " to offset " + std::to_string(current_byte_offset));
        }

        inFile.read(reinterpret_cast<char*>(data_ptr_to_fill), static_cast<std::streamsize>(bytes_to_read));
        if (!inFile || inFile.gcount() != static_cast<std::streamsize>(bytes_to_read)) {
            inFile.close();
            throw std::runtime_error("Error reading MLP weight data from component file: " + specificComponentBinFile + ", W_idx: " + std::to_string(i));
        }
        current_byte_offset += static_cast<std::streamoff>(bytes_to_read);
    }
    inFile.close();
}
