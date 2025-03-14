
#include "include/model_fs.hpp"
#include <iostream>
#include <cstdio>
#include <stdexcept>

/**
 * @brief Deserialize matrix from binary file
 * @param a matrix to deserialize into
 * @param file FILE pointer to read from
 */
void deserialiseMAT(mat& a, FILE* file, int row, int col) {
    if (!file) {
        throw std::runtime_error("File pointer is null");
    }
    
    // Read matrix elements row by row with size verification
    for (int i = 0; i < row; i++) {
        if (a.a[i].size() != static_cast<size_t>(col)) {
            a.a[i].resize(col);
        }
        size_t read = fread(a.a[i].data(), sizeof(double), col, file);
        if (read != static_cast<size_t>(a.col)) {
            throw std::runtime_error("Matrix data read error: expected " + 
                                   std::to_string(a.col) + " elements, got " + 
                                   std::to_string(read));
        }
    }
}

/**
 * @brief Deserialize MLP from binary file
 * @param a MLP to deserialize into
 * @param file FILE pointer to read from
 */
void deserialiseMLP(mlp& a, FILE* file, int layers, int neurons) {
    if (!file) {
        throw std::runtime_error("File pointer is null");
    }

    // Ensure weights vector is properly sized
    if (a.weights.size() != layers) {
        a.weights.resize(layers);
        for (auto& layer : a.weights) {
            layer.resize(neurons, std::vector<double>(neurons));
        }
    }
    
    // Read weights layer by layer with verification
    for (unsigned int l = 0; l < layers; ++l) {
        for (unsigned int i = 0; i < neurons; ++i) {
            if (a.weights[l][i].size() != neurons) {
                a.weights[l][i].resize(neurons);
            }
            size_t read = fread(a.weights[l][i].data(), sizeof(double), neurons, file);
            if (read != neurons) {
                throw std::runtime_error("MLP weights read error at layer " + std::to_string(l) + 
                                            ", neuron " + std::to_string(i));
            }
        }
    }
}

/**
 * @brief Deserialize model from binary file
 * @param a model to deserialize into
 */
void deserialiseModel(model& a) {
    if (!a.file) {
        throw std::runtime_error("Model file pointer is null");
    }

    FILE* file = a.file;
}
