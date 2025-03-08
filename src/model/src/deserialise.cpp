
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
    
    // Read and verify model metadata
    int metadata[11];  // Array to hold all metadata values
    size_t read = fread(metadata, sizeof(int), 11, file);
    if (read != 11) {
        throw std::runtime_error("Failed to read model metadata");
    }

    // Assign metadata with validation
    a.tCount = metadata[0];
    a.m = metadata[2];
    a.x = metadata[3];
    a.y = metadata[4];
    a.n = metadata[5];
    a.d = metadata[6];
    a.h = metadata[7];
    a.l = metadata[8];
    a.totalParams = metadata[9];
    a.total = metadata[10];

    // Validate critical dimensions
    if (a.m <= 0 || a.x <= 0 || a.y <= 0 || a.n <= 0 || 
        a.d <= 0 || a.h <= 0 || a.l <= 0) {
        throw std::runtime_error("Invalid model dimensions in metadata");
    }

    // Deserialize transformer
    transformer& T = a.T;

    // Verify transformer block structure
    if (T.b.size() != static_cast<size_t>(a.m)) {
        throw std::runtime_error("Transformer block count mismatch");
    }

    // Deserialize attention blocks with validation
    for (auto& block : T.b) {
        if (block.b.size() != static_cast<size_t>(a.y)) {
            throw std::runtime_error("Attention layer count mismatch");
        }

        for (auto& attention_layer : block.b) {
            if (attention_layer.size() != static_cast<size_t>(a.x)) {
                throw std::runtime_error("Attention count mismatch in layer");
            }

            for (auto& att : attention_layer) {
                // Verify matrix dimensions before deserializing
                if (att.MQ.row != a.h || att.MQ.col != a.d ||
                    att.MK.row != a.h || att.MK.col != a.d ||
                    att.MV.row != a.h || att.MV.col != a.d ||
                    att.MH.row != a.h || att.MH.col != a.d) {
                    throw std::runtime_error("Attention matrix dimensions mismatch");
                }

                // Verify MLP dimensions
                if (att.ver.layers != static_cast<unsigned int>(a.l) || 
                    att.hor.layers != static_cast<unsigned int>(a.l) ||
                    att.ver.neurons != static_cast<unsigned int>(a.d) || 
                    att.hor.neurons != static_cast<unsigned int>(a.d)) {
                    throw std::runtime_error("MLP dimensions mismatch");
                }

                /**
                // Deserialize matrices
                deserialiseMAT(att.MQ, file, int, int);
                deserialiseMAT(att.MK, file);
                deserialiseMAT(att.MV, file);
                deserialiseMAT(att.MH, file);

                // Deserialize MLPs
                deserialiseMLP(att.ver, file);
                deserialiseMLP(att.hor, file);
                 */
            }
        }
    }
}
