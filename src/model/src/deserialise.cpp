
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
        size_t read = fread(a.a[i].data(), sizeof(float), col, file);
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
            layer.resize(neurons, std::vector<float>(neurons));
        }
    }
    
    // Read weights layer by layer with verification
    for (unsigned int l = 0; l < layers; ++l) {
        for (unsigned int i = 0; i < neurons; ++i) {
            if (a.weights[l][i].size() != neurons) {
                a.weights[l][i].resize(neurons);
            }
            size_t read = fread(a.weights[l][i].data(), sizeof(float), neurons, file);
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
    
    // Read model metadata
    fread(&a.m, sizeof(int), 1, file);            // number of blocks
    fread(&a.x, sizeof(int), 1, file);            // incomplete attentions
    fread(&a.y, sizeof(int), 1, file);            // layers of partial attention
    fread(&a.n, sizeof(int), 1, file);            // total tokens
    fread(&a.d, sizeof(int), 1, file);            // token dimension
    fread(&a.h, sizeof(int), 1, file);            // height of matrices
    fread(&a.l, sizeof(int), 1, file);            // layers of mlp
    fread(&a.totalParams, sizeof(int), 1, file);  // total parameters
    fread(&a.total, sizeof(int), 1, file);        // total token limit
    
    // Initialize the transformer based on read parameters
    a.T = transformer(a.m, a.x, a.y, a.n, a.d, a.h, a.l);
    
    // Deserialize transformer by deserializing attention class
    for (auto& block : a.T.t) {
        // For each block, deserialize its attention layers
        for (auto& alay : block.b) {
            // For each attention layer, deserialize each attention's 4mat and 2mlp
            for (auto& att : alay) {
                // Deserialize matrices
                deserialiseMAT(att.MQ, file, a.d, a.h);   // qkrow * qkcol
                deserialiseMAT(att.MK, file, a.d, a.h);   // qkrow * qkcol
                deserialiseMAT(att.MV, file, a.h, a.d);   // vhrow * vhcol
                deserialiseMAT(att.MH, file, a.h, a.d);   // vhrow * vhcol

                // Deserialize MLPs
                deserialiseMLP(att.ver, file, a.l, a.d);  // (d * d) * l
                deserialiseMLP(att.hor, file, a.l, a.d);  // (d * d) * l
            }
        }
    }
    
    // Update model info structure with the loaded parameters
    /**
    a.info.d = a.d;
    a.info.qkrow = a.d;
    a.info.qkcol = a.h;
    a.info.vhrow = a.h;
    a.info.vhcol = a.d;
    a.info.layer = a.l;
    a.info.nIAs = a.y;
    a.info.nPAs = a.x;
    a.info.nCAs = a.m;
    a.info.paramsIA = (4 * a.h * a.d) + (2 * a.d * a.d * a.l);
    a.info.totalParams = a.totalParams;
     */
    
    std::cout << "Model deserialized successfully." << std::endl;
}
