
#include "include/model_fs.hpp"
#include <iostream>
#include <cstdio>
#include <stdexcept>

/**
 * @brief serialising MAT class
 * @param a matrix to be serialised
 * @param file FILE pointer to write to
 */
void serialiseMAT(const mat& a, FILE* file) {
    if (!file) {
        throw std::runtime_error("File pointer is null");
    }
    
    // Write matrix elements row by row in binary format
    for (int i = 0; i < a.row; i++) {
        fwrite(a.a[i].data(), sizeof(double), a.col, file);
    }
}

/**
 * @brief serialising MLP class
 * @param a mlp to be serialised
 * @param file FILE pointer to write to
 * @param in number of neurons in each layer
 * @param layers number of layers in MLP
 */
void serialiseMLP(const mlp& a, FILE* file, int in, int layers) {
    if (!file) {
        throw std::runtime_error("File pointer is null");
    }
    
    // Write weights layer by layer
    for (unsigned int l = 0; l < layers; ++l) {
        for (unsigned int i = 0; i < in; ++i) {
            fwrite(a.weights[l][i].data(), sizeof(double), in, file);
        }
    }
}

/**
 * @brief serialising model parameters
 * @param a model to be serialised
 */
void serialiseModel(const model& a) {
    if (!a.file) {
        throw std::runtime_error("Model file pointer is null");
    }

    // from file to a.file via pointer reference
    FILE* file = a.file;        // pointer reference to original file

    // Write model metadata
    fwrite(&a.tCount, sizeof(int), 1, file);       // transformer count
    fwrite(&a.m, sizeof(int), 1, file);            // number of blocks
    fwrite(&a.x, sizeof(int), 1, file);            // incomplete attentions
    fwrite(&a.y, sizeof(int), 1, file);            // layers of partial attention
    fwrite(&a.n, sizeof(int), 1, file);            // total tokens
    fwrite(&a.d, sizeof(int), 1, file);            // token dimension
    fwrite(&a.h, sizeof(int), 1, file);            // height of matrices
    fwrite(&a.l, sizeof(int), 1, file);            // layers of mlp
    fwrite(&a.totalParams, sizeof(int), 1, file);  // total parameters
    fwrite(&a.total, sizeof(int), 1, file);        // total token limit

    // Serialize transformer based on transformer count
    if (a.tCount == 1) {
        // Serialize single transformer
        const transformer& T = a.T;
        for (const auto& block : T.b) {
            // For each block, serialize its attention layers
            for (const auto& alay : block.b) {
                // For each attention layer, serialise each attention's 4mat and 2mlp
                for (const auto& att : alay) {
                    // Serialize matrices
                    serialiseMAT(att.MQ, file);   // qkrow * qkcol
                    serialiseMAT(att.MK, file);   // qkrow * qkcol
                    serialiseMAT(att.MV, file);   // vhrow * vhcol
                    serialiseMAT(att.MH, file);   // vhrow * vhcol

                    // Serialize MLPs
                    serialiseMLP(att.ver, file, a.d, a.l);  // (d * d) * l
                    serialiseMLP(att.hor, file, a.d, a.l);  // (d * d) * l
                }
            }
        }
    } else {
        // Serialize multiple transformers
        for (const auto& T : a.Tg) {
            for (const auto& block : T.b) {
                // For each block, serialize its attention layers
                for (const auto& alay : block.b) {
                    // For each attention layer, serialise each attention's 4mat and 2mlp
                    for (const auto& att : alay) {
                        // Serialize matrices
                        serialiseMAT(att.MQ, file);   // qkrow * qkcol
                        serialiseMAT(att.MK, file);   // qkrow * qkcol
                        serialiseMAT(att.MV, file);   // vhrow * vhcol
                        serialiseMAT(att.MH, file);   // vhrow * vhcol

                        // Serialize MLPs
                        serialiseMLP(att.ver, file, a.d, a.l);  // (d * d) * l
                        serialiseMLP(att.hor, file, a.d, a.l);  // (d * d) * l
                    }
                }
            }
        }
    }
    
    std::cout << "Model serialized successfully." << std::endl;
}
