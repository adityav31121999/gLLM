
// model.cpp: implementation of Model class
#include "include/model.hpp"
#include "include/model_fs.hpp"
#include <iostream>
#include <stdexcept>
#include <filesystem>
#include "model.hpp"

/**
 * @brief Constructor for single transformer model
 * @param m number of blocks
 * @param x number of incomplete attentions in each partial attention
 * @param y number of layers of partial attention for complete attention block
 * @param n total tokens for each attention head
 * @param d token dimension
 * @param h height of MQ, MK and columns of MV, MH
 * @param l layers of mlp
 */
model::model(int m, int x, int y, int n, int d, int h, int l) {
    // initiate a memory block of float/double for this model
    this->m = m;
    this->x = x;
    this->y = y;
    this->n = n;
    this->d = d;
    this->h = h;
    this->l = l;
    totalParams = m * x * y * ((4 * h * d) + (2 * d * d * l));
    // allocate float value block of size totalParams to file
    allocateMemory();
}

/**
 * @brief Constructor for multiple transformer model
 * @param tCount transformer count
 * @param m number of blocks
 * @param x number of incomplete attentions in each partial attention
 * @param y number of layers of partial attention for complete attention block
 * @param n total tokens for each attention head
 * @param d token dimension
 * @param h height of MQ, MK and columns of MV, MH
 * @param l layers of mlp
 */
model::model(int tCount, int m, int x, int y, int n, int d, int h, int l) {
    // initiate a memory block of float/double for this model
    this->tCount = tCount;
    this->m = m;
    this->x = x;
    this->y = y;
    this->n = n;
    this->d = d;
    this->h = h;
    this->l = l;
    totalParams = tCount * m * x * y * ((4 * h * d) + (2 * d * d * l));
    // allocate float value block of size totalParams to file
    allocateMemory();
}

/**
 * @brief allocate block of memory for given number of float values
 */
void model::allocateMemory() {
    // Allocate memory for the float values
    float* floatArray = new float[totalParams];

    // Initialize the float values (for example purposes)
    for (int i = 0; i < totalParams; ++i) {
        floatArray[i] = static_cast<float>(0);
    }

    // Open the file in binary write mode
    errno_t err;
    err = fopen_s(&file, "output.bin", "wb");
    if (err != 0) {
        std::cerr << "Error opening file!" << std::endl;
        delete[] floatArray; // Free the allocated memory
        return;
    }

    // Write the float values to the file in binary format
    fwrite(floatArray, sizeof(float), totalParams, file);

    // Close the file
    fclose(file);

    // Free the allocated memory
    delete[] floatArray;

    std::cout << "Float values written to file in binary format successfully." << std::endl;
}
