
// model.cpp: implementation of Model class
#include "include/model.hpp"
#include "include/model_fs.hpp"
#include <iostream>
#include <stdexcept>
#include <filesystem>

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
model::model(int m, int x, int y, int n, int d, int h, int l, int vocab) :
    m(m), x(x), y(y), n(n), d(d), h(h), l(l) {
    totalParams = m * x * y * ((4 * h * d) + (2 * d * d * l));
    T = transformer(m, x, y, n, d, h, l, vocab);  // transformer(int m, int x, int y, int n, int d, int h, int l);
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
model::model(int tCount, int m, int x, int y, int n, int d, int h, int l, int vocab) :
    tCount(tCount), m(m), x(x), y(y), n(n), d(d), h(h), l(l) {
    totalParams = tCount * m * x * y * ((4 * h * d) + (2 * d * d * l));
    Tg = std::vector<transformer>(tCount, transformer(m, x, y, n, d, h, l, vocab));
    // allocate float value block of size totalParams to file
    allocateMemory();
}

/**
 * @brief Constructor for single transformer model with learning rate
 * @param m number of blocks
 * @param x number of incomplete attentions in each partial attention
 * @param y number of layers of partial attention for complete attention block
 * @param n total tokens for each attention head
 * @param d token dimension
 * @param h height of MQ, MK and columns of MV, MH
 * @param l layers of mlp
 * @param learning learning rate for MLPs
 */
model::model(int m, int x, int y, int n, int d, int h, int l, float learning, int vocab) :
    m(m), x(x), y(y), n(n), d(d), h(h), l(l), learning(learning) {
    totalParams = m * x * y * ((4 * h * d) + (2 * d * d * l));
    T = transformer(m, x, y, n, d, h, l, vocab);
    T.setLearning(learning);  // Set learning rate for the transformer
    // allocate float value block of size totalParams to file
    allocateMemory();
}

/**
 * @brief Constructor for multiple transformer model with learning rate
 * @param tCount transformer count
 * @param m number of blocks
 * @param x number of incomplete attentions in each partial attention
 * @param y number of layers of partial attention for complete attention block
 * @param n total tokens for each attention head
 * @param d token dimension
 * @param h height of MQ, MK and columns of MV, MH
 * @param l layers of mlp
 * @param learning learning rate for MLPs
 */
model::model(int tCount, int m, int x, int y, int n, int d, int h, int l, float learning, int vocab) :
    tCount(tCount), m(m), x(x), y(y), n(n), d(d), h(h), l(l), learning(learning) {
    totalParams = tCount * m * x * y * ((4 * h * d) + (2 * d * d * l));
    Tg = std::vector<transformer>(tCount, transformer(m, x, y, n, d, h, l, vocab));
    // Set learning rate for all transformers
    for (auto& t : Tg) {
        t.setLearning(learning);
    }
    // allocate float value block of size totalParams to file
    allocateMemory();
}

/**
 * @brief allocate block of memory for given number of float values
 */
void model::allocateMemory() {
    // allocate float value block of size totalParams to file
    std::filesystem::path p = "model.bin";
    std::ofstream ofs(p, std::ios::binary | std::ios::out);
    if (!ofs) {
        throw std::runtime_error("Failed to open file for writing.");
    }
    std::vector<float> buffer(totalParams, 0.0f);
    ofs.write(reinterpret_cast<const char*>(buffer.data()), buffer.size() * sizeof(float));
    if (!ofs) {
        throw std::runtime_error("Failed to write to file.");
    }
    ofs.close();
    std::cout << "Float values written to file in binary format successfully." << std::endl;
}
