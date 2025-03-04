
// model.cpp: implementation of Model class
#include "include/model.hpp"
#include "include/model_fs.hpp"
#include <iostream>
#include <stdexcept>
#include <filesystem>
#include "model.hpp"

/**
 * @brief Constructor for single language model
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
    // allocate float value block of size totalParams
}

/**
 * @brief Constructor for multiple language model
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
    // allocate float value block of size totalParams
}

/**
 * @brief Constructor for multipurpose model with multiple language model
 * @param mCount model count
 * @param tCount transformer count
 * @param m number of blocks
 * @param x number of incomplete attentions in each partial attention
 * @param y number of layers of partial attention for complete attention block
 * @param n total tokens for each attention head
 * @param d token dimension
 * @param h height of MQ, MK and columns of MV, MH
 * @param l layers of mlp
 */
model::model(int mCount, int tCount, int m, int x, int y, int n, int d, int h, int l) {
    // initiate a memory block of float/double for this model
    this->mCount = mCount;
    this->tCount = tCount;
    this->m = m;
    this->x = x;
    this->y = y;
    this->n = n;
    this->d = d;
    this->h = h;
    this->l = l;
    totalParams = mCount * tCount * m * x * y * ((4 * h * d) + (2 * d * d * l));
    // allocate float value block of size totalParams
}
