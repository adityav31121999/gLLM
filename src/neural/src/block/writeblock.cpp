
#include "include/block.hpp"
#include <fstream>
#include <filesystem>
#include <random>

/**
 * @brief provide random values to all the matrices and mlps in block
 * @param type type of initialisation (He, Xavier, Glorot)
 * @note this function is used for random value initialisation
 */
void block::randomValuesForBlock(int type) {
    unsigned long long offset = 0;
    int row = MATHEIGHTS;
    int col = EMBEDDING;
    int mRow = EMBEDDING;
    int mCol = EMBEDDING;
    int layer = LAYERS_MLP-1;
    float scale1 = 0.0f;
    float scale2 = 0.0f;

    // Input features = col, Output features = row
    float fan_in_MQ_MK = static_cast<float>(col);
    float fan_out_MQ_MK = static_cast<float>(row);
    // Input features = row, Output features = col
    float fan_in_MV_MH = static_cast<float>(row);
    float fan_out_MV_MH = static_cast<float>(col);
    // Input features = mCol, Output features = mRow
    float fan_in_MLP = static_cast<float>(mCol);
    float fan_out_MLP = static_cast<float>(mRow);

    // --- He Initialization (Best for ReLU and its variants) ---
    // Normal Distribution: stdDev = sqrt(2 / fan_in)
    if(type == 1) {
        if (fan_in_MQ_MK > 0) {
            scale1 = std::sqrt(2.0f / fan_in_MQ_MK);
        } else {
            // Handle error: fan_in is zero or negative. Assign a safe default or throw.
            scale1 = 0.01f; // Small arbitrary positive value to prevent division by zero/NaN
        }

        if (fan_in_MLP > 0) {
            scale2 = std::sqrt(2.0f / fan_in_MLP);
        } else {
            scale2 = 0.01f;
        }
    }
    // --- Xavier/Glorot Initialization (Good for Sigmoid, Tanh, or general linear layers) ---
    // Normal Distribution: stdDev = sqrt(2 / (fan_in + fan_out))
    else if(type == 2) {
        if ((fan_in_MQ_MK + fan_out_MQ_MK) > 0) {
            scale1 = std::sqrt(2.0f / (fan_in_MQ_MK + fan_out_MQ_MK));
        } else {
            scale1 = 0.01f;
        }

        if ((fan_in_MLP + fan_out_MLP) > 0) {
            scale2 = std::sqrt(2.0f / (fan_in_MLP + fan_out_MLP));
        } else {
            scale2 = 0.01f;
        }
    }
    // --- LeCun Initialization (Less common, for Sigmoid/Tanh, only considers fan_in) ---
    // Normal Distribution: stdDev = sqrt(1 / fan_in)
    else if (type == 3) { // type == 2 (or default for any other value of type)
        if (fan_in_MQ_MK > 0) {
            scale1 = std::sqrt(1.0f / fan_in_MQ_MK);
        } else {
            scale1 = 0.01f;
        }

        if (fan_in_MLP > 0) {
            scale2 = std::sqrt(1.0f / fan_in_MLP);
        } else {
            scale2 = 0.01f;
        }
    }
    else {
        std::cout << "Not selected He/Xavier/LeCunn Initialisation, going with ";
        scale1 = 0.01f;
        scale2 = 0.0025f;
        std::cout << "scale1 = " << scale1 << " and scale2 = " << scale2 << std::endl;
    }
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist1(0, scale1);
    std::normal_distribution<float> dist2(0, scale2);
    for (int i = 0; i < x; i++) {
        for (int j = 0; j < y; j++) {
            // add random values to mapped data of MQ, MK, MV, MH
            for(int k = 0; k < b[i][j].MQ.row; k++) {
                for(int l = 0; l < b[i][j].MQ.col; l++) {
                    b[i][j].MQ(k, l) = dist1(gen);
                    b[i][j].MK(k, l) = dist1(gen);
                    b[i][j].MV(l, k) = dist1(gen);
                    b[i][j].MH(l, k) = dist1(gen);
                }
            }
            // add random values to weights to hor and ver mlps
            for(int k = 0; k < b[i][j].hor.weights.size(); k++) {
                for(int l = 0; l < b[i][j].hor.weights[k].row; l++) {
                    for(int m = 0; m < b[i][j].hor.weights[k].col; m++) {
                        b[i][j].hor.weights[k](l, m) = dist2(gen);
                    }
                }
            }
            for(int k = 0; k < b[i][j].ver.weights.size(); k++) {
                for(int l = 0; l < b[i][j].ver.weights[k].row; l++) {
                    for(int m = 0; m < b[i][j].ver.weights[k].col; m++) {
                        b[i][j].ver.weights[k](l, m) = dist2(gen);
                    }
                }
            }
            // serialise these values to the bin file
            b[i][j].MQ.serialise(offset, blockFilePath);
            b[i][j].MK.serialise(offset+row+col, blockFilePath);
            b[i][j].MV.serialise(offset+(2*(row+col)), blockFilePath);
            b[i][j].MH.serialise(offset+(3*(row+col)), blockFilePath);
            b[i][j].hor.serialise(offset+(4*(row+col)), blockFilePath);
            b[i][j].ver.serialise(offset+(4*(row+col))+mRow+mCol, blockFilePath);
            offset +=(b[i][j].MQ.row*b[i][j].MQ.col) 
                    + (b[i][j].MK.row*b[i][j].MK.col) 
                    + (b[i][j].MV.row*b[i][j].MV.col)
                    + (b[i][j].MH.row*b[i][j].MH.col) 
                    + (b[i][j].hor.weights.size()*b[i][j].hor.weights[0].row*b[i][j].hor.weights[0].col)
                    + (b[i][j].ver.weights.size()*b[i][j].ver.weights[0].row*b[i][j].ver.weights[0].col) 
                    + (b[i][j].K.row*b[i][j].K.col)
                    + (b[i][j].Q.row*b[i][j].Q.col) 
                    + b[i][j].EH.size() 
                    + (b[i][j].KdotQ.row*b[i][j].KdotQ.col)
                    + (b[i][j].EV.row*b[i][j].EV.col);
        }
    }
    std::cout << "Random Values initialised for first block of transformer" << std::endl;
}


/**
 * serialise block
 */
void block::serialise(const std::string& locationofbinfile) {
    int offset1 = 0;
    for(int i = 0; i < x; i ++) {
        for(int j = 0; j < y; j ++) {
            b[i][j].serialise(offset1, locationofbinfile);
            offset1 +=(b[i][j].MQ.row*b[i][j].MQ.col)   // 4 * row * col
                    + (b[i][j].MK.row*b[i][j].MK.col) 
                    + (b[i][j].MV.row*b[i][j].MV.col)
                    + (b[i][j].MH.row*b[i][j].MH.col) 
                    + (b[i][j].hor.weights.size()*b[i][j].hor.weights[0].row*b[i][j].hor.weights[0].col)
                    + (b[i][j].ver.weights.size()*b[i][j].ver.weights[0].row*b[i][j].ver.weights[0].col) 
                    + (b[i][j].K.row*b[i][j].K.col)
                    + (b[i][j].Q.row*b[i][j].Q.col) 
                    + b[i][j].EH.size() 
                    + (b[i][j].KdotQ.row*b[i][j].KdotQ.col)
                    + (b[i][j].EV.row*b[i][j].EV.col);
        }
    }
    std::cout << "Serialised block to file: " << locationofbinfile << std::endl;
}


/**
 * deserialise block
 */
void block::deserialise(const std::string& locationofbinfile) {
    int offset1 = 0;
    for(int i = 0; i < x; i ++) {
        for(int j = 0; j < y; j ++) {
            //std::cout << "deserialising block " << i << " " << j << " " << offset1 << std::endl;
            b[i][j].deserialise(offset1, locationofbinfile);
            offset1 +=(b[i][j].MQ.row*b[i][j].MQ.col) 
                    + (b[i][j].MK.row*b[i][j].MK.col) 
                    + (b[i][j].MV.row*b[i][j].MV.col)
                    + (b[i][j].MH.row*b[i][j].MH.col) 
                    + (b[i][j].hor.weights.size()*b[i][j].hor.weights[0].row*b[i][j].hor.weights[0].col)
                    + (b[i][j].ver.weights.size()*b[i][j].ver.weights[0].row*b[i][j].ver.weights[0].col) 
                    + (b[i][j].K.row*b[i][j].K.col)
                    + (b[i][j].Q.row*b[i][j].Q.col) 
                    + b[i][j].EH.size() 
                    + (b[i][j].KdotQ.row*b[i][j].KdotQ.col)
                    + (b[i][j].EV.row*b[i][j].EV.col);
        }
    }
    std::cout << "Deserialised block from file: " << locationofbinfile << std::endl;
}
