
#include "include/block.hpp"
#include <fstream>
#include <filesystem>
#include <random>

/**
 * @brief provide random values to all the matrices and mlps in block
 * @param min minimum value for random number generation
 * @param max maximum value for random number generation
 * @note this function is used for training
 */
void block::randomValuesForBlock(float min, float max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min, max);
    long long int offset = 0;
    int row = b[0][0].MQ.row;
    int col = b[0][0].MQ.col;
    int mRow = b[0][0].hor.weights[0].row;
    int mCol = b[0][0].hor.weights[0].col;
    int layer = b[0][0].hor.weights.size();
    for (int i = 0; i < x; i++) {
        for (int j = 0; j < y; j++) {
            // add random values to mapped data of MQ, MK, MV, MH
            for(int k = 0; k < b[i][j].MQ.row; k++) {
                for(int l = 0; l < b[i][j].MQ.col; l++) {
                    b[i][j].MQ(k, l) = dis(gen);
                    b[i][j].MK(k, l) = dis(gen);
                    b[i][j].MV(l, k) = dis(gen);
                    b[i][j].MH(l, k) = dis(gen);
                }
            }
            // add random values to weights to hor and ver mlps
            for(int k = 0; k < b[i][j].hor.weights.size(); k++) {
                for(int l = 0; l < b[i][j].hor.weights[k].row; l++) {
                    for(int m = 0; m < b[i][j].hor.weights[k].col; m++) {
                        b[i][j].hor.weights[k](l, m) = dis(gen);
                        b[i][j].ver.weights[k](l, m) = dis(gen);
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
                    + (b[i][j].KdotQ.row*b[i][j].KdotQ.col);
        }
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
void block::serialise(const std::string& locationofbinfile) {
    int offset1 = 0;
    for(int i = 0; i < x; i ++) {
        for(int j = 0; j < y; j ++) {
            b[i][j].serialise(offset1, locationofbinfile);
            offset1 +=(b[i][j].MQ.row*b[i][j].MQ.col) 
                    + (b[i][j].MK.row*b[i][j].MK.col) 
                    + (b[i][j].MV.row*b[i][j].MV.col)
                    + (b[i][j].MH.row*b[i][j].MH.col) 
                    + (b[i][j].hor.weights.size()*b[i][j].hor.weights[0].row*b[i][j].hor.weights[0].col)
                    + (b[i][j].ver.weights.size()*b[i][j].ver.weights[0].row*b[i][j].ver.weights[0].col) 
                    + (b[i][j].K.row*b[i][j].K.col)
                    + (b[i][j].Q.row*b[i][j].Q.col) 
                    + b[i][j].EH.size() 
                    + (b[i][j].KdotQ.row*b[i][j].KdotQ.col);
        }
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
