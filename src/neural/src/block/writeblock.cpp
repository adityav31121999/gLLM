#include "include/block.hpp"
#include <fstream>
#include <filesystem>
#include <random>

/**
 * @brief provide random values to all the matrices and mlps in block.
 *      1. n = 1 : normal distribution with mean = x1 and std. deviation = x2
 *      2. n = 2 : uniform distribution with range = x1 to x2
 *      3. n = 3 : exponential distribution with mean = x1
 *      4. n = 4 : poisson distribution with mean = x1
 *      5. n = 5 : gamma distribution with shape = x1 and scale = x2
 * @param x1 first value
 * @param x2 second value
 * @note this function is used for training
 */
void block::randomValuesForBlock(float x1, float x2, int n) {
    if (n == 1) {
        std::cout << "Values of block initialised with normal_distribution, mean & std. deviation: " << x1 << " and " << x2 << std::endl;
    }
    else if (n == 2) {
        std::cout << "Values of block initialised with uniform_real_distribution, range: " << x1 << " and " << x2 << std::endl;
    }
    else if (n == 3) {
        std::cout << "Values of block initialised with exponential_distribution, mean: " << x1 << std::endl;
    }
    else if (n == 4) {
        std::cout << "Values of block initialised with poisson_distribution, mean: " << x1 << std::endl;
    }
    else if (n == 5) {
        std::cout << "Values of block initialised with gamma_distribution, shape & scale: " << x1 << " and " << x2 << std::endl;
    }
    else {
        throw std::runtime_error("randomValuesForBlock: Invalid distribution type 'n'. Must be:\n"
                                 "1: normal distribution with mean = x1 and std. deviation = x2\n"
                                 "2: uniform distribution with range = x1 to x2\n"
                                 "3: exponential distribution with mean = x1\n"
                                 "4: poisson distribution with mean = x1\n"
                                 "5: gamma distribution with shape = x1 and scale = x2");
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    unsigned long long offset = 0;
    int row = b[0][0].MQ.row;
    int col = b[0][0].MQ.col;
    int mRow = b[0][0].hor.weights[0].row;
    int mCol = b[0][0].hor.weights[0].col;
    int layer = b[0][0].hor.weights.size();

    for (int i = 0; i < x; i++) {
        for (int j = 0; j < y; j++) {
            std::function<float()> random_generator;
            if(n == 1) {
                std::normal_distribution<float> normal_dis(x1, x2);
                random_generator = [&]() { return normal_dis(gen); };
                // std::cout << "Values of block initialised with normal_distribution, mean std. deviation: " << x1 << " and " << x2 << std::endl;
            }
            else if(n == 2) {
                std::uniform_real_distribution<float> uniform_dis(x1, x2);
                random_generator = [&]() { return uniform_dis(gen); };
                // std::cout << "Values of block initialised with uniform_real_distribution, range: " << x1 << " and " << x2 << std::endl;
            }
            else if (n == 3) {
                std::exponential_distribution<float> exp_dis(1.0f / x1);
                random_generator = [&]() { return exp_dis(gen); };
                // std::cout << "Values of block initialised with exponential_distribution, mean: " << x1 << std::endl;
            }
            else if (n == 4) {
                std::poisson_distribution<int> poisson_dis(x1);
                random_generator = [&]() { return static_cast<float>(poisson_dis(gen)); };
                // std::cout << "Values of block initialised with poisson_distribution, mean: " << x1 << std::endl;
            }
            else if (n == 5) {
                std::gamma_distribution<float> gamma_dis(x1, x2);
                random_generator = [&]() { return static_cast<float>(gamma_dis(gen)); };
                // std::cout << "Values of block initialised with gamma_distribution, shape and scale: " << x1 << " and " << x2 << std::endl;
            }

            // add random values to mapped data of MQ, MK, MV, MH
            for(int k = 0; k < b[i][j].MQ.row; k++) {
                for(int l = 0; l < b[i][j].MQ.col; l++) {
                    b[i][j].MQ(k, l) = random_generator();
                    b[i][j].MK(k, l) = random_generator();
                    b[i][j].MV(l, k) = random_generator();
                    b[i][j].MH(l, k) = random_generator();
                }
            }

            // add random values to weights to hor and ver mlps
            for(int k = 0; k < b[i][j].hor.weights.size(); k++) {
                for(int l = 0; l < b[i][j].hor.weights[k].row; l++) {
                    for(int m = 0; m < b[i][j].hor.weights[k].col; m++) {
                        b[i][j].hor.weights[k](l, m) = random_generator();
                        b[i][j].ver.weights[k](l, m) = random_generator();
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
    serialise(blockFilePath);
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
    std::cout << "Serialised block to file: " << locationofbinfile << std::endl;
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
