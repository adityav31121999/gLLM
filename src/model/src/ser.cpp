
#include "include/model.hpp"
#include <iostream>
#include <stdexcept>
#include <filesystem>
#include <neural.hpp>

/**
 * total values to be stored in each bin file:
 * 1. MQ, MK, MV, MH : m*x*y*matheight*d each
 * 2. hor, ver : m*x*y*d*d*l each
 * 3. QK, KH, QV : m*x*y*d*d each
 */

// calculate offset for model layout and its components
void model::calculateAndSetLayout() {
    matOffset = matheight * d;      // for single mat
    cacheOffset = d * d;            // for single cache
    mlpOffset = d * d * l;          // for single mlp
    // for training
    attentionOffset = (4*matOffset) + (2*mlpOffset) + d + (n*d) + (2*n*d) * (n*n);
    blockOffset = (x * y * (attentionOffset + (n * d))) + (n*d);
}


/**
 * @brief after training the first block, share all the values to other blocks
 */
void model::copy1toOhterBlocks()
{
    // create a .bin as common.bin
    // hold all the values of MQ, MK, MV, MH, hor and ver
    std::ofstream common(baseDir + "/common.bin", std::ios::binary);
    
    // total values = x*y*(4*row*col+2*layer*d*d)
    long long int offset = x * y * (4 * matOffset + 2 * mlpOffset);
    long long int of = 0;
    std::vector<float> commonData;
    common.write(reinterpret_cast<char*>(commonData.data()), offset * sizeof(float));
    common.close();
    for(int j = 0; j < x; j++) {
        for(int k = 0; k < y; k++) {
            T.t[0].b[j][k].MQ.serialise(of, baseDir + "/common.bin");
            T.t[0].b[j][k].MK.serialise(of+matOffset, baseDir + "/common.bin");
            T.t[0].b[j][k].MV.serialise(of+2*matOffset, baseDir + "/common.bin");
            T.t[0].b[j][k].MH.serialise(of+3*matOffset, baseDir + "/common.bin");
            T.t[0].b[j][k].hor.serialise(of+4*matOffset, baseDir + "/common.bin");
            T.t[0].b[j][k].ver.serialise(of+4*matOffset+mlpOffset, baseDir + "/common.bin");
        }
        of += 4 * matOffset + 2 * mlpOffset;
    }
    std::cout << "common.bin created. Common Kowledge Stored." << std::endl;

    // shared common knowledge
    T.t[0].serialise(baseDir + "/block_1.bin");
    std::cout << "Block 1 serialised. Knowledge Shared." << std::endl;
    for(int i = 1; i < m; i++) {
        for(int j = 0; j < x; j++) {
            for(int k = 0; k < y; k++) {
                T.t[i].b[j][k].MQ = T.t[0].b[j][k].MQ;
                T.t[i].b[j][k].MK = T.t[0].b[j][k].MK;
                T.t[i].b[j][k].MV = T.t[0].b[j][k].MV;
                T.t[i].b[j][k].MH = T.t[0].b[j][k].MH;
                T.t[i].b[j][k].hor.weights = T.t[0].b[j][k].hor.weights;
                T.t[i].b[j][k].ver.weights = T.t[0].b[j][k].ver.weights;
            }
        }
        T.t[i].serialise(baseDir + "/block_" + std::to_string(i+1) + ".bin");
        std::cout << "Block " << i+1 << " serialised. Knowledge Shared." << std::endl;
    }

    // allocate the common knowledge to discrete bins
    mat cache(d, d);
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < x; j++) {
            for(int k = 0; k < y; k++) {
                T.t[i].b[j][k].MQ.serialise(i*k*j*matOffset, baseDir + "/bin/MQ.bin");
                T.t[i].b[j][k].MK.serialise(i*k*j*matOffset, baseDir + "/bin/MK.bin");
                T.t[i].b[j][k].MV.serialise(i*k*j*matOffset, baseDir + "/bin/MV.bin");
                T.t[i].b[j][k].MH.serialise(i*k*j*matOffset, baseDir + "/bin/MH.bin");
                T.t[i].b[j][k].hor.serialise(i*k*j*mlpOffset, baseDir + "/bin/hor.bin");
                T.t[i].b[j][k].ver.serialise(i*k*j*mlpOffset, baseDir + "/bin/ver.bin");
                cache = T.t[i].b[j][k].MQ * T.t[i].b[j][k].MK;
                cache.serialise(i*k*j*cacheOffset, baseDir + "/bin/QK.bin");
                cache = T.t[i].b[j][k].MK * T.t[i].b[j][k].MH;
                cache.serialise(i*k*j*cacheOffset, baseDir + "/bin/KH.bin");
                cache = T.t[i].b[j][k].MQ * T.t[i].b[j][k].MV;
                cache.serialise(i*k*j*cacheOffset, baseDir + "/bin/QV.bin");
            }
        }
    }
    std::cout << "All discrete bins serialised." << std::endl;
    std::cout << "All blocks serialised. Knowledge Shared." << std::endl;
}


/**
 * @brief number of float values currently available in the model file
 * @param blockCount current block in the full context (1-based index)
 * @param paCount current row of attention heads in block (1-based index)
 * @param attentionCount current attention head in pa (1-based index)
 * @param matCount which matrix in use (MQ = 1, MK = 2, MV = 3, MH = 4, use 5 when all mats are filled)
 * @param mlpCount which mlp in use (hor = 1, ver = 2, use 5 when mlp needed to be filled)
 */
int model::getOffset(int blockCount, int paCount, int attentionCount, int matCount, int mlpCount)
{
    int offset = (blockOffset)*(blockCount-1) + attentionCount*paCount*attentionOffset + (matCount-1)*matOffset + (mlpCount-1)*mlpOffset;
    return offset;
}


/**
 * @brief create the binary storage files for the model
 * @param locationOfModel location of the model folder
 */
void model::serialise()
{
    mat cache(d, d);
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < x; j++) {
            for(int k = 0; k < y; k++) {
                T.t[0].b[j][k].MQ.serialise(k*j*matOffset, baseDir + "/bin/MQ.bin");
                T.t[0].b[j][k].MK.serialise(k*j*matOffset, baseDir + "/bin/MK.bin");
                T.t[0].b[j][k].MV.serialise(k*j*matOffset, baseDir + "/bin/MV.bin");
                T.t[0].b[j][k].MH.serialise(k*j*matOffset, baseDir + "/bin/MH.bin");
                T.t[0].b[j][k].hor.serialise(k*j*mlpOffset, baseDir + "/bin/hor.bin");
                T.t[0].b[j][k].ver.serialise(k*j*mlpOffset, baseDir + "/bin/ver.bin");
                cache = T.t[0].b[j][k].MQ*T.t[0].b[j][k].MK;
                cache.serialise(k*j*cacheOffset, baseDir + "/bin/QK.bin");
                cache = T.t[0].b[j][k].MK*T.t[0].b[j][k].MH;
                cache.serialise(k*j*cacheOffset, baseDir + "/bin/KH.bin");
                cache = T.t[0].b[j][k].MQ*T.t[0].b[j][k].MV;
                cache.serialise(k*j*cacheOffset, baseDir + "/bin/QV.bin");
            }
        }
    }
    std::cout << "common.bin created. Common Kowledge Stored." << std::endl;
    for(int i = 0; i < m; i++) {
        T.t[i].serialise(baseDir + "/block" + std::to_string(i+1) + ".bin");
        std::cout << "Block " << i+1 << " serialised." << std::endl;
    }
}
