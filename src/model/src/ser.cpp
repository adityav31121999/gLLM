
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


// calculate offset for model layout and its components
void model::calculateAndSetLayout() {
    matOffset = matheight * d;      // for single mat values
    cacheOffset = d * d;            // for single cache values
    mlpOffset = d * d * l;          // for single mlp values
    // for training
    attentionOffset = (4*matOffset) + (2*mlpOffset) + d + (n*d) + (2*n*d) * (n*n);
    blockOffset = (x * y * (attentionOffset + (n * d))) + (n*d);
}

/**
 * @brief make common bin with all matrices and mlp weights and caches
 * @param path2folderOfAllBins path to bin files
 */
void transformer::makeCommon(std::string &path2folderOfAllBins)
{
    // Create/truncate common.bin. The serialise methods will append to it.
    std::string commonBinPath = path2folderOfAllBins + "/common.bin";
    std::ofstream commonFileStream(commonBinPath, std::ios::binary | std::ios::trunc);
    if (!commonFileStream.is_open()) {
        throw std::runtime_error("Failed to open/create common.bin for writing: " + commonBinPath);
    }
    commonFileStream.close(); // Close it now. serialise calls will reopen in append mode.

    unsigned long long single_mlp_params_from_instance = 0;
    if (x > 0 && y > 0) {
        if (blocks.empty()) {
            throw std::runtime_error("transformer::makeCommon: Transformer has no blocks ( T.blocks is empty) when x and y are positive.");
        }
        if (blocks[0].b.empty() || blocks[0].b[0].empty()) {
            throw std::runtime_error("transformer::makeCommon: Block 0's attention structure (b) is empty or first row is empty when x and y are positive.");
        }
        // Access parameters from the first head of the first partial attention layer of the first block
        single_mlp_params_from_instance = blocks[0].b[0][0].hor.params;
    }
    // If x or y is 0, totalparams will correctly be 0.

    unsigned long long totalparams = static_cast<unsigned long long>(x) * y *
                               ((4 * static_cast<unsigned long long>(h) * d) +
                                (2 * single_mlp_params_from_instance));

    std::cout << "Size of Common Bin FIle: " << std::endl;
    std::cout << "\tTotal Parameters: " << totalparams << std::endl;
    std::cout << "\tSize(MiBs): " << static_cast<float>(sizeof(float) * totalparams) / (1000.0f * 1000.0f) << std::endl;
    std::cout << "\tSize(MBs): " << static_cast<float>(sizeof(float) * totalparams) / (1024.0f * 1024.0f) << std::endl;

    // The 'conceptual_offset' is for logical tracking if needed, as serialise appends.
    // The offset parameter to mat::serialise is 0, assuming it appends.
    if (!blocks.empty()) { // Proceed only if there's at least one block
        unsigned long long of = 0;
        unsigned long long bo = NUMBER_OF_PA*NUMBER_OF_HEADS*(4*EMBEDDING*CONTEXT_WIN + 3*EMBEDDING*EMBEDDING + 2*EMBEDDING*EMBEDDING*(LAYERS_MLP - 1));
        for(int i = 0; i < m; i++) {
            of = i * bo;
            for (int j = 0; j < x; j++) {
                for (int k = 0; k < y; k++) {
                    if (j < blocks[i].b.size() && k < blocks[i].b[j].size()) {
                        blocks[i].b[j][k].MQ.serialise(of, commonBinPath); of += matOffset;
                        blocks[i].b[j][k].MK.serialise(of, commonBinPath); of += matOffset;
                        blocks[i].b[j][k].MV.serialise(of, commonBinPath); of += matOffset;
                        blocks[i].b[j][k].MH.serialise(of, commonBinPath); of += matOffset;
                        for(int l = 0; l < LAYERS_MLP-1; l++) {
                            blocks[i].b[j][k].hor.weights[l].serialise(of, commonBinPath); of += cacheOffset;
                        }
                        for(int l = 0; l < LAYERS_MLP-1; l++) {
                            blocks[i].b[j][k].ver.weights[l].serialise(of, commonBinPath); of += cacheOffset;
                        }
                        blocks[i].b[j][k].qkCache.serialise(of, commonBinPath); of += cacheOffset;
                        blocks[i].b[j][k].khCache.serialise(of, commonBinPath); of += cacheOffset;
                        blocks[i].b[j][k].qvCache.serialise(of, commonBinPath); of += cacheOffset;
                    }
                    else {
                        std::cerr << "Access out of bounds for block[" << i << "].b[" << j << "][" << k << "]" << std::endl;
                        break;
                    }
                }
            }
        }
        std::cout << "common.bin populated with shared weights from block 0." << std::endl;
    }
    else if (x > 0 && y > 0) { // x and y suggest there should be data, but  T.blocks is empty
        std::cerr << "Warning: transformer::makeCommon: x and y are positive, but  T.blocks is empty. common.bin will be empty." << std::endl;
    }
}


/**
 * @brief create the binary storage files for the model
 * @param locationOfModel location of the model folder
 */
void model::serialise()
{
    mat cache(d, d);
    // serialise all mats, mlps and caches
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < x; j++) {
            for(int k = 0; k < y; k++) {
                std::cout << "Serialising: Block " << i+1 << " | Partial Attention " << j+1 << " | Attention Head " << k+1 << std::endl;
                T.blocks[i].b[j][k].MQ.serialise(i*matOffset + k*j*matOffset, baseDir + "/bin/MQ.bin");
                T.blocks[i].b[j][k].MK.serialise(i*matOffset + k*j*matOffset, baseDir + "/bin/MK.bin");
                T.blocks[i].b[j][k].MV.serialise(i*matOffset + k*j*matOffset, baseDir + "/bin/MV.bin");
                T.blocks[i].b[j][k].MH.serialise(i*matOffset + k*j*matOffset, baseDir + "/bin/MH.bin");
                T.blocks[i].b[j][k].hor.serialise(i*mlpOffset + k*j*mlpOffset, baseDir + "/bin/hor.bin");
                T.blocks[i].b[j][k].ver.serialise(i*mlpOffset + k*j*mlpOffset, baseDir + "/bin/ver.bin");
                cache = T.blocks[i].b[j][k].MQ * T.blocks[i].b[j][k].MK;
                cache.serialise(i*cacheOffset + k*j*cacheOffset, baseDir + "/bin/QK.bin");
                cache =  T.blocks[i].b[j][k].MK * T.blocks[i].b[j][k].MH;
                cache.serialise(i*cacheOffset + k*j*cacheOffset, baseDir + "/bin/KH.bin");
                cache =  T.blocks[i].b[j][k].MQ * T.blocks[i].b[j][k].MV;
                cache.serialise(i*cacheOffset + k*j*cacheOffset, baseDir + "/bin/QV.bin");
            }
        }
    }
    // serialsie all blocks
    for(int i = 0; i < m; i++) {
        T.blocks[i].serialise(baseDir + "/block" + std::to_string(i+1) + ".bin");
        std::cout << "Block " << i+1 << " serialised." << std::endl;
    }
    // make common bin
    std::string common = baseDir + "/bin/common.bin";
    T.makeCommon(common);
    std::cout << "Common bin serialised." << std::endl;
}
