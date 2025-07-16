
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
    matOffset = matheight * d;      // for single mat
    cacheOffset = d * d;            // for single cache
    mlpOffset = d * d * l;          // for single mlp
    // for training
    attentionOffset = (4*matOffset) + (2*mlpOffset) + d + (n*d) + (2*n*d) * (n*n);
    blockOffset = (x * y * (attentionOffset + (n * d))) + (n*d);
}


void transformer::makeCommon(std::string &path2folderOfAllBins)
{
    // Create/truncate common.bin. The serialise methods will append to it.
    std::string commonBinPath = path2folderOfAllBins + "/common.bin";
    std::ofstream commonFileStream(commonBinPath, std::ios::binary | std::ios::trunc);
    if (!commonFileStream.is_open()) {
        throw std::runtime_error("Failed to open/create common.bin for writing: " + commonBinPath);
    }
    commonFileStream.close(); // Close it now. serialise calls will reopen in append mode.

    long long int single_mlp_params_from_instance = 0;
    if (this->x > 0 && this->y > 0) {
        if (this->t.empty()) {
             throw std::runtime_error("transformer::makeCommon: Transformer has no blocks (T.t is empty) when x and y are positive.");
        }
        if (this->t[0].b.empty() || this->t[0].b[0].empty()) {
            throw std::runtime_error("transformer::makeCommon: Block 0's attention structure (b) is empty or first row is empty when x and y are positive.");
        }
        // Access parameters from the first head of the first partial attention layer of the first block
        single_mlp_params_from_instance = t[0].b[0][0].hor.params;
    }
    // If x or y is 0, totalparams will correctly be 0.

    long long int totalparams = static_cast<long long int>(this->x) * this->y *
                               ((4 * static_cast<long long int>(this->h) * this->d) +
                                (2 * single_mlp_params_from_instance));

    std::cout << "Size of Common Bin FIle: " << std::endl;
    std::cout << "\tTotal Parameters: " << totalparams << std::endl;
    std::cout << "\tSize(MiBs): " << static_cast<float>(sizeof(float) * totalparams) / (1000.0f * 1000.0f) << std::endl;
    std::cout << "\tSize(MBs): " << static_cast<float>(sizeof(float) * totalparams) / (1024.0f * 1024.0f) << std::endl;

    // The 'conceptual_offset' is for logical tracking if needed, as serialise appends.
    // The offset parameter to mat::serialise is 0, assuming it appends.
    if (!this->t.empty()) { // Proceed only if there's at least one block
        for (int j = 0; j < this->x; j++) {
            for (int k = 0; k < this->y; k++) {
                // Ensure T.t[0].b[j][k] is valid before calling methods on it.
                // This should be guaranteed if x and y match the dimensions of T.t[0].b
                if (j < t[0].b.size() && k < t[0].b[j].size()) {
                    t[0].b[j][k].MQ.serialise(j * k * matOffset, commonBinPath);
                    t[0].b[j][k].MK.serialise(j * k * matOffset, commonBinPath);
                    t[0].b[j][k].MV.serialise(j * k * matOffset, commonBinPath);
                    t[0].b[j][k].MH.serialise(j * k * matOffset, commonBinPath);
                    t[0].b[j][k].hor.serialise4train(commonBinPath);
                    t[0].b[j][k].ver.serialise4train(commonBinPath);
                }
                else {
                    throw std::out_of_range("transformer::makeCommon: Attempted to access T.t[0].b out of bounds during serialization.");
                }
            }
        }
        std::cout << "common.bin populated with shared weights from block 0." << std::endl;
    }
    else if (this->x > 0 && this->y > 0) { // x and y suggest there should be data, but T.t is empty
        std::cerr << "Warning: transformer::makeCommon: x and y are positive, but T.t is empty. common.bin will be empty." << std::endl;
    }
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
                std::cout << "Serialising: Block " << i+1 << " | Partial Attention " << j+1 << " | Attention Head " << k+1 << " -> MQ -> ";
                T.t[0].b[j][k].MQ.serialise(k*j*matOffset, baseDir + "/bin/MQ.bin");
                std::cout << "MQ -> ";
                T.t[0].b[j][k].MK.serialise(k*j*matOffset, baseDir + "/bin/MK.bin");
                std::cout << "MK -> ";
                T.t[0].b[j][k].MV.serialise(k*j*matOffset, baseDir + "/bin/MV.bin");
                std::cout << "MV -> ";
                T.t[0].b[j][k].MH.serialise(k*j*matOffset, baseDir + "/bin/MH.bin");
                std::cout << "MH -> ";
                T.t[0].b[j][k].hor.serialise(k*j*mlpOffset, baseDir + "/bin/hor.bin");
                std::cout << "hor -> ";
                T.t[0].b[j][k].ver.serialise(k*j*mlpOffset, baseDir + "/bin/ver.bin");
                std::cout << "ver -> ";
                cache.mult_A_Bt(T.t[0].b[j][k].MQ,T.t[0].b[j][k].MK);
                std::cout << "QK' -> ";
                cache.serialise(k*j*cacheOffset, baseDir + "/bin/QK.bin");
                cache = T.t[0].b[j][k].MK*T.t[0].b[j][k].MH;
                std::cout << "KH -> ";
                cache.serialise(k*j*cacheOffset, baseDir + "/bin/KH.bin");
                cache = T.t[0].b[j][k].MQ*T.t[0].b[j][k].MV;
                std::cout << "QV" << std::endl;
                cache.serialise(k*j*cacheOffset, baseDir + "/bin/QV.bin");
            }
        }
    }
    for(int i = 0; i < m; i++) {
        T.t[i].serialise(baseDir + "/block" + std::to_string(i+1) + ".bin");
        std::cout << "Block " << i+1 << " serialised." << std::endl;
    }
}
