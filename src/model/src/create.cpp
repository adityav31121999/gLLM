
#include "include/model.hpp"
#include <iostream>
#include <stdexcept>
#include <filesystem>
#include <neural.hpp>

/**
 * @brief create the binary storage files for the model
 * @param locationOfModel location of the model folder
 */
void create(std::string &locationOfALLbins)
{
    // create the bin files at model location, or in one f its folder
    // create the model folder at locationOfModel
    if (!std::filesystem::exists(locationOfALLbins)) {
        std::filesystem::create_directory(locationOfALLbins);
    }
    // create the bin files at model location
    std::filesystem::create_directory(locationOfALLbins + "/bin");
    // create matrix files
    std::ofstream mqFile(locationOfALLbins + "/bin/MQ.bin", std::ios::binary);
    std::ofstream mkFile(locationOfALLbins + "/bin/MK.bin", std::ios::binary);
    std::ofstream mhFile(locationOfALLbins + "/bin/MH.bin", std::ios::binary);
    std::ofstream mvFile(locationOfALLbins + "/bin/MV.bin", std::ios::binary);
    
    // create MLP files
    std::ofstream horFile(locationOfALLbins + "/bin/hor.bin", std::ios::binary);
    std::ofstream verFile(locationOfALLbins + "/bin/ver.bin", std::ios::binary);
    
    // create cache files
    std::ofstream qkFile(locationOfALLbins + "/bin/QK.bin", std::ios::binary);
    std::ofstream khFile(locationOfALLbins + "/bin/KH.bin", std::ios::binary);
    std::ofstream qvFile(locationOfALLbins + "/bin/QV.bin", std::ios::binary);

    // create file for Matrices, MLPs and Caches
    std::ofstream commonBin(locationOfALLbins + "/bin/common.bin", std::ios::binary);
    std::ofstream block1stSpecific(locationOfALLbins + "/bin/block1stSpecific.bin", std::ios::binary);

    // Initialize files with correct dimensions
    long long int mat_size = static_cast<long long int>(NUMBER_OF_BLOCKS * NUMBER_OF_HEADS * NUMBER_OF_PA * EMBEDDING * MATHEIGHTS);
    long long int mlp_size = static_cast<long long int>(NUMBER_OF_BLOCKS * NUMBER_OF_HEADS * NUMBER_OF_PA * EMBEDDING * EMBEDDING * LAYERS_MLP);
    long long int cache_size = static_cast<long long int>(NUMBER_OF_BLOCKS * NUMBER_OF_HEADS * NUMBER_OF_PA * EMBEDDING * EMBEDDING);
    long long int totalParams = (4*NUMBER_OF_BLOCKS * NUMBER_OF_HEADS * NUMBER_OF_PA * EMBEDDING * MATHEIGHTS
                               + 2*NUMBER_OF_BLOCKS * NUMBER_OF_HEADS * NUMBER_OF_PA * EMBEDDING * EMBEDDING * LAYERS_MLP
                               + 3*NUMBER_OF_BLOCKS * NUMBER_OF_HEADS * NUMBER_OF_PA * EMBEDDING * EMBEDDING);
    long long int trainableParams = (4*NUMBER_OF_BLOCKS * NUMBER_OF_HEADS * NUMBER_OF_PA * EMBEDDING * MATHEIGHTS
                                   + 2*NUMBER_OF_BLOCKS * NUMBER_OF_HEADS * NUMBER_OF_PA * EMBEDDING * EMBEDDING * LAYERS_MLP);
    long long int inferenceParams = (2*NUMBER_OF_BLOCKS * NUMBER_OF_HEADS * NUMBER_OF_PA * EMBEDDING * EMBEDDING * LAYERS_MLP
                                   + 3*NUMBER_OF_BLOCKS * NUMBER_OF_HEADS * NUMBER_OF_PA * EMBEDDING * EMBEDDING);
    long long int block1stParams = (4*NUMBER_OF_HEADS * NUMBER_OF_PA * EMBEDDING * MATHEIGHTS
                               + 2*NUMBER_OF_HEADS * NUMBER_OF_PA * EMBEDDING * EMBEDDING * LAYERS_MLP
                               + 3*NUMBER_OF_HEADS * NUMBER_OF_PA * EMBEDDING * EMBEDDING);

    std::cout << "      PARAMETERS                 Total Values     SIZE(MiBs)      SIZE(MBs)" << std::endl;
    std::cout << "Trained Weight Matrix         :  " << mat_size << "\t   " << static_cast<float>(sizeof(float)*mat_size)/(1000*1000) << "\t    " << static_cast<float>(sizeof(float)*mat_size)/(1024*1024) << std::endl;
    std::cout << "Trained MLP Weight            :  " << mlp_size << "\t   " << static_cast<float>(sizeof(float)*mlp_size)/(1000*1000) << "\t    " << static_cast<float>(sizeof(float)*mlp_size)/(1024*1024) << std::endl;
    std::cout << "Compressed Weight             :  " << cache_size << "\t   " << static_cast<float>(sizeof(float)*cache_size)/(1000*1000) << "\t    " << static_cast<float>(sizeof(float)*cache_size)/(1024*1024) << std::endl;
    std::cout << "Total (Matrices + MLPs)       :  " << trainableParams << "\t   " << static_cast<float>(sizeof(float)*trainableParams)/(1000*1000) << "\t    " << static_cast<float>(sizeof(float)*trainableParams)/(1024*1024) << std::endl;
    std::cout << "Total (Caches + MLPs)         :  " << inferenceParams << "\t   " << static_cast<float>(sizeof(float)*inferenceParams)/(1000*1000) << "\t    " << static_cast<float>(sizeof(float)*inferenceParams)/(1024*1024) << std::endl;
    std::cout << "Total (Common Bin)            :  " << totalParams << "\t   " << static_cast<float>(sizeof(float)*totalParams)/(1000*1000) << "\t    " << static_cast<float>(sizeof(float)*totalParams)/(1024*1024) << std::endl;
    std::cout << "Total (First Block)           :  " << block1stParams << "\t   " << static_cast<float>(sizeof(float)*block1stParams)/(1000*1000) << "\t    " << static_cast<float>(sizeof(float)*block1stParams)/(1024*1024) << std::endl;

    // Initialize MQ, MK, MH, MV files
    std::vector<float> data(mat_size, 0.0f);
    mqFile.write(reinterpret_cast<char*>(data.data()), mat_size * sizeof(float));
    mkFile.write(reinterpret_cast<char*>(data.data()), mat_size * sizeof(float));
    mhFile.write(reinterpret_cast<char*>(data.data()), mat_size * sizeof(float));
    mvFile.write(reinterpret_cast<char*>(data.data()), mat_size * sizeof(float));

    // Initialize hor, ver files
    data.resize(mlp_size, 0.0f);
    horFile.write(reinterpret_cast<char*>(data.data()), mlp_size * sizeof(float));
    verFile.write(reinterpret_cast<char*>(data.data()), mlp_size * sizeof(float));

    // Initialize QK, KH, QV files
    data.resize(cache_size, 0.0f);
    qkFile.write(reinterpret_cast<char*>(data.data()), cache_size * sizeof(float));
    khFile.write(reinterpret_cast<char*>(data.data()), cache_size * sizeof(float));
    qvFile.write(reinterpret_cast<char*>(data.data()), cache_size * sizeof(float));

    // Initialize common bin file
    data.resize(totalParams, 0.0f);
    commonBin.write(reinterpret_cast<char*>(data.data()), totalParams * sizeof(float));
    data.resize(block1stParams, 0.0f);
    block1stSpecific.write(reinterpret_cast<char*>(data.data()), block1stParams * sizeof(float));

    // Close all files
    mqFile.close();
    mkFile.close();
    mhFile.close();
    mvFile.close();
    horFile.close();
    verFile.close();
    qkFile.close();
    khFile.close();
    qvFile.close();
    commonBin.close();
    block1stSpecific.close();
    std::cout << "Bins for mat, cache and mlp created." << std::endl;
    std::cout << "Common Bin file created." << std::endl;
}


/**
 * @brief create the binary storage files for the model
 * @param locationOfModel location of the model folder
 */
void create(std::string &locationOfALLbins, int totalBlocks)
{
    // create the bin files at model location, or in one f its folder
    // create the model folder at locationOfModel
    if (!std::filesystem::exists(locationOfALLbins)) {
        std::filesystem::create_directory(locationOfALLbins);
    }
    // create the bin files at model location
    // std::filesystem::create_directory(locationOfALLbins);
    
    for(int i = 0; i < totalBlocks; i++) {
        // create bin files
        std::string blockFilePath = locationOfALLbins + "/bin/block_" + std::to_string(i) + ".bin";
        std::ofstream blockFile(blockFilePath, std::ios::binary);
        if (!blockFile) {
            std::cerr << "Error: Could not create block file " << blockFilePath << std::endl;
            continue;
        }
    }

    std::cout << "ALL Block Files Created." << std::endl;

    // create the bin files at model location, or in one f its folder
    // create the model folder at locationOfModel
    // if (!std::filesystem::exists(locationOfALLbins)) {
       //  std::filesystem::create_directory(locationOfALLbins);
    // }

    // create the bin files at model location
    std::filesystem::create_directory(locationOfALLbins + "/bin");

    // create matrix files
    std::ofstream mqFile(locationOfALLbins + "/bin/MQ.bin", std::ios::binary);
    std::ofstream mkFile(locationOfALLbins + "/bin/MK.bin", std::ios::binary);
    std::ofstream mhFile(locationOfALLbins + "/bin/MH.bin", std::ios::binary);
    std::ofstream mvFile(locationOfALLbins + "/bin/MV.bin", std::ios::binary);
    
    // create MLP files
    std::ofstream horFile(locationOfALLbins + "/bin/hor.bin", std::ios::binary);
    std::ofstream verFile(locationOfALLbins + "/bin/ver.bin", std::ios::binary);
    
    // create cache files
    std::ofstream qkFile(locationOfALLbins + "/bin/QK.bin", std::ios::binary);
    std::ofstream khFile(locationOfALLbins + "/bin/KH.bin", std::ios::binary);
    std::ofstream qvFile(locationOfALLbins + "/bin/QV.bin", std::ios::binary);

    // create file for Matrices, MLPs and Caches
    std::ofstream commonBin(locationOfALLbins + "/bin/common.bin", std::ios::binary);
    std::ofstream block1stSpecific(locationOfALLbins + "/bin/block1stSpecific.bin", std::ios::binary);

    // Initialize files with correct dimensions
    long long int mat_size = static_cast<long long int>(NUMBER_OF_BLOCKS * NUMBER_OF_HEADS * NUMBER_OF_PA * EMBEDDING * MATHEIGHTS);
    long long int mlp_size = static_cast<long long int>(NUMBER_OF_BLOCKS * NUMBER_OF_HEADS * NUMBER_OF_PA * EMBEDDING * EMBEDDING * LAYERS_MLP);
    long long int cache_size = static_cast<long long int>(NUMBER_OF_BLOCKS * NUMBER_OF_HEADS * NUMBER_OF_PA * EMBEDDING * EMBEDDING);
    long long int totalParams = (4*NUMBER_OF_BLOCKS * NUMBER_OF_HEADS * NUMBER_OF_PA * EMBEDDING * MATHEIGHTS
                               + 2*NUMBER_OF_BLOCKS * NUMBER_OF_HEADS * NUMBER_OF_PA * EMBEDDING * EMBEDDING * LAYERS_MLP
                               + 3*NUMBER_OF_BLOCKS * NUMBER_OF_HEADS * NUMBER_OF_PA * EMBEDDING * EMBEDDING);
    long long int trainableParams = (4*NUMBER_OF_BLOCKS * NUMBER_OF_HEADS * NUMBER_OF_PA * EMBEDDING * MATHEIGHTS
                                   + 2*NUMBER_OF_BLOCKS * NUMBER_OF_HEADS * NUMBER_OF_PA * EMBEDDING * EMBEDDING * LAYERS_MLP);
    long long int inferenceParams = (2*NUMBER_OF_BLOCKS * NUMBER_OF_HEADS * NUMBER_OF_PA * EMBEDDING * EMBEDDING * LAYERS_MLP
                                   + 3*NUMBER_OF_BLOCKS * NUMBER_OF_HEADS * NUMBER_OF_PA * EMBEDDING * EMBEDDING);
    long long int block1stParams = (4*NUMBER_OF_HEADS * NUMBER_OF_PA * EMBEDDING * MATHEIGHTS
                               + 2*NUMBER_OF_HEADS * NUMBER_OF_PA * EMBEDDING * EMBEDDING * LAYERS_MLP
                               + 3*NUMBER_OF_HEADS * NUMBER_OF_PA * EMBEDDING * EMBEDDING);

    std::cout << "      PARAMETERS                  COUNT           SIZE(MiBs)      SIZE(MBs)" << std::endl;
    std::cout << "Trained Weight Matrix         :  " << mat_size << "\t   " << static_cast<float>(sizeof(float)*mat_size)/(1000*1000) << "\t    " << static_cast<float>(sizeof(float)*mat_size)/(1024*1024) << std::endl;
    std::cout << "Trained MLP Weight            :  " << mlp_size << "\t   " << static_cast<float>(sizeof(float)*mlp_size)/(1000*1000) << "\t    " << static_cast<float>(sizeof(float)*mlp_size)/(1024*1024) << std::endl;
    std::cout << "Compressed Weight             :  " << cache_size << "\t   " << static_cast<float>(sizeof(float)*cache_size)/(1000*1000) << "\t    " << static_cast<float>(sizeof(float)*cache_size)/(1024*1024) << std::endl;
    std::cout << "Total (Matrices + MLPs)       :  " << trainableParams << "\t   " << static_cast<float>(sizeof(float)*trainableParams)/(1000*1000) << "\t    " << static_cast<float>(sizeof(float)*trainableParams)/(1024*1024) << std::endl;
    std::cout << "Total (Caches + MLPs)         :  " << inferenceParams << "\t   " << static_cast<float>(sizeof(float)*inferenceParams)/(1000*1000) << "\t    " << static_cast<float>(sizeof(float)*inferenceParams)/(1024*1024) << std::endl;
    std::cout << "Total (Common Bin)            :  " << totalParams << "\t   " << static_cast<float>(sizeof(float)*totalParams)/(1000*1000) << "\t    " << static_cast<float>(sizeof(float)*totalParams)/(1024*1024) << std::endl;

    // Initialize MQ, MK, MH, MV files
    std::vector<float> data(mat_size, 0.0f);
    mqFile.write(reinterpret_cast<char*>(data.data()), mat_size * sizeof(float));
    mkFile.write(reinterpret_cast<char*>(data.data()), mat_size * sizeof(float));
    mhFile.write(reinterpret_cast<char*>(data.data()), mat_size * sizeof(float));
    mvFile.write(reinterpret_cast<char*>(data.data()), mat_size * sizeof(float));

    // Initialize hor, ver files
    data.resize(mlp_size, 0.0f);
    horFile.write(reinterpret_cast<char*>(data.data()), mlp_size * sizeof(float));
    verFile.write(reinterpret_cast<char*>(data.data()), mlp_size * sizeof(float));

    // Initialize QK, KH, QV files
    data.resize(cache_size, 0.0f);
    qkFile.write(reinterpret_cast<char*>(data.data()), cache_size * sizeof(float));
    khFile.write(reinterpret_cast<char*>(data.data()), cache_size * sizeof(float));
    qvFile.write(reinterpret_cast<char*>(data.data()), cache_size * sizeof(float));

    // Initialize common bin file
    data.resize(totalParams, 0.0f);
    commonBin.write(reinterpret_cast<char*>(data.data()), totalParams * sizeof(float));
    data.resize(block1stParams, 0.0f);
    block1stSpecific.write(reinterpret_cast<char*>(data.data()), block1stParams * sizeof(float));

    // Close all files
    mqFile.close();
    mkFile.close();
    mhFile.close();
    mvFile.close();
    horFile.close();
    verFile.close();
    qkFile.close();
    khFile.close();
    qvFile.close();
    commonBin.close();
    block1stSpecific.close();
    std::cout << "Bins for mat, cache and mlp created." << std::endl;
    std::cout << "Common Bin file created." << std::endl;
}
