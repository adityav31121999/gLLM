
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
    
    // Initialize files with correct dimensions
    size_t mat_size = static_cast<size_t>(NUMBER_OF_BLOCKS * NUMBER_OF_HEADS * NUMBER_OF_PA * EMBEDDING * MATHEIGHTS);
    size_t mlp_size = static_cast<size_t>(NUMBER_OF_BLOCKS * NUMBER_OF_HEADS * NUMBER_OF_PA * EMBEDDING * EMBEDDING * LAYERS_MLP);
    size_t cache_size = static_cast<size_t>(NUMBER_OF_BLOCKS * NUMBER_OF_HEADS * NUMBER_OF_PA * EMBEDDING * EMBEDDING);

    // Initialize MQ, MK, MH, MV files
    std::vector<float> mq_data(mat_size, 0.0f);
    mqFile.write(reinterpret_cast<char*>(mq_data.data()), mat_size * sizeof(float));
    mkFile.write(reinterpret_cast<char*>(mq_data.data()), mat_size * sizeof(float));
    mhFile.write(reinterpret_cast<char*>(mq_data.data()), mat_size * sizeof(float));
    mvFile.write(reinterpret_cast<char*>(mq_data.data()), mat_size * sizeof(float));
    
    // Initialize hor, ver files
    std::vector<float> mlp_data(mlp_size, 0.0f);
    horFile.write(reinterpret_cast<char*>(mlp_data.data()), mlp_size * sizeof(float));
    verFile.write(reinterpret_cast<char*>(mlp_data.data()), mlp_size * sizeof(float));
    
    // Initialize QK, KH, QV files
    std::vector<float> cache_data(cache_size, 0.0f);
    qkFile.write(reinterpret_cast<char*>(cache_data.data()), cache_size * sizeof(float));
    khFile.write(reinterpret_cast<char*>(cache_data.data()), cache_size * sizeof(float));
    qvFile.write(reinterpret_cast<char*>(cache_data.data()), cache_size * sizeof(float));

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

    std::cout << "Bins for mat, cache and mlp created." << std::endl;
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
    std::filesystem::create_directory(locationOfALLbins);
    
    for(int i = 0; i < totalBlocks; i++) {
        // create bin files
        std::string blockFilePath = locationOfALLbins + "/bin/block_" + std::to_string(i) + ".bin";
        std::ofstream blockFile(blockFilePath, std::ios::binary);
        if (!blockFile) {
            std::cerr << "Error: Could not create block file " << blockFilePath << std::endl;
            continue;
        }
    }

    std::cout << "Blocks created." << std::endl;
}
