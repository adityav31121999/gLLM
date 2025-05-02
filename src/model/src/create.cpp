
// model.cpp: implementation of Model class
#include "include/model.hpp"
#include <iostream>
#include <stdexcept>
#include <filesystem>
#include <neural.hpp>

/**
 * total values to be stored in each bin file:
 * 1. MQ, MK, MV, MH: m*x*y*matheight*d each
 * 2. hor, ver: m*x*y*d*d*l each
 * 3. QK, KH, QV: m*x*y*d*d each
 */

 /**
  * @brief create the binary storage files for the model
  * @param locationOfModel location of the model folder
  */
void model::create(std::string &locationOfModel)
{
    // create the bin files at model location, or in one f its folder
    // create the model folder at locationOfModel
    if (!std::filesystem::exists(locationOfModel)) {
        std::filesystem::create_directory(locationOfModel);
    }
    // create the bin files at model location
    std::filesystem::create_directory(locationOfModel + "/bin");
    // create matrix files
    std::ofstream mqFile(locationOfModel + "/bin/MQ.bin", std::ios::binary);
    std::ofstream mkFile(locationOfModel + "/bin/MK.bin", std::ios::binary);
    std::ofstream mhFile(locationOfModel + "/bin/MH.bin", std::ios::binary);
    std::ofstream mvFile(locationOfModel + "/bin/MV.bin", std::ios::binary);
    
    // create MLP files
    std::ofstream horFile(locationOfModel + "/bin/hor.bin", std::ios::binary);
    std::ofstream verFile(locationOfModel + "/bin/ver.bin", std::ios::binary);
    
    // create cache files
    std::ofstream qkFile(locationOfModel + "/bin/QK.bin", std::ios::binary);
    std::ofstream khFile(locationOfModel + "/bin/KH.bin", std::ios::binary);
    std::ofstream qvFile(locationOfModel + "/bin/QV.bin", std::ios::binary);
    
    // Initialize files with correct dimensions
    size_t mat_size = static_cast<size_t>( m * x * y * matheight * d);
    size_t mlp_size = static_cast<size_t>(m * x * y * d * d * l);
    size_t cache_size = static_cast<size_t>(m * x * y * d * d);
    
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
}
