
// model.cpp: implementation of Model class
#include "include/model.hpp"
#include "include/model_fs.hpp"
#include <iostream>
#include <stdexcept>
#include <filesystem>

/**
 * @brief Save model to binary file. This function opens the model file, 
 *      serializes the model, and closes the file
 */
void model::save() {
    // Check if file path is set in model info
    std::string filePath = "model.bin";
    if (!info.modelName.empty()) {
        filePath = info.modelName + ".bin";
    }
    
    // Open file for writing
    errno_t err = fopen_s(&file, filePath.c_str(), "wb");
    if (err != 0)
    if (!file) {
        throw std::runtime_error("Failed to open file for writing: " + filePath);
    }
    
    try {
        // Serialize model to file
        serialiseModel(*this);
        
        // Close file
        fclose(file);
        file = nullptr;
        
        std::cout << "Model saved to " << filePath << " successfully." << std::endl;
    }
    catch (const std::exception& e) {
        // Close file on error
        if (file) {
            fclose(file);
            file = nullptr;
        }
        throw std::runtime_error("Error saving model: " + std::string(e.what()));
    }
}

/**
 * @brief Save model to binary file. This function opens the model file, 
 *      serializes the model, and closes the file
 * @param locationOfModel save model at this location
 */
void model::save(std::string& locationOfModel) {
    // Ensure the directory exists
    if (!std::filesystem::exists(locationOfModel)) {
        if (!std::filesystem::create_directories(locationOfModel)) {
            throw std::runtime_error("Failed to create directory: " + locationOfModel);
        }
    }

    // Construct the full file path
    std::string filePath = locationOfModel + "/" + info.modelName + ".bin";
    if (info.modelName.empty()) {
        filePath = locationOfModel + "/model.bin";
    }

    // Open file for writing
    errno_t err = fopen_s(&file, filePath.c_str(), "wb");
    if (err != 0 || !file) {
        throw std::runtime_error("Failed to open file for writing: " + filePath);
    }

    try {
        // Serialize model to file
        serialiseModel(*this);

        // Close file
        fclose(file);
        file = nullptr;

        std::cout << "Model saved to " << filePath << " successfully." << std::endl;
    }
    catch (const std::exception& e) {
        if (file) fclose(file);
        throw std::runtime_error("Error saving model: " + std::string(e.what()));
    }
}
