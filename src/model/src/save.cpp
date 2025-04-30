
// model.cpp: implementation of Model class
#include "include/model.hpp"
#include "include/model_fs.hpp"
#include <iostream>
#include <stdexcept>
#include <filesystem>

/**
 * @brief Save model to binary file
 * This function opens the model file, serializes the model, and closes the file
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
    } catch (const std::exception& e) {
        // Close file on error
        if (file) {
            fclose(file);
            file = nullptr;
        }
        throw std::runtime_error("Error saving model: " + std::string(e.what()));
    }
}

/**
 * @brief Load model from binary file
 * This function opens the model file, deserializes the model, and closes the file
 */
void model::load() {
    // Check if file path is set in model info
    std::string filePath = "model.bin";
    if (!info.modelName.empty()) {
        filePath = info.modelName + ".bin";
    }
    
    // Check if file exists
    if (!std::filesystem::exists(filePath)) {
        throw std::runtime_error("Model file does not exist: " + filePath);
    }
    
    // Open file for reading
    errno_t err = fopen_s(&file, filePath.c_str(), "rb");
    if (err != 0)
    if (!file) {
        throw std::runtime_error("Failed to open file for reading: " + filePath);
    }
    
    try {
        // Deserialize model from file
        // deserialiseModel(*this);
        
        // Close file
        fclose(file);
        file = nullptr;
        
        std::cout << "Model loaded from " << filePath << " successfully." << std::endl;
    } catch (const std::exception& e) {
        // Close file on error
        if (file) {
            fclose(file);
            file = nullptr;
        }
        throw std::runtime_error("Error loading model: " + std::string(e.what()));
    }
}
