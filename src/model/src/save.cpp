
// model.cpp: implementation of Model class
#include "include/model.hpp"
#include <iostream>
#include <stdexcept>
#include <filesystem>
#include <cstdio> // Include for fprintf, fclose, etc.

/**
 * @brief Writes the model metadata from the 'info' struct to the file
 *        pointed to by the 'file' member variable. Assumes 'file' is open
 *        in a suitable text write mode.
 */
void model::addMetadataToFile()
{
    if (!file) {
        // Or handle this more gracefully depending on how save() calls it
        throw std::runtime_error("Error: Metadata file handle is NULL in addMetadataToFile.");
    }

    // Write metadata in a key: value format
    fprintf(file, "modelName: %s\n", info.modelName.c_str());
    fprintf(file, "version: %s\n", info.version.c_str());
    fprintf(file, "author: %s\n", info.author.c_str());
    fprintf(file, "date: %s\n", info.date.c_str());
    fprintf(file, "attentionMech: %s\n", info.attentionMech.c_str());
    fprintf(file, "modelArch: %s\n", info.modelArch.c_str());
    fprintf(file, "license: %s\n", info.license.c_str());
    fprintf(file, "d: %d\n", info.d);
    fprintf(file, "vocab: %d\n", info.vocab);
    fprintf(file, "qkrow: %d\n", info.qkrow); // Assuming these are correctly populated in info
    fprintf(file, "qkcol: %d\n", info.qkcol);
    fprintf(file, "vhrow: %d\n", info.vhrow);
    fprintf(file, "vhcol: %d\n", info.vhcol);
    fprintf(file, "m: %d\n", info.m);
    fprintf(file, "x: %d\n", info.x);
    fprintf(file, "y: %d\n", info.y);
    fprintf(file, "n: %d\n", info.n);
    fprintf(file, "h: %d\n", info.h); // Same as matheight? Ensure consistency
    fprintf(file, "l: %d\n", info.l);
    fprintf(file, "matheight: %d\n", info.matheight);
    fprintf(file, "totalParams: %d\n", info.totalParams);
    fprintf(file, "totalContext: %d\n", info.totalContext);
    fprintf(file, "learning: %f\n", info.learning);
    fprintf(file, "attentionType: %d\n", info.attentionType); // 0 for cross, 1 for self
}

/**
 * @brief Save model to binary file. This function opens the model file, 
 *      serializes the model, and closes the file
 */
void model::save() {
    // Default directory for saving parameters and metadata
    std::string saveDirectory = "./model_data";
    std::filesystem::create_directories(saveDirectory); // Ensure directory exists

    // Determine metadata file path
    std::string metadataFilePath = saveDirectory + "/model_meta.txt"; // Default name
    if (!info.modelName.empty()) {
        metadataFilePath = saveDirectory + "/" + info.modelName + "_meta.txt";
    }

    // Open metadata file for writing (text mode)
    errno_t err = fopen_s(&file, metadataFilePath.c_str(), "w"); // Use "w" for text write
    if (err != 0 || !file) {
        throw std::runtime_error("Failed to open metadata file for writing: " + metadataFilePath);
    }

    try {
        // Write metadata to the text file
        addMetadataToFile();
        // Close metadata file
        fclose(file);
        file = nullptr;

        // Serialize parameters to separate .bin files in the default directory
        serialise(); // This calls serialise("./model_data") internally

        std::cout << "Model metadata saved to " << metadataFilePath << std::endl;
        std::cout << "Model parameters saved to directory " << saveDirectory << " successfully." << std::endl;
    }
    catch (const std::exception& e) {
        // Close file on error
        if (file) {
            fclose(file);
            file = nullptr;
        }
        // Rethrow or handle more specifically
        throw std::runtime_error("Error during model save process: " + std::string(e.what()));
    }
}

/**
 * @brief Saves the model metadata to a .txt file and parameters to separate .bin files
 *        within the specified directory.
 * @param locationOfModel The directory where metadata and parameter files will be saved.
 */
void model::save(std::string& locationOfModel) {
    // Ensure the directory exists
    try {
        std::filesystem::create_directories(locationOfModel);
    } 
    catch (const std::filesystem::filesystem_error& e) {
        throw std::runtime_error("Failed to create directory '" + locationOfModel + "': " + e.what());
    }

    // Determine metadata file path within the specified directory
    std::string metadataFilePath = locationOfModel + "/model_meta.txt"; // Default name
    if (info.modelName.empty()) {
        // Use a default if modelName is empty, or throw an error
        // For now, using a default:
        metadataFilePath = locationOfModel + "/default_model_meta.txt";
    } 
    else {
        metadataFilePath = locationOfModel + "/" + info.modelName + "_meta.txt";
    }

    // Open metadata file for writing (text mode)
    errno_t err = fopen_s(&file, metadataFilePath.c_str(), "w"); // Use "w" for text write
    if (err != 0 || !file) {
        throw std::runtime_error("Failed to open metadata file for writing: " + metadataFilePath);
    }

    try {
        // Write metadata to the text file
        addMetadataToFile();
        // Close metadata file
        fclose(file);
        file = nullptr;
        // Serialize parameters to separate .bin files in the specified directory
        serialise(locationOfModel);

        std::cout << "Model metadata saved to " << metadataFilePath << std::endl;
        std::cout << "Model parameters saved to directory " << locationOfModel << " successfully." << std::endl;
    }
    catch (const std::exception& e) {
        if (file) fclose(file);
        throw std::runtime_error("Error saving model: " + std::string(e.what()));
    }
}
