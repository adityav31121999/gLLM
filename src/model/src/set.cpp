
#include "include/model.hpp" // Adjust path as per your project structure
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept> // For std::runtime_error, std::invalid_argument
#include <iostream>  // For error reporting, if not using exceptions for everything
#include <locale>    // For std::isspace with locale


// Helper to ensure file has a certain size.
int ensure_file_size_basic(FILE* fp, size_t required_size) {
    if (!fp) return -1;
    long current_pos = ftell(fp);
    if (fseek(fp, 0, SEEK_END) != 0) return -1;
    long current_size = ftell(fp);
    if (current_size < 0) return -1;

    if (static_cast<size_t>(current_size) < required_size) {
        if (fseek(fp, required_size - 1, SEEK_SET) != 0) return -1;
        if (fwrite("", 1, 1, fp) != 1) return -1; // Write a single byte to extend
        if (fflush(fp) != 0) return -1;
    }
    // Restore original position or rewind
    if (current_pos != -1) fseek(fp, current_pos, SEEK_SET);
    else rewind(fp);
    return 0;
}

void model::setLearning(float learning) {
    this->learning = learning;
    T.setLearning(learning);
}

void model::setVocab(int vocab) {
    info.vocab = vocab;
}

void model::setModelName(const std::string& modelName) {
    info.modelName = modelName;
}

void model::setVersion(const std::string& version) {
    info.version = version;
}

void model::setAuthor(const std::string& author) {
    info.author = author;
}

void model::setDate(const std::string& date) {
    info.date = date;
}

void model::setLicense(const std::string& license) {
    info.license = license;
}

void model::setInfo(modelDataInfo& info) {
    this->info = info;
}

void model::setInfo(std::string& modelName, std::string& version, std::string& author, 
                   std::string& date, std::string& modelArch, std::string& license, 
                   std::string& trainingData) {
    info.modelName = modelName;
    info.version = version;
    info.author = author;
    info.date = date;
    info.modelArch = modelArch;
    info.license = license;
}
