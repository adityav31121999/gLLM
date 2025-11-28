
#include "include/model.hpp"
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <thread>
#include <mutex>

/**
 * @brief Count the number of lines in a CSV file.
 * @param filename The name of the CSV file.
 * @return The number of lines in the file, or -1 if an error occurs.
 */
unsigned long long countLinesInCSV(const std::string& filename) 
{
    // Open the file
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return -1; // Return -1 to indicate an error
    }
    // Count the number of lines
    unsigned long long lineCount = 0;
    std::string line;
    while (std::getline(file, line)) {
        lineCount++;
    }
    // Close the file
    file.close();
    return lineCount;
}

/**
 * @brief Count the number of lines in a CSV file.
 * @param filename The name of the CSV file.
 * @return The number of lines in the file, or -1 if an error occurs.
 */
unsigned long long countLineInTXT(const std::string& filename) 
{
    // Open the file
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return -1; // Return -1 to indicate an error
    }
    // Count the number of lines
    int lineCount = 0;
    std::string line;
    while (std::getline(file, line)) {
        lineCount++;
    }
    // Close the file
    file.close();
    return lineCount;
}

/**
 * @brief Create a CSV file from token embeddings.
 * @param tokens The vector of tokens.
 * @param tokenEmbed The vector of token embeddings.
 * @param csvFilePath The path to the CSV file to be created.
 */
void makeCSV(std::vector<std::string>& tokens, mat& tokenEmbed, const std::string& csvFilePath) 
{
    std::ofstream csvFile(csvFilePath);
    if (!csvFile.is_open()) {
        std::cerr << "Error: Could not open CSV file: " << csvFilePath << std::endl;
        return;
    }

    for (size_t i = 0; i < tokens.size(); ++i) {
        // first column is the token
        csvFile << tokens[i];
        // subsequent columns are the token embeddings
        for (const auto& value : tokenEmbed(i)) {
            csvFile << "," << value;
        }
        csvFile << "\n";
    }

    csvFile.close();
}
