
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
long long int countLinesInCSV(const std::string& filename) 
{
    // Open the file
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return -1; // Return -1 to indicate an error
    }
    // Count the number of lines
    long long int lineCount = 0;
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
int countLineInTXT(const std::string& filename) 
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
 * @brief Create a TXT file from a CSV file.
 * @param csvFilePath The path to the CSV file.
 * @param txtFilesPath The path to the TXT files.
 * @param totalLines The total number of lines in the CSV file.
 * @param totalGroups The number of groups to split the data into.
 */
void makeTXTfromCSV(const std::string& csvFilePath, std::vector<std::string>& txtFilesPath, long long int totalLines, 
    int totalGroups) 
{
    if (totalLines <= 0) {
        std::cout << "No lines to process from CSV." << std::endl;
        return;
    }
    if (totalGroups <= 0) {
        std::cerr << "Error: totalGroups must be positive." << std::endl;
        return;
    }
    if (txtFilesPath.empty()) {
        std::cerr << "Error: txtFilesPath is empty, no output files specified." << std::endl;
        return;
    }

    std::ifstream csvFile(csvFilePath);
    if (!csvFile.is_open()) {
        std::cerr << "Error: Could not open CSV file: " << csvFilePath << std::endl;
        return;
    }

    long long int linesPerGroup = totalLines / totalGroups;         // Lines per group (integer division)
    long long int linesRemaining = totalLines % totalGroups;        // = totalLines - (linesPerGroup * totalGroups)

    std::string line;
    int groupIndex = 0;
    long long int overallLineCount = 0; // 1-based counter for total lines processed from CSV
    for(groupIndex = 0; groupIndex <= totalGroups; ++groupIndex) {
        // check for txt file path
        std::ofstream txtFile(txtFilesPath[groupIndex]);
        if (!txtFile.is_open()) {
            std::cerr << "Error: Could not open TXT file: " << txtFilesPath[groupIndex] << std::endl;
            continue;
        }
        std::cout << "Writing to file: " << txtFilesPath[groupIndex] << std::endl;

        long long int linesToWrite = (groupIndex == totalGroups) ? linesRemaining : linesPerGroup;
        for (long long int j = 0; j < linesToWrite && std::getline(csvFile, line); ++j) {
            txtFile << line << "\n\n"; // Write the line to the TXT file
            overallLineCount++;
        }

        txtFile.close();
        if(!csvFile) 
            break;
    }

    csvFile.close();
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
