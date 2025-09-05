#include "include/model.hpp"
#include <filesystem>
#include <sstream> // For std::istringstream
#include <iomanip> // For std::put_time
#include <fstream>
#include <neural.hpp>
#include <maths.hpp>
#include <chrono>

/**
 * @brief train first block using training data from txt file
 * @param txtFileLocation location of txt file (with each line as a sentence, 
 *          between two lines there is an empty line)
 */
void model::trainBlockSentence(const std::string& txtFileLocation) 
{
    // basic checks for file
    if (txtFileLocation.empty()) {
        throw std::runtime_error("Training data file path cannot be empty!");
    }
    if (!std::filesystem::exists(txtFileLocation) || !std::filesystem::is_regular_file(txtFileLocation)) {
        throw std::runtime_error("Training data file not found or is not a regular file: " + txtFileLocation);
    }
    if (std::filesystem::path(txtFileLocation).extension() != ".txt") {
        throw std::runtime_error("Training data file must be a .txt file: " + txtFileLocation);
    }

    TrainingSessionData sessionData;
    bool sessionFileExistsAndIsValid = false;
    int startLineForCurrentFile = 0;

    // Attempt to load session data
    if (!currentChatLogPath.empty() && std::filesystem::exists(currentChatLogPath)) {
        if (std::filesystem::is_regular_file(currentChatLogPath)) {
            std::ifstream tempInfoFile(currentChatLogPath);
            if (tempInfoFile.is_open()) {
                if (tempInfoFile.peek() != std::ifstream::traits_type::eof()) { // Check if file is not empty
                    if (sessionData.load(currentChatLogPath)) {
                        sessionFileExistsAndIsValid = true;
                        std::cout << "Successfully loaded session data from: " << currentChatLogPath;
                        std::cout << " \nPrevious file: " << sessionData.lastTrainingFileName
                                  << ",\nLines processed in it: " << sessionData.linesProcessedInLastFile;
                        std::cout << "\nCumulative Lines Trained: " << sessionData.cumulativeTotalLinesTrained
                                  << ",\nTokens: " << sessionData.cumulativeTotalTokensProcessed
                                  << ",\nTrainCount: " << sessionData.cumulativeTotalTrainCount;
                        std::cout << "\nLast VocabSize: " << sessionData.vocabSizeSnapshot
                                  << ",\nLast BlockCount: " << sessionData.lastBlockCountState
                                  << ",\nLast EpochCount: " << sessionData.lastEpochCountState << std::endl;
                    }
                     else {
                        std::cerr << "Warning: Failed to parse session data from " << currentChatLogPath 
                                  << ". Starting with fresh session values." << std::endl;
                    }
                } 
                else {
                    std::cout << "Session data file " << currentChatLogPath << " is empty. Starting with fresh session values." << std::endl;
                }
                tempInfoFile.close();
            } 
            else {
                std::cerr << "Warning: Could not open session data file " << currentChatLogPath 
                          << " for reading. Starting with fresh session values." << std::endl;
            }
        }
        else {
            std::cerr << "Warning: Session data path " << currentChatLogPath 
                       << " is not a regular file. Starting with fresh session values." << std::endl;
        }
    }
    else {
        if (currentChatLogPath.empty()) {
            std::cout << "Session data file path (currentChatLogPath) is not set. Starting with fresh session values." << std::endl;
        }
        else {
            std::cout << "Session data file " << currentChatLogPath << " not found. Starting with fresh session values." << std::endl;
        }
    }

    // Initialize model state based on session data or defaults
    this->totalTokens = sessionFileExistsAndIsValid ? sessionData.cumulativeTotalTokensProcessed : 0;
    this->T.trainCount = sessionFileExistsAndIsValid ? sessionData.cumulativeTotalTrainCount : 0;
    this->T.blockCount = 0;
    this->T.epochCount = 0;
    int currentLine = 0;

    if (sessionFileExistsAndIsValid && sessionData.lastTrainingFileName == txtFileLocation) {
        startLineForCurrentFile = sessionData.linesProcessedInLastFile;
        std::cout << "Resuming training for " << txtFileLocation << " from line " << startLineForCurrentFile << "." << std::endl;
    }

    // open file and read line by line
    std::ifstream file(txtFileLocation);
    if (!file.is_open()) {
        throw std::runtime_error("Error opening training data file: " + txtFileLocation);
    }
    long long int numberOfLines = countLineInTXT(txtFileLocation);
    if (numberOfLines <= 0) {
        throw std::runtime_error("No training data found in the specified file!");
    }
    std::cout << "Total training lines: " << numberOfLines << std::endl;
    std::string line;

    if (startLineForCurrentFile >= numberOfLines && sessionFileExistsAndIsValid && 
        sessionData.lastTrainingFileName == txtFileLocation) 
    {
        std::cout << "All lines in " << txtFileLocation << " were already processed according to session data. Skipping." << std::endl;
        return;
    }
    // Start timing here
    auto startTime = std::chrono::high_resolution_clock::now();

    // get tokens
    std::vector<std::string> linesOfFile;
    while (std::getline(file, line)) {
        if(line.empty() == 1) {
            continue;
        }
        linesOfFile.push_back(line);
    }
    std::vector<std::string> tokensOfLine;      // hold tokens of single line for each process

    unsigned long long initialCumulativeLinesTrainedForSession = sessionFileExistsAndIsValid ? sessionData.cumulativeTotalLinesTrained : 0;
    unsigned long long linesProcessedInThisRun = 0;

    // tokenise each line and povide their respective emebeddings
    for(long long int k = startLineForCurrentFile; k < numberOfLines; k++)
    {
        tokensOfLine.clear();
        T.currentTokenCount = 0;
        int tok = 0;
        T.currentTokenCount = 0;
        std::cout << "Current Sentence is:- " << linesOfFile[k] << std::endl;
        TOK.splitSentence(linesOfFile[k], tokensOfLine);
        tokensOfLine.back() = "</s>";          // sentece terminator
        for(auto& tokens:tokensOfLine) {
            std::cout << tokens << "  ";
        }
        std::cout << std::endl;
        std::vector<std::vector<float>> sentenceEmbeddings(tokensOfLine.size(), std::vector<float>(EMBEDDING, 0.0f));
        std::cout << "Embeddings for tokens: " << sentenceEmbeddings.size() << std::endl;
        // get embeddings for prompt
        for(int j = 0; j < tokensOfLine.size(); j++) {
            // set embeddings and EVs
            sentenceEmbeddings[j] = TOK.getEmbeddingForToken(tokensOfLine[j]);
            int actual_row_in_ev = (T.currentTokenCount + j) % CONTEXT_WIN;
            for(int m1 = 0; m1 < x; m1++) {
                for(int m2 = 0; m2 < y; m2++) {
                    std::vector<float> v(EMBEDDING, 0.0f);
                    v = T.t[0].b[m1][m2].EV(actual_row_in_ev);
                    v += sentenceEmbeddings[j];
                    T.t[0].b[m1][m2].EV.addRow(v, actual_row_in_ev);
                }
            }
        }
        tok += tokensOfLine.size();
        // train the first block
        if(tok < CONTEXT_WIN)
        {
            #ifdef USE_CUDA
                std::cout << "Using CUDA Implementation" << std::endl;
                this->T.cuTrain(sentenceEmbeddings, tokensOfLine);
            #elif USE_OPENCL
                std::cout << "Using OpenCL Implementation" << std::endl;
                this->T.clTrain(sentenceEmbeddings, tokensOfLine);
            #elif USE_CPU
                std::cout << "Using C++ Implementation" << std::endl;
                this->T.train(sentenceEmbeddings, tokensOfLine);
            #endif
        }
        else {
            throw std::runtime_error("LOCAL CONTEXT LIMIT REACHED");
        }
        totalTokens += tok;
        linesProcessedInThisRun++;

        // Update and save session data after each line
        sessionData.lastTrainingFileName = txtFileLocation;
        sessionData.linesProcessedInLastFile = k + 1; // We just finished line k (0-indexed)
        sessionData.cumulativeTotalLinesTrained = initialCumulativeLinesTrainedForSession + linesProcessedInThisRun;
        sessionData.cumulativeTotalTokensProcessed = this->totalTokens;
        sessionData.cumulativeTotalTrainCount = this->T.trainCount;
        sessionData.lastBlockCountState = this->T.blockCount;
        sessionData.lastEpochCountState = this->T.epochCount;
        sessionData.vocabSizeSnapshot = this->T.vocabsize;
        if (!currentChatLogPath.empty()) {
            sessionData.save(currentChatLogPath);
        }

        std::cout << "complete " << k << "th part. Progress saved." << std::endl;
        if(T.blockCount == 1) 
            T.t[0].serialise(T.t[0].blockFilePath);
        std::cout << "              ---------------- To New LINE --------------              " << std::endl;
    }
    std::cout << "Training complete for file " << txtFileLocation << std::endl;
    newChat();

    // copy to other blocks and serialise
    // End timing here
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "Total training time for file " << txtFileLocation << ": " << duration.count() << " ms" << std::endl;
    std::cout << "-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-" << std::endl;
}


/**
 * @brief train all blocks using training data from txt file
 * @param txtFile path 2 txt file for training 2 or more or all blocks
 */
void model::trainModelSentence(const std::string& txtFileLocation) 
{
    // basic checks for file
    if (txtFileLocation.empty()) {
        throw std::runtime_error("Training data file path cannot be empty!");
    }
    if (!std::filesystem::exists(txtFileLocation) || !std::filesystem::is_regular_file(txtFileLocation)) {
        throw std::runtime_error("Training data file not found or is not a regular file: " + txtFileLocation);
    }
    if (std::filesystem::path(txtFileLocation).extension() != ".txt") {
        throw std::runtime_error("Training data file must be a .txt file: " + txtFileLocation);
    }

    TrainingSessionData sessionData;
    bool sessionFileExistsAndIsValid = false;
    int startLineForCurrentFile = 0;

    // Attempt to load session data
    if (!currentChatLogPath.empty() && std::filesystem::exists(currentChatLogPath)) {
        if (std::filesystem::is_regular_file(currentChatLogPath)) {
            std::ifstream tempInfoFile(currentChatLogPath);
            if (tempInfoFile.is_open()) {
                if (tempInfoFile.peek() != std::ifstream::traits_type::eof()) { // Check if file is not empty
                    if (sessionData.load(currentChatLogPath)) {
                        sessionFileExistsAndIsValid = true;
                        std::cout << "Successfully loaded session data from: " << currentChatLogPath;
                        std::cout << " \nPrevious file: " << sessionData.lastTrainingFileName
                                  << ",\nLines processed in it: " << sessionData.linesProcessedInLastFile;
                        std::cout << "\nCumulative Lines Trained: " << sessionData.cumulativeTotalLinesTrained
                                  << ",\nTokens: " << sessionData.cumulativeTotalTokensProcessed
                                  << ",\nTrainCount: " << sessionData.cumulativeTotalTrainCount;
                        std::cout << "\nLast VocabSize: " << sessionData.vocabSizeSnapshot
                                  << ",\nLast BlockCount: " << sessionData.lastBlockCountState
                                  << ",\nLast EpochCount: " << sessionData.lastEpochCountState << std::endl;
                    }
                     else {
                        std::cerr << "Warning: Failed to parse session data from " << currentChatLogPath 
                                  << ". Starting with fresh session values." << std::endl;
                    }
                } 
                else {
                    std::cout << "Session data file " << currentChatLogPath << " is empty. Starting with fresh session values." << std::endl;
                }
                tempInfoFile.close();
            } 
            else {
                std::cerr << "Warning: Could not open session data file " << currentChatLogPath 
                          << " for reading. Starting with fresh session values." << std::endl;
            }
        }
        else {
            std::cerr << "Warning: Session data path " << currentChatLogPath 
                       << " is not a regular file. Starting with fresh session values." << std::endl;
        }
    }
    else {
        if (currentChatLogPath.empty()) {
            std::cout << "Session data file path (currentChatLogPath) is not set. Starting with fresh session values." << std::endl;
        }
        else {
            std::cout << "Session data file " << currentChatLogPath << " not found. Starting with fresh session values." << std::endl;
        }
    }

    // Initialize model state based on session data or defaults
    this->totalTokens = sessionFileExistsAndIsValid ? sessionData.cumulativeTotalTokensProcessed : 0;
    this->T.trainCount = sessionFileExistsAndIsValid ? sessionData.cumulativeTotalTrainCount : 0;
    this->T.blockCount = 0;
    this->T.epochCount = 0;
    int currentLine = 0;

    if (sessionFileExistsAndIsValid && sessionData.lastTrainingFileName == txtFileLocation) {
        startLineForCurrentFile = sessionData.linesProcessedInLastFile;
        std::cout << "Resuming training for " << txtFileLocation << " from line " << startLineForCurrentFile << "." << std::endl;
    }

    // open file and read line by line
    std::ifstream file(txtFileLocation);
    if (!file.is_open()) {
        throw std::runtime_error("Error opening training data file: " + txtFileLocation);
    }
    long long int numberOfLines = countLineInTXT(txtFileLocation);
    if (numberOfLines <= 0) {
        throw std::runtime_error("No training data found in the specified file!");
    }
    std::cout << "Total training lines: " << numberOfLines << std::endl;
    std::string line;

    if (startLineForCurrentFile >= numberOfLines && sessionFileExistsAndIsValid && 
        sessionData.lastTrainingFileName == txtFileLocation) 
    {
        std::cout << "All lines in " << txtFileLocation << " were already processed according to session data. Skipping." << std::endl;
        return;
    }
    // Start timing here
    auto startTime = std::chrono::high_resolution_clock::now();

    // get tokens
    std::vector<std::string> linesOfFile;
    while (std::getline(file, line)) {
        if(line.empty() == 1) {
            continue;
        }
        linesOfFile.push_back(line);
    }
    std::vector<std::string> tokensOfLine;              // hold tokens of single line for each process

    long long initialCumulativeLinesTrainedForSession = sessionFileExistsAndIsValid ? sessionData.cumulativeTotalLinesTrained : 0;
    long long linesProcessedInThisRun = 0;

    // tokenise each line and povide their respective emebeddings
    for(long long int k = startLineForCurrentFile; k < numberOfLines; k++)
    {
        tokensOfLine.clear();
        T.currentTokenCount = 0;
        int tok = 0;
        T.currentTokenCount = 0;
        TOK.splitSentence(linesOfFile[k], tokensOfLine);
        tokensOfLine.back() = "</s>";          // sentece terminator
        std::vector<std::vector<float>> sentenceEmbeddings;
        std::vector<std::string> responseTokens;
        // get embeddings for prompt
        for(int j = 0; j < tokensOfLine.size(); j++) {
            std::vector<float> embed(d, 0.0f);
            this->T.getEmbedding(tokensOfLine[j], embed);
            sentenceEmbeddings.push_back(embed);
            int actual_row_in_ev = (T.currentTokenCount + j) % CONTEXT_WIN;
            for(int m1 = 0; m1 < x; m1++) {
                for(int m2 = 0; m2 < y; m2++) {
                    std::vector<float> v(EMBEDDING, 0.0f);
                    v = T.t[0].b[m1][m2].EV(actual_row_in_ev);
                    v += embed;
                    T.t[0].b[m1][m2].EV.addRow(v, actual_row_in_ev);
                }
            }
        }
        tok += tokensOfLine.size();
        // train the first block
        if(tok < FULL_CONTEXT)
        {
        #ifdef USE_CUDA
            std::cout << "Using CUDA Implementation" << std::endl;
            this->T.cuTrain(sentenceEmbeddings, responseTokens);
        #elif USE_OPENCL
            std::cout << "Using OpenCL Implementation" << std::endl;
            this->T.clTrain(sentenceEmbeddings, responseTokens);
        #elif USE_CPU
            std::cout << "Using C++ Implementation" << std::endl;
            this->T.train(sentenceEmbeddings, responseTokens);
        #endif
        }
        else {
            throw std::runtime_error("LOCAL CONTEXT LIMIT REACHED");
        }
        totalTokens += tok;
        linesProcessedInThisRun++;

        // Update and save session data after each line
        sessionData.lastTrainingFileName = txtFileLocation;
        sessionData.linesProcessedInLastFile = k + 1; // We just finished line k (0-indexed)
        sessionData.cumulativeTotalLinesTrained = initialCumulativeLinesTrainedForSession + linesProcessedInThisRun;
        sessionData.cumulativeTotalTokensProcessed = this->totalTokens;
        sessionData.cumulativeTotalTrainCount = this->T.trainCount;
        sessionData.lastBlockCountState = this->T.blockCount;
        sessionData.lastEpochCountState = this->T.epochCount;
        sessionData.vocabSizeSnapshot = this->T.vocabsize;
        if (!currentChatLogPath.empty()) {
            sessionData.save(currentChatLogPath);
        }

        std::cout << "complete " << k << "th part. Progress saved." << std::endl;
        if(T.blockCount == 1) T.t[0].serialise(T.t[0].blockFilePath);
    }
    std::cout << "Training complete for file " << txtFileLocation << std::endl;
    newChat();

    // copy to other blocks and serialise
    // End timing here
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "Total training time for file " << txtFileLocation << ": " << duration.count() << " ms" << std::endl;
    std::cout << "-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-" << std::endl;
}
