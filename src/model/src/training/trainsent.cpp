#include "include/model.hpp"
#include <filesystem>
#include <sstream>
#include <iomanip>
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
        throw std::runtime_error("trainBlockSentence: Training data file path cannot be empty!");
    }
    if (!std::filesystem::exists(txtFileLocation) || !std::filesystem::is_regular_file(txtFileLocation)) {
        throw std::runtime_error("trainBlockSentence: Training data file not found or is not a regular file: " + txtFileLocation);
    }
    if (std::filesystem::path(txtFileLocation).extension() != ".txt") {
        throw std::runtime_error("trainBlockSentence: Training data file must be a .txt file: " + txtFileLocation);
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
                        std::cout << "trainBlockSentence: Successfully loaded session data from: " << currentChatLogPath;
                        std::cout << " \nPrevious file: " << sessionData.lastTrainingFileName
                                  << ",\nLines processed in it: " << sessionData.linesProcessedInLastFile;
                        std::cout << "\nCumulative Lines Trained: " << sessionData.cumulativeTotalLinesTrained
                                  << ",\nTokens: " << sessionData.cumulativeTotalTokensProcessed
                                  << ",\nTrainCount: " << sessionData.cumulativeTotalTrainCount;
                        std::cout << "\nLast VocabSize: " << sessionData.vocabSizeSnapshot
                                  << ",\nLast BlockCount: " << sessionData.lastBlockCountState
                                  << ",\nLast EpochCount: " << sessionData.lastEpochCountState << std::endl;
                        // deserialise all blocks
                        T.t[0].deserialise(T.t[0].blockFilePath);
                    }
                     else {
                        std::cerr << "trainBlockSentence: Warning: Failed to parse session data from " << currentChatLogPath 
                                  << ". Starting with fresh session values." << std::endl;
                    }
                } 
                else {
                    std::cout << "trainBlockSentence: Session data file " << currentChatLogPath << " is empty. Starting with fresh session values." << std::endl;
                }
                tempInfoFile.close();
            } 
            else {
                std::cerr << "trainBlockSentence: Warning: Could not open session data file " << currentChatLogPath 
                          << " for reading. Starting with fresh session values." << std::endl;
            }
        }
        else {
            std::cerr << "trainBlockSentence: Warning: Session data path " << currentChatLogPath 
                       << " is not a regular file. Starting with fresh session values." << std::endl;
        }
    }
    else {
        if (currentChatLogPath.empty()) {
            std::cout << "trainBlockSentence: Session data file path (currentChatLogPath) is not set. Starting with fresh session values." << std::endl;
        }
        else {
            std::cout << "trainBlockSentence: Session data file " << currentChatLogPath << " not found. Starting with fresh session values." << std::endl;
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
        std::cout << "trainBlockSentence: Resuming training for " << txtFileLocation << " from line " << startLineForCurrentFile << "." << std::endl;
    }

    // open file and read line by line
    std::ifstream file(txtFileLocation);
    if (!file.is_open()) {
        throw std::runtime_error("trainBlockSentence: Error opening training data file: " + txtFileLocation);
    }
    long long int numberOfLines = countLineInTXT(txtFileLocation);
    if (numberOfLines <= 0) {
        throw std::runtime_error("trainBlockSentence: No training data found in the specified file!");
    }
    std::cout << "trainBlockSentence: Total training lines: " << numberOfLines << std::endl;
    std::string line;

    if (startLineForCurrentFile >= numberOfLines && sessionFileExistsAndIsValid && 
        sessionData.lastTrainingFileName == txtFileLocation) {
        std::cout << "trainBlockSentence: All lines in " << txtFileLocation << " were already processed according to session data. Skipping this file." << std::endl;
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
        tokensOfLine.back() = "<@#0>";          // sentece terminator
        std::vector<std::vector<float>> sentenceEmbeddings(tokensOfLine.size(), std::vector<float>(EMBEDDING, 0.0f));
        std::vector<std::string> responseTokens;
        // get embeddings for prompt
        for(int j = 0; j < tokensOfLine.size(); j++) {
            std::vector<float> embed(d, 0.0f);          // embedding (loaded from tokeniser data)
            T.getEmbedding(tokensOfLine[j], embed);
            positionalEmbedding(embed, sentenceEmbeddings[j], j);
        }
        // Corrected logging to show the actual embedding vector for the first token
        std::cout << "trainBlockSentence: total tokens in vocabulary: " << T.tokens.size() << std::endl;
        tok += tokensOfLine.size();
        // train the first block
        if(tok < CONTEXT_WIN)
        {
            std::cout << "Training line " << k << " with total number of tokens in it " << tokensOfLine.size() << std::endl;
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
            throw std::runtime_error("trainBlockSentence: LOCAL CONTEXT LIMIT REACHED");
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
        sessionData.lambdaL1 = this->lambda_L1;
        sessionData.lambdaL2 = this->lambda_L2;
        sessionData.currentLearning = this->T.learning;
        sessionData.totalLearning = this->T.totalLearning;
        sessionData.adLearning = this->T.totalLearning/this->T.trainCount;
        sessionData.cumulativeError = this->T.error;
        if (!currentChatLogPath.empty()) {
            sessionData.save(currentChatLogPath);
        }

        std::cout << "complete " << k << "th part. Progress saved." << std::endl;
        if(T.blockCount == 1) T.t[0].serialise(T.t[0].blockFilePath);
        std::cout << "-----------------------------------------------------------------------" << std::endl;
    }
    std::cout << "trainBlockSentence: Training complete for file " << txtFileLocation << std::endl;
    newChat();

    // copy to other blocks and serialise
    // End timing here
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "trainBlockSentence: Total training time for file " << txtFileLocation << ": " << duration.count() << " ms" << std::endl;
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
    T.tokens = TOK.getTokens();
    std::vector<float> vec(EMBEDDING, 0.0f);
    std::transform(vec.begin(), vec.end(), vec.begin(), [j = 10](float i){ 
        return terminatorEmbed(i, j); 
    });
    T.embeddings.addRow(vec, T.tokens.size());
    vocabsize = T.tokens.size() + 1;
    // deserialise all blocks
    for(int i = 0; i < m; i++) T.t[i].deserialise(T.t[i].blockFilePath);

    // tokenise each line and povide their respective emebeddings
    for(long long int k = startLineForCurrentFile; k < numberOfLines; k++)
    {
        tokensOfLine.clear();
        T.currentTokenCount = 0;
        int tok = 0;
        T.currentTokenCount = 0;
        TOK.splitSentence(linesOfFile[k], tokensOfLine);
        std::vector<std::vector<float>> sentenceEmbeddings(tokensOfLine.size(), std::vector<float>(EMBEDDING, 0.0f));
        std::vector<std::string> responseTokens;
        // get embeddings for prompt
        for(int j = 0; j < tokensOfLine.size(); j++) {
            std::vector<float> embed(d, 0.0f);
            embed = TOK.getEmbeddingForToken(tokensOfLine[j]);
            sentenceEmbeddings[j] = embed;
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
        tokensOfLine.back() = "<@#0>";          // sentece terminator
        sentenceEmbeddings.back() = vec;        // append the 
        // Corrected logging to show the actual embedding vector for the first token
        std::cout << "total tokens in vocabulary: " << T.tokens.size() << std::endl;
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
        sessionData.lambdaL1 = this->lambda_L1;
        sessionData.lambdaL2 = this->lambda_L2;
        sessionData.currentLearning = this->T.learning;
        sessionData.totalLearning = this->T.totalLearning;
        sessionData.adLearning = this->T.totalLearning/this->T.trainCount;
        sessionData.cumulativeError = this->T.error;
        if (!currentChatLogPath.empty()) {
            sessionData.save(currentChatLogPath);
        }

        std::cout << "complete " << k << "th part. Progress saved." << std::endl;
        newChat();
    }
    std::cout << "Training complete for file " << txtFileLocation << std::endl;
    serialise();
    // copy to other blocks and serialise
    // End timing here
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "Total training time for file " << txtFileLocation << ": " << duration.count() << " ms" << std::endl;
    std::cout << "-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-" << std::endl;
}
