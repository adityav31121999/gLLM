
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
void model::trainBlockPR(const std::string& txtFileLocation) 
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
                        std::cout << "Successfully loaded session data from: " << currentChatLogPath << std::endl;
                        std::cout << " \nPrevious file: " << sessionData.lastTrainingFileName
                                  << ",\nLines processed in it: " << sessionData.linesProcessedInLastFile << std::endl;
                        std::cout << "\nCumulative Lines Trained: " << sessionData.cumulativeTotalLinesTrained
                                  << ",\nTokens: " << sessionData.cumulativeTotalTokensProcessed
                                  << ",\nTrainCount: " << sessionData.cumulativeTotalTrainCount << std::endl;
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
    unsigned long long numberOfLines = countLineInTXT(txtFileLocation);
    if (numberOfLines <= 0) {
        throw std::runtime_error("No training data found in the specified file!");
    }
    std::cout << "Total training lines: " << numberOfLines << std::endl;
    std::string line;

    if (startLineForCurrentFile >= numberOfLines && sessionFileExistsAndIsValid && sessionData.lastTrainingFileName == txtFileLocation) {
        std::cout << "All lines in " << txtFileLocation << " were already processed according to session data. Skipping." << std::endl;
        return;
    }
    // Start timing here
    auto startTime = std::chrono::high_resolution_clock::now();

    // get tokens
    std::vector<std::string> linesOfFile, tokensOfFile;
    while (std::getline(file, line)) {
        if(line.empty() == 1) {
            continue;
        }
        linesOfFile.push_back(line);
    }
    std::vector<std::vector<std::string>> oddSentence, evenSentence;

    long long initialCumulativeLinesTrainedForSession = sessionFileExistsAndIsValid ? sessionData.cumulativeTotalLinesTrained : 0;
    long long linesProcessedInThisRun = 0;
    bool sortIt = 0;    // off
    for(unsigned long long k = startLineForCurrentFile; k < numberOfLines; k++)
    {
        tokensOfFile.clear();
        oddSentence.clear();
        evenSentence.clear();
        T.currentTokenCount = 0;
        splitLine2SubSentences(linesOfFile[k], tokensOfFile);

        // Ensure tokensOfFile contains pairs of sub-sentences (prompt, response)
        if (tokensOfFile.size() % 2 != 0) {
            std::cerr << "Warning: Line " << k << " in file " << txtFileLocation 
                      << " resulted in an odd number of sub-sentences (" << tokensOfFile.size() 
                      << "). Skipping this line." << std::endl;
            continue; // Skip to the next line from the input file
        }

        if (tokensOfFile.empty() && k < numberOfLines) { // k < numberOfLines to ensure it's not just end of file
            std::cout << "Info: Line " << k << " in file " << txtFileLocation 
                      << " produced no tokens. Skipping." << std::endl;
            continue; // Skip to the next line
        }

        int num_pairs = tokensOfFile.size() / 2;
        if (num_pairs > 0) {
            oddSentence.resize(num_pairs);
            evenSentence.resize(num_pairs);
        }
        else if (!tokensOfFile.empty()) {
            std::cerr << "Warning: Line " << k << " has tokens but num_pairs is 0. Skipping." << std::endl;
            continue;
        }
        std::cout << "In line " << k << ". Number of pairs of Odd-Even Lines are: " << num_pairs << "." << std::endl;
        int tok = 0;
        T.currentTokenCount = 0;
        // Loop based on the number of pairs identified
        for(int i = 0; i < num_pairs; i++) {
            TOK.splitSentence(tokensOfFile[2*i], oddSentence[i]);    // WILL TAKE EVEN INDEX (odd NUMBERED SENTENCE)
            TOK.splitSentence(tokensOfFile[2*i+1], evenSentence[i]); // WILL TAKE ODD INDEX (odd NUMBERED SENTENCE)
            evenSentence[i].resize(evenSentence[i].size() + 1);
            evenSentence[i].back() = "<@#0>";
            std::vector<std::vector<float>> promptEmbeddings, responseEmbeddings;
            std::vector<std::string> responseTokens;
            // get embeddings for prompt
            for(int j = 0; j < oddSentence[i].size(); j++) {
                std::vector<float> embed = TOK.getEmbeddingForToken(oddSentence[i][j]);
                if (embed.empty()) {
                    std::cerr << "Warning: No embedding found for prompt token '" << oddSentence[i][j] << "'. Using a zero vector." << std::endl;
                    embed.assign(d, 0.0f);
                }
                promptEmbeddings.push_back(embed);
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
            // get embedding for response
            for(int j = 0; j < evenSentence[i].size(); j++) {
                std::vector<float> embed = TOK.getEmbeddingForToken(evenSentence[i][j]);
                if (embed.empty()) {
                    std::cerr << "Warning: No embedding found for response token '" << evenSentence[i][j] << "'. Using a zero vector." << std::endl;
                    embed.assign(d, 0.0f);
                }
                responseEmbeddings.push_back(embed);
                responseTokens.push_back(evenSentence[i][j]);
            }
            tok += promptEmbeddings.size() + responseEmbeddings.size();
            // train the first block
            if(tok < CONTEXT_WIN)
            {
                #ifdef USE_CUDA
                    std::cout << "Using CUDA Implementation" << std::endl;
                    this->T.cuTrain(promptEmbeddings, responseEmbeddings, responseTokens);
                #elif USE_OPENCL
                    std::cout << "Using OpenCL Implementation" << std::endl;
                    this->T.clTrain(promptEmbeddings, responseEmbeddings, responseTokens);
                #elif USE_CPU
                    std::cout << "Using C++ Implementation" << std::endl;
                    this->T.train(promptEmbeddings, responseEmbeddings, responseTokens);
                #endif
            }
            else {
                throw std::runtime_error("LOCAL CONTEXT LIMIT REACHED");
            }
            totalTokens += tok;
        }
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
void model::trainModelPR(const std::string& txtFileLocation) 
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
                        std::cout << "Successfully loaded session data from: " << currentChatLogPath << std::endl;
                        std::cout << " \nPrevious file: " << sessionData.lastTrainingFileName
                                  << ",\nLines processed in it: " << sessionData.linesProcessedInLastFile << std::endl;
                        std::cout << "\nCumulative Lines Trained: " << sessionData.cumulativeTotalLinesTrained
                                  << ",\nTokens: " << sessionData.cumulativeTotalTokensProcessed
                                  << ",\nTrainCount: " << sessionData.cumulativeTotalTrainCount << std::endl;
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
    unsigned long long numberOfLines = countLineInTXT(txtFileLocation);
    if (numberOfLines <= 0) {
        throw std::runtime_error("No training data found in the specified file!");
    }
    std::cout << "Total training lines: " << numberOfLines << std::endl;
    std::string line;

    if (startLineForCurrentFile >= numberOfLines && sessionFileExistsAndIsValid && sessionData.lastTrainingFileName == txtFileLocation) {
        std::cout << "All lines in " << txtFileLocation << " were already processed according to session data. Skipping." << std::endl;
        return;
    }
    // Start timing here
    auto startTime = std::chrono::high_resolution_clock::now();

    // get tokens
    std::vector<std::string> linesOfFile, tokensOfFile;
    while (std::getline(file, line)) {
        if(line.empty() == 1) {
            continue;
        }
        linesOfFile.push_back(line);
    }
    std::vector<std::vector<std::string>> oddSentence, evenSentence;

    long long initialCumulativeLinesTrainedForSession = sessionFileExistsAndIsValid ? sessionData.cumulativeTotalLinesTrained : 0;
    long long linesProcessedInThisRun = 0;
    bool sortIt = 0;    // off
    for(unsigned long long k = startLineForCurrentFile; k < numberOfLines; k++)
    {
        tokensOfFile.clear();
        oddSentence.clear();
        evenSentence.clear();
        T.currentTokenCount = 0;
        splitLine2SubSentences(linesOfFile[k], tokensOfFile);

        // Ensure tokensOfFile contains pairs of sub-sentences (prompt, response)
        if (tokensOfFile.size() % 2 != 0) {
            std::cerr << "Warning: Line " << k << " in file " << txtFileLocation 
                      << " resulted in an odd number of sub-sentences (" << tokensOfFile.size() 
                      << "). Skipping this line." << std::endl;
            continue; // Skip to the next line from the input file
        }

        if (tokensOfFile.empty() && k < numberOfLines) {
            std::cout << "Info: Line " << k << " in file " << txtFileLocation 
                      << " produced no tokens. Skipping." << std::endl;
            continue; // Skip to the next line
        }

        int num_pairs = tokensOfFile.size() / 2;
        if (num_pairs > 0) {
            oddSentence.resize(num_pairs);
            evenSentence.resize(num_pairs);
        }
        else if (!tokensOfFile.empty()) {
            std::cerr << "Warning: Line " << k << " has tokens but num_pairs is 0. Skipping." << std::endl;
            continue;
        }
        std::cout << "In line " << k << ". Number of pairs of Odd-Even Lines are: " << num_pairs << "." << std::endl;
        int tok = 0;
        T.currentTokenCount = 0;
        // Loop based on the number of pairs identified
        for(int i = 0; i < num_pairs; i++) {
            TOK.splitSentence(tokensOfFile[2*i], oddSentence[i]);    // WILL TAKE EVEN INDEX (odd NUMBERED SENTENCE)
            TOK.splitSentence(tokensOfFile[2*i+1], evenSentence[i]); // WILL TAKE ODD INDEX (odd NUMBERED SENTENCE)
            evenSentence[i].resize(evenSentence[i].size() + 1);
            evenSentence[i].back() = "<@#0>";
            std::vector<std::vector<float>> promptEmbeddings, responseEmbeddings;
            std::vector<std::string> responseTokens;
            // get embeddings for prompt
            for(int j = 0; j < oddSentence[i].size(); j++) {
                std::vector<float> embed = TOK.getEmbeddingForToken(oddSentence[i][j]);
                if (embed.empty()) {
                    std::cerr << "Warning: No embedding found for prompt token '" << oddSentence[i][j] << "'. Using a zero vector." << std::endl;
                    embed.assign(d, 0.0f);
                }
                promptEmbeddings.push_back(embed);
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
            // get embedding for response
            for(int j = 0; j < evenSentence[i].size(); j++) {
                std::vector<float> embed = TOK.getEmbeddingForToken(evenSentence[i][j]);
                if (embed.empty()) {
                    std::cerr << "Warning: No embedding found for response token '" << evenSentence[i][j] << "'. Using a zero vector." << std::endl;
                    embed.assign(d, 0.0f);
                }
                responseEmbeddings.push_back(embed);
                responseTokens.push_back(evenSentence[i][j]);
            }
            tok += promptEmbeddings.size() + responseEmbeddings.size();
            // train the first block
            if(tok < FULL_CONTEXT)
            {
                #ifdef USE_CUDA
                    std::cout << "Using CUDA Implementation" << std::endl;
                    this->T.cuTrain(promptEmbeddings, responseEmbeddings, responseTokens);
                #elif USE_OPENCL
                    std::cout << "Using OpenCL Implementation" << std::endl;
                    this->T.clTrain(promptEmbeddings, responseEmbeddings, responseTokens);
                #elif USE_CPU
                    std::cout << "Using C++ Implementation" << std::endl;
                    this->T.train(promptEmbeddings, responseEmbeddings, responseTokens);
                #endif
            }
            else {
                throw std::runtime_error("LOCAL CONTEXT LIMIT REACHED");
            }
            totalTokens += tok;
        }
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
