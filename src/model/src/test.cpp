#include "include/model.hpp"
#include <string>
#include <filesystem>
#include <fstream>


/**
 * @brief test model on prompt-response pairs
 * @param context_window context window allowed for testing
 * @param context use embeddings or de-embedding matrix for prediction
 */
void model::test(const std::string &txtFileLocation, int context_window, bool context)
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
    getSessionData(sessionData, sessionFileExistsAndIsValid, startLineForCurrentFile);

    // Initialize model state based on session data or defaults
    totalTokens = sessionFileExistsAndIsValid ? sessionData.cumulativeTotalTokensProcessed : 0;
    T.trainCount = sessionFileExistsAndIsValid ? sessionData.cumulativeTotalTrainCount : 0;
    T.blockCount = 0;
    T.epochCount = 0;
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

    unsigned long long initialCumulativeLinesTrainedForSession = sessionFileExistsAndIsValid ? sessionData.cumulativeTotalLinesTrained : 0;
    unsigned long long linesProcessedInThisRun = 0;
    std::cout << "Starting sequence-sequence Training with trainSeq2Seq :)" << std::endl;
    for(unsigned long long k = startLineForCurrentFile; k < numberOfLines; k++)
    {
        tokensOfFile.clear();
        oddSentence.clear();
        evenSentence.clear();
        T.currentTokenCount = 0;
        // split line to multiple sentences
        splitLine2SubSentences(linesOfFile[k], tokensOfFile);

        // Ensure tokensOfFile contains pairs of sub-sentences (sequence1, sequence2)
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
            // sequence first tokens and indices for embeddings
            TOK.splitSentence(tokensOfFile[2*i], oddSentence[i]);
            oddSentence[i].push_back("</s>");
            std::vector<std::vector<float>> sequence1Embeddings, sequence2Embeddings;
            T.indexVec.resize(oddSentence[i].size(), 0);
            T.getIndexOfAllTokens(oddSentence[i], T.indexVec);
            // get embeddings for sequence1
            sequence1Embeddings.clear();
            for(int j = 0; j < oddSentence[i].size(); j++) {
                std::vector<float> embed(d, 0.0f);
                embed = T.embeddings(T.indexVec[j]);
                sequence1Embeddings.push_back(embed);
            }

            // sequence second tokens and indices for embeddings
            TOK.splitSentence(tokensOfFile[2*i+1], evenSentence[i]);
            evenSentence[i].push_back("</s>");
            T.indexVec.clear();
            T.indexVec.resize(evenSentence[i].size(), 0);
            T.getIndexOfAllTokens(evenSentence[i], T.indexVec);

            // get embedding for sequence2
            sequence2Embeddings.clear();
            std::vector<std::string> sequence2Tokens;
            for(int j = 0; j < evenSentence[i].size(); j++) {
                std::vector<float> embed(d, 0.0f);
                embed = T.embeddings(T.indexVec[j]);
                sequence2Embeddings.push_back(embed);
                sequence2Tokens.push_back(evenSentence[i][j]);
            }

            // total tokens in this pair
            tok += sequence1Embeddings.size() + sequence2Embeddings.size();
            std::cout << "Current Token Count: " << T.currentTokenCount 
                      << " | No. of Tokens in Sequence " << 2*i + 1 <<": " << oddSentence[i].size() 
                      << " | No. of Tokens in Sequence " << 2*i + 2 << ": " << evenSentence[i].size() << std::endl;
            std::cout << "Sequence " << 2*i + 1 << " ---> ";
            for(int ch = 0; ch < oddSentence[i].size(); ch++)
                std::cout << oddSentence[i][ch] << "  ";
            std::cout << std::endl;
            std::cout << "Sequence " << 2*i + 2 << " ---> ";
            for(int ch = 0; ch < evenSentence[i].size(); ch++)
                std::cout << evenSentence[i][ch] << "  ";
            std::cout << std::endl;

            // train the first block
            if(tok < context_window)
            {
                #ifdef USE_CU
                    std::cout << "Using CUDA Implementation" << std::endl;
                #elif USE_CL
                    std::cout << "Using OpenCL Implementation" << std::endl;
                #elif USE_CPU
                    std::cout << "Using C++ Implementation" << std::endl;
                #endif
                std::cout << "---> Training complete for Pair " << i + 1 << "." << std::endl;

                if(contextTrain == 1) {
                    TOK.saveEmbeddings(TOK.path2token + "/_embeddings_only.csv", T.deEmbeddings.mapped_data);
                    TOK.savedeEmbeddings(TOK.path2token + "/_deEmbeddings_only.csv", T.deEmbeddings.mapped_data);
                }
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

        setSessionData(sessionData, k, linesProcessedInThisRun);
        if (!currentChatLogPath.empty()) {
            sessionData.save(currentChatLogPath);
        }

        std::cout << "Completed " << k << "th part. Progress saved." << std::endl;
        for(int i = 0; i < T.currentTokenCount % CONTEXT_WIN; i++){
            T.blocks[i].serialise(T.blocks[i].blockFilePath);
        }
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