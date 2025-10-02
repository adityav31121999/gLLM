#include "include/model.hpp" // Adjust path as per your project structure
#include <filesystem>
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
    learning = learning;
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
    info = info;
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

void model::setTokens2Transformer()
{
    T.embeddings = TOK.getEmbeddings();
    T.deEmbeddings = TOK.getDeEmbeddings();
    T.tokens = TOK.getTokens();
    T.vocabsize = TOK.getVocabularySize();
    // sort the  T.tokens lexicographically in descending order of their length
    std::sort(T.tokens.begin(),  T.tokens.end(), [](const std::string& a, const std::string& b) {
        if(a.length() != b.length()) {
            // larger length first
            return a.length() > b.length();
        }
        // alphabetically
        return a < b;
    });
    /*
    for(int i = 0; i <  T.tokens.size(); i++) {
        std::cout <<  T.tokens[i] << " | ";
    }
    */
    std::cout << "Tokeniser data set to transformer :)" << std::endl;
}


/**
 * @brief get previous sessions data
 * @param sessionFileExistsAndIsValid check for file existence
 * @param startLineForCurrentFile get line from previous session to continue training from there
 */
void model::getSessionData(TrainingSessionData& sessionData, bool& sessionFileExistsAndIsValid, 
            int& startLineForCurrentFile)
{
    // get session from model.currentChatLogPath
    sessionFileExistsAndIsValid = 0;
    startLineForCurrentFile = 0;
    // Attempt to load session data
    if (!currentChatLogPath.empty() && std::filesystem::exists(currentChatLogPath)) {
        if (std::filesystem::is_regular_file(currentChatLogPath)) {
            std::ifstream tempInfoFile(currentChatLogPath);
            if (tempInfoFile.is_open()) {
                if (tempInfoFile.peek() != std::ifstream::traits_type::eof()) {
                    // Check if file is not empty
                    if (sessionData.load(currentChatLogPath)) {
                        sessionFileExistsAndIsValid = true;
                        std::cout << "Successfully loaded session data from: " << currentChatLogPath << std::endl;
                        std::cout << "  Previous file: " << sessionData.lastTrainingFileName << std::endl;
                        std::cout << "  Lines processed in it: " << sessionData.linesProcessedInLastFile << std::endl;
                        std::cout << "  Cumulative Lines Trained: " << sessionData.cumulativeTotalLinesTrained << std::endl;
                        std::cout << "  Tokens: " << sessionData.cumulativeTotalTokensProcessed << std::endl;
                        std::cout << "  TrainCount: " << sessionData.cumulativeTotalTrainCount << std::endl;
                        std::cout << "  Last VocabSize: " << sessionData.vocabSizeSnapshot << std::endl;
                        std::cout << "  Last BlockCount: " << sessionData.lastBlockCountState << std::endl;
                        std::cout << "  Last EpochCount: " << sessionData.lastEpochCountState << std::endl;
                        std::cout << "  Last LearningRate: " << sessionData.lastLearningRateRecorded << std::endl;

                        // Restore model and transformer state from sessionData ---
                        totalTokens = sessionData.cumulativeTotalTokensProcessed;
                        T.trainCount = sessionData.cumulativeTotalTrainCount;
                        T.blockCount = sessionData.lastBlockCountState;
                        T.epochCount = sessionData.lastEpochCountState;
                        T.vocabsize = sessionData.vocabSizeSnapshot;
                        vocabsize = sessionData.vocabSizeSnapshot; // also in model

                        // Restore learning rates and errors
                        learning = sessionData.lastLearningRateRecorded; // also in model
                        T.totalLearning = sessionData.totalLearning;
                        T.averageLearningRate = sessionData.averageLearningRate;

                        // Restore totals and averages for metrics
                        T.perplexityCE = sessionData.perplexityCE;
                        T.perplexityBCE = sessionData.perplexityBCE;
                        T.avgBCELoss = sessionData.avgBCELoss;
                        T.avgCELoss = sessionData.avgCELoss;
                        T.averagePerplexity = sessionData.averagePerplexity;
                        T.totalBCELoss = sessionData.totalBCELoss;
                        T.totalCELoss = sessionData.totalCELoss;
                        T.error = sessionData.cumulativeError;
                        T.totalCEPerplexity = sessionData.totalCEPerplexity;
                        T.totalBCEPerplexity = sessionData.totalBCEPerplexity;

                        // Restore booleans
                        T.isSelf = sessionData.isSelf;
                        isSelf = sessionData.isSelf;
                        T.contextTrain = sessionData.contextTrain;
                        contextTrain = sessionData.contextTrain;
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
}


/**
 * @brief set session data values
 * @param sessionData store current session values here
 * @param k current line processed in this session
 * @param linesProcessedIn1session total lines processed
 */
void model::setSessionData(TrainingSessionData& sessionData, int k, int linesProcessedIn1Session)
{
    // set session in model.currentChatLogPath
    sessionData.linesProcessedInLastFile = k + 1;
    sessionData.cumulativeTotalTokensProcessed = totalTokens;
    sessionData.cumulativeTotalTrainCount = T.trainCount;
    sessionData.lastBlockCountState = T.blockCount;
    sessionData.lastEpochCountState = T.epochCount;
    sessionData.vocabSizeSnapshot = T.vocabsize;

    // Learning rates and errors
    sessionData.lastLearningRateRecorded = T.learning;
    sessionData.totalLearning = T.totalLearning;
    sessionData.averageLearningRate = T.averageLearningRate;

    // Totals and averages for metrics
    sessionData.perplexityCE = T.perplexityCE;
    sessionData.perplexityBCE = T.perplexityBCE;
    sessionData.avgBCELoss = T.avgBCELoss;
    sessionData.avgCELoss = T.avgCELoss;
    sessionData.averagePerplexity = T.averagePerplexity;
    sessionData.totalBCELoss = T.totalBCELoss;
    sessionData.totalCELoss = T.totalCELoss;
    sessionData.cumulativeError = T.totalBCELoss + T.totalCELoss;
    sessionData.totalCEPerplexity = T.totalCEPerplexity;
    sessionData.totalBCEPerplexity = T.totalBCEPerplexity;

    // Booleans
    sessionData.isSelf = T.isSelf;
    sessionData.contextTrain = T.contextTrain;

    std::cout << "Lines Processes in this session: " << linesProcessedIn1Session << std::endl;
    std::cout << "Progress saved to " << currentChatLogPath << std::endl;
}
