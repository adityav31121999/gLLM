#ifndef INFO_HPP
#define INFO_HPP 1
#include <filesystem>
#include <fstream>
#include <string>
#include <iostream>
#include <vector>

// metadata for model and data information
typedef struct modelDataInfo {
    // model information
    std::string modelName;          // model name
    std::string version;            // model version
    std::string author;             // author of model
    std::string date;               // any date as per author
    std::string attentionMech;      // attention mechanism
    std::string modelArch;          // architecture of model
    std::string license;            // license for model use

    // data information and distribution
    int d;                  // dimension of embedding
    int vocab;              // vocabulary size
    int qkrow;              // matrix MQ and MK rows
    int qkcol;              // matrix MQ and MK columns
    int vhrow;              // matrix MV and MH rows
    int vhcol;              // matrix MV and MH columns
    int m;                  // number of blocks
    int x;                  // number of incomplete attentions in each partial attention
    int y;                  // number of layers of partial attention for complete attention block
    int n;                  // total tokens for each attention head
    int h;                  // height of MQ, MK and columns of MV, MH
    int l;                  // layers of mlp
    int matheight;          // height of MQ, MK and columns of MV, MH
    int totalParams;        // total parameters of transformer
    int totalContext;       // total tokenLimit -> t*count m * n
    unsigned int tokens;    // total tokens (words and punctutations) for training, testing and validation
    float learning;         // learning rate
    bool attentionType;     // if self attention or cross attention
    bool isSelf;            // if self attention or cross attention
    bool contextTrain;      // contextualised training = 1 else static training
} modelDataInfo;

struct scores {
    // confusion matrix
    std::vector<std::vector<float>> confusionMatrix;

    // 
};

// for training session data
struct TrainingSessionData {
    std::string lastTrainingFileName;                           // Name of the last training file
    int linesProcessedInLastFile = 0;                           // Lines processed in the lastTrainingFileName
    int totalLines = 0;                                         // total lines available in last training file
    int lastBlockCountState = 0;                                // Snapshot of T.blockCount at last save
    int lastEpochCountState = 0;                                // Snapshot of T.epochCount at last save
    unsigned int vocabSizeSnapshot = 0;                         // Snapshot of T.vocabsize

    unsigned long long cumulativeTotalLinesTrained = 0;         // Total lines trained across all files/sessions
    unsigned long long cumulativeTotalTokensProcessed = 0;      // Total tokens processed (model::totalTokens)
    unsigned long long cumulativeTotalTrainCount = 0;           // Total train count ( T.trainCount)

    // Errors and Learning Rates
    float prev_error = 0;                                            // previous iteration's error (T.prev_error)
    float pErr2 = 0;                                            // previous to previous iteration's error (T.pErr2)
    float lastLearningRateRecorded = 0;                         // Last learning rate recorded while training (T.learning)
    double totalLearning = 0;                                   // total learning for all updates (adaptive learning) (T.totalLearning)
    float averageLearningRate = 0;                              // Average learning rate throughout training of transformer (T.averageLearningRate)

    // Perplexity and Entropy Metrics
    float perplexityCE = 0;                                     // current perplexity (T.perplexityCE)
    float perplexityBCE = 0;                                    // current perplexity (T.perplexityBCE)
    float avgBCELoss = 0;                                       // average bce (T.avgBCELoss)
    float avgCELoss = 0;                                        // average ce (T.avgCELoss)
    float averagePerplexity = 0;                                // average perplexity throughout training (T.averagePerplexity)
    double totalBCELoss = 0;                                    // total BCE error for all updates (adaptive learning) (T.totalBCELoss)
    double totalCELoss = 0;                                     // total CE error for all updates (adaptive learning) (T.totalCELoss)
    double totalCEPerplexity = 0;                               // total CE perplexity for all updates (adaptive learning) (T.totalCEPerplexity)
    double totalBCEPerplexity = 0;                              // total BCE perplexity for all updates (adaptive learning) (T.totalBCEPerplexity)
    double cumulativeError = 0;                                 // total error throughout training (T.error)

    // Booleans
    bool isSelf = false;                                        // if self attention or cross attention
    bool contextTrain = false;                                  // contextualised training = 1 else static training

    std::string lastSaveTimestamp;                              // Timestamp of last progress saved

    bool load(const std::string& filepath) {
        std::ifstream ifs(filepath);
        if (!ifs.is_open()) return false;
        std::string line;
        
        auto get_value = [&](auto& var) {
            if (std::getline(ifs, line)) {
                std::istringstream ss(line);
                ss >> var;
                return !ss.fail();
            }
            return false;
        };

        if (!std::getline(ifs, lastTrainingFileName)) return false;
        if (!get_value(linesProcessedInLastFile)) return false;
        if (!get_value(totalLines)) return false;
        if (!get_value(cumulativeTotalLinesTrained)) return false;
        if (!get_value(cumulativeTotalTokensProcessed)) return false;
        if (!get_value(lastBlockCountState)) return false;
        if (!get_value(lastEpochCountState)) return false;
        if (!get_value(cumulativeTotalTrainCount)) return false;
        if (!get_value(vocabSizeSnapshot)) return false;
        if (!get_value(lastLearningRateRecorded)) return false;
        if (!get_value(totalLearning)) return false;
        if (!get_value(averageLearningRate)) return false;
        if (!get_value(prev_error)) return false;
        if (!get_value(pErr2)) return false;
        if (!get_value(perplexityCE)) return false;
        if (!get_value(perplexityBCE)) return false;
        if (!get_value(avgBCELoss)) return false;
        if (!get_value(avgCELoss)) return false;
        if (!get_value(averagePerplexity)) return false;
        if (!get_value(totalBCELoss)) return false;
        if (!get_value(totalCELoss)) return false;
        if (!get_value(cumulativeError)) return false;
        if (!get_value(totalCEPerplexity)) return false;
        if (!get_value(totalBCEPerplexity)) return false;
        if (!get_value(isSelf)) return false;
        if (!get_value(contextTrain)) return false;
        if (!std::getline(ifs, lastSaveTimestamp)) return false;

        return ifs.eof() || ifs.peek() == EOF; // Ensure all expected data was read
    }

    void save(const std::string& filepath) {
        std::ofstream ofs(filepath);
        if (!ofs.is_open()) { std::cerr << "Error: Could not open session data file for writing: " << filepath << std::endl; return; }
        ofs << lastTrainingFileName << std::endl;
        ofs << linesProcessedInLastFile << std::endl;
        ofs << totalLines << std::endl;
        ofs << cumulativeTotalLinesTrained << std::endl;
        ofs << cumulativeTotalTokensProcessed << std::endl;
        ofs << lastBlockCountState << std::endl;
        ofs << lastEpochCountState << std::endl;
        ofs << cumulativeTotalTrainCount << std::endl;
        ofs << vocabSizeSnapshot << std::endl;
        ofs << lastLearningRateRecorded << std::endl;
        ofs << totalLearning << std::endl;
        ofs << averageLearningRate << std::endl;
        ofs << prev_error << std::endl;
        ofs << pErr2 << std::endl;
        ofs << perplexityCE << std::endl;
        ofs << perplexityBCE << std::endl;
        ofs << avgBCELoss << std::endl;
        ofs << avgCELoss << std::endl;
        ofs << averagePerplexity << std::endl;
        ofs << totalBCELoss << std::endl;
        ofs << totalCELoss << std::endl;
        ofs << cumulativeError << std::endl;
        ofs << totalCEPerplexity << std::endl;
        ofs << totalBCEPerplexity << std::endl;
        ofs << isSelf << std::endl;
        ofs << contextTrain << std::endl;

        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss_time;
        ss_time << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %H:%M:%S");
        ofs << ss_time.str() << std::endl;
    }
};

#endif