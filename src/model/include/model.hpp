// model class for file
#ifndef MODEL_HPP
#define MODEL_HPP 1

#include "tokenise.hpp"
#include <fstream>
#include <string_view>
#include <sstream>
#include <string>
#include <vector>
#include <maths.hpp>
#include <neural.hpp>
#include <chrono>

#define MECH "SHADY-ATTENTION"      // attention mechanism
#define ARCH "DIVIDED-CONTEXT"      // model architecture

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


/**
 * Model parameters are stored in a single binary file pointed to by `modelPath`.
 * This file, managed by modelFILE, is used for training, contains:
 * 1. `modelDataInfo` struct: Metadata about the model, stored at the beginning of the file.
 * 2. `transformer T` data: All matrices and MLP weights for the main transformer, memory-mapped.
 * 3. `block b` data: All matrices and MLP weights for the common block, memory-mapped.
 * All `mat` and `mlp` objects within these components will map to specific regions
 * within this single `modelFILE`.
 * The actual numerical parameters are stored in separate .bin files for inference:
 * 1. Matrices: MQ.bin, MK.bin, MH.bin, MV.bin
 * 2. MLPs: hor.bin, ver.bin
 * 3. Caches: QK.bin, KH.bin, QV.bin
 */


/**
 * @brief Model Class for storing transformers. Uses transformer class to store all the parametes
 * trained and to be trained. This helps in keeping all values together and accessing the values 
 * easily.
 */
class model {
public:
    int m;                      // number of blocks
    int x;                      // number of incomplete attentions in each partial attention
    int y;                      // number of layers of partial attention for complete attention block
    int matheight;              // height of MQ, MK and columns of MV, MH
    int n;                      // total tokens for each attention head
    int d;                      // token dimension
    int l;                      // layers of mlp
    int total;                  // total tokenLimit -> t*count m * n
    unsigned int vocabsize;     // total vocabulary size
    float learning;             // learning rate for MLPs
    float lambda_L1;            // lambda for L1 penalty
    float lambda_L2;            // lambda for L2 penalty
    bool isSelf;                // if self attention or cross attention
    bool toTrain;               // if training of model, set to 1, or use of model, set to 0
    bool contextTrain;          // contextualised training = 1 else static training

// files to hold data
    modelDataInfo info;     // model info
    FILE *metadata = nullptr;       // .txt file for model metadata
    TrainingSessionData trainInfo;  // training session data
    FILE *chat = nullptr;           // .txt file to save chat

// paths
    std::string baseDir;            // Base directory for model files (e.g., D:/train)
    std::string currentChatLogPath; // Stores the path of the currently open chat log file

// using these strings, embeddings are provided to the transformer t (for training and application)
    std::string userSequence1;                 // user sequence1
    std::vector<std::string> tinput;        // token input
    std::vector<std::string> expected;      // expected token output
    std::vector<std::string> toutput;       // predicted token output
    std::vector<std::string> chatToken;     // Input + Expected/Output + Terminator

// offsets
    unsigned long long matOffset;                // matrix offset
    unsigned long long mlpOffset;                // mlp offset
    unsigned long long cacheOffset;              // cache offset
    unsigned long long attentionOffset;          // attention offset
    unsigned long long blockOffset;              // block offset
    unsigned long long totalParams;              // total parameters of transformer
    unsigned long long totalTokens;              // total tokens used for training, testing and validation
    unsigned long long totalTrainingCount;       // total training count obtained from epochs based on token training

    tokeniser TOK;          // tokeniser
    transformer T;          // transformer

    // default constructor
#ifdef USE_OPENCL
    OpenCLContext& clcontext;
    model(OpenCLContext& context, const std::string& baseDirectory, const std::string& tokenDirectory, int m, int x, int y, int n, int d, int matheight, 
        int l, float learning, float lambda_L1, float lambda_L2, bool isSelfAttention, bool toTrainModel, bool contextTrainModel);
    model(OpenCLContext& context, const std::string& modelName, const std::string& baseDirectory, const std::string& tokenDirectory, int m, int x, int y, int n, int d, int matheight, 
        int l, float learning, float lambda_L1, float lambda_L2, bool isSelfAttention, bool toTrainModel, bool contextTrainModel);
#elif USE_CUDA || USE_CPU
    model() = default;
    model(const std::string& baseDirectory, const std::string& tokenDirectory, int m, int x, int y, int n, int d, int matheight, 
        int l, float learning, float lambda_L1, float lambda_L2, bool isSelfAttention, bool toTrainModel, bool contextTrainModel);
    model(const std::string& modelName, const std::string& baseDirectory, const std::string& tokenDirectory, int m, int x, int y, int n, int d, int matheight, 
        int l, float learning, float lambda_L1, float lambda_L2, bool isSelfAttention, bool toTrainModel, bool contextTrainModel);
#endif

    void setLearning(float learning);
    void setVocab(int vocab);
    void setModelName(const std::string& modelName);
    void setVersion(const std::string& version);
    void setAuthor(const std::string& author);
    void setDate(const std::string& date);
    void setLicense(const std::string& license);
    void setInfo(modelDataInfo& info);
    void setInfo(std::string& modelName, std::string& version, std::string& author, std::string& date, std::string& modelArch, 
                    std::string& license, std::string& trainingData);
    void setEmbeddingFromCSV(const std::string& path2file);
    void makeEmbedding(std::string& path2file);
    void setTokens2Transformer();
    void getSessionData(TrainingSessionData& sessionData, bool& sessionFileExistsAndIsValid, int& startLineForCurrentFile);
    void setSessionData(TrainingSessionData& sessionData, int k, int linesProcessedIn1Session);

    // train seq2seq and sequence with context in consideration
    void trainSeq2Seq(const std::string& txtFileLocation, int context_window, bool contextTrain);
    void trainSequence(const std::string& txtFileLocation, int context_window, bool contextTrain);
    void test(const std::string& txtFileLocation, int context_window, bool contextTrain);

    // get offsets for layout and components
    void calculateAndSetLayout();
    int getOffset(int blockCount, int paCount, int attentionCount, int matCount, int mlpCount);
    void fetchmat(mat& a, int blockCount, int x, int y, const std::string& trainLocation);        // cache and mat
    void fetchmlp(mlp& network, int blockCount, int x, int y, const std::string& trainLocation);  // mlp
    void fetchForInference(const std::string& binDirectory);    // fetch for inference (mlp and cache)
    void fetchForTraining(const std::string& binDirectory);     // fetch for training (matrix and mlp)
    void serialise();

    // chat with model
    void runModel(const std::string& binDirectory);     // run transformer for conversation
    void takeInput();       // take required input for transformer
    void newChat();         // for new chat clear everything and set all to 0
    void endChat();         // end chat, save parameters and clear all the memory, exit transformer
    void saveChat();        // save chat to file

    // Destructor to ensure modelFILE is closed
    ~model() {
        if (metadata) {
            fclose(metadata);
            metadata = nullptr;
        }
        if (chat) { // Assuming 'chat' is the FILE* for the current chat log
            fclose(chat);
            chat = nullptr;
        }
    }
};

// words and embeddings

unsigned long long countLinesInCSV(const std::string& filename);
unsigned long long countLineInTXT(const std::string& filename);

static bool is_sub_sentence_delimiter(char c);
static bool is_digit(char c);

void tokenize_with_numbers(const std::string& str, std::vector<std::string>& tokens, bool& sortIt);
void splitLine2SubSentences(std::string& line, std::vector<std::string>& subSentences);
void textSplit(std::string& path2file, std::vector<std::string>& tokensOfFile, std::vector<std::vector<std::string>>& oddSentence, 
                std::vector<std::vector<std::string>>& evenSentence);

void create(std::string &locationOfALLbins);
void makeCSV(std::vector<std::string>& tokens, mat& tokenEmbed, const std::string& csvFilePath);

#endif
