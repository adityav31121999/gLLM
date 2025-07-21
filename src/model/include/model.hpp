// model class for file
#ifndef MODEL_HPP
#define MODEL_HPP 1

#include "tokenise.hpp"
#include <string>
#include <sstream>
#include <vector>
#include <maths.hpp>
#include <neural.hpp>
#include <chrono>
#include <cmath>

#define MECH "SHADY-ATTENTION"      // attention mechanism
#define ARCH "DIVIDED-CONTEXT"      // model architecture
#define M_PI 3.14159                // no-so on-point value of pi

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
    unsigned long long tokens;   // total tokens (words and punctutations) for training, testing and validation
    float learning;         // learning rate
    bool attentionType;     // if self attention or cross attention
} modelDataInfo;


// for training session data
struct TrainingSessionData {
    std::string lastTrainingFileName;               // Name of the last training file
    int linesProcessedInLastFile = 0;               // Lines processed in the lastTrainingFileName
    int totalLines = 0;                             // total lines available in last training file
    long long cumulativeTotalLinesTrained = 0;      // Total lines trained across all files/sessions
    long long cumulativeTotalTokensProcessed = 0;   // Total tokens processed (model::totalTokens)
    int lastBlockCountState = 0;                    // Snapshot of T.blockCount at last save
    int lastEpochCountState = 0;                    // Snapshot of T.epochCount at last save
    long long cumulativeTotalTrainCount = 0;        // Total train count (T.trainCount)
    long long vocabSizeSnapshot = 0;                // Snapshot of T.vocabsize
    float lambdaL1;                                 // L1 penalty
    float lambdaL2;                                 // L2 penalty
    float epsilon;                                  // epsilon for adam optimiser
    float currentLearning;                          // current line's learning rate
    double totalLearning;                           // total learning rate from previous trainings
    double adLearning;                              // average of total learning from last session
    float cumulativeError = 0;                      // total error throughout training
    unsigned long long t_adam_steps = 0;            // total adam steps
    std::string lastSaveTimestamp;                  // Timestamp of last progress saved

    bool load(const std::string& filepath) {
        std::ifstream ifs(filepath);
        if (!ifs.is_open()) return false;
        std::string line;
        
        // Read data in the order it's saved
        if (std::getline(ifs, line)) lastTrainingFileName = line; else return false;
        if (std::getline(ifs, line)) { std::istringstream ss(line); ss >> linesProcessedInLastFile; if(ss.fail()) return false; } else return false;
        if (std::getline(ifs, line)) { std::istringstream ss(line); ss >> totalLines; if(ss.fail()) return false; } else return false;
        if (std::getline(ifs, line)) { std::istringstream ss(line); ss >> cumulativeTotalLinesTrained; if(ss.fail()) return false; } else return false;
        if (std::getline(ifs, line)) { std::istringstream ss(line); ss >> cumulativeTotalTokensProcessed; if(ss.fail()) return false; } else return false;
        if (std::getline(ifs, line)) { std::istringstream ss(line); ss >> lastBlockCountState; if(ss.fail()) return false; } else return false;
        if (std::getline(ifs, line)) { std::istringstream ss(line); ss >> lastEpochCountState; if(ss.fail()) return false; } else return false;
        if (std::getline(ifs, line)) { std::istringstream ss(line); ss >> cumulativeTotalTrainCount; if(ss.fail()) return false; } else return false;
        if (std::getline(ifs, line)) { std::istringstream ss(line); ss >> vocabSizeSnapshot; if(ss.fail()) return false; } else return false;
        if (std::getline(ifs, line)) { std::istringstream ss(line); ss >> lambdaL1; if(ss.fail()) return false; } else return false;
        if (std::getline(ifs, line)) { std::istringstream ss(line); ss >> lambdaL2; if(ss.fail()) return false; } else return false;
        if (std::getline(ifs, line)) { std::istringstream ss(line); ss >> epsilon; if(ss.fail()) return false; } else return false;
        if (std::getline(ifs, line)) { std::istringstream ss(line); ss >> currentLearning; if(ss.fail()) return false; } else return false;
        if (std::getline(ifs, line)) { std::istringstream ss(line); ss >> totalLearning; if(ss.fail()) return false; } else return false;
        if (std::getline(ifs, line)) { std::istringstream ss(line); ss >> adLearning; if(ss.fail()) return false; } else return false;
        if (std::getline(ifs, line)) { std::istringstream ss(line); ss >> cumulativeError; if(ss.fail()) return false; } else return false;
        if (std::getline(ifs, line)) { std::istringstream ss(line); ss >> t_adam_steps; if(ss.fail()) return false; } else return false;
        if (std::getline(ifs, line)) lastSaveTimestamp = line; else return false;

        return ifs.eof() || ifs.peek() == EOF; // Ensure all expected data was read
    }

    void save(const std::string& filepath) {
        std::ofstream ofs(filepath);
        if (!ofs.is_open()) {
            std::cerr << "Error: Could not open session data file for writing: " << filepath << std::endl;
            return;
        }
        ofs << lastTrainingFileName << std::endl;
        ofs << linesProcessedInLastFile << std::endl;
        ofs << totalLines << std::endl;
        ofs << cumulativeTotalLinesTrained << std::endl;
        ofs << cumulativeTotalTokensProcessed << std::endl;
        ofs << lastBlockCountState << std::endl;
        ofs << lastEpochCountState << std::endl;
        ofs << cumulativeTotalTrainCount << std::endl;
        ofs << vocabSizeSnapshot << std::endl;
        ofs << lambdaL1 << std::endl;
        ofs << lambdaL2 << std::endl;
        ofs << epsilon << std::endl;
        ofs << currentLearning << std::endl;
        ofs << totalLearning << std::endl;
        ofs << adLearning << std::endl;
        ofs << cumulativeError << std::endl;
        ofs << t_adam_steps << std::endl;
        
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
    int m;                  // number of blocks
    int x;                  // number of incomplete attentions in each partial attention
    int y;                  // number of layers of partial attention for complete attention block
    int matheight;          // height of MQ, MK and columns of MV, MH
    int n;                  // total tokens for each attention head
    int d;                  // token dimension
    int l;                  // layers of mlp
    int total;              // total tokenLimit -> t*count m * n
    float learning;         // learning rate
    double totalLearning;   // total learning for all updates (adaptive learning)
    double adLearning;      // = T.totalLearning/T.trainCount (average adaptive learning)
    float lambda_L1;        // lambda for L1 penalty
    float lambda_L2;        // lambda for L2 penalty
    float epsilon;          // epsilon for adam optimiser
    bool isSelf;            // if self attention = 1 or cross attention = 0
    bool toTrain;           // if training of model, set to 1, or inference, set to 0
    unsigned long long t_step_adam;         // for time steps, everytime adam optimiser is used

// files to hold data
    modelDataInfo info;     // model info
    FILE *metadata = nullptr;       // .txt file for model metadata
    TrainingSessionData trainInfo;  // training session data
    FILE *chat = nullptr;           // .txt file to save chat

// paths
    std::string baseDir;            // Base directory for model files (e.g., D:/train)
    std::string currentChatLogPath; // Stores the path of the currently open chat log file

// using these strings, embeddings are provided to the transformer t (for training and application)
    std::string userPrompt;                     // user prompt
    std::vector<std::string> tinput;            // token input
    std::vector<std::string> expected;          // expected token output
    std::vector<std::string> toutput;           // predicted token output
    std::vector<std::string> chatToken;         // Input + Expected/Output + Terminator

// offsets
    unsigned long long matOffset;                // matrix offset
    unsigned long long mlpOffset;                // mlp offset
    unsigned long long cacheOffset;              // cache offset
    unsigned long long attentionOffset;          // attention offset
    unsigned long long blockOffset;              // block offset
    unsigned long long totalParams;              // total parameters of transformer
    unsigned long long totalTokens;              // total tokens used for training, testing and validation
    unsigned long long vocabsize;                // total vocabulary size
    unsigned long long totalTrainingCount;       // total training count obtained from epochs based on token training
    tokeniser TOK;          // tokeniser
    transformer T;          // transformer

    // default constructor
#ifdef USE_OPENCL
    OpenCLContext& clcontext;
    model(OpenCLContext& context, const std::string& baseDirectory, int m, int x, int y, int n, int d, int matheight, 
        int l, float learning, float lambda_L1, float lambda_L2, bool isSelfAttention, bool toTrainModel,
        const std::string& tokeniserPath);
#elif USE_CUDA || USE_CPU
    model() = default;
    model(const std::string& baseDirectory, int m, int x, int y, int n, int d, int matHeightParam, int l, float learning, 
        float lambda_L1, float lambda_L2, bool isSelfAttention, bool toTrainModel, const std::string& tokeniserPath);
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
    void positionalEmbedding(const std::vector<float>& originalmbedding, std::vector<float>& newEmbedding, int position);
    void setTokenAndEmbeddingForTransformer(tokeniser& tok);

    // train first block on promp-response and sentences
    void trainBlockPR(const std::string& trainingDataFolder);
    void trainBlockSentence(const std::string& trainingDataFolder);
    void testBlockPR(const std::string& testDataFolder);
    void testBlockSentence(const std::string& testDataFolder);

    // train and test model on promp-response and sentences
    void trainModelPR(const std::string& trainingDataFolder);
    void trainModelSentence(const std::string& trainingDataFolder);
    void testModelPR(const std::string& testDataFolder);
    void testModelSentence(const std::string& testDataFolder);

    // get offsets for layout and components
    void calculateAndSetLayout();
    int getOffset(int blockCount, int paCount, int attentionCount, int matCount, int mlpCount);

    void serialise();               // serialise whole model
    void fetchmat(mat& a, int blockCount, int x, int y, const std::string& trainLocation);          // cache and mat
    void fetchmlp(mlp& network, int blockCount, int x, int y, const std::string& trainLocation);    // mlp
    void fetchForInference(const std::string& binDirectory);    // fetch for inference (mlp and cache)
    void fetchForTraining(const std::string& binDirectory);     // fetch for training (matrix and mlp)

    // chat with model
    void runModel(const std::string& binDirectory);     // run transformer for conversation
    void takeInput();               // take required input for transformer
    void newChat();                 // for new chat clear everything and set all to 0
    void endChat();                 // end chat, save parameters and clear all the memory, exit transformer
    void saveChat();                // save chat to file

    // lambda = i(sin(i) + cos(pi - i))
    static constexpr auto terminatorEmbed = [](float i, int j) -> float {
        return static_cast<float>(j * (std::sin(i) + std::cos(M_PI - i)));
    };

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
void splitLine2SubSentences(std::string& line, std::vector<std::string>& subSentences);
void textSplit(std::string& path2file, std::vector<std::string>& tokensOfFile, std::vector<std::vector<std::string>>& oddSentence, 
                std::vector<std::vector<std::string>>& evenSentence);
void create(std::string &locationOfALLbins);
void makeCSV(std::vector<std::string>& tokens, mat& tokenEmbed, const std::string& csvFilePath);

#endif
