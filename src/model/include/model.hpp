// model class for file
#ifndef MODEL_HPP
#define MODEL_HPP 1
#include "tokenise.hpp"
#include "info.hpp"
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
    std::string allFiles;           // csv of all training .txt files

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
#ifdef USE_CL
    OpenCLContext& clcontext;
    model(OpenCLContext& context, const std::string& baseDirectory, const std::string& tokenDirectory, int m, int x, int y, int n, int d, int matheight, 
        int l, float learning, float lambda_L1, float lambda_L2, bool isSelfAttention, bool toTrainModel, bool contextTrainModel);
    model(OpenCLContext& context, const std::string& modelName, const std::string& baseDirectory, const std::string& tokenDirectory, int m, int x, int y, int n, int d, int matheight, 
        int l, float learning, float lambda_L1, float lambda_L2, bool isSelfAttention, bool toTrainModel, bool contextTrainModel);
#elif USE_CU || USE_CPU
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
    void train(const std::string& path2txtDir, int context_window, bool contextTrain, bool trainType);
    void test(const std::string& path2txtDir, int context_window, bool contextTrain);

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
