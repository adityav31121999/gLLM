
// model class for file
#ifndef MODEL_HPP
#define MODEL_HPP 1

#include <string>
#include <vector>
#include <maths.hpp>
#include <neural.hpp>

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
    long long int tokens;   // total tokens (words and punctutations) for training, testing and validation
    float learning;         // learning rate
    bool attentionType;     // if self attention or cross attention
} modelDataInfo;

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
    float learning;         // learning rate for MLPs
    bool isSelf;            // if self attention or cross attention
    bool toTrain;           // if training of model, set to 1, or use of model, set to 0
    transformer T;          // model with 1 transformer
    modelDataInfo info;     // model info
    FILE *metadata = nullptr;       // .txt file for model metadata
    FILE *chat = nullptr;           // .txt file to save chat
    std::string baseDir;            // Base directory for model files (e.g., D:/train)
    std::string currentChatLogPath; // Stores the path of the currently open chat log file

// using these strings, embeddings are provided to the transformer t (for training and application)
    std::string userPrompt;                 // user prompt
    std::vector<std::string> tinput;        // token input
    std::vector<std::string> expected;      // expected token output
    std::vector<std::string> toutput;       // predicted token output
    std::vector<std::string> token;         // Input + Expected/Output + Terminator

    long long int matOffset;                // matrix offset
    long long int mlpOffset;                // mlp offset
    long long int cacheOffset;              // cache offset
    long long int attentionOffset;          // attention offset
    long long int blockOffset;              // block offset
    long long int totalParams;              // total parameters of transformer
    long long int totalTokens;              // total tokens used for training, testing and validation
    long long int vocabsize;                // total vocabulary size

    // default constructor
#ifdef USE_OPENCL
    OpenCLContext& clcontext;
    model(const std::string& baseDirectory, OpenCLContext& context, int m, int x, int y, int n, int d, int matheight, int l, long long int vocab, bool isSelfAttention, bool toTrainModel);
    model(const std::string& baseDirectory, OpenCLContext& context, int m, int x, int y, int n, int d, int matheight, int l, float learning, long long int vocab, bool isSelfAttention, bool toTrainModel);
#elif USE_CUDA || USE_CPU
    model() = default;
    model(const std::string& baseDirectory, int m, int x, int y, int n, int d, int matHeightParam, int l, long long int vocab, bool isSelfAttention, bool toTrainModel);
    model(const std::string& baseDirectory, int m, int x, int y, int n, int d, int matHeightParam, int l, float learning, long long int vocab, bool isSelfAttention, bool toTrainModel);
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
    void setEmbeddingFromBin(const std::string& path2file);
    void makeEmbedding(std::string& path2file);

    // for common knowledge training usin first block
    void train1stBlock(std::vector<std::vector<float>>& prompt, std::vector<std::vector<float>>& response, std::vector<std::string> rString);
    void test1stBlock(std::vector<std::vector<float>>& prompt, std::vector<std::vector<float>>& response, std::vector<std::string> rString);
    void validate1stBlock(std::vector<std::vector<float>>& prompt, std::vector<std::vector<float>>& response, std::vector<std::string> rString);
    void trainBlock(const std::string& trainingDataFolder);
    void testBlock(const std::string& testDataFolder);
    void validateBlock(const std::string& validationDataFolder);
    void copy1toOhterBlocks();

    // train model
    void trainModel(const std::string& trainingDataFolder);
    void testModel(const std::string& testDataFolder);
    void validateModel(const std::string& validationDataFolder);

    // get offsets for layout and components
    void calculateAndSetLayout();
    int getOffset(int blockCount, int paCount, int attentionCount, int matCount, int mlpCount);

    // fetch components from different bins for inference
    void fetchmat(mat& a, int blockCount, int x, int y, std::string& trainLocation);        // cache and mat
    void fetchmlp(mlp& network, int blockCount, int x, int y, std::string& trainLocation);  // mlp
    void serialise();           // serialise the model

    // chat with model
    void runModel(const std::string& binDirectory);     // run transformer for conversation
    void takeInput();       // take required input for transformer
    void newChat();         // for new chat clear everything and set all to 0
    void endChat();         // end chat, save parameters and clear all the memory, exit transformer
    void saveChat();        // save chat to file
    void inferenceAttention();
    void inferenceParallel();
    void inferenceBlock();

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

// tokens management and embedding

static bool is_sub_sentence_delimiter(char c);
static bool is_digit(char c);
void textSplit(std::string& path2file, std::vector<std::string>& tokensOfFile, std::vector<std::vector<std::string>>& oddSentence, 
                std::vector<std::vector<std::string>>& evenSentence);
void tokenize(const std::string& str, std::vector<std::string>& tokens);
void tokenize_with_numbers(const std::string& str, std::vector<std::string>& tokens);
void tokenize(std::string& line, std::vector<std::string>& tokensOfFile, std::vector<std::vector<std::string>>& oddSentence, 
                std::vector<std::vector<std::string>>& evenSentence);
void splitLine2SubSentences(std::string& line, std::vector<std::string>& subSentences);


// binary files

void create(std::string &locationOfALLbins);
void create(std::string &locationOfALLbins, int totalBlocks);

// csv and txt support functions

long long int countLinesInCSV(const std::string& filename);
int countLineInTXT(const std::string& filename);
void makeTXTfromCSV(const std::string& csvFilePath, std::vector<std::string>& txtFilePath, long long int totalLines, int totalGroups);
void makeCSV(std::vector<std::string>& tokens, mat& tokenEmbed, const std::string& csvFilePath);

#endif
