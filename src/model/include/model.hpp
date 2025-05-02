
// model class for file
#ifndef MODEL_HPP
#define MODEL_HPP 1

#include <string>
#include <vector>
#include <maths.hpp>
#include <neural.hpp>

/**
 * Model will have several .bin files for storing all prameters in binary
 * format. These files are of Matrices, NLPs and Caches:
 * 1. Matrices: MQ.bin, MK.bin, MH.bin, MV.bin
 * 2. MLPs: hor.bin, ver.bin
 * 3. Caches: QK.bin, KH.bin, QV.bin
 */

#define MECH "SHADY-ATTENTION"
#define ARCH "DIVIDED-CONTEXT"
#define EXTENSION ".lm"

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
    int totalParams;        // total parameters of transformer
    int totalContext;       // total tokenLimit -> t*count m * n
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
    int h;                  // height of MQ, MK and columns of MV, MH
    int l;                  // layers of mlp
    int totalParams;        // total parameters of transformer
    int total;              // total tokenLimit -> t*count m * n
    float learning;         // learning rate for MLPs
    bool isSelf;            // if self attention or cross attention
    bool toTrain;           // if training of model, set to 1, or use of model, set to 0
    transformer T;         // model with 1 transformer
    modelDataInfo info;     // model info
    FILE *file;             // file where all data is to be stored
    FILE *chat;             // .txt file to save chat

// using these strings, embeddings are provided to the transformer t (for training and application)
    std::string userPrompt;                 // user prompt
    std::vector<std::string> tinput;        // token input
    std::vector<std::string> expected;      // expected token output
    std::vector<std::string> toutput;       // predicted token output
    // Hold all input, generated or predicted tokens till TERMINATOR MEETS (Input + Expected/Output + Terminator)
    std::vector<std::string> token;

    #define MATOFFSET d*matheight       // number of elements in a matrix
    #define MLPOFFSET d*d*l             // number of elements in a mlp hidden weights
    #define CACHEOFFSET d*d             // number of elements in a cache

    // default constructor
#ifdef USE_OPENCL
    OpenCLContext& clcontext;   // reference to class for opencl context for accessing all kernels
    model(OpenCLContext& context, int m, int x, int y, int n, int d, int h, int l, int vocab);
    model(OpenCLContext& context, int m, int x, int y, int n, int d, int h, int l, float learning, int vocab);
    model(OpenCLContext& context, int m, int x, int y, int n, int d, int h, int l, int vocab, bool isSelfAttention, bool toTrainModel);
    model(OpenCLContext& context, int m, int x, int y, int n, int d, int h, int l, float learning, int vocab, bool isSelfAttention, bool toTrainModel);
#elif USE_CUDA || USE_CPU
    model() = default;
    model(int m, int x, int y, int n, int d, int h, int l, int vocab);
    model(int m, int x, int y, int n, int d, int h, int l, float learning, int vocab);
    model(int m, int x, int y, int n, int d, int h, int l, int vocab, bool isSelfAttention, bool toTrainModel);
    model(int m, int x, int y, int n, int d, int h, int l, float learning, int vocab, bool isSelfAttention, bool toTrainModel);
#endif

    void setLearning(float learning);
    void setVocab(int vocab);
    void setModelName(std::string& modelName);
    void setVersion(std::string& version);
    void setAuthor(std::string& author);
    void setDate(std::string& date);
    void setLicense(std::string& license);
    void setInfo(modelDataInfo& info);
    void setInfo(std::string& modelName, std::string& version, std::string& author, std::string& date, std::string& modelArch, 
                    std::string& license, std::string& trainingData);

    // model related functions
    void create(std::string& locationOfModel);
    void save();
    void save(std::string& locationOfModel);

    // for common knowledge training usin first block
    void train1stBlock(std::vector<std::vector<float>>& prompt, std::vector<std::vector<float>>& response, std::vector<std::string> rString);
    void test1stBlock(std::vector<std::vector<float>>& prompt, std::vector<std::vector<float>>& response, std::vector<std::string> rString);
    void validate1stBlock(std::vector<std::vector<float>>& prompt, std::vector<std::vector<float>>& response, std::vector<std::string> rString);
    void trainBlock(const std::string& trainingDataFolder);
    void testBlock(const std::string& testDataFolder);
    void validateBlock(const std::string& validationDataFolder);

    // train model
    void train(const std::string& trainingDataFolder);
    void test(const std::string& testDataFolder);
    void validate(const std::string& validationDataFolder);

    // chat with model
    void runModel();  // run transformer for conversation
    void takeInput();       // take required input for transformer
    void newChat();         // for new chat clear everything and set all to 0
    void endChat();         // end chat, save parameters and clear all the memory, exit transformer
    void saveChat();        // save chat to file
    void loadChat();        // load chat from file

    // default destructor
    ~model() = default;
};

// tokens management and embedding

void textSplit(std::string& path2file, std::vector<std::string>& tokensOfFile, std::vector<std::vector<std::string>>& oddSentence, 
                std::vector<std::vector<std::string>>& evenSentence);

#endif
