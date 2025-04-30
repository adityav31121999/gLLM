
// model class for file
#ifndef MODEL_HPP
#define MODEL_HPP 1

#include <string>
#include <vector>
#include <maths.hpp>
#include <neural.hpp>

#define ARCH "SHADY-ATTENTION"
#define EXTENSION ".lm"

// metadata for model and data information
typedef struct modelDataInfo {
    // model information
    std::string modelName;          // model name
    std::string version;            // model version
    std::string author;             // author of model
    std::string date;               // any date as per author
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
 * Model will have several .bin files for storing all prameters in binary
 * format. These files are of Matrices, NLPs and Caches:
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
    std::string path2model; // path to model for use and training

// using these strings, embeddings are provided to the transformer t (for training and application)
    std::vector<std::string> tinput;        // token input
    std::vector<std::string> expected;      // expected token output
    std::vector<std::string> toutput;       // predicted token output
    // Hold all input, generated or predicted tokens till TERMINATOR MEETS (Input + Expected/Output + Terminator)
    std::vector<std::string> token;

    // default constructor
#ifdef USE_OPENCL
    OpenCLContext& clcontext;   // reference to class for opencl context for accessing all kernels
    model(OpenCLContext& context, int m, int x, int y, int n, int d, int h, int l, int vocab);
    model(OpenCLContext& context, int m, int x, int y, int n, int d, int h, int l, float learning, int vocab);
    model(OpenCLContext& context, int m, int x, int y, int n, int d, int h, int l, int vocab, bool isSelfAttention, bool toTrainModel);
    model(OpenCLContext& context, int m, int x, int y, int n, int d, int h, int l, float learning, int vocab, bool isSelfAttention, bool toTrainModel);
#elif USE_CUDA || USE_CPU
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
    void setEmbedding(std::string&);
    void getToken(std::vector<float>&);

    void allocateMemory();
    void load();
    void load(std::string& from, std::string& to);
    void save();
    void train();

    // chats
    void takeInput();       // take required input for transformer
    void runTransformer();  // run transformer for conversation
    void newChat();         // for new chat clear everything and set all to 0
    void endChat();         // end chat and clear all the memory
    void continueChat();    // continue chat with the model
    void previousResponse();    // get previous response from the model
    void nextResponse();        // get next response from the model
    void saveChat();        // save chat to file
    void loadChat();        // load chat from file

    // default destructor
    ~model() = default;
};

// tokens management and embedding

void alternateSplit(std::vector<std::string>& token, std::vector<std::string>& prompt, std::vector<std::string>& response);
void continuousSplit(std::vector<std::string>& token, std::vector<std::string>& prompt, std::vector<std::string>& response);
void QNASplit(std::vector<std::string>& token, std::vector<std::string>& prompt, std::vector<std::string>& response);
void sentenceSplit(std::vector<std::string>& token, std::vector<std::string>& prompt, std::vector<std::string>& response);
void punctutationSplit(std::vector<std::string>& token, std::vector<std::string>& prompt, std::vector<std::string>& response);
void fillInTheBlanks(std::vector<std::string>& token, std::vector<std::string>& prompt, std::vector<std::string>& response);

#endif
