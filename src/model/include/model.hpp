
// model class for file
#ifndef MODEL_HPP
#define MODEL_HPP 1

#include <map>
#include <string>
#include <vector>
#include <neural.hpp>

#define ARCH "SHADY-ATTENTION"

// metadata for model and data information
typedef struct modelDataInfo {
    // model information
    std::string modelName;          // model name
    std::string version;            // model version
    std::string author;             // author of model
    std::string date;               // any date as per author
    std::string modelArch;          // architecture of model
    // std::string description;        // model description
    // std::string capability;         // model capabilities
    // std::string limitations;        // model limitations
    std::string license;            // license for model use
    // std::string trainingData;       // data used to train model

    // data information and distribution
    int d;              // dimension of embedding
    int qkrow;          // matrix MQ and MK rows
    int qkcol;          // matrix MQ and MK columns
    int vhrow;          // matrix MV and MH rows
    int vhcol;          // matrix MV and MH columns
    int layer;          // number of layers in MLP hor and ver
    int nIAs;           // number of attention in PA
    int nPAs;           // number of PA in CA
    int nCAs;           // number of CA in transformer
    int paramsIA;       // total params in one attention
    int totalParams;    // total parameters in model
} modelDataInfo;


/**
 * @brief Model Class for storing transformers. Uses transformer class to store all the parametes
 * trained and to be trained. This helps in keeping all values together and accessing the values 
 * easily.
 */
class model {
public:
    int tCount;     // transformer count
    int m;          // number of blocks
    int x;          // number of incomplete attentions in each partial attention
    int y;          // number of layers of partial attention for complete attention block
    int n;          // total tokens for each attention head
    int d;          // token dimension
    int h;          // height of MQ, MK and columns of MV, MH
    int l;          // layers of mlp
    int totalParams;        // total parameters of transformer
    int total;      // total tokenLimit -> t*count m * n
    float learning;        // learning rate for MLPs
    transformer T;  // model with 1 transformer
    std::vector<transformer> Tg;        // model with tcount transformer
    std::vector<std::vector<std::vector<block>>> b;     // attention block (1 or many)
    std::vector<std::vector<std::vector<std::vector<std::vector<attention>>>>> att;      // inbetween tokens transfer
    modelDataInfo info;     // model info
    FILE *file;     // file where all data is to be stored

    /**
     * each attention has 2 mlps with 3d vector of dimensions 'l' 2d vectors of size d x d
     * and 2 matrices of qkrow x qkcol and 2 matrices of vhrow x vhcol, the matrices are need
     * to be serialised first and then mlp: MQ, MK, MH, MV, hor, ver
     */

    // default constructor
    model() = default;
    model(int m, int x, int y, int n, int d, int h, int l);
    model(int tCount, int m, int x, int y, int n, int d, int h, int l);
    model(int m, int x, int y, int n, int d, int h, int l, float learning);
    model(int tCount, int m, int x, int y, int n, int d, int h, int l, float learning);

    void train();
    void load();
    void load(std::string& from, std::string& to);
    void save();
    void allocateMemory();

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

// provide token embedding from a csv file
// provide token embedding from model binary file
// perform dembedding by searching maximum value of probability

#endif
