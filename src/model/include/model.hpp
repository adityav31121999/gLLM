
// model class for file
#ifndef MODEL_HPP
#define MODEL_HPP 1

#include <map>
#include <string>
#include <vector>
#include <neural.hpp>

#define ARCH "SHADY-ATTENTION"

/**
 * Model will have several .bin files for storing all prameters in binary
 * format. These files are of Matrices, NLPs and Caches:
 * 1. Matrices: MQ.bin, MK.bin, MH.bin, MV.bin
 * 2. MLPs: hor.bin, ver.bin
 * 3. Caches: QK.bin, KH.bin, QV.bin
 */

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
    int vocab;          // vocabulary size
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
    model(int m, int x, int y, int n, int d, int h, int l, int vocab);
    model(int tCount, int m, int x, int y, int n, int d, int h, int l, int vocab);
    model(int m, int x, int y, int n, int d, int h, int l, float learning, int vocab);
    model(int tCount, int m, int x, int y, int n, int d, int h, int l, float learning, int vocab);

    void setLearning(float learning);
    void setVocab(int vocab);
    void setModelName(std::string& modelName);
    void setVersion(std::string& version);
    void setAuthor(std::string& author);
    void setDate(std::string& date);
    void setModelArch(std::string& modelArch);
    void setLicense(std::string& license);
    void setTrainingData(std::string& trainingData);
    void setInfo(modelDataInfo& info);
    void setInfo(std::string& modelName, std::string& version, std::string& author, std::string& date, 
                    std::string& modelArch, std::string& license, std::string& trainingData);

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

#ifdef USE_CUDA
    // cuda implementation
#elif USE_OPENCL
    // opencl implementation    
#endif

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


// model filesystem
#ifndef MODEL_FS_HPP
#define MODEL_FS_HPP 1z

#include <cstdint>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <maths.hpp>
#include <neural.hpp>
#include "model.hpp"

// serialisation and deserilisation of model parameters

void serialiseMAT(const mat& a, FILE* file);
void serialiseMLP(const mlp& a, FILE* file, int, int);
void serialiseModel(const model& a);

void deserialiseMAT(mat& a, FILE* file, int, int);
void deserialiseMLP(mlp& a, FILE* file, int, int);
void deserialiseModel(model& a);

void loadModel(model& a, std::string& from);
void loadModel(model& a, FILE* file);
void loadModel(std::string& from, std::string& to);
void loadModel(model& a, std::string& from, std::string& to);
void loadModel(FILE* file, std::string& from, std::string& to);
void loadModel(model& a, FILE* file, std::string& from, std::string& to);

void saveModel(model& a, std::string& to);
void saveModel(model& a, FILE* file);
void saveModel(std::string& from, std::string& to);
void saveModel(model& a, std::string& from, std::string& to);
void saveModel(FILE* file, std::string& from, std::string& to);
void saveModel(model& a, FILE* file, std::string& from, std::string& to);

void serialiseModel(model& a, FILE* file);
void serialiseModel(model& a, std::string& to);
void serialiseModel(model& a, std::string& from, std::string& to);

void deserialiseModel(model& a, FILE* file);
void deserialiseModel(model& a, std::string& from);
void deserialiseModel(model& a, std::string& from, std::string& to);

#endif
