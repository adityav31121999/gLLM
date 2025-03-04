
// model class for file
#ifndef MODEL_HPP
#define MODEL_HPP 1

#include <map>
#include <string>
#include <vector>
#include "neural.hpp"

// metadata for model information
typedef struct modelInfo {
    std::string modelName;          // model name
    std::string version;            // model version
    std::string author;             // author of model
    std::string date;               // any date as per author
    std::string modelArch;          // architecture of model
    std::string description;        // model description
    std::string capability;         // model capabilities
    std::string limitations;        // model limitations
    std::string license;            // license for model use
    std::string trainingData;        // data used to train model
} modelInfo;

// metadata for data information and distributib
typedef struct dataInfo {
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
} dataInfo;

/**
 * @brief Model Class
 */
class model {
public:
    int tCount;     // transformer count
    int mCount;     // model count
    int m;          // number of blocks
    int x;          // number of incomplete attentions in each partial attention
    int y;          // number of layers of partial attention for complete attention block
    int n;          // total tokens for each attention head
    int d;          // token dimension
    int h;          // height of MQ, MK and columns of MV, MH
    int l;          // layers of mlp
    int totalParams;    // total parameters of transformer
    int total;      // total tokenLimit -> m * n
    transformer T;  // transformer
    modelInfo info;     // model info
    dataInfo dinfo;     // data info

    // default constructor
    model() = default;
    model(int m, int x, int y, int n, int d, int h, int l);
    model(int tCount, int m, int x, int y, int n, int d, int h, int l);
    model(int mCount, int tCount, int m, int x, int y, int n, int d, int h, int l);

    void train();
    void load();
    void save();

    // default destructor
    ~model() = default;
};

#endif
