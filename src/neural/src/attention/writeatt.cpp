
#include "include/attention.hpp"
#include <fstream>
#include <filesystem>

/**
 * FOR TRAINING:
 * serialise all the mats and mlps and vectors in this way:
 * MQ, MK, MV, MH
 * hor, ver
 * K, Q, KdotQ
 * EH, EV
 */
void attention::serialise(int offset, const std::string& locationofbinfile) {
    // matrices
    MQ.serialise(offset, locationofbinfile);
    MK.serialise(offset+(MQ.row*MQ.col), locationofbinfile);
    MV.serialise(offset+(MQ.row*MQ.col)+(MK.row*MK.col), locationofbinfile);
    MH.serialise(offset+(MQ.row*MQ.col)+(MK.row*MK.col)+(MV.row*MV.col), locationofbinfile);
    // mlps
    hor.serialise(offset+(MQ.row*MQ.col)+(MK.row*MK.col)+(MV.row*MV.col)+(MH.row*MH.col), locationofbinfile);
    ver.serialise(offset+(MQ.row*MQ.col)+(MK.row*MK.col)+(MV.row*MV.col)+(MH.row*MH.col)+(hor.weights.size()*hor.weights[0].row*hor.weights[0].col), 
                    locationofbinfile);
    // vectors
    K.serialise(offset+(MQ.row*MQ.col)+(MK.row*MK.col)+(MV.row*MV.col)+(MH.row*MH.col)+(hor.weights.size()*hor.weights[0].row*hor.weights[0].col)+
                    (ver.weights.size()*ver.weights[0].row*ver.weights[0].col), locationofbinfile);
    Q.serialise(offset+(MQ.row*MQ.col)+(MK.row*MK.col)+(MV.row*MV.col)+(MH.row*MH.col)+(hor.weights.size()*hor.weights[0].row*hor.weights[0].col)+
                    (ver.weights.size()*ver.weights[0].row*ver.weights[0].col)+(K.row*K.col), locationofbinfile);
    // score
    KdotQ.serialise(offset+(MQ.row*MQ.col)+(MK.row*MK.col)+(MV.row*MV.col)+(MH.row*MH.col)+(hor.weights.size()*hor.weights[0].row*hor.weights[0].col)+
                    (ver.weights.size()*ver.weights[0].row*ver.weights[0].col)+(K.row*K.col)+(Q.row*Q.col), locationofbinfile);
    // Horizontal retention
    std::ofstream ehFile(locationofbinfile, std::ios::binary | std::ios::in | std::ios::out);
    if (ehFile.is_open()) {
        ehFile.seekp(offset+(MQ.row*MQ.col)+(MK.row*MK.col)+(MV.row*MV.col)+(MH.row*MH.col)+(hor.weights.size()*hor.weights[0].row*hor.weights[0].col)+
                        (ver.weights.size()*ver.weights[0].row*ver.weights[0].col)+(K.row*K.col)+(Q.row*Q.col)+(KdotQ.row*KdotQ.col), std::ios::beg);
        ehFile.write(reinterpret_cast<const char*>(EH.data()), EH.size() * sizeof(float));
        ehFile.close();
    }
    // vertical retention
    EV.serialise(offset+(MQ.row*MQ.col)+(MK.row*MK.col)+(MV.row*MV.col)+(MH.row*MH.col)+(hor.weights.size()*hor.weights[0].row*hor.weights[0].col)+
                    (ver.weights.size()*ver.weights[0].row*ver.weights[0].col)+(K.row*K.col)+(Q.row*Q.col)+EH.size()+(KdotQ.row*KdotQ.col), 
                    locationofbinfile);
}


/**
 * FOR TRAINING:
 * serialise all the mats and mlps and vectors in this way:
 * MQ, MK, MV, MH
 * hor, ver
 * K, Q, KdotQ
 * EH, EV
 */
void attention::deserialise(int offset, const std::string& locationofbinfile) {
    // matrices
    MQ.deserialise(offset, locationofbinfile);
    MK.deserialise(offset+(MQ.row*MQ.col), locationofbinfile);
    MV.deserialise(offset+(MQ.row*MQ.col)+(MK.row*MK.col), locationofbinfile);
    MH.serialise(offset+(MQ.row*MQ.col)+(MK.row*MK.col)+(MV.row*MV.col), locationofbinfile);
    // mlps
    hor.deserialise(offset+(MQ.row*MQ.col)+(MK.row*MK.col)+(MV.row*MV.col)+(MH.row*MH.col), locationofbinfile);
    ver.deserialise(offset+(MQ.row*MQ.col)+(MK.row*MK.col)+(MV.row*MV.col)+(MH.row*MH.col)+(hor.weights.size()*hor.weights[0].row*hor.weights[0].col), 
                    locationofbinfile);
    // vectors
    K.deserialise(offset+(MQ.row*MQ.col)+(MK.row*MK.col)+(MV.row*MV.col)+(MH.row*MH.col)+(hor.weights.size()*hor.weights[0].row*hor.weights[0].col)+
                    (ver.weights.size()*ver.weights[0].row*ver.weights[0].col), locationofbinfile);
    Q.deserialise(offset+(MQ.row*MQ.col)+(MK.row*MK.col)+(MV.row*MV.col)+(MH.row*MH.col)+(hor.weights.size()*hor.weights[0].row*hor.weights[0].col)+
                    (ver.weights.size()*ver.weights[0].row*ver.weights[0].col)+(K.row*K.col), locationofbinfile);
    // scores
    KdotQ.deserialise(offset+(MQ.row*MQ.col)+(MK.row*MK.col)+(MV.row*MV.col)+(MH.row*MH.col)+(hor.weights.size()*hor.weights[0].row*hor.weights[0].col)+
                    (ver.weights.size()*ver.weights[0].row*ver.weights[0].col)+(K.row*K.col)+(Q.row*Q.col), locationofbinfile);
    // horizontal retention
    std::ifstream ehFile(locationofbinfile, std::ios::binary);
    if (ehFile.is_open()) {
        ehFile.seekg(offset+(MQ.row*MQ.col)+(MK.row*MK.col)+(MV.row*MV.col)+(MH.row*MH.col)+(hor.weights.size()*hor.weights[0].row*hor.weights[0].col)+
                        (ver.weights.size()*ver.weights[0].row*ver.weights[0].col)+(K.row*K.col)+(Q.row*Q.col)+(KdotQ.row*KdotQ.col),
                        std::ios::beg);
        ehFile.read(reinterpret_cast<char*>(EH.data()), EH.size() * sizeof(float));
        ehFile.close();
    }
    // vertical retention
    EV.deserialise(offset+(MQ.row*MQ.col)+(MK.row*MK.col)+(MV.row*MV.col)+(MH.row*MH.col)+(hor.weights.size()*hor.weights[0].row*hor.weights[0].col)+
                    (ver.weights.size()*ver.weights[0].row*ver.weights[0].col)+(K.row*K.col)+(Q.row*Q.col)+EH.size()+(KdotQ.row*KdotQ.col), 
                    locationofbinfile);
}
