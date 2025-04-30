
// model filesystem
#ifndef MODEL_FS_HPP
#define MODEL_FS_HPP 1z

#include <cstdint>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include "model.hpp"

// serialisation and deserilisation of model parameters

void serialiseMAT(const mat& a, FILE* file);
void serialiseMLP(const mlp& a, FILE* file, int, int);
void serialiseModel(const model& a);

void deserialiseMAT(mat& a, FILE* file, int, int);
void deserialiseMLP(mlp& a, FILE* file, int, int);
#ifndef USE_OPENCL
void deserialiseModel(model& a);
#else
void deserialiseModel(model& a, OpenCLContext& context);
#endif

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
