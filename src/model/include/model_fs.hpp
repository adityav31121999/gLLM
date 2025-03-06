
// model filesystem
#ifndef MODEL_FS_HPP
#define MODEL_FS_HPP 1z

#include <cstdint>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <maths.hpp>
#include <neural.hpp>
#include "model.hpp"

// serialisation and deserilisation of model parameters

void serialiseMAT(const mat& a, FILE* file);
void serialiseMLP(const mlp& a, FILE* file);
void serialiseModel(const model& a);

void deserialiseMAT(mat& a, FILE* file);
void deserialiseMLP(mlp& a, FILE* file);
void deserialiseModel(model& a);

#endif
