
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

void serialiseMAT(mat&, int&);
void serialiseMLP(mlp&, int&);
void serialiseModel(model&);

void deserialiseMAT(mat&, int&);
void deserialiseMLP(mat&, int&);
void deserialiseModel(model&);

#endif
