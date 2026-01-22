// basic.hpp: header source of basic library
#ifndef BASIC_HPP
#define BASIC_HPP 1
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <functional>
#include <atomic>
#include <random>

// Define MAXFLOAT if not implicitly available
#ifndef MAXFLOAT
    #define MAXFLOAT 3.402823466e+38F
#endif

// vect.cpp

bool operator==(std::vector<float>, std::vector<float>);
bool operator!=(std::vector<float>, std::vector<float>);
std::vector<float> operator+(std::vector<float>, std::vector<float>);
std::vector<float> operator-(std::vector<float>, std::vector<float>);
std::vector<float> operator*(std::vector<float>, float);
std::vector<float> operator*(float, std::vector<float>);
std::vector<float> operator/(std::vector<float>, float);
std::vector<float> operator+=(std::vector<float>, std::vector<float>);
std::vector<float> operator-=(std::vector<float>, std::vector<float>);
std::vector<float> operator*=(std::vector<float>, float);
std::vector<float> operator*=(float, std::vector<float>);
std::vector<float> operator/=(std::vector<float>, float);
std::vector<std::vector<float>> operator+(std::vector<std::vector<float>>, std::vector<std::vector<float>>);
std::vector<std::vector<float>> operator-(std::vector<std::vector<float>>, std::vector<std::vector<float>>);
std::vector<std::vector<float>> operator*(std::vector<std::vector<float>>, float y);
std::vector<std::vector<float>> operator/(std::vector<std::vector<float>>, float y);
std::vector<std::vector<float>> operator+=(std::vector<std::vector<float>>, std::vector<std::vector<float>>);
std::vector<std::vector<float>> operator-=(std::vector<std::vector<float>>, std::vector<std::vector<float>>);
std::vector<std::vector<float>> operator*=(std::vector<std::vector<float>>, float y);
std::vector<std::vector<float>> operator/=(std::vector<std::vector<float>>, float y);

float vdotv2val(std::vector<float>, std::vector<float>);
float vdotv2scal(std::vector<float> , std::vector<float>);
float MAE(std::vector<float>&, std::vector<float>&);
float MSE(std::vector<float>&, std::vector<float>&);
float crossEntropy(std::vector<float>&, std::vector<float>&);
float binaryCrossEntropy(std::vector<float>&, std::vector<float>&);
float categoricalCrossEntropy(std::vector<std::vector<float>>& y_true, std::vector<std::vector<float>>& y_pred);
float sum(std::vector<float>);
float sum(std::vector<std::vector<float>>);
float product(std::vector<float>);
float product(std::vector<std::vector<float>>);

std::vector<float> error(std::vector<float>&, std::vector<float>&);
std::vector<float> power(std::vector<float>, float);
std::vector<float> sumofrow(std::vector<std::vector<float>>);
std::vector<float> sumofcol(std::vector<std::vector<float>>);
std::vector<float> vxv2v(std::vector<float>, std::vector<float>);
std::vector<float> vdotv2v(std::vector<float>, std::vector<float>);
std::vector<float> vxmat2vec(std::vector<float>, std::vector<std::vector<float>>);
std::vector<float> mat2vec(std::vector<std::vector<float>>);
std::vector<std::vector<float>> vec2mat(std::vector<float>, unsigned int, unsigned int);
std::vector<std::vector<float>> vdotmat2mat(std::vector<float>, std::vector<std::vector<float>>);
std::vector<std::vector<float>> vxv2mat(std::vector<float>, std::vector<float>);
std::vector<std::vector<float>> iproduct(std::vector<std::vector<float>>);
std::vector<std::vector<float>> power(std::vector<std::vector<float>>, float);

// activations.cpp

float sigmoid(const float& x);
float sigmoidder(const float& x);
float softsign(const float& x);
std::vector<float> sigmoid(const std::vector<float>& x);
std::vector<float> sigmoidder(const std::vector<float>& x);
std::vector<std::vector<float>> sigmoid(const std::vector<std::vector<float>>& x);
std::vector<std::vector<float>> sigmoidder(const std::vector<std::vector<float>>& x);
std::vector<float> softmax(const std::vector<float>& x);
std::vector<float> softmax(const std::vector<float>& x, float temp);
std::vector<float> softmaxder(const std::vector<float>& x, float temp);
std::vector<std::vector<float>> softmax(const std::vector<std::vector<float>>& x, float temp);
std::vector<std::vector<float>> softmaxder(const std::vector<std::vector<float>>& x, float temp);
float ReLU(const float& x);
float ReLUder(const float& x);
std::vector<float> ReLU(const std::vector<float>& x);
std::vector<float> ReLUder(const std::vector<float>& x);
std::vector<float> LOTA(const std::vector<float>& y);
std::vector<float> LOTAder(const std::vector<float>& y);

// weights.cpp

void randomweights(std::vector<std::vector<float>>);
void jumbledwbs(std::vector<std::vector<float>>);
void ijbasedwbs(std::vector<std::vector<float>>);
void Random(std::vector<std::vector<float>>);

#endif