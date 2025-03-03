
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


// vect.cpp

std::vector<double> operator+(std::vector<double>, std::vector<double>);
std::vector<double> operator-(std::vector<double>, std::vector<double>);
std::vector<double> operator*(std::vector<double>, double);
std::vector<double> operator*(double, std::vector<double>);
std::vector<double> operator/(std::vector<double>, double);
std::vector<std::vector<double>> operator+(std::vector<std::vector<double>>, std::vector<std::vector<double>>);
std::vector<std::vector<double>> operator-(std::vector<std::vector<double>>, std::vector<std::vector<double>>);
std::vector<std::vector<double>> operator*(std::vector<std::vector<double>>, double y);
std::vector<std::vector<double>> operator/(std::vector<std::vector<double>>, double y);
bool operator==(std::vector<double>, std::vector<double>);
bool operator!=(std::vector<double>, std::vector<double>);

std::vector<double> error(std::vector<double>, std::vector<double>);
double errorofv(std::vector<double> , std::vector<double> );
std::vector<double> percenterrorofvec(std::vector<double> , std::vector<double>);
std::vector<double> gradient_descent(std::vector<double>, std::vector<double>, double);
double gradientdesc1(std::vector<double>, std::vector<double>);

double sum(std::vector<double>);
double sum(std::vector<std::vector<double>>);
double product(std::vector<double>);
double product(std::vector<std::vector<double>>);
std::vector<std::vector<double>> iproduct(std::vector<std::vector<double>>);
std::vector<double> power(std::vector<double>, double);
std::vector<std::vector<double>> power(std::vector<std::vector<double>>, double);
std::vector<double> sumofrow(std::vector<std::vector<double>>);
std::vector<double> sumofcol(std::vector<std::vector<double>>);
std::vector<std::vector<double>> vxv2mat(std::vector<double>, std::vector<double>);
std::vector<double> vxv2v(std::vector<double>, std::vector<double>);
std::vector<double> vdotv2v(std::vector<double>, std::vector<double>);
double vdotv2val(std::vector<double>, std::vector<double>);
double vdotv2scal(std::vector<double> , std::vector<double>);
double MSE(std::vector<double>, std::vector<double>);
std::vector<std::vector<double>> vdotmat2mat(std::vector<double>, std::vector<std::vector<double>>);
std::vector<double> vxmat2vec(std::vector<double>, std::vector<std::vector<double>>);
std::vector<std::vector<double>> kronecker(std::vector<std::vector<double>>, std::vector<std::vector<double>>);
std::vector<std::vector<double>> kronecker(std::vector<std::vector<double>>, std::vector<double>);
std::vector<std::vector<double>> hadamard(std::vector<std::vector<double>>, std::vector<std::vector<double>>);
std::vector<std::vector<double>> vec2mat(std::vector<double>, unsigned int, unsigned int);
std::vector<double> mat2vec(std::vector<std::vector<double>>);

// activations.cpp

double sigmoid(double);
double sigmoidder(double);
std::vector<double> sigmoidv(std::vector<double>);
std::vector<double> sigmoidvder(std::vector<double>);
std::vector<std::vector<double>> sigmoid(std::vector<std::vector<double>>);
std::vector<std::vector<double>> sigmoidder(std::vector<std::vector<double>>);
std::vector<double> softmax(std::vector<double>, double);
std::vector<double> softmaxder(std::vector<double>, double);
std::vector<std::vector<double>> softmax(std::vector<std::vector<double>>, double);
std::vector<std::vector<double>> softmaxder(std::vector<std::vector<double>>, double);
double ReLU(double);
double ReLUder(double);
std::vector<double> ReLUv(std::vector<double>);
std::vector<double> ReLUvder(std::vector<double>);
double SeLU(double);
double SeLUder(double);
std::vector<double> SeLUv(std::vector<double>);
std::vector<double> SeLUvder(std::vector<double>);
std::vector<double> LOTA(std::vector<double> y);
std::vector<std::vector<double>> LOTA(std::vector<std::vector<double>>);
std::vector<std::vector<double>> LOTA(std::vector<std::vector<double>>, int);
std::vector<double> LOTAder(std::vector<double> y);
std::vector<std::vector<double>> LOTAder(std::vector<std::vector<double>>);

// weights.cpp

void randomweights(std::vector<std::vector<double>>);
void jumbledwbs(std::vector<std::vector<double>>);
void ijbasedwbs(std::vector<std::vector<double>>);
void Random(std::vector<std::vector<double>>);

// functions1.cpp

unsigned int hcf(unsigned int, unsigned int);
unsigned int lcm(unsigned int, unsigned int);

// functions2.cpp

long long int factorial(int);
long long int nCr(int, int);
long long int nPr(int, int);


#endif
