
// stat.hpp : header source of statistics library
#ifndef STAT_HPP
#define STAT_HPP 1

#include <vector>

// basic functions of statistics

double mean(std::vector<double> a);
double median(std::vector<double> a);
double mode(std::vector<double> a);
double variance(std::vector<double> a);
double standardDeviation(std::vector<double> a);
double covariance(std::vector<double> a, std::vector<double> b);
double correlation(std::vector<double> a, std::vector<double> b);
double spearman(std::vector<double> a, std::vector<double> b);
double rank(double val, std::vector<double> vec);
double percentile(double p, std::vector<double> vec);
double quartile(double q, std::vector<double> vec);
double interquartileRange(std::vector<double> vec);
double zScore(double val, std::vector<double> vec);
double outlier(double val, std::vector<double> vec);
std::vector<double> outlier(std::vector<double> vec);


#endif
