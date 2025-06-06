
// stat.hpp : header source of statistics library
#ifndef STAT_HPP
#define STAT_HPP 1

#include <vector>

// basic functions of statistics

float mean(std::vector<float> a);
float median(std::vector<float> a);
float mode(std::vector<float> a);
float variance(std::vector<float> a);
float standardDeviation(std::vector<float> a);
float covariance(std::vector<float> a, std::vector<float> b);
float correlation(std::vector<float> a, std::vector<float> b);
float spearman(std::vector<float> a, std::vector<float> b);
float rank(float val, std::vector<float> vec);
float percentile(float p, std::vector<float> vec);
float quartile(float q, std::vector<float> vec);
float interquartileRange(std::vector<float> vec);
float zScore(float val, std::vector<float> vec);
float outlier(float val, std::vector<float> vec);
std::vector<float> outlier(std::vector<float> vec);


#endif
