
// basic functions of statistics
#include "stats.hpp"
#include <functional>
#include <algorithm>
#include <numeric>
#include <set>
#include <stdexcept>

/**
 * @brief Calculates the mean of a vector of doubles.
 * @param a pointer to vector of doubles
 * @return the mean of the vector
 */
double mean(std::vector<double> a) {
    // Mean
    return std::accumulate(a.begin(), a.end(), 0.0) / a.size();
}

/**
 * @brief Calculates the median of a vector of doubles. The median is the 
 * middle value in an ordered list of numbers. If the list has an even number 
 * of numbers, the median is the average of the two middle values.
 * @param a pointer to vector of doubles
 * @return the median of the vector
 */
double median(std::vector<double> a) {
    // Sort the vector
    std::sort(a.begin(), a.end());
    // Calculate the median
    if (a.size() % 2 == 0) {
        // Even number of elements, return the average of the two middle values
        return (a.at(a.size() / 2 - 1) + a.at(a.size() / 2)) / 2;
    } else {
        // Odd number of elements, return the middle value
        return a.at(a.size() / 2);
    }
}

/**
 * @brief Calculates the mode of a vector of doubles.
 * The mode is the most frequently occurring value in the vector.
 * @param a pointer to vector of doubles
 * @return the mode of the vector
 */
double mode(std::vector<double> a) {
    // Create a set of unique values in the vector
    std::set<double> s(a.begin(), a.end());
    std::vector<double> unique(s.begin(), s.end()); 
    std::vector<int> counts(unique.size());
    // Count the occurrences of each unique value
    for (int i = 0; i < unique.size(); i++) {
        counts[i] = std::count(a.begin(), a.end(), unique[i]);
    }
    return unique[std::distance(counts.begin(), std::max_element(counts.begin(), counts.end()))];
}

/**
 * @brief Calculates the variance of a vector of doubles.
 * @details Variance is a measure of the spread of the numbers in a dataset.
 * It is calculated as the average of the squared differences from the mean.
 * @param a pointer to vector of doubles
 * @return the variance of the vector
 */
double variance(std::vector<double> a) {
    double m = mean(a); // Calculate mean of the vector
    double sum = 0; // Initialize sum of squared differences
    for (int i = 0; i < a.size(); i++) {
        // Calculate squared difference from the mean
        sum += std::pow(a.at(i) - m, 2);
    }
    return sum / a.size(); // Return average of squared differences
}

/**
 * @brief Calculates the standard deviation of a vector of doubles.
 * @details The standard deviation is the square root of the variance.
 * @param a pointer to vector of doubles
 * @return the standard deviation of the vector
 */
double standardDeviation(std::vector<double> a) {
    // Standard deviation
    return sqrt(variance(a));
}

/**
 * @brief Calculates the covariance between two vectors of doubles.
 * @details Covariance is a measure of the joint variability of two random variables.
 * @param a pointer to the first vector of doubles
 * @param b pointer to the second vector of doubles
 * @return the covariance between the two vectors
 */
double covariance(std::vector<double> a, std::vector<double> b) {
    double sum = 0; // Initialize sum for covariance calculation
    double meanA = mean(a); // Calculate mean of vector a
    double meanB = mean(b); // Calculate mean of vector b
    // Iterate through each element of the vectors
    for (int i = 0; i < a.size(); i++) {
        // Calculate the product of deviations from the mean and add to sum
        sum += (a.at(i) - meanA) * (b.at(i) - meanB);
    }
    // Return the average of the products of deviations
    return sum / a.size();
}

/**
 * @brief Calculates the correlation coefficient between two vectors of doubles.
 * @details The correlation coefficient is a measure of the linear relationship
 * between two random variables. It is calculated as the covariance divided by
 * the product of the standard deviations of the two vectors.
 * @param a pointer to the first vector of doubles
 * @param b pointer to the second vector of doubles
 * @return the correlation coefficient between the two vectors
 */
double correlation(std::vector<double> a, std::vector<double> b) {
    // Calculate the correlation coefficient
    return covariance(a, b) / (standardDeviation(a) * standardDeviation(b));
}

/**
 * @brief Calculates the rank of a value in a vector. The rank of a value is the 
 * number of elements in the vector that are smaller than the value.
 * @param val the value to calculate the rank of
 * @param vec the vector to calculate the rank from
 * @return the rank of the value in the vector
 */
double rank(double val, std::vector<double> vec) {
    int count = 0;
    for (int i = 0; i < vec.size(); i++) {
        if (vec.at(i) < val) count++;
    }
    return count;
}

/**
 * @brief Calculates the Spearman's rank correlation coefficient between two vectors of doubles.
 * Spearman's rank correlation coefficient is a measure of the monotonic relationship between two variables.
 * It evaluates how well the relationship between two variables can be described by a monotonic function.
 * @param a pointer to the first vector of doubles
 * @param b pointer to the second vector of doubles
 * @return the Spearman's rank correlation coefficient between the two vectors
 */
double spearman(std::vector<double> a, std::vector<double> b) {
    // Create rank vectors for each input vector
    std::vector<double> rankA, rankB;
    for (int i = 0; i < a.size(); i++) {
        rankA.push_back(rank(a.at(i), a));
        rankB.push_back(rank(b.at(i), b));
    }
    // Calculate the Pearson correlation between the two rank vectors
    return correlation(rankA, rankB);
}

/**
 * @brief Calculates the expectation of a vector of doubles. The expectation is 
 * calculated as the weighted sum of elements in the vector, where the weights are 
 * provided by a corresponding vector of probabilities.
 * @param vec A pointer to the vector of doubles.
 * @param prob A pointer to the vector of probabilities.
 * @return The expectation of the vector.
 * @throws std::invalid_argument if the sum of probabilities is not equal to 1.
 */
double expectation(std::vector<double> vec, std::vector<double> prob) {
    // Check if the sum of probabilities is equal to 1
    if (std::accumulate(prob.begin(), prob.end(), 0.0) == 1.0) {
        // Calculate the weighted sum using inner product
        return std::inner_product(vec.begin(), vec.end(), prob.begin(), 0.0);
    } else {
        // Throw an exception if probabilities do not sum to 1
        throw std::invalid_argument("The sum of probabilities must be equal to 1.");
    }
}

/**
 * @brief Calculates the Pearson correlation coefficient between two vectors of doubles.
 * The Pearson correlation coefficient measures the linear relationship between two variables.
 * It evaluates how well the relationship between two variables can be described by a straight line.
 * @param a pointer to the first vector of doubles
 * @param b pointer to the second vector of doubles
 * @return the Pearson correlation coefficient between the two vectors
 */
double pearson(std::vector<double> a, std::vector<double> b) {
    return correlation(a, b)/(standardDeviation(a)*standardDeviation(b));
}

/**
 * @brief Calculates the percentile of a vector of doubles.
 * The percentile is the value at a given percentage of the sorted vector.
 * @param p The percentage of the percentile to calculate.
 * @param vec A pointer to the vector of doubles.
 * @return The value at the calculated percentile.
 */
double percentile(double p, std::vector<double> vec) {
    // Calculate the index of the percentile
    double index = p / 100.0 * (vec.size() - 1);
    // Sort the vector
    std::sort(vec.begin(), vec.end());
    // Return the value at the calculated index
    return vec.at(index);
}

/**
 * @brief Calculates the quartile of a vector of doubles.
 * The quartile is the value at a given percentage of the sorted vector.
 * @param q The percentage of the quartile to calculate.
 * @param vec A pointer to the vector of doubles.
 * @return The value at the calculated quartile.
 */
double quartile(double q, std::vector<double> vec) {
    // Calculate the index of the quartile
    double index = q / 100.0 * (vec.size() - 1);
    // Return the value at the calculated index
    return vec.at(index);
}


/**
 * @brief Calculates the interquartile range of a vector of doubles.
 * The interquartile range is the difference between the 75th percentile (Q3) and the 25th percentile (Q1).
 * @param vec A pointer to the vector of doubles.
 * @return The interquartile range of the vector.
 */
double interquartileRange(std::vector<double> vec) {
    // Calculate the 75th percentile (Q3)
    double q3 = quartile(75, vec);
    // Calculate the 25th percentile (Q1)
    double q1 = quartile(25, vec);
    // Return the interquartile range
    return q3 - q1;
}

/**
 * @brief Calculates the z-score of a value in a vector of doubles.
 * The z-score is calculated as (val - mean) / standardDeviation.
 * @param val The value for which the z-score is calculated.
 * @param vec A pointer to the vector of doubles.
 * @return The z-score of the value.
 */
double zScore(double val, std::vector<double> vec) {
    // Calculate the mean and standard deviation of the vector
    double m = mean(vec);
    double stdDev = standardDeviation(vec);
    // Calculate the z-score
    return (val - m) / stdDev;
}

/**
 * @brief Determines if a value is an outlier in a vector of doubles.
 * This function uses both the z-score and the interquartile range (IQR) method to determine if a value is an outlier. 
 * It calculates the z-score of the value and checks it against the IQR-based bounds. 
 * If the value is an outlier, the z-score is returned; otherwise, 0 is returned.
 * @param val The value to check for being an outlier.
 * @param vec A pointer to the vector of doubles.
 * @return The z-score of the value if it is an outlier, otherwise 0.
 */
double outlier(double val, std::vector<double> vec) {
    // Calculate the z-score of the value
    double z = zScore(val, vec);
    
    // Calculate the interquartile range of the vector
    double iqr = interquartileRange(vec);
    
    // Calculate the lower and upper bounds for outliers using IQR
    double lowerBound = quartile(25, vec) - 1.5 * iqr;
    double upperBound = quartile(75, vec) + 1.5 * iqr;
    
    // Check if the value is an outlier based on IQR
    if (val < lowerBound || val > upperBound) {
        return z;
    } else {
        return 0;
    }
}

/**
 * @brief Check for outliers in the vector
 * This function takes a vector as input and returns a vector of z-scores of values that are outliers.
 * An outlier is defined as a value that is 1.5 times the interquartile range (IQR) away from the 25th percentile (Q1) or 75th percentile (Q3).
 * @param vec The vector to be checked for outliers
 * @return A vector of z-scores of outliers
 */
std::vector<double> outlier(std::vector<double> vec) {
    // Calculate the mean and standard deviation of the vector
    double m = mean(vec);
    double stdDev = standardDeviation(vec);
    // Calculate the z-scores of all values in the vector
    std::vector<double> zScores;
    for (int i = 0; i < vec.size(); i++) {
        zScores.push_back((vec.at(i) - m) / stdDev);
    }
    // Calculate the interquartile range of the vector
    double iqr = interquartileRange(vec);
    // Calculate the lower and upper bounds for outliers
    double lowerBound = quartile(25, vec) - 1.5 * iqr;
    double upperBound = quartile(75, vec) + 1.5 * iqr;
    // Check for outliers and return the z-scores
    std::vector<double> outliers;
    for (int i = 0; i < vec.size(); i++) {
        if (vec.at(i) < lowerBound || vec.at(i) > upperBound) {
            outliers.push_back(zScores[i]);
        }
    }
    return outliers;
}
