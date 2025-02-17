
// for recurrence polynomials
#include "poly.hpp"

/**
 * @brief Constructs a vector of recurrence polynomials from a vector of polynomials 
 * and a vector of scalars with a specified number of terms.
 * This function takes a vector of polynomials and a vector of scalars, and adds a specified number of
 * recurrence polynomials to the vector. The recurrence polynomials are determined by the scalars in the
 * vector, which are used to compute a sequence of polynomials. Each polynomial in the sequence is
 * computed by multiplying the current polynomial in the sequence by the current scalar in the vector.
 * @tparam t1 The type of the x-value of the polynomials.
 * @tparam t2 The type of the y-value of the polynomials.
 * @param polynomial A vector of polynomials.
 * @param recurrence A vector of scalars.
 * @param terms The number of terms to add.
 * @return A vector of recurrence polynomials.
 */
template <typename t1, typename t2>
std::vector<poly<t1, t2>> recurrence(std::vector<poly<t1, t2>> polynomial, std::vector<t1> recurrence, unsigned int terms) {
    // define a new polynomial vector
    std::vector<poly<t1, t2>> result;
    // loop through the polynomial vector
    for (unsigned int i = 0; i < polynomial.size(); i++)
        result[i] = polynomial[i];
    // add the recurrence vector to the polynomial vector
    for(unsigned int i = 0; i < terms; i++) {
        // loop through the recurrence vector
        for(unsigned int j = 0; j < recurrence.size(); j++) {
            // multiply the current polynomial in the sequence by the current scalar in the vector
            result[i + polynomial.size()] += recurrence[i] * polynomial[i];
        }
    }
    // return the result
    return result;
}


/**
 * @brief Constructs a vector of Chebyshev polynomials from a vector of polynomials.
 * This function takes a vector of polynomials and adds a specified number of Chebyshev 
 * polynomials to it. The sequence starts with the given polynomials and generates subsequent
 * Chebyshev polynomials using the recurrence relation.
 * @tparam t1 The type of the x-value of the polynomials.
 * @tparam t2 The type of the y-value of the polynomials.
 * @param polynomial A vector of polynomials.
 * @param terms The number of Chebyshev polynomials to add.
 * @return A vector of Chebyshev polynomials.
 */
template <typename t1, typename t2>
std::vector<poly<t1, t2>> chebyshev(std::vector<poly<t1, t2>> polynomial, unsigned int terms) {
    // Define a new polynomial vector to store the result
    std::vector<poly<t1, t2>> result;
    // Initialize the result with the initial polynomials
    for (unsigned int i = 0; i < polynomial.size(); i++) {
        result.push_back(polynomial[i]);
    }
    // Generate Chebyshev polynomials using the recurrence relation
    for(unsigned int i = polynomial.size(); i < polynomial.size() + terms; i++) {
        // T_n(x) = 2*x*T_{n-1}(x) - T_{n-2}(x)
        // Here, we assume polynomial[i] represents T_{n-1}(x) and polynomial[i-1] represents T_{n-2}(x)
        result.push_back(2 * result[i - 1] - result[i - 2]);
    }
    // Return the vector of Chebyshev polynomials
    return result;
}
