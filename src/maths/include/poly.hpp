
// poly.hpp : header source of polynomial library
#ifndef POLY_HPP
#define POLY_HPP 1

#include <vector>

/**
 * @brief A class representing a polynomial. This class represents a polynomial, 
 *    and provides methods for adding, subtracting, multiplying, and dividing polynomials. 
 *    It also provides a method for computing the value of the polynomial at a given point. 
 *    P(X) = {0, 1, 2, 3, 4, 5, 6, 7, ..... n-1, n} <- Powers (n+1 elements)
 *           {C, C1, C2, C3, C4,................Cn} <- Coefficients of powers
 *    ex.: 3x^4 + 4x^2 + 5x + 6 => {6, 5, 4, 0, 3} <- Coefficients of powers
 *                                 {1, 1, 1, 0, 1} <- Powers (0 represent the absence if ith power and its coefficient)
 * @tparam t The type of the x-value of the polynomial
 */
template <typename t1, typename t2>
class poly {
public:
// member variables
    unsigned int order;                 // order of polynomial, order = largest power of x in polynomial
    std::vector<t1> coeffs;             // coefficients of polynomial
    std::vector<int> powersofx;         // powers of x in polynomial, x can be matrix and values
    std::vector<t2> powersofxvalues;    // values of powers of x (x^n)

// member functions
    // constructors

    /**
     * @brief Default constructor.
     */
    poly() : order(0), coeffs(1), powersofx(1) {}

    /**
     * @brief Constructs a polynomial with a specified order and sets all coefficients to 0.
     * @param o The order of the polynomial.
     */
    poly(unsigned int o) {
        order = o;
        coeffs.resize(o + 1, 0.0);
        powersofx.resize(o + 1, 0.0);
    }

    /**
     * @brief Constructs a polynomial with a specified order and coefficients.
     * @param o The order of the polynomial.
     * @param c A vector containing the coefficients of the polynomial.
     */
    poly(unsigned int o, std::vector<t1> c) : order(o), coeffs(c) {
        for(auto i: c)
            powersofx.push_back((i != 0) ? 1 : 0);
    }

    /**
     * @brief Constructs a polynomial with a specified order, coefficients, and powers of x.
     * @param o The order of the polynomial.
     * @param c A vector containing the coefficients of the polynomial.
     * @param p A vector containing the powers of x in the polynomial.
     */
    poly(unsigned int o, std::vector<t1> c, std::vector<int> p) : order(o), coeffs(c), powersofx(p) {}

    /**
     * @brief Copy constructor.
     * @param[in] p The poly object to be copied.
     */
    poly(const poly& p) {
        order = p.order;
        coeffs = p.coeffs;
        powersofx = p.powersofx;
    }

    // operators

    poly operator=(const poly);
    poly operator+(const poly);
    poly operator-(const poly);
    poly operator*(const poly);
    poly operator*(const float);
    poly operator/(const float);
    poly power(poly, unsigned int);

    poly dp_dx(poly);
    t2 poly_x_(t2);

    ~poly() {}      // default destructor
};


// recurrence polynomial

template <typename t1, typename t2> std::vector<poly<t1, t2>> recurrence(std::vector<poly<t1, t2>>, std::vector<t1, t2>, unsigned int);
template <typename t1, typename t2> std::vector<poly<t1, t2>> chebyshev(std::vector<poly<t1, t2>>, unsigned int);

// 



#endif
