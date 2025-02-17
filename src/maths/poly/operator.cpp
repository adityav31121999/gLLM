
// operator.cpp : file for operator operations for polynomials
#include <algorithm>
#include "poly.hpp"

/**
 * @brief Overload of the = operator. 
 * @param[in] p Polynomial to be assigned
 * @return Polynomial with assigned values
 */
template <typename t1, typename t2> poly<t1, t2> poly<t1, t2>::operator=(const poly p) {
    this->order = p.order;
    this->coeffs = p.coeffs;
    this->powersofx = p.powersofx;
    return *this;
}


/**
 * @brief Overload of the + operator. 
 * @param[in] p Polynomial to be added
 * @return New polynomial with the added values
 */
template <typename t1, typename t2> poly<t1, t2> poly<t1, t2>::operator+(const poly p) {
    // fix the order for the polynomial
    int s = 0;
    if(this->order == p.order)
        s = this->order;
    else if(this->order > p.order)
        s = this->order;
    else if(this->order < p.order)
        s = p.order;
    poly r(s);
    // now add the polynomials
    for (int i = 0; i <= p.order; i++) {
        // add coefficients
        r.coeffs.push_back(this->coeffs[i] + p.coeffs[i]);
        // set power of x for this coefficient
        if(r.coeffs[i] == 0) 
            r.powersofx.push_back(0);
        else
            r.powersofx.push_back(1);
    }
    return r;
}


/**
 * @brief Overload of the - operator. 
 * @param[in] p Polynomial to be added
 * @return New polynomial with the added values
 */
template <typename t1, typename t2>
inline poly<t1, t2> poly<t1, t2>::operator-(const poly p) {
    poly r;
    // fix the order for the polynomial
    if(this->order == p.order) {
        r(this->order);
        int s = this->order;
    }
    else if(this->order > p.order) {
        r(this->order);
        int s = this->order;
    }
    else if(this->order < p.order) {
        r(p.order);
        int s = p.order;
    }
    // now add the polynomials
    for (int i = 0; i <= p.order; i++) {
        // add coefficients
        r.coeffs.push_back(this->coeffs[i] - p.coeffs[i]);
        // set power of x for this coefficient
        if(r.coeffs[i] == 0) 
            r.powersofx.push_back(0);
        else
            r.powersofx.push_back(1);
    }
    return r;
}


/**
 * @brief Overload of the * operator. Multiply the polynomial by a double.
 * @param[in] p Double to be multiplied to the polynomial
 * @return New polynomial with the multiplied double
 */
template <typename t1, typename t2>
inline poly<t1, t2> poly<t1, t2>::operator*(const double p) {
    poly r(this->order, this->coeffs);
    std::transform(this->coeffs.begin(), this->coeffs.end(), this->coeffs.begin(), 
                  [p](double val) { return val * p; });
    return r;
}


/**
 * @brief Overload of the / operator. Divide the polynomial by a double.
 * @param[in] p Double to be divided from the polynomial
 * @return New polynomial with the divided double
 */
template <typename t1, typename t2>
inline poly<t1, t2> poly<t1, t2>::operator/(const double p) {
    poly r(this->order, this->coeffs);
    std::transform(this->coeffs.begin(), this->coeffs.end(), this->coeffs.begin(), 
                  [p](double val) { return val / p; });
    return r;
}


/**
 * @brief Overload of the * operator. Multiply two polynomials.
 *      This method multiplies two polynomials by computing the product 
 *      of the coefficients of the two polynomials.
 * @param[in] p Polynomial to be multiplied
 * @return New polynomial with the product of the two input polynomials
 */
template <typename t1, typename t2>
poly<t1, t2> poly<t1, t2>::operator*(const poly<t1, t2> p) {
    // Create a new polynomial with the same order as this
    poly r;
    r.order = this->order + p.order + 2;
    r.coeffs.resize(r.order + 3, 0.0);
    r.powersofx.resize(r.order + 3, 0.0);
    // Create a matrix of coefficients
    std::vector<std::vector<t1>> a(this->order + 1, std::vector<t1>(this->order + p.order + 3, 0.0));
    std::vector<std::vector<t1>> b(this->order + 1, std::vector<t1>(p.order + 1, 0.0));
    // Compute the product of the coefficients of the two polynomials
    for (int i = 0; i <= this->order; i++) {
        for (int j = 0; j <= p.order; j++) {
            b[i][j] = this->coeffs[i] * p.coeffs[j];
        }
    }
    // Compute the sum of the columns of the matrix
    for(int i = 0; i <= this->order; i++) {
        for(int j = 0; j <= p.order; j++) {
            a[i][i+j] = b[i][j];
        }
    }
    r.coeffs = sumofcol(a);
    for (int i = 0; i <= r.order; i++) {
        r.powersofx[i] = (r.coeffs[i] != 0.0) ? 1.0 : 0.0;
    }
    // Return the new polynomial
    return r;
}


/**
 * @brief Raise a polynomial to a power
 * @param[in] p Polynomial to be raised to a power
 * @param[in] n Power to which the polynomial should be raised
 * @return A new polynomial which is the result of raising the input polynomial to the specified power
 */
template <typename t1, typename t2>
poly<t1, t2> poly<t1, t2>::power(poly<t1, t2> p, unsigned int n) {
    // create a new polynomial with the same coefficients as p
    poly r(p);
    // loop n times
    for(int i = 0; i < n; i++) {
        // multiply r by p
        r = r * p;
    }
    // return the new polynomial
    return r;
}
