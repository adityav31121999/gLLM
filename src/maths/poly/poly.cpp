
// some polynomial functions
#include "include/poly.hpp"


/**
 * @brief Compute the value of the polynomial at a given point x.
 * @param[in] x The value of x at which the polynomial should be evaluated.
 * @return p(x)
 */
template <typename t1, typename t2>
t2 poly<t1, t2>::poly_x_(t2 x) {
    // compute the value of the polynomial at x
    t2 sum = t2(0);
    t2 px = t2(1);
    for(int i = 0; i <= this->order; i++) {
        // add the ith term of the polynomial to the sum
        sum += this->coeffs[i] * px;
        // multiply px by x to prepare for the next term
        if(i == this->order - 1)
            break;
        px *= x;
    }
    // return the computed value of the polynomial
    return sum;
}


/**
 * @brief Compute the derivative of the polynomial.
 * @param[in] p The polynomial whose derivative is to be computed.
 * @return A new polynomial representing the derivative of the input polynomial.
 */
template <typename t1, typename t2>
poly<t1, t2> poly<t1, t2>::dp_dx(poly p) {
    // dp/dx = p'
    poly p_(p.order - 1);
    // Resize the coefficient and powers of x vectors to match the new order
    p_.coeffs.resize(p.order);
    p_.powersofx.resize(p.order);
    // Compute the derivative coefficients
    for(int i = 0; i < p_.order; i++) {
        // Derivative of x^i is i*x^(i-1), so multiply by i
        p_.coeffs[i] = (i + 1) * p.coeffs[i + 1];
    }
    for(int i = 0; i < p_.order; i++) {
        p_.powersofx[i] = (coeffs[i] == 0) ? 0 : 1;
    }
    // Return the resultant polynomial representing the derivative
    return p_;
}
