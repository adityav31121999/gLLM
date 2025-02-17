
// basic functions of arithmetics
#include "include/basic.hpp"

/**
 * @brief Calculate the highest common factor of two numbers using the Euclidean Algorithm.
 * @param a The first number.
 * @param b The second number.
 * @return The highest common factor of the two numbers.
 */
unsigned int hcf(unsigned int a, unsigned int b) {
    while (b != 0) {
        int temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

/**
 * @brief Calculate the lowest common multiple of two numbers.
 * @param a The first number.
 * @param b The second number.
 * @return The lowest common multiple of the two numbers.
 */
unsigned int lcm(unsigned int a, unsigned int b) {
    return (a * b) / hcf(a, b);
}
