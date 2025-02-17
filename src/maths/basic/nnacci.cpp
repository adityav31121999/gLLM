
#include "include/nnacci.hpp"

#include <vector>
#include <numeric> // For std::accumulate
#include <iostream>
#include <algorithm> // For std::generate
#include <random> // For random number generation
#include <stdexcept> // For std::out_of_range
#include "nnacci.hpp"

/**
 * @brief Copy assignment operator
 * @param[in] other the object to be copied into this
 * @return a reference to this
 */
template <typename t> nnacci<t> &nnacci<t>::operator=(const nnacci<t> &other) {
    if (this != &other) {
        values = other.values;
        nvalue = other.nvalue;
        new_values = other.new_values;
    }
    return *this;
}

/**
 * @brief Move assignment operator
 * @param[in] other the object to be moved into this
 * @return a reference to this
 */
template <typename t> nnacci<t> &nnacci<t>::operator=(nnacci<t> &&other) noexcept {
    if (this != &other) {
        nvalue = other.nvalue;
        values = std::move(other.values);
        new_values = std::move(other.new_values);
    }
    return *this;
}

/**
 * @brief Binary plus operator
 * @param[in] other the object to be added to this
 * @return a new object that is the sum of this and the other
 */
template <typename t> nnacci<t> nnacci<t>::operator+(const nnacci<t> other) {
    nnacci result = *this;
    result += other;
    return result;
}
/**
 * @brief Binary minus operator
 * @param[in] other the object to be subtracted from this
 * @return a new object that is the difference of this and the other
 */
template <typename t> nnacci<t> nnacci<t>::operator-(const nnacci<t> other) {
    nnacci result = *this;
    result -= other;
    return result;
}

/**
 * @brief Binary multiplication operator
 * @param[in] other the object to be multiplied with this
 * @return a new object that is the product of this and the other
 */
template <typename t> nnacci<t> nnacci<t>::operator*(const nnacci<t> other) {
    nnacci result = *this;
    result *= other;
    return result;
}

/**
 * @brief Binary division operator
 * @param[in] other the object to divide this by
 * @return a new object that is the quotient of this and the other
 */
template <typename t> nnacci<t> nnacci<t>::operator/(const nnacci<t> other) {
    nnacci result = *this;
    result /= other;
    return result;
}

/**
 * @brief Binary modulus operator
 * @param[in] other the object to be divided modulo this
 * @return a new object that is the remainder of this divided by the other
 */
template <typename t> nnacci<t> nnacci<t>::operator%(const nnacci<t> other) {
    nnacci result = *this;
    result %= other;
    return result;
}

/**
 * @brief Add the values of two nnacci objects element-wise
 * @param[in] other the object to be added
 * @return the modified object
 */
template <typename t> nnacci<t> nnacci<t>::operator+=(const nnacci<t> other) {
    if (values.size() != other.values.size() || new_values.size() != other.new_values.size()) {
        throw std::invalid_argument("Vectors must be of the same size");
    }
    std::transform(values.begin(), values.end(), other.values.begin(), values.begin(), std::plus<t>());
    std::transform(new_values.begin(), new_values.end(), other.new_values.begin(), new_values.begin(), std::plus<t>());
    return *this;
}

/**
 * @brief Subtract the values of two nnacci objects element-wise
 * @param[in] other the object to be subtracted
 * @return the modified object
 */
template <typename t> nnacci<t> nnacci<t>::operator-=(const nnacci<t> other) {
    if (values.size() != other.values.size() || new_values.size() != other.new_values.size()) {
        throw std::invalid_argument("Vectors must be of the same size");
    }
    std::transform(values.begin(), values.end(), other.values.begin(), values.begin(), std::minus<t>());
    std::transform(new_values.begin(), new_values.end(), other.new_values.begin(), new_values.begin(), std::minus<t>());
    return *this;
}

/**
 * @brief Multiply the values of two nnacci objects element-wise
 * @param[in] other the object to be multiplied
 * @return the modified object
 */
template <typename t> nnacci<t> nnacci<t>::operator*=(const nnacci other) {
    if (values.size() != other.values.size() || new_values.size() != other.new_values.size()) {
        throw std::invalid_argument("Vectors must be of the same size");
    }
    std::transform(values.begin(), values.end(), other.values.begin(), values.begin(), std::multiplies<t>());
    std::transform(new_values.begin(), new_values.end(), other.new_values.begin(), new_values.begin(), std::multiplies<t>());
    return *this;
}

/**
 * @brief Divide the values of two nnacci objects element-wise
 * @param[in] other the object to divide by
 * @return the modified object
 */
template <typename t> nnacci<t> nnacci<t>::operator/=(const nnacci other) {
    if (values.size() != other.values.size() || new_values.size() != other.new_values.size()) {
        throw std::invalid_argument("Vectors must be of the same size");
    }
    std::transform(values.begin(), values.end(), other.values.begin(), values.begin(), std::divides<t>());
    std::transform(new_values.begin(), new_values.end(), other.new_values.begin(), new_values.begin(), std::divides<t>());
    return *this;
}

/**
 * @brief Compute the modulus of two nnacci objects element-wise
 * @param[in] other the object to divide by
 * @return the modified object
 */
template <typename t> nnacci<t> nnacci<t>::operator%=(const nnacci other) {
    if (values.size() != other.values.size() || new_values.size() != other.new_values.size()) {
        throw std::invalid_argument("Vectors must be of the same size");
    }
    std::transform(values.begin(), values.end(), other.values.begin(), values.begin(), std::modulus<t>());
    std::transform(new_values.begin(), new_values.end(), other.new_values.begin(), new_values.begin(), std::modulus<t>());
    return *this;
}

/**
 * @brief Access the element at index in the nnacci object
 * @param[in] index the index of the element to access
 * @return the element at index
 * @throws std::out_of_range if index is out of range
 */
template <typename t> t nnacci<t>::operator[](int index) const {
    if (index < 0 || index >= new_values.size()) {
        throw std::out_of_range("Index out of range");
    }
    return new_values[index];
}

/**
 * @brief Access the element at index in the nnacci object
 * @param[in] index the index of the element to access
 * @return the element at index
 * @throws std::out_of_range if index is out of range
 */
template <typename t> t nnacci<t>::operator[](int index) {
    if (index < 0 || index >= new_values.size()) {
        throw std::out_of_range("Index out of range");
    }
    return new_values[index];
}

template <typename t> t nnacci<t>::operator()(int index) const {
    return (*this)[index];
}

template <typename t> t nnacci<t>::operator()(int index) {
    return (*this)[index];
}

template <typename t> bool nnacci<t>::operator==(const nnacci other) const {
    return values == other.values && new_values == other.new_values;
}

template <typename t> bool nnacci<t>::operator!=(const nnacci other) const {
    return !(*this == other);
}

template <typename t> bool nnacci<t>::operator>(const nnacci other) const {
    return std::lexicographical_compare(new_values.begin(), new_values.end(), other.new_values.begin(), other.new_values.end(), std::greater<t>());
}

template <typename t> bool nnacci<t>::operator<(const nnacci other) const {
    return std::lexicographical_compare(new_values.begin(), new_values.end(), other.new_values.begin(), other.new_values.end(), std::less<t>());
}

template <typename t> bool nnacci<t>::operator>=(const nnacci other) const {
    return !(*this < other);
}

template <typename t> bool nnacci<t>::operator<=(const nnacci other) const {
    return !(*this > other);
}

template <typename t> bool nnacci<t>::operator==(const t other) const {
    return std::all_of(new_values.begin(), new_values.end(), [other](const t &val) { return val == other; });
}

template <typename t> bool nnacci<t>::operator!=(const t other) const {
    return !(*this == other);
}

template <typename t> bool nnacci<t>::operator>(const t other) const {
    return std::all_of(new_values.begin(), new_values.end(), [other](const t &val) { return val > other; });
}

template <typename t> bool nnacci<t>::operator<(const t other) const {
    return std::all_of(new_values.begin(), new_values.end(), [other](const t &val) { return val < other; });
}

template <typename t> bool nnacci<t>::operator>=(const t other) const {
    return std::all_of(new_values.begin(), new_values.end(), [other](const t &val) { return val >= other; });
}

template <typename t> bool nnacci<t>::operator<=(const t other) const {
    return std::all_of(new_values.begin(), new_values.end(), [other](const t &val) { return val <= other; });
}

template <typename t> void nnacci<t>::createval() {
    // Create nnacci sequence from values vector
    if (nvalue < 1) return;

    new_values = values;
    for (size_t i = nvalue; i < values.size(); ++i) {
        t sum = std::accumulate(new_values.end() - nvalue, new_values.end(), t{});
        new_values.push_back(sum);
    }
}

template <typename t> void nnacci<t>::randomval() {
    // Create random values as seed for nnacci sequence
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 100);

    values.clear();
    for (int i = 0; i < nvalue; ++i) {
        values.push_back(dis(gen));
    }
}

template <typename t> void nnacci<t>::printval() const {
    // Print vector of values
    for (const auto &val : values) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

template <typename t> void nnacci<t>::printnval() const {
    // Print nvalue
    std::cout << "nvalue: " << nvalue << std::endl;
}

template <typename t> void nnacci<t>::printnewval() const {
    // Print new values after operation
    for (const auto &val : new_values) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

template <typename t> void nnacci<t>::setn(int n) {
    nvalue = n;
}

template <typename t> void nnacci<t>::setval(const std::vector<t> &other) {
    values = other;
    nvalue = values.size();
}

template <typename t> void nnacci<t>::setnewval(const std::vector<t> &other) {
    new_values = other;
}

template <typename t> int nnacci<t>::getn() const {
    return nvalue;
}

template <typename t> const std::vector<t> &nnacci<t>::getval() const {
    return values;
}

template <typename t> const std::vector<t> &nnacci<t>::getnewval() const {
    return new_values;
}


