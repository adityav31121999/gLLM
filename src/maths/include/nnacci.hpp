
// nnacci.hpp: header source for nnacci sequeces
#ifndef NNACCI_HPP
#define NNACCI_HPP 1

#include <vector>

/**
 * @brief CLASS: n-nacci sequence
 * @param t The type of data
 */
template <typename t>
class nnacci {
private:
    int nvalue;                     // total values to be recursed on
    std::vector<t> values;          // original values
    std::vector<t> new_values;      // new values made with original values

public:
    // default constructor
    nnacci() = default;

    // constructor with value
    nnacci(int value) : nvalue(value) {
        values.resize(nvalue);
    }

    // constructor with vector
    nnacci(std::vector<t> values) : values(values) {
        nvalue = values.size();
    }

    // copy constructor
    nnacci(const nnacci<t> &other){
        values = other.values;
        nvalue = values.size();
        new_values = other.new_values;
    }

    // move constructor
    nnacci(nnacci<t> &&other) {
        nvalue = values.size();
        values = std::move(other.values);
        new_values = std::move(other.new_values);
    }

    void createNewValues();

    nnacci &operator=(const nnacci &other);
    nnacci &operator=(nnacci &&other);

    nnacci operator+(const nnacci other);
    nnacci operator-(const nnacci other);
    nnacci operator*(const nnacci other);
    nnacci operator/(const nnacci other);
    nnacci operator%(const nnacci other);
    nnacci operator+=(const nnacci other);
    nnacci operator-=(const nnacci other);
    nnacci operator*=(const nnacci other);
    nnacci operator/=(const nnacci other);
    nnacci operator%=(const nnacci other);

    t operator[](int index) const;
    t operator[](int index);
    t operator()(int index) const;
    t operator()(int index);

    bool operator==(const nnacci other) const;
    bool operator!=(const nnacci other) const;
    bool operator>(const nnacci other) const;
    bool operator<(const nnacci other) const;
    bool operator>=(const nnacci other) const;
    bool operator<=(const nnacci other) const;
    bool operator==(const t other) const;
    bool operator!=(const t other) const;
    bool operator>(const t other) const;
    bool operator<(const t other) const;
    bool operator>=(const t other) const;
    bool operator<=(const t other) const;

    void createval();               // create nnacci sequence from values vector
    void randomval();               // create random values as seed for nnacci sequence
    void printval() const;          // vector of values
    void printnval() const;         // nvalue
    void printnewval() const;       // new values after operation

    void setn(int n);
    void setval(const std::vector<t> &other);
    void setnewval(const std::vector<t> &other);

    int getn() const;
    const std::vector<t> &getval() const;
    const std::vector<t> &getnewval() const;

    ~nnacci() = default;
};


#endif
