
// vect.cpp: all vector related functions
#include "include/basic.hpp"


/**
 * @brief Checks if two vectors are equal.
 * This function takes two vectors of floats as an input and returns true if they are equal and false otherwise.
 * @param a first vector
 * @param b second vector
 * @return true if the vectors are equal, false otherwise
 */
bool operator==(std::vector<float> a, std::vector<float> b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (int i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Checks if two vectors are not equal.
 * This function takes two vectors of floats as an input and returns true if they are not equal and false otherwise.
 * @param a first vector
 * @param b second vector
 * @return true if the vectors are not equal, false otherwise
 */
bool operator!=(std::vector<float> a, std::vector<float> b) {
    return !(a == b);
}

/**
 * @brief Overloaded addition operator for vectors. This function takes two vectors 
 * as an input and returns a vector where each element is the sum of the corresponding 
 * elements of the input vectors.
 * @param x first vector
 * @param y second vector
 * @return a vector where each element is the sum of the corresponding elements of the 
 * input vectors
 */
std::vector<float> operator+(std::vector<float>x, std::vector<float> y) {
    if(x.size() != y.size()) {
        throw std::runtime_error("Vectors must be of the same size");
    }
    std::vector<float> result(x.size());
    std::transform(x.begin(), x.end(), y.begin(), result.begin(), [](const auto& i, const auto& j) { return i+j; });
    return result;
}

/**
 * @brief Overloaded subtraction operator for vectors. This function takes two vectors as 
 * an input and returns a vector where each element is the difference between the corresponding 
 * elements of the input vectors.
 * @param x first vector
 * @param y second vector
 * @return a vector where each element is the difference between the corresponding elements of 
 * the input vectors
 */
std::vector<float> operator-(std::vector<float>x, std::vector<float> y) {
    if(x.size() != y.size()) {
        throw std::runtime_error("Vectors must be of the same size");
    }
    std::vector<float> result(x.size());
    std::transform(x.begin(), x.end(), y.begin(), result.begin(), [](const auto& i, const auto& j) { return i-j; });
    return result;
}

/**
 * @brief Overloaded multiplication operator for vectors. This function takes a vector and a number
 * as an input and returns a vector where each element is the product of the corresponding
 * elements of the input vector and number.
 * @param x vector
 * @param y number
 * @return a vector where each element is the product of the corresponding elements of the
 * input vector and number
 */
std::vector<float> operator*(std::vector<float> x, float y) {
    std::vector<float> result(x.size());
    std::transform(x.begin(), x.end(), result.begin(), [&y](const auto& i) { return i*y; });
    return result;
}

/**
 * @brief Overloaded multiplication operator for vectors. This function takes a vector and a number
 * as an input and returns a vector where each element is the product of the corresponding
 * elements of the input vector and number.
 * @param a number
 * @param b vector
 * @return a vector where each element is the product of the corresponding elements of the
 * input vector and number
 */
std::vector<float> operator*(float a, std::vector<float> b) {
    std::vector<float> c(b.size(), 0.0);
    for(int i = 0; i < b.size(); i++) {
        c[i] = b[i] * a;
    }
    return c;
}

/**
 * @brief Overloaded division operator for vectors. This function takes a vector and a number
 * as an input and returns a vector where each element is the division of the corresponding
 * elements of the input vector and number.
 * @param x vector
 * @param y number
 * @return a vector where each element is the division of the corresponding elements of the
 * input vector and number
 */
std::vector<float> operator/(std::vector<float> x, float y) {
    if(y == 0) {
        throw std::runtime_error("Division by zero is undefined");
    }
    std::vector<float> result(x.size());
    std::transform(x.begin(), x.end(), result.begin(), [&y](const auto& i) { return i/y; });
    return result;
}


/**
 * @brief Overloaded addition operator for matrix. This function takes two matrices as an input 
 * and returns a matrix where each element is the sum of the corresponding elements of the input 
 * matrices.
 * @param x first matrix
 * @param y second matrix
 * @return a matrix where each element is the sum of the corresponding elements of the input 
 * matrices
 */
std::vector<std::vector<float>> operator+(std::vector<std::vector<float>> x, std::vector<std::vector<float>> y) {
    if(x.size() != y.size() && x[0].size() != y[0].size()) {
        throw std::runtime_error("Matrices must be of the same size");
    }
    std::vector<std::vector<float>> a(x.size(), std::vector<float>(x[0].size()));
    for(int i = 0; i < x.size(); ++i) {
        std::transform(x[i].begin(), x[i].end(), y[i].begin(), a[i].begin(), [](const auto& j, const auto& k) { return j+k; });
    }
    // Return the resulting matrix
    return a;
}


/**
 * @brief Overloaded addition operator for matrix. This function takes two matrices as an input and 
 * returns a matrix where each element is the sum of the corresponding elements of the input matrices.
 * @param x first matrix
 * @param y second matrix
 * @return a matrix where each element is the sum of the corresponding elements of the input matrices
 */
std::vector<std::vector<float>> operator-(std::vector<std::vector<float>> x, std::vector<std::vector<float>> y) {
    if(x.size() != y.size() && x[0].size() != y[0].size()) {
        throw std::runtime_error("Matrices must be of the same size");
    }
    std::vector<std::vector<float>> a(x.size(), std::vector<float>(x[0].size()));
    for(int i = 0; i < x.size(); ++i) {
        std::transform(x[i].begin(), x[i].end(), y[i].begin(), a[i].begin(), [](const auto& j, const auto& k) { return j-k; });
    }
    // Return the resulting matrix
    return a;
}

/**
 * @brief Overloaded multiplication operator for matrix. This function takes a matrix and a number as an
 * input and returns a matrix where each element is the product of the corresponding elements of
 * the input matrix and the number.
 * @param x matrix
 * @param y number
 * @return a matrix where each element is the product of the corresponding elements of the input
 * matrix and the number
 */
std::vector<std::vector<float>> operator*(std::vector<std::vector<float>> x, float y) {
    // Create a new matrix where each element is the product of the corresponding elements of the input matrix and the number
    std::vector<std::vector<float>> a(x.size(), std::vector<float>(x[0].size()));
    for(int i = 0; i < x.size(); ++i) {
        // Use the std::transform algorithm to fill the matrix with the product of the corresponding elements of the input matrix and the number
        std::transform(x[i].begin(), x[i].end(), a[i].begin(), [&y](const auto& j) { return j*y; });
    }
    // Return the resulting matrix
    return a;
}

/**
 * @brief Overloaded division operator for matrix. This function takes a matrix and a number as an
 * input and returns a matrix where each element is the division of the corresponding elements of
 * the input matrix by the number.
 * @param x matrix
 * @param y number
 * @return a matrix where each element is the division of the corresponding elements of the input
 * matrix by the number
 */
std::vector<std::vector<float>> operator/(std::vector<std::vector<float>> x, float y) {
    if(y == 0) {
        throw std::runtime_error("Division by zero is undefined");
    }
    std::vector<std::vector<float>> a(x.size(), std::vector<float>(x[0].size()));
    for(int i = 0; i < x.size(); ++i) {
        std::transform(x[i].begin(), x[i].end(), a[i].begin(), [&y](const auto& j) { return j/y; });
    }
    // Return the resulting matrix
    return a;
}

/**
 * @brief Convert a 1D vector to a 2D matrix. This function takes a 1D vector of floats 
 * and two unsigned integers as input and returns a 2D vector of floats. The two unsigned 
 * integers represent the number of rows and columns in the output matrix.
 * @param vec 1D vector of floats
 * @param row number of rows in the output matrix
 * @param col number of columns in the output matrix
 * @return a 2D vector of floats
 */
std::vector<std::vector<float>> vec2mat(std::vector<float> vec, unsigned int row, unsigned int col) {
    // Check if the size of the input vector is equal to the product of row and col
    if (vec.size() != row * col) {
        throw std::runtime_error("Vector size is not equal to row * col");
    }
    // Create a 2D vector of floats with the given number of rows and columns
    std::vector<std::vector<float>> m = std::vector(row, std::vector<float>(col, 0.0));
    // Iterate over the elements of the input vector and assign them to the corresponding elements of the output matrix
    for (unsigned int i = 0; i < row; ++i) {
        for (unsigned int j = 0; j < col; ++j) {
            m[i][j] = vec[i * col + j];
        }
    }
    // Return the resulting matrix
    return m;
}

/**
 * @brief Convert a 1D vector to a 2D matrix. This function takes a 1D vector of floats 
 * and two unsigned integers as input and returns a 2D vector of floats. The two unsigned 
 * integers represent the number of rows and columns in the output matrix.
 * @param vec 1D vector of floats
 * @param row number of rows in the output matrix
 * @param col number of columns in the output matrix
 * @return a 2D vector of floats
 */
std::vector<float> mat2vec(std::vector<std::vector<float>> m) {
    // Create a 2D vector of floats with the given number of rows and columns
    std::vector<float> vec(m.size() * m[0].size());
    // Iterate over the elements of the input vector and assign them to the corresponding elements of the output matrix
    for (unsigned int i = 0; i < m.size(); ++i) {
        for (unsigned int j = 0; j < m[0].size(); ++j) {
            vec[i * m[0].size() + j] = m[i][j];
        }
    }
    // Return the resulting matrix
    return vec;
}

/**
 * @brief Calculates the sum of the elements in a vector.
 * This function takes a vector of floats as an input and returns the sum of all its elements.
 * @param a vector of floats
 * @return the sum of the elements in the vector
 */
float sum(std::vector<float> a) {
    return std::accumulate(a.begin(), a.end(), 0.0);
}

/**
 * @brief Calculates the sum of all elements in a 2D vector.
 * This function takes a 2D vector of floats as input and returns the sum of all its elements.
 * Internally, it computes the sum of each row and then sums up these row sums.
 * @param a 2D vector of floats
 * @return the sum of all elements in the 2D vector
 */
float sum(std::vector<std::vector<float>> a) {
    // Sum the elements of each row using sumofrow and then sum those results
    return sum(sumofrow(a));
}

/**
 * @brief Calculates the product of the elements in a vector.
 * This function takes a vector of floats as an input and returns the product of all its elements.
 * @param a vector of floats
 * @return the product of the elements in the vector
 */
float product(std::vector<float> a) {
    return static_cast<float>(std::accumulate(a.begin(), a.end(), 1.0, std::multiplies<float>()));
}

/**
 * @brief Calculates the product of all elements in a 2D vector.
 * This function takes a 2D vector of floats as input and returns the product of all its elements.
 * Internally, it first calculates the sum of each row using sumofrow and then calculates the product of these row sums.
 * @param a 2D vector of floats
 * @return the product of all elements in the 2D vector
 */
float product(std::vector<std::vector<float>> b) {
    float a = 1;
    for(int i = 0; i < b.size(); i++) {
        a *= std::accumulate(b[i].begin(), b[i].end(), 1.0, std::multiplies<float>());
    }
    return a;
}

/**
 * @brief Calculate the inner product of a vector of vector with itself 
 * and form a square matrix of its dot products.
 * @param a The input matrix
 * @return The product of the matrix with itself
 * @throws std::runtime_error if the input matrix is empty
 */
std::vector<std::vector<float>> iproduct(std::vector<std::vector<float>> a) {
    if(a.empty()) 
        throw std::runtime_error("embeddings must not be empty");

    std::vector<std::vector<float>> c(a.size(), std::vector<float>(a.size(), 0.0));
    
    for(size_t i = 0; i < a.size(); i++) {
        for(size_t j = 0; j < a.size(); j++) {
            c[i][j] = std::inner_product(a[i].begin(), a[i].end(), a[j].begin(), 0.0);
        }
    }
    
    return c;
}

/**
 * @brief Calculate the product of two vector of vectors and form a square 
 * matrix of dot products of each combination of two vectors.
 * @param a The first matrix
 * @param b The second matrix
 * @return The product of the two matrices
 * @throws std::runtime_error if the rows of the matrices are not of equal sizes
 */
std::vector<std::vector<float>> iproduct(std::vector<std::vector<float>> a, std::vector<std::vector<float>> b) {
    if(a.empty() || b.empty() || a[0].size() != b[0].size()) 
        throw std::runtime_error("Rows must be of equal sizes");

    std::vector<std::vector<float>> c(a.size(), std::vector<float>(b[0].size(), 0.0));
    
    for(size_t i = 0; i < a.size(); i++) {
        for(size_t j = 0; j < b.size(); j++) {
            c[i][j] = std::inner_product(a[i].begin(), a[i].end(), b[j].begin(), 0.0);
        }
    }
    
    return c;
}

/**
 * @brief Calculates the power of each element in a vector.
 * This function takes a vector of floats as an input and returns a new vector where each element is the 
 * result of raising the corresponding element of the input vector to the power of y.
 * @param x vector of floats
 * @param y power to raise the elements of x to
 * @return a new vector where each element is the result of raising the corresponding element of x to the power of y
 */
std::vector<float> power(std::vector<float> x, float y) {
    std::vector<float> result(x.size());
    std::transform(x.begin(), x.end(), result.begin(),
                   [y](float value) { return std::pow(value, y); });
    return result;
}

/**
 * @brief Calculates the power of each element in a 2D vector.
 * This function takes a 2D vector of floats as an input and returns a new 2D vector where each element is the 
 * result of raising the corresponding element of the input vector to the power of y.
 * @param x 2D vector of floats
 * @param y power to raise the elements of x to
 * @return a new 2D vector where each element is the result of raising the corresponding element of x to the power of y
 */
std::vector<std::vector<float>> power(std::vector<std::vector<float>> x, float y) {
    std::vector<std::vector<float>> result(x.size(), std::vector<float>(x[0].size()));
    std::transform(x.begin(), x.end(), result.begin(),
                   [y](const auto& row) { return power(row, y); });
    return result;
}

/**
 * @brief Sum of each row in a 2D vector. This function takes a 2D vector as an input and returns 
 * a 1D vector where each element is the sum of each row in the input vector.
 * @param a 2D vector
 * @return a vector of sums of each row
 */
std::vector<float> sumofrow(std::vector<std::vector<float>> a) {
    std::vector<float> b(a.size());
    // use std::transform to sum each row
    std::transform(a.begin(), a.end(), b.begin(),
                   // lambda to sum each row
                   [](const auto& row) { return std::accumulate(row.begin(), row.end(), 0.0f); });
    // return the result
    return b;
}

/**
 * @brief Sum of each column in a 2D vector
 * @param a 2D vector
 * @return a vector of sums of each column
 */
std::vector<float> sumofcol(std::vector<std::vector<float>> a) {
    // Create a vector to hold the sum of each column
    std::vector<float> b(a[0].size());
    // Iterate through each column
    for (int i = 0; i < a[0].size(); i++) {
        // Iterate through each row in the column
        for (int j = 0; j < a.size(); j++) {
            // Add the element in the column to the sum
            b[i] += a[j][i];
        }
    }
    // Return the vector of sums
    return b;
}

//----------------------//

/**
 * @brief Calculate the error between two vectors
 * This function takes two vectors as an input and returns a new vector
 * where each element is the difference between the corresponding elements
 * of the input vectors divided by 100.
 * @param x first vector
 * @param y second vector
 * @return a vector of errors
 */
std::vector<float> error(std::vector<float> x, std::vector<float> y) {
    if(x.size() != y.size()) {
        throw std::runtime_error("Vectors must be of the same size");
    }
    // Create a vector to hold the errors
    std::vector<float> b(x.size());
    // Use std::transform to calculate the error
    std::transform(x.begin(), x.end(), y.begin(), b.begin(), \
                // lambda to calculate the error
                [](const auto& i, const auto& j) { return (i - j); });
    // Return the vector of errors
    return b;
}

/**
 * @brief Calculate the mean error between two vectors.
 * @details This function computes the mean error by taking the difference between corresponding elements of two vectors
 * and averaging the result. It is commonly used to evaluate the performance of a neural network by comparing the 
 * predicted and actual output vectors.
 * @param[in] v1 The first vector (predicted values).
 * @param[in] v2 The second vector (actual values).
 * @return The mean error between the two vectors.
 */
float errorofv(std::vector<float> v1, std::vector<float> v2) {
    float error = 0.0;
    for(int i = 0; i < v1.size(); i++) {
        // Accumulate the error as the difference between corresponding elements
        error += v1[i] - v2[i];
    }
    // Return the mean error
    return error*100 / v1.size();
}

/**
 * @brief Calculate the Mean Squared Error (MSE) between two vectors. This function 
 * takes two vectors as input and returns the mean squared error between them.
 * The mean squared error is calculated as the average of the squared differences 
 * between the corresponding elements of the two vectors.
 * @param a The first vector.
 * @param b The second vector.
 * @return The mean squared error between the two vectors.
 */
float MSE(std::vector<float> a, std::vector<float> b) {
    if(a.size() != b.size())
        throw std::runtime_error("Same Size Vectors are ALLOWED only");
    float sum = 0.0;
    for(int i = 0; i < a.size(); i++) {
        // Calculate the squared difference between the elements of the two vectors
        sum += pow(a[i] - b[i], 2);
    }
    return sum / a.size();
}

/**
 * @brief Calculates the error between two vectors as a fraction of the first vector.
 * @param v1 The first vector: expected.
 * @param v2 The second vector: produced.
 * @return A vector containing the error between the two vectors as a fraction of the first vector.
 */
std::vector<float> percenterrorofvec(std::vector<float> v1, std::vector<float> v2) {
    if(v1.size() != v2.size()) {
        throw std::runtime_error("Vectors must be of the same size, enlarge or shorten any one.");
    }
    std::vector<float> error;
    for(int i = 0; i < v1.size(); i++) {
        // Calculate the error between the two vectors as a fraction of the first vector.
        error.push_back((v1[i] - v2[i])*100/v1[i]);
    }
    return error;
}

/**
 * @brief Gradient descent for a given output vector and expected value vector
 * @param y_true Output vector
 * @param y_pred Expected value vector
 * @param size Size of the vectors
 * @return The gradient descent output
 */
std::vector<float> gradient_descent(std::vector<float> y_true, std::vector<float> y_pred, float learning_rate) {
    if(y_true.size() != y_pred.size()) {
        throw std::runtime_error("Vectors must be of the same size");
    }
    std::vector<float> dw(y_true.size(), 0.0);
    for (size_t i = 0; i < y_true.size(); i++) {
        dw[i] = learning_rate * (y_true[i] - y_pred[i]);
    }
    return dw;
}

/**
 * @brief Calculate the error between two vectors and return the sum of the errors
 * This function takes two vectors as an input and returns the sum of the errors
 * between the two vectors. The error is calculated as the difference between
 * the corresponding elements of the input vectors divided by 100.
 * @param x first vector
 * @param y second vector
 * @return the sum of the errors
 */
float gradientdesc1(std::vector<float> x, std::vector<float> y) {
    if(x.size() != y.size()) {
        throw std::runtime_error("Vectors must be of the same size");
    }
    // Create a vector to hold the errors
    std::vector<float> b(x.size());
    // Use std::transform to calculate the error
    std::transform(x.begin(), x.end(), y.begin(), b.begin(), \
                // lambda to calculate the error
                [](const auto& i, const auto& j) { return (i - j)/100; });
    // Return the vector of errors
    return (std::accumulate(b.begin(), b.end(), 0.0)/b.size());
}

/**
 * @brief Create a matrix from two vectors
 * This function takes two vectors as input and returns a matrix
 * where each element is the product of the corresponding elements
 * of the input vectors.
 * @param x first vector
 * @param y second vector
 * @return a matrix where each element is the product of the corresponding elements of the input vectors
 */
std::vector<std::vector<float>> vxv2mat(std::vector<float> x, std::vector<float> y) {
    if(x.size() != y.size()) {
        throw std::runtime_error("Vectors must be of the same size");
    }
    // Create a matrix to hold the result
    std::vector<std::vector<float>> a(x.size(), std::vector<float>(y.size()));
    // Iterate through each element in the matrix and calculate the product
    for (int i = 0; i < x.size(); i++) {
        for (int j = 0; j < y.size(); j++) {
            // Calculate the product of the corresponding elements of the input vectors
            a[i][j] = x[i] * y[j];
        }
    }
    // Return the resulting matrix
    return a;
}

/**
 * @brief Calculate the cross product of two vectors and return the resulting vector.
 * This function takes two vectors as an input and returns a new vector where each
 * element is the product of the corresponding elements of the input vectors.
 * @param x first vector
 * @param y second vector
 * @return a new vector where each element is the product of the corresponding
 * elements of the input vectors
 */
std::vector<float> vxv2v(std::vector<float> x, std::vector<float> y) {
    if(x.size() != y.size()) {
        throw std::runtime_error("Vectors must be of the same size");
    }
    // Create a matrix to hold the result
    std::vector<std::vector<float>> a(vxv2mat(x, y));
    // Return the resulting matrix
    return sumofrow(a);
}

/**
 * @brief Calculate the inner/dot product of two vectors and return the resulting vector.
 * This function takes two vectors as an input and returns a new vector where each
 * element is the product of the corresponding elements of the input vectors.
 * @param x first vector
 * @param y second vector
 * @return a new vector where each element is the product of the corresponding
 * elements of the input vectors
 */
std::vector<float> vdotv2v(std::vector<float> x, std::vector<float> y) {
    if(x.size() != y.size()) {
        throw std::runtime_error("Vectors must be of the same size");
    }
    // Create a vector to hold the result
    std::vector<float> a(x.size());
    // Iterate through each element in the vector and calculate the product
    for (int i = 0; i < x.size(); i++) {
        // Calculate the product of the corresponding elements of the input vectors
        a[i] = x[i] * y[i];
    }
    // Return the resulting vector
    return a;
}

/**
 * @brief Calculate the dot product of two vectors and return the sum of the products.
 * This function takes two vectors as an input and returns the sum of the products
 * of the corresponding elements of the input vectors.
 * @param x first vector
 * @param y second vector
 * @return the sum of the products of the corresponding elements of the input vectors
 */
float vdotv2val(std::vector<float> x, std::vector<float> y) {
    if(x.size() != y.size()) {
        throw std::runtime_error("Vectors must be of the same size");
    }
    // Initialize the sum of products to 0
    float a = 0;
    for (int i = 0; i < x.size(); i++) {
        a += x[i] * y[i];
    }
    // Return the sum of the products
    return a;
}

/**
 * @brief Vector dot product of two vectors and a scalar value
 * @details This function takes two vectors and returns the dot product of the two vectors. The dot product is the sum of the
 * product of each element of the two vectors. The function is used to calculate the output of a layer in a neural network.
 * @param[in] v1 The first vector
 * @param[in] v2 The second vector
 * @return The dot product of the two vectors
 */
float vdotv2scal(std::vector<float> v1, std::vector<float> v2) {
    float sum = 0;
    for(int i = 0; i < v1.size(); i++) {
        sum += v1[i] * v2[i];
    }
    return sum;
}

/**
 * @brief Create a matrix from two vectors
 * This function takes two vectors as input and returns a matrix
 * where each element is the product of the corresponding elements
 * of the input vectors.
 * @param x first vector
 * @param y second vector
 * @return a matrix where each element is the product of the corresponding elements of the input vectors
 */
std::vector<std::vector<float>> vdotmat2mat(std::vector<float> x, std::vector<std::vector<float>> y) {
    if(x.size() != y[0].size()) {
        throw std::runtime_error("Vectors must be of the same size");
    }
    // Create a matrix to hold the result
    std::vector<std::vector<float>> a(x.size(), std::vector<float>(y[0].size()));
    // Iterate through each element in the matrix and calculate the product
    for (int i = 0; i < x.size(); i++) {
        // Calculate the product of the corresponding elements of the input vectors
        a[i] = vdotv2v(x, y[i]);
    }
    // Return the resulting matrix
    return a;
}

/**
 * @brief Calculate the product of a vector and a matrix. This function takes a vector 
 * and a matrix as input and returns a new vector where each element is the sum of the 
 * products of the corresponding elements of the input vector and matrix.
 * @param x input vector
 * @param y input matrix
 * @return a new vector where each element is the sum of the products of the corresponding
 * elements of the input vector and matrix
 */
std::vector<float> vxmat2vec(std::vector<float> x, std::vector<std::vector<float>> y) {
    if(x.size() != y[0].size()) {
        throw std::runtime_error("Vectors must be of the same size");
    }
    // Create a matrix to hold the result
    std::vector<float> a(x.size());
    // Iterate through each element in the matrix and calculate the product
    for (int i = 0; i < x.size(); i++) {
        for(int j = 0; j < y[i].size(); j++) {
            a[i] += x[j] * y[j][i];
        }
    }
    // Return the resulting matrix
    return a;
}

/**
 * @brief Computes the Kronecker product of two matrices.
 * This function takes two matrices as input and returns their Kronecker product.
 * The Kronecker product is a block matrix where each block is the product of an element
 * from the first matrix and the entire second matrix.
 * @param a First matrix
 * @param b Second matrix
 * @return The Kronecker product of the two matrices
 */
std::vector<std::vector<float>> kronecker(std::vector<std::vector<float>> a, std::vector<std::vector<float>> b) {
    // Initialize the resulting matrix with appropriate dimensions
    std::vector<std::vector<float>> c(a.size() * b.size(), std::vector<float>(a[0].size() * b[0].size()));
    // Iterate over each element in the first matrix
    for(int i = 0; i < a.size(); i++) {
        for(int k = 0; k < a[0].size(); k++) {
            // For each element in the first matrix, iterate over each element in the second matrix
            for(int j = 0; j < b.size(); j++) {
                for(int l = 0; l < b[0].size(); l++) {
                    // Compute the Kronecker product element
                    c[i * b.size() + j][k * b[0].size() + l] = a[i][k] * b[j][l];
                }
            }
        }
    }
    // Return the resulting Kronecker product matrix
    return c;
}

/**
 * @brief Computes the Kronecker product of a matrix and a vector.
 * This function takes a matrix and a vector as input and returns their Kronecker product.
 * The Kronecker product is a block matrix where each block is the product of an element
 * from the matrix and the vector.
 * @param a Matrix (2D vector)
 * @param b Vector (1D vector)
 * @return The Kronecker product of the matrix and the vector
 */
std::vector<std::vector<float>> kronecker(std::vector<std::vector<float>> a, std::vector<float> b) {
    // Initialize the resulting matrix with appropriate dimensions
    std::vector<std::vector<float>> c(a.size(), std::vector<float>(a[0].size() * b.size()));
    // Iterate over each row in the matrix
    for(int i = 0; i < a.size(); i++) {
        // Iterate over each element in the vector
        for(int j = 0; j < b.size(); j++) {
            // Iterate over each column in the matrix
            for(int k = 0; k < a[0].size(); k++) {
                // Compute the Kronecker product element
                c[i][k * b.size() + j] = a[i][k] * b[j];
            }
        }
    }
    // Return the resulting Kronecker product matrix
    return c;
}

/**
 * @brief Computes the Hadamard product of two matrices.
 * @details The Hadamard product, or element-wise multiplication, takes two matrices of the same dimensions 
 * and produces a matrix where each element is the product of the corresponding elements in the input matrices.
 * @param a First matrix (2D vector)
 * @param b Second matrix (2D vector)
 * @return The Hadamard product of the input matrices
 * @throws std::runtime_error if the dimensions of the matrices do not match
 */
std::vector<std::vector<float>> hadamard(std::vector<std::vector<float>> a, std::vector<std::vector<float>> b) {
    // Check if the dimensions of the input matrices match
    if (a.size() != b.size() || a[0].size() != b[0].size()) {
        throw std::runtime_error("Matrices must be of the same dimensions");
    }
    // Create a matrix to store the result with the same dimensions as the input matrices
    std::vector<std::vector<float>> c(a.size(), std::vector<float>(a[0].size()));
    // Iterate through each element of the matrices
    for(int i = 0; i < a.size(); i++) {
        for(int j = 0; j < a[0].size(); j++) {
            // Compute the element-wise product
            c[i][j] = a[i][j] * b[i][j];
        }
    }
    // Return the resulting Hadamard product matrix
    return c;
}
