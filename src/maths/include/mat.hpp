
// matrix.hpp : header source of matrix library
#ifndef MAT_HPP
#define MAT_HPP 1

#include <vector>

/**
 * @brief CLASS: Matrix class
 * @param r number of rows
 * @param c number of columns
 * @param a matrix / 2d vector for coefficients of linear equations
 */
class mat {
public:
    // for other classes to access this class
    int row;
    int col;
    std::vector<std::vector<double>> a;     // coefficient matrix

    // default constructor
    mat():row(0), col(0) {a = {{0}};}

    /**
     * @brief Constructor for matrix of size x*y
     * @param x number of rows
     * @param y number of columns
     */
    mat(int x, int y) {
        row = x;
        col = y;
        a.resize(x, std::vector<double>(y, 0.0));
    }

    /**
     * @brief Constructor for square matrix of size x*x
     * @param x number of rows and columns
     */
    mat(int x) {
        row = col = x;
        a.resize(x, std::vector<double>(x, 0.0));
    }

    /**
     * @brief Constructor for matrix from a 2D vector
     * @param b 2D vector of doubles representing the matrix
     */
    mat(std::vector<std::vector<double>> b) {
        // Set the number of rows and columns from the input 2D vector
        row = b.size();
        col = b[0].size();
        // Copy the input matrix into the matrix member variable
        a = b;
    }

    /**
     * @brief Copy constructor for matrix
     * @param b matrix to be copied from
     */
    mat(mat &b) {
        row = b.row;       // copy number of rows
        col = b.col;       // copy number of columns
        a = b.a;       // copy coefficient matrix
    }

    
    /**
     * @brief Move constructor for matrix
     * @param b matrix to be moved from
     * @note This constructor is marked as noexcept as it is guaranteed to not throw any exceptions
     */
    mat(mat&& b) noexcept {
        // Move the number of rows and columns from the input matrix
        row = std::exchange(b.row, 0);
        col = std::exchange(b.col, 0);
        // Move the coefficient matrix from the input matrix
        a = std::move(b.a);
    }
    
    /**
     * @brief Copy constructor for matrix.
     * @param b The matrix to be copied from.
     * This constructor creates a deep copy of the input matrix.
     * It copies the number of rows, columns and the coefficient matrix from the input matrix.
     */
    mat(const mat &b) {
        row = b.row;       // Copy number of rows
        col = b.col;       // Copy number of columns
        a = b.a;           // Copy coefficient matrix
    }

    int getrow() { return row; };       // get row
    int getcol() { return col; };       // get column
    std::vector<std::vector<double>> geta() { return a; };       // get matrix

    double& operator()(int i, int j);   // operator overload for accessing element
    mat operator=(mat);                 // assignement operator overload
    mat operator+(mat);                 // addition operator overload
    mat operator-(mat);                 // subtraction operator overload
    mat operator=(std::vector<std::vector<double>>);        // assignment operator overload
    mat operator+(std::vector<std::vector<double>>);        // Addition operator overload
    mat operator-(std::vector<std::vector<double>>);        // subtraction operator overload
    mat operator*(double);              // multiplication operator overload for value
    mat operator*(mat);                 // multiplication operator overload for matrix (fmmlt)
    mat operator/(double);              // division operator overload for value
    mat operator/(mat);                 // division operator overload for value
    mat operator+=(mat);                // addition operator overload
    mat operator-=(mat);                // subtraction operator overload
    mat operator+=(std::vector<std::vector<double>>);       // Addition operator overload
    mat operator-=(std::vector<std::vector<double>>);       // subtraction operator overload
    mat operator*=(double);             // multiplication operator overload for value
    mat operator*=(mat);                // multiplication operator overload for matrix (fmmlt)
    mat operator/=(double);             // division operator overload for value
    mat operator/=(mat);                // division operator overload for value
    mat imat(int);                      // identity matrix
    mat inva();                         // additive inverse of matrix
    mat inverse();                      // inverse of matrix using adjoint of matrix
    mat adjoint();                      // adjoint of matrix
    mat gaussjordan();                  // inverse of matrix using gauss jordan elimination method
    mat transpose();                    // new matrix as transpose of matrix
    mat cofac();                        // cofactor of matrix
    mat Random(int, int);        // initialise values of matrices
    mat resize(int row, int col);       // resize the matrix by row and col

    double det2();                      // determinant of 2x2 matrix
    double det3();                      // determinant of 3x3 matrix
    double det4();                      // determinant of 4x4 matrix
    double detn();                      // determinant of nxn matrix
    double det();                       // determinant of square matrix
    double trace();                     // trace of square matrix

    void trnsps();                      // transpose the current matrix
    bool ifsquare();                    // check if matrix is square
    bool ifrectangular();               // check if matrix is rectangular
    bool ifsymmetric();                 // check if matrix is symmetric
    bool ifidentity();                  // check if matrix is identity
    bool ifdiagonal();                  // check if matrix is diagonal
    bool ifupper();                     // check if matrix is upper triangular
    bool iflower();                     // check if matrix is lower triangular
    bool ifskew();                      // check if matrix is skew-symmetric

    ~mat() {};                          // destructor for matrix class
};

mat rowechelon(mat a);
mat rowechelon(std::vector<std::vector<double>> a);
mat submat(mat, unsigned int, unsigned int);
mat submat(std::vector<std::vector<double>>, unsigned int, unsigned int);
mat minor(mat a);
mat minor(std::vector<std::vector<double>>);

double *householder(double*, int, int);
double *householderTransform(double*, int, int);

std::pair<mat*, mat*> makeOrthogonalMatrix(mat*);
mat jacobian(mat *a);
double jacobianval(mat *a);


#endif
