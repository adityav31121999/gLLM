
# src/maths/CMakeLists.txt
cmake_minimum_required(VERSION 3.30.0 FATAL_ERROR)
project(maths C CXX)

include_directories(include)

add_library(maths STATIC
    # basic files
    basic/functions1.cpp
    basic/functions2.cpp
    basic/vect.cpp
    basic/weights.cpp
    basic/activations.cpp

    # mat files
    mat/det.cpp
    mat/eigenval.cpp
    mat/inverse.cpp
    mat/jacobian.cpp
    mat/operators.cpp
    mat/prop.cpp
    mat/qrdecom.cpp
    mat/svd.cpp
    mat/submat.cpp

    # poly files
    poly/poly.cpp
    poly/operator.cpp
    poly/recurrence.cpp

    # stats files
    stats/basic.cpp
)
