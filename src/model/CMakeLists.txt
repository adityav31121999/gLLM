
# src/maths/CMakeLists.txt
cmake_minimum_required(VERSION 3.30.0 FATAL_ERROR)
project(model C CXX)

add_library(model STATIC
    src/model.cpp
    src/create.cpp
    src/load.cpp
    src/run.cpp
    src/save.cpp
)

target_link_libraries(model
    maths
    neural
)
