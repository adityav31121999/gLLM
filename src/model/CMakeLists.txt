
# src/model/CMakeLists.txt
cmake_minimum_required(VERSION 3.25.0 FATAL_ERROR)
project(model C CXX)

add_library(model STATIC
    src/model.cpp
    src/create.cpp
    src/fetch.cpp
    src/set.cpp
    src/csvsup.cpp
    src/token.cpp
    src/ser.cpp

    src/training/train.cpp
    src/training/test.cpp
    src/training/validate.cpp

    src/inference/chat.cpp
    src/inference/run.cpp
)

target_link_libraries(model
    PUBLIC # Important: Make OpenCL and CUDA linking public
        ${OpenCL_LIBRARIES}
        ${CUDAToolkit_LIBRARIES}
    memorymap
    maths
    neural
)
