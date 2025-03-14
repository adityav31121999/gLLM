
# src/model/CMakeLists.txt
cmake_minimum_required(VERSION 3.30.0 FATAL_ERROR)
project(script C CXX)

include_directories(include)

add_library(script STATIC
    src/model.cpp
    src/use.cpp
    src/chat.cpp
    src/train.cpp
)

target_link_libraries(script
    PUBLIC # Important: Make OpenCL and CUDA linking public
        ${OpenCL_LIBRARIES}
        ${CUDAToolkit_LIBRARIES}
    maths
    neural
    model
)
