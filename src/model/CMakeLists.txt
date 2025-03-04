
# src/model/CMakeLists.txt
cmake_minimum_required(VERSION 3.30.0 FATAL_ERROR)
project(model C CXX)

add_library(model STATIC
    src/model.cpp
    src/create.cpp
    src/run.cpp
    src/save.cpp
    src/serialise.cpp
    src/deserialise.cpp

    src/train/train.cpp
    src/train/fine.cpp

    src/use/load.cpp
    src/use/act.cpp
)

target_link_libraries(model
    PUBLIC # Important: Make OpenCL and CUDA linking public
        ${OpenCL_LIBRARIES}
        ${CUDAToolkit_LIBRARIES}
    maths
    neural
)
