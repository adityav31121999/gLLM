
# src/model/CMakeLists.txt
cmake_minimum_required(VERSION 3.30.0 FATAL_ERROR)
project(model C CXX)

include_directories(include)

add_library(model STATIC
    # tokenization and corpus handling
    src/tokens/tokeniser.cpp
    src/tokens/clHost.cpp
    src/tokens/embedding.cpp
    src/tokens/split.cpp
    src/tokens/corpus.cpp
    src/tokens/token.cu
    src/tokens/token.cl
    src/tokens/group.cpp
    src/tokens/pairstats.cpp
    src/tokens/merge.cpp
    src/tokens/readFiles.cpp
    src/tokens/utility.cpp
    src/tokens/values.cpp

    # model core functionality
    src/model.cpp
    src/create.cpp
    src/fetch.cpp
    src/set.cpp
    src/csvsup.cpp
    src/ser.cpp
    src/token.cpp

    # training, testing, inference
    src/training/trainpr.cpp
    src/training/trainsent.cpp
    src/training/trainbatch.cpp
    src/testing/testpr.cpp
    src/testing/testsent.cpp
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
