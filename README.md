
# gLLM - AI/ML Library for creating LLMs

![alt text](gLLMicon.svg)

## INTRO
- Library for SLM, LLM and LCM
  - SLM: small language model (**<2B** Parameters)
  - LLM: large language model (**3B-450B** Parameters)
  - LCM: large concept model (**>40B** Parameters)
- **VERSION**: 0.0.0.1
- **LICENSE**: MIT
- **PROGRAMMING LANGUAGES**: C, C++, OpenCL, CUDA
  - *C VERSION*: 17
  - *C++ VERSION*: 20
  - *OpenCL VERSION*: 300
  - *CUDA*: 12.6
- **PROJECT BUILD SYSTEM**: CMake
- **Model Architecture**: Shady Attention Mechanism

### Main Functionalities
- Data preprocessing and tokenization
- Neural network training and evaluation
- Model creation and operations
- GPU-accelerated computation (via OpenCL for AMD and CUDA for NVidia)
- Grammer and Context Based Training

### Build System
- Uses CMake for cross-platform build configuration
- Supports debug and release builds
- Outputs to bin directory with system-specific naming

## Project Structure
- **maths**: Mathematical Library for AI/ML
- **neural**: Neural Network Library
- **model**: Model Library
- **script**: CLI script

### src/maths
- *basic*: Basic Mathematical Functions
- *mat*: Mathematical Functions for Matrix Operations
- *stats*: Mathematical Functions for Statistics
- *maths.hpp*: Main Header

### src/neural
- *mlp*: Multi-Layer Perceptron
- *attention*: Attention Mechanism
- *block*: Attention Block
- *transformer*: Transformer Structure
- *neural.hpp*: Main Header

### src/model
- *model.hpp*: Main Header

### src/script
- *script.hpp*: Main Header

### bin
- Output directory for compiled binaries: .lib and .dll

## MODEL STRUCTURE
 
**MODEL METADATA**:
```
{
  "name": "user-defined-name",
  "version": "user-defined-version",
  "license": "user-defined-license",
  "description": "user-defined-description",
  "author": "user-defined-author",
  "architecture": "Dual Pseudo-Attention Mechanism",
  "data": {
    "data_type": "can be any of following:
        int2, int4, int8, int16, int32, int64,
        float16, float32, float64, double
  }
}
```
**MODEL SERIALISATION**:
```

```

### NOTE: for making an application with this library

```
# add executable
add_executable(project_name "main.cpp" 
                "gLLM.h"
                "gLLM/src/model/include/maths.hpp"
                "gLLM/src/neural/include/neural.hpp"
                "gLLM/src/model/include/model.hpp"
                "gLLM/src/script/include/script.hpp"
)

# bind private libraries
target_include_directories(gLLM
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/gLLM//src/maths/include
    ${CMAKE_CURRENT_SOURCE_DIR}/gLLM//src/neural/include
    ${CMAKE_CURRENT_SOURCE_DIR}/gLLM//src/model/include
    ${CMAKE_CURRENT_SOURCE_DIR}/gLLM//src/script/include
)

# link 3rd party libraries
target_link_libraries(gLLM
  PRIVATE
    ${OpenCL_LIBRARIES}
    ${CUDAToolkit LIBS}
  maths
  model
  neural
  script
)
```
