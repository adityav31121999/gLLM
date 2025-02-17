
# gLLM - AI/ML Library for creating LLMs

![alt text](gLLMicon.svg)

## INTRO
- Library for mLM, SLM, LLM and LCM
  - SLM: small language model (**<2B** Parameters)
  - LLM: large language model (**3B-450B** Parameters)
  - LCM: large concept model (**>40B** Parameters)
- **VERSION**: 0.0.0.1
- **LICENSE**: MIT
- **PROGRAMMING LANGUAGES**: C, C++, OpenCL
  - *C VERSION*: C17
  - *C++ VERSION*: C++20
  - *OpenCL VERSION*: OpenCL 3
- **PROJECT BUILD SYSTEM**: CMake
- **Model Architecture**: Dual Pseudo-Attention Mechanism

### Main Functionalities
- Data preprocessing and tokenization
- Neural network training and evaluation
- Model creation and operations
- GPU-accelerated computation (via OpenCL)
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
- *poly*: Mathematical Functions for Polynomials
- *stats*: Mathematical Functions for Statistics
- *maths.hpp*: Main Header

### src/neural
- *grammer*: Grammer Library
- *attention*: Attention Mechanism
- *mlp*: Multi-Layer Perceptron
- *snn*: String Neural Network
- *polynn*: Polynomial Neural Network
- *transformer*: Transformer
- *neural.hpp*: Main Header

### src/model
- *model.hpp*: Main Header

### src/script
- *script.hpp*: Main Header

### bin
- Output directory for compiled binaries: .lib and .dll

## MODEL STRUCTURE

**Model Name**:
- If possible for everyone's usefullness and easy understanding, use this syntax for naming of model:
  - **Use Cases**: Use cases as per T, C, G, L, P
    - *T*: translation
    - *C*: conversation
    - *G*: generative and reasoning
    - *L*: natural languages
    - *P*: programming languages
  - **Model Name**: user-defined-name
  - **Model Version**: user-defined-version
  - **Extension**:
    - **.gllm**: default
    - **.xyz**: user-defined extension (for all types of models) 
  - **Name**: 
    - ex: name_CP3_V.clm:
      - C for conversation, P for programmig language, 3 for number of programming languages with V as version and extension .clm
    - ex: name_TCGL10P4_2.0.3.2.rizz:
      - C for conversation, T for translation, G for generative and reasoning, L for natural languages, 10 for natural languages, P for programming languages, 2.0.3.2 version and extension .rizz
 
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
        float32, float64, double,
        complexint2, complexint4, complexint8, complexint16, complexint32, complexint64,
        complexfloat32, complexfloat64, complexdouble",
  }
}
```

### Important for making an application with this library

```
# add executable
add_executable(gLLM "main.cpp" 
                "gLLM.h"
                "src/model/include/maths.hpp"
                "src/neural/include/neural.hpp"
                "src/model/include/model.hpp"
                "src/script/include/script.hpp"
)

# bind private libraries
target_include_directories(gLLM
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/maths/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/neural/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/model/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/script/include
)

# link 3rd party libraries
target_link_libraries(gLLM
PRIVATE
	${OpenCL_LIBRARIES}
    maths
    model
    neural
    script
)
```
