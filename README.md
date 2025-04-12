
# gLLM - AI/ML Library for creating LLMs

![alt text](gLLMicon.svg)

## INTRO
- Library for LLMs
- **VERSION**: 0.0.0.1
- **LICENSE**: MIT License
- **PROGRAMMING LANGUAGES**: C, C++, OpenCL, CUDA
  - *C VERSION*: 17
  - *C++ VERSION*: 20
  - *OpenCL VERSION*: 300
  - *CUDA*: 12.6
- **PROJECT BUILD SYSTEM**: CMake
- **Model Architecture**: Shady Attention Mechanism (DENSE)

## Project Structure
- **maths**: Mathematical Library for AI/ML
- **neural**: Neural Network Library
- **model**: Model Library

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

### bin
- Output directory for compiled binaries: .lib and .dll

## Mechanism
- This mechanism is a modification to Attention Mechanism defined in paper "Attention Is All You Need" by VASWANI et. al. (2017).
- Also, I would direct all the readers to 3BLUE1BROWN YouTube channel where the Deep Learning Playlist is created and the inspiration has been drawn from there.

### Main IDEA:
- The main idea is to break long context into various small equal parts (Context Window) and introduce two new matrices for Horizontal and Vertical retention in place of Value matrix, having horizontal retention vector for token prediction and vertical retention vectors for context retention in next block.
- These two new matrices are taken from value matrix as V = Up_projection x Down_projection (refer to playlist) and this V is replaced by MH and MV with two MLPs for forward propagation in horizontal direction and forward propagation in vertical direction.
- The new mechanism has structure three main components as head, block and transformer.
- Head is the primary attention mechanism. Block is the 2d vector of Heads with each row being termed as partial attention and each column as parallel. Transformer is the vector of such blocks. So, the structure is 3d vector of heads, with head being a primary unit where majority of process takes place.
- Head is referred as incomplete attention, rows of head as partial attention and Block as complete attention due to the nature of final prediction being obtained by summing up the partial attentions.
- Transformer is referred as FULL context since it comprises of all blocks with equal context window.

### Components
**HEAD**:
- MQ, MK: Query and Key matrices
- MV, MH: Vertical and Horizontal retention matrices
- hor, ver: MLPs for horizontal and vertical propagation
- qkCache, qvCache, khCache: Caches for model operation
- K, Q: vectors of Key and Query vectors
- KdotQ: grid for dot product of Key and Query
- EH: horizontal retention vector for token prediction
- EV: vector of vertical retention vectors for context retention
- dh: delta for EH
- dv: delta for EV[i] where i is the current iteration
- isSelf: boolean to check attention type (1 for self, 0 for cross)
- inTraining: boolean to check training (1) or use (0)
- tokenCount: current tokenCount in the context window of head

**BLOCK**:
- x, y: number of rows and columns
- error: error for block
- isSelfattention: boolean to check attention type (1 for self, 0 for cross)
- inTraining: boolean to check training (1) or use (0)
- str: string to hold the token to compare with TERMINATE ('@#0')
- EH: combined or concatanated value of partial attention for token prediction
- EV: collection of all Head EVs
- probability: probability vector for token prediction
- b: 2d vector of Heads (attention class) of dimension 'x' rows (Partial Attentions) and 'y' columns (Parallels)

**TRANSFORMER**:
- m: number of blocks
- x: number of rows
- y: number of columns
- n: context window for each head
- d: token embedding dimension
- h: rows of MQ, MK and columns of MV, MH
- l: hidden weight layers of mlp
- epochs: allowable iteration for training of block and mlp
- totalParams: total parameters of transformer
- learning: rate of learning for training of mlp and transformer
- error: error for transformer
- isSelf: boolean to check attention type (1 for self, 0 for cross)
- inTraining: boolean to check training (1) or use (0)
- isTerminate: 1 for terminate ('@#0') and 0 for continuation
- blockCount: number of block which is in use
- epochCount: epoch counter for training
- promptCount: tokens in user prompt
- currentTokenCount: number of tokens generated or in use in full context

### Training

### OPERATION

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
- Model have specifically named files for Matrices, MLPs and caches and store the values represented by their name in binary format
- These values are stored in .bin files
- These files are:
  - Matrices: MQ.bin, MK.bin, MV.bin,  MH.bin (For Training only)
  - MLPs: hor.bin, ver.bin (For Training and Use)
  - Caches: QK.bin, QV.bin, KH.bin (For Use only)
- To access them head offset and block offset must be known
- This table gives the total values, dimension, single offset and block offset of each file
------------------------------------------------------------------------------------------------------
| NAME |  DIM1  |  DIM2  |  DIM3  | QUANTITY |  TOTAL PARAMETERS  |  SINGLE OFFSET  |  BLOCK OFFSET  |
|------|--------|--------|--------|----------|--------------------|-----------------|----------------|
|  MQ  |   h    |    d   |   1    |x * y * m |      h.d.x.y.m     |       h*d       |    h.d.x.y     |
|  MK  |   h    |    d   |   1    |x * y * m |      h.d.x.y.m     |       h*d       |    h.d.x.y     |
|  MV  |   d    |    h   |   1    |x * y * m |      d.h.x.y.m     |       d*h       |    d.h.x.y     |
|  MH  |   d    |    h   |   1    |x * y * m |      d.h.x.y.m     |       d*h       |    d.h.x.y     |
|  hor |   d    |    d   |   l    |x * y * m |     d.d.l.x.y.m    |     d * l * d   |   d.d.l.x.y    |
|  ver |   d    |    d   |   l    |x * y * m |     d.d.l.x.y.m    |     d * l * d   |   d.d.l.x.y    |
|  QK  |   d    |    d   |   1    |x * y * m |      d.d.x.y.m     |       d*d       |    d.d.x.y     |
|  QV  |   d    |    d   |   1    |x * y * m |      d.d.x.y.m     |       d*d       |    d.d.x.y     |
|  KH  |   d    |    d   |   1    |x * y * m |      d.d.x.y.m     |       d*d       |    d.d.x.y     |
------------------------------------------------------------------------------------------------------
- Here single offset refers to total number of values in single object i.e., Matrix, MLP or cache
- Block Offset refers to total number of values of specific object in the single block i.e., number of object (matrix or mlp or cache) * single offset = x * y * single offset
- Following is the serialisation of MQ.bin file as example:
  - Q[i][j][k] represent MQ of kth head of jth row of ith block
```
.bin File:
M[1][1][1] = --------------------------------------------------------------
M[1][1][2] = --------------------------------------------------------------
M[1][1][3] = --------------------------------------------------------------
    |               |               |               |               |
M[1][x][y] = --------------------------------------------------------------
M[2][1][1] = --------------------------------------------------------------
M[2][1][2] = --------------------------------------------------------------
M[2][1][1] = --------------------------------------------------------------
    |               |               |               |               |
M[m][x][y] = --------------------------------------------------------------
```

### NOTE: for making an application with this library
While making application based on this mechanism and project, just use main CMakeLists.txt and update it with following code and add a folder to create the application with a main.cpp file.

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

## IMPORTANT NOTE:

- I would like to give huge credits to AI models and AgenticIDEs that I have used to build cuda and opencl operations.
- GROK: For Backpropagation of blocks (A big problem was that how should I reflect the change from error to mlp to the matrices like MQ, MK, MV and MH from block to block and most of the time without affecting the horizontal operations)
- FOR CUDA AND OPENCL:
  - GEMINI and GEMINI code assist
  - CLAUDE SONNET in TRAE, WINDSURF and CURSOR (not much due to paywall)
  - DEEPSEEK in TRAE and WINDSURF
  - MISTRAL
  - ChatGPT
  - COPILOT
