
# gLLM - AI/ML Library for creating LLMs

![alt text](gLLMicon.svg)

- This idea is due to the need to train a AI Model on gaming gpu without causing heavy burden on VRAM and GPU compute capability.

## INTRO
- Library for LLMs
- **VERSION**: 0.1.1.1 (Memory mapped matrices)
- **LICENSE**: MIT License
- **PROGRAMMING LANGUAGES**: C, C++, OpenCL, CUDA
  - *C VERSION*: 17
  - *C++ VERSION*: 20
  - *OpenCL VERSION*: 300
  - *CUDA*: 12.6
- **PROJECT BUILD SYSTEM**: CMake
- **Model Architecture**:
  - Attention Mechanism: Retention Mechanism
  - Transformer Architecture: Divided context
  - Neural Connections: Dense

## Project Structure
- **memorymap**: Memory Mapping for large objects like mat and mlp
- **maths**: Mathematical Library for LLM
- **neural**: Neural Network Library for LLM
- **model**: Model Library for LLM

### src/memorymap
- *memory_map.h*: Provides C functions for memory-mapping files, enabling efficient file I/O by mapping file contents directly into the process's address space. Includes functions to open, create/resize, and close mapped files.

### src/maths
- *basic.hpp*: Basic Mathematical Functions
- *mat.hpp*: Mathematical Functions for Matrix Operations
- *stats.hpp*: Mathematical Functions for Statistics
- *maths.hpp*: Main Header

### src/neural
- *mlp.hpp*: Multi-Layer Perceptron
- *attention.hpp*: Attention Mechanism
- *block.hpp*: Attention Block
- *transformer.hpp*: Transformer Structure
- *neural.hpp*: Main Header

### src/model
- *model.hpp*: Main Header for model class
- *tokenise.hpp*: Header for tokenisation process

### bin
- Output directory for compiled binaries: .lib (static) and .dll (dynamic)

## Mechanism
- This mechanism is a modification to Attention Mechanism defined in paper "Attention Is All You Need" by VASWANI et. al. (2017).
- Also, I would direct all the readers to 3BLUE1BROWN YouTube channel where the Deep Learning Playlist is available and the main inspiration about the idea came from there.
- This modification provides retention of context via horizontal flow using EH and vertical flow using EV.
- The losses are decreased through multiple iterations of EV for maximum retention. While EH keeps updating Head by Head, EV is processed and updated at every head and all its rows get ReLUed MLP output.

### Main IDEA:
- The main idea is to break long context into various small equal parts (Context Window) and introduce two new matrices for Horizontal and Vertical retention in place of Value matrix, having horizontal retention vector for token prediction and vertical retention vectors for context retention in next block.
- These two new matrices are taken from value matrix as V = Up_projection x Down_projection (refer to playlist) and this V is replaced by MH and MV with two MLPs for forward propagation in horizontal direction and forward propagation in vertical direction.
- The new mechanism has structure three main components as head, block and transformer.
- Head is the primary attention mechanism. Block is the 2d vector of Heads with each row being termed as partial attention and each column as parallel. Transformer is the vector of such blocks. So, the structure is 3d vector of heads, with head being a primary unit where majority of process takes place.
- Head is referred as incomplete attention, rows of head as partial attention and Block as complete attention due to the nature of final prediction being obtained by summing up the partial attentions.
- Transformer is referred as FULL context since it comprises of all blocks with equal context window.

### Components
**MLP**:
- Multilayer Perceptron defined in mlp.hpp
- `status`: Boolean, 1 if completely trained, 0 otherwise.
- `num_layers`: Total number of layers (input, hidden, output).
- `layer_sizes`: Vector storing the number of neurons in each layer.
- `epochs`: Number of training epochs.
- `learning_rate`: Learning rate for training.
- `input`, `output`, `expected`: Vectors for input data, model output, and expected output.
- `weights`: Vector of `mat` objects, representing weight matrices between layers (memory-mapped).
- `hlayers`: Vector of vectors for hidden layer outputs (typically RAM-based).
- `activations`: Vector of vectors for activations of each layer.
- `gweights`: Vector of `mat` objects for gradient matrices corresponding to weights (memory-mapped).
- `params`: Total number of parameters in the MLP.
- `initializeWeightsFromSharedMap()`: Method to initialize weights from a shared memory-mapped region, allowing MLPs to use segments of a larger pre-allocated file.

**HEAD**:
- Attention class defined in attention.hpp
- `isSelfAttention`: Boolean, 1 for self-attention, 0 for cross-attention.
- `inTraining`: Boolean, 1 during training, 0 during inference.
- `tokenCount`: Integer, current number of tokens processed by this head within its context window.
- `MQ`, `MK`: `mat` objects, projection matrices. As per `modelDataInfo` and common interpretation for `K_vec = Token_vec * MK_matrix`, these are `EMBEDDING x MATHEIGHTS` (i.e., `d x h`). Stored as `MATHEIGHTS x EMBEDDING` if `modelDataInfo` dictates storage format.
- `MV`, `MH`: `mat` objects, projection matrices, typically `EMBEDDING x MATHEIGHTS` (i.e., `d x h`). Stored as `EMBEDDING x MATHEIGHTS`.
- `ver`, `hor`: `mlp` objects for vertical and horizontal propagation paths. MLPs typically operate on `EMBEDDING` dimension vectors.
- `qkCache`, `qvCache`, `khCache`: `mat` objects storing pre-computed matrix products for inference. Based on `attention.hpp` comments and typical usage, these are `EMBEDDING x EMBEDDING` (i.e., `d x d`).
- `K`, `Q`: `mat` objects storing Key and Query vectors derived from input tokens (e.g., `CONTEXT_WIN x MATHEIGHTS`). These are memory-mapped.
- `KdotQ`: `mat` object storing the attention scores (dot products of K and Q vectors), typically `CONTEXT_WIN x CONTEXT_WIN`. This is memory-mapped.
- `EH`: `std::vector<float>` (e.g., size `EMBEDDING`), the horizontal retention vector.
- `EV`: `mat` object (e.g., `CONTEXT_WIN x EMBEDDING`), the vertical retention matrix.
- `dh`, `dv`: `std::vector<float>` (e.g., size `EMBEDDING`), deltas for backpropagation.
- `learning_rate`: Learning rate specific to the attention mechanism.
- `params`: Total number of parameters within this attention head.

**BLOCK**:
- Attention block defined in block.hpp
- Referred to as LOCAL CONTEXT.
- `x`, `y`: Integers representing the dimensions of the attention head grid (x: partial attention layers, y: parallels/heads per layer).
- `tokenCount`: Number of tokens currently processed within this block's context window (e.g., `CONTEXT_WIN`).
- `error`: Floating point value for block error.
- `isSelfAttention`: Boolean, for all heads within the block.
- `inTraining`: Boolean, mode for the block.
- `EV`: `std::vector<std::vector<std::vector<std::vector<float>>>>`, manages vertical retention data from heads.
- `tokForBlock`: `mat` object (e.g., `CONTEXT_WIN x EMBEDDING`), stores token embeddings for this block.
- `b`: `std::vector<std::vector<attention>>`, a 2D vector of `attention` head objects.
- `blockFilePath`, `blockOffset`: String and long int for data persistence.
- `params`: Total number of parameters within this block.

**TRANSFORMER**:
- Transformer defined in transformer.hpp
- Referred to as FULL CONTEXT.
- `isSelf`: Boolean, global attention type.
- `inTraining`: Boolean, global mode.
- `m`: Integer, number of `block` objects.
- `x`: Integer, partial attention layers per `block`.
- `y`: Integer, parallel attention heads per partial attention layer.
- `n`: Integer, context window size per head (e.g., `CONTEXT_WIN`).
- `d`: Integer, token embedding dimension (e.g., `EMBEDDING`).
- `h`: Integer, dimension for projection matrices (e.g., `MATHEIGHTS`).
- `l`: Integer, number of layers in MLPs (e.g., `LAYERS_MLP`).
- `epochs`: Default training epochs.
- `learning`: Default learning rate.
- `blockCount`: Tracks active block.
- `promptCount`: Number of prompt tokens.
- `currentTokenCount`: Total tokens in transformer's context.
- `error`: Aggregated transformer error.
- `vocabsize`: Vocabulary size.
- `isTerminate`: Termination flag.
- `t`: `std::vector<block>`, sequence of blocks.
- `tokens`: `std::vector<std::string>`, vocabulary.
- `embeddings`: `mat` (`vocabsize x d`), memory-mapped token embeddings.
- `tokenEmbed`: `mat` (`currentTokenCount x d`), memory-mapped current sequence embeddings.
- `EVuse`: Stores vertical retention vectors from the last block for multi-turn context.
- `tokForBlock`: `mat` (`n x d`), memory-mapped tokens for the current block during inference.
- `params`: Long long int, total parameters.

## MODEL STRUCTURE

**MODEL**:
- `m, x, y, n, d, l`: Integers defining transformer architecture (blocks, attention layout, context window, embedding dim, MLP layers).
- `matheight`: Corresponds to `h` (MATHEIGHTS) in transformer/attention.
- `learning`: Default learning rate.
- `isSelf`: Boolean, default attention type.
- `toTrain`: Boolean, indicates if model is for training or inference.
- `T`: The main `transformer` object.
- `info`: `modelDataInfo` struct holding metadata.
- `metadata`, `chat`: `FILE*` pointers for model metadata and chat log.
- `baseDir`: String, base directory for model files.
- `currentChatLogPath`: Path to current chat log.
- `userPrompt`, `tinput`, `expected`, `toutput`, `token`: `std::vector<std::string>` for tokenized inputs/outputs.
- `matOffset`, `mlpOffset`, `cacheOffset`, `attentionOffset`, `blockOffset`: Offsets for organizing data within a single large model binary file (primarily for training with memory-mapped components).
- `totalParams`: Total parameters of the model.
- `vocabsize`: Vocabulary size.
- `calculateAndSetLayout()`: Method to determine memory layout for components.
- Manages model lifecycle, training, inference, and serialization.
- Tokeniser: BPE-based tokenisation

**MODEL METADATA**:
- Stored in a `modelDataInfo` struct, typically at the beginning of the main model binary file or a separate metadata file.
- Contains: `modelName`, `version`, `author`, `date`, `attentionMech`, `modelArch`, `license`.
- Dimensions: `d` (embedding), `vocab` (vocabulary size).
- Matrix dimensions: `qkrow`, `qkcol` (for MQ/MK, e.g., `MATHEIGHTS`, `EMBEDDING`).
- Matrix dimensions: `vhrow`, `vhcol` (for MV/MH, e.g., `EMBEDDING`, `MATHEIGHTS`).
- Transformer structure: `m, x, y, n, h, l` (blocks, attention layout, context window, `MATHEIGHTS`, MLP layers).
- Other info: `totalParams`, `totalContext` (`m * n`), `tokens` (dataset size), `learning` rate, `attentionType`.

**MODEL SERIALISATION**:
- Model have specifically named files for Matrices, MLPs and caches and store the values represented by their name in binary format
- These values are stored in .bin files
- These files are:
  - Matrices: MQ.bin, MK.bin, MV.bin,  MH.bin (For Training only)
  - MLPs: hor.bin, ver.bin (For Training and Use)
  - Caches: QK.bin, QV.bin, KH.bin (For Use only)
- To access them head offset and block offset must be known
- This table gives the total values, dimension, single offset and block offset of each file
--------------------------------------------------------------------------
|NAME|DIM1|DIM2|DIM3|QUANTITY|TOTAL PARAMETERS|SINGLE OFFSET|BLOCK OFFSET|
|----|----|----|----|--------|----------------|-------------|------------|
|MQ  |h   |d   |1   |x.y.m   |h.d.x.y.m       |h*d          |h.d.x.y     |
|MK  |h   |d   |1   |x.y.m   |h.d.x.y.m       |h*d          |h.d.x.y     |
|MV  |d   |h   |1   |x.y.m   |d.h.x.y.m       |d*h          |d.h.x.y     |
|MH  |d   |h   |1   |x.y.m   |d.h.x.y.m       |d*h          |d.h.x.y     |
|hor |d   |d   |l   |x.y.m   |d.d.l.x.y.m     |d* d *l      |d.d.l.x.y   |
|ver |d   |d   |l   |x.y.m   |d.d.l.x.y.m     |d* d *l      |d.d.l.x.y   |
|QK  |d   |d   |1   |x.y.m   |d.d.x.y.m       |d*d          |d.d.x.y     |
|QV  |d   |d   |1   |x.y.m   |d.d.x.y.m       |d*d          |d.d.x.y     |
|KH  |d   |d   |1   |x.y.m   |d.d.x.y.m       |d*d          |d.d.x.y     |
--------------------------------------------------------------------------
- Here single offset refers to total number of values in single object i.e., Matrix, MLP or cache
- Block Offset refers to total number of values of specific object in the single block i.e., number of object (matrix or mlp or cache) * single offset = x * y * single offset
- In the table: `h` refers to `MATHEIGHTS` (e.g., 1024), `d` refers to `EMBEDDING` dimension (e.g., 64), and `l` refers to the number of weight matrices in an MLP (e.g., `LAYERS_MLP - 1`).
- Following is the serialisation of MQ.bin file as example:
  - Q[i][j][k] represent MQ of kth head of jth row of ith block
```
.bin File:
Q[1][1][1] = --------------------------------------------------------------
Q[1][1][2] = --------------------------------------------------------------
Q[1][1][3] = --------------------------------------------------------------
    |               |               |               |               |
Q[1][x][y] = --------------------------------------------------------------
Q[2][1][1] = --------------------------------------------------------------
Q[2][1][2] = --------------------------------------------------------------
Q[2][1][1] = --------------------------------------------------------------
    |               |               |               |               |
Q[m][x][y] = --------------------------------------------------------------
```
- Similarly, all other matrices and MLP weights are serialised.

## Advantage
- In transformers, where to increase context means, you have to increase attention score grid and computation linearly, this can cause toll on RAM/VRAM. If the Using DCA, the context can be increased without increasing the memory requirement, and also computation gets reduced.
- Another way to look at this is that, if we want to create a long context LLM, we can make DCA based model with multiple smalled context, and get a model with similar context length without causing load on GPUs for computation.

## IMPORTANT NOTE:

- I would like to give huge credits to AI models and AgenticIDEs that I have used to build cuda and opencl operations.
- GROK: For Backpropagation of blocks (A big problem was that how should I reflect the change from error to mlp to the matrices like MQ, MK, MV and MH from block to block and most of the time without affecting the horizontal operations)
- FOR CUDA AND OPENCL:
  - GEMINI code assist
  - CLAUDE SONNET and DEEPSEEK in TRAE, WINDSURF
  - MISTRAL
  - ChatGPT
  - COPILOT

## Refernce
-  Vaswani, Ashish, Shazeer, Noam, Parmar, Niki, Uszkoreit, Jakob, Jones, Llion, Gomez, Aidan N., Kaiser, Lukasz, and Polosukhin, Illia Attention is all you need. 2017. https://doi.org/10.48550/arXiv.1706.03762
