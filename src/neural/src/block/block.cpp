
#include "include/block.hpp"
#include <stdexcept>
#include <string> // Required for std::to_string and string manipulations
#include <cstring>
#include <fstream>
#include <filesystem>

#ifndef USE_OPENCL

/**
 * @brief Constructor for complete attention block - NO OpenCL
 * @param x_layers number of partial attentions (layers) in block
 * @param y_heads number of attention heads in each partial attention
 * @param n_tokens number of tokens for each attention head (context window)
 * @param d_embed dimension of each token (embedding dimension)
 * @param h_internal height of MQ, MK matrices (internal dimension)
 * @param l_mlp layers of mlp within each attention head
 * @param vocab vocabulary size (unused)
 * @param attentionType attention type of heads, true if self and false if cross
 * @param trainMode Training (true) or Inference (false)
 * @param blockCount The index of this block, used for unique file naming.
 * @param blockFilePath_param The base path for the block's data file.
 */
block::block(int x_layers, int y_heads, int n_tokens, int d_embed, int h_internal, int l_mlp, long long int vocab, bool attentionType, 
    bool trainMode, int blockCount, const std::string blockFilePath_param) : // Changed to pass by value
    x(x_layers), y(y_heads), error(0.0f),
    isSelfAttention(attentionType), inTraining(trainMode),
    blockFilePath([&blockFilePath_param, blockCount]() {
        std::string base = blockFilePath_param;
        // Ensure the directory path ends with a separator
        if (!base.empty() && base.back() != '/' && base.back() != '\\') {
            base += '/'; // Use forward slash for consistency, Windows handles it
        }
        return base + "block_" + std::to_string(blockCount) + ".bin";
    }()),
    blockOffset(0LL)
{
    if (x <= 0 || y <= 0 || n_tokens <= 0 || d_embed <= 0 || h_internal <= 0 || l_mlp <= 0) {
        throw std::invalid_argument("Block dimensions must be positive in OpenCL constructor.");
    }
    b.resize(x, std::vector<attention>(y, attention(n_tokens, d_embed, h_internal, l_mlp, attentionType, trainMode)));
    EV.resize(x, std::vector<std::vector<std::vector<float>>>(y, std::vector<std::vector<float>>(n_tokens, std::vector<float>(d_embed, 0.0f))));
    tokForBlock = mat(n_tokens, d_embed);

    params = (x * y * (b[0][0].params + (n_tokens * d_embed))) + (d_embed * n_tokens);
    long long int totalBlockSize = params * sizeof(float);

    // File handling logic
    // Declare variables before conditional compilation block
    bool open_for_read_write_existing = false;
    long long int existing_file_size = 0;
    FILE* test_file = fopen(this->blockFilePath.c_str(), "rb");

    if (test_file) {
        #if defined(_WIN32) || defined(_WIN64)
            if (_fseeki64(test_file, 0LL, SEEK_END) == 0) {
                existing_file_size = _ftelli64(test_file); // Assign to the outer-scoped variable
            } else {
                std::cout << "FILESIZE WILL REMAIN 0 (_fseeki64 failed)" << std::endl;
            }
        #else // Assuming POSIX-like environment (Linux, macOS)
            if (fseeko64(test_file, 0LL, SEEK_END) == 0) {
                existing_file_size = ftello64(test_file); // Assign to the outer-scoped variable
            } else {
                // Handle fseeko64 failure? Log a warning?
                std::cout << "FILESIZE WILL REMAIN 0 (fseeko64 failed)" << std::endl;
            }
        #endif
            fclose(test_file);
    }

    if (open_for_read_write_existing) {
        blockFile = fopen(this->blockFilePath.c_str(), "rb+");
        if (!blockFile) {
            throw std::runtime_error("Could not open existing block file for read/write: " + this->blockFilePath);
        }
        std::cout << "BLOCK " << blockCount << " opened existing file: " << this->blockFilePath << std::endl;
    }
    else { // File does not exist, or exists but size mismatches
        blockFile = fopen(this->blockFilePath.c_str(), "wb+");
        if (!blockFile) {
            throw std::runtime_error("Could not create/truncate block file for writing: " + this->blockFilePath);
        }

        if (totalBlockSize > 0) {
        #if defined(_WIN32) || defined(_WIN64)
            if (_fseeki64(blockFile, totalBlockSize - 1, SEEK_SET) != 0) {
                fclose(blockFile);
                throw std::runtime_error("_fseeki64 failed to seek in block file to preallocate: " + this->blockFilePath);
            }
        #else // Assuming POSIX-like environment (Linux, macOS)
            if (fseeko64(blockFile, totalBlockSize - 1, SEEK_SET) != 0) {
                fclose(blockFile);
                throw std::runtime_error("_fseeki64 failed to seek in block file to preallocate: " + this->blockFilePath);
            }
        #endif
            if (fputc(0, blockFile) == EOF) {
                fclose(blockFile);
                throw std::runtime_error("fputc failed to write byte for preallocation: " + this->blockFilePath);
            }
        }

        if (fflush(blockFile) != 0) {
            fclose(blockFile);
            throw std::runtime_error("fflush failed after preallocation: " + this->blockFilePath);
        }
        
        if (test_file) { // File existed but size mismatched
            std::cout << "BLOCK " << blockCount << " truncated and recreated file due to size mismatch: " << this->blockFilePath << std::endl;
        } 
        else { // File did not exist
            std::cout << "BLOCK " << blockCount << " created new file: " << this->blockFilePath << std::endl;
        }
    }

    rewind(blockFile);
    std::cout << "BLOCK " << blockCount << " file prepared. Block parameters: " << params << ". Size of File: " << static_cast<float>(params * sizeof(float)) / (1000 * 1000) << " MiBs (" 
                                        << static_cast<float>(params * sizeof(float)) / (1024 * 1024) << " MBs)" << std::endl;
}

#else

#include <CL/cl.hpp>

/**
 * @brief Constructor for complete attention block - WITH OpenCL
 * @param context Reference to the shared OpenCL context.
 * @param x_layers number of partial attentions (layers) in block
 * @param y_heads number of attention heads in each partial attention
 * @param n_tokens number of tokens for each attention head (context window)
 * @param d_embed dimension of each token (embedding dimension)
 * @param h_internal height of MQ, MK matrices (internal dimension)
 * @param l_mlp layers of mlp within each attention head
 * @param vocab vocabulary size (unused)
 * @param attentionType attention type of heads, true if self and false if cross
 * @param trainMode Training (true) or Inference (false)
 * @param blockCount The index of this block, used for unique file naming.
 * @param blockFilePath_param The base path for the block's data file.
 */
block::block(OpenCLContext& context, int x_layers, int y_heads, int n_tokens, int d_embed, int h_internal, int l_mlp, long long int vocab,
    bool attentionType, bool trainMode, int blockCount, const std::string& blockFilePath_param) :
    clcontext(context), x(x_layers), y(y_heads), error(0.0f),
    isSelfAttention(attentionType), inTraining(trainMode),
        blockFilePath([&blockFilePath_param, blockCount]() {
        // Correctly construct path from directory
        std::string base = blockFilePath_param;
        // Ensure the directory path ends with a separator
        if (!base.empty() && base.back() != '/' && base.back() != '\\') {
            base += '/'; // Use forward slash for consistency, Windows handles it
        }
        return base + "block_" + std::to_string(blockCount) + ".bin";
    }()),
    blockOffset(0LL)

{
    if (x <= 0 || y <= 0 || n_tokens <= 0 || d_embed <= 0 || h_internal <= 0 || l_mlp <= 0) {
        throw std::invalid_argument("Block dimensions must be positive in OpenCL constructor.");
    }
    b.resize(x, std::vector<attention>(y, attention(context, n_tokens, d_embed, h_internal, l_mlp, attentionType, trainMode)));
    EV.resize(x, std::vector<std::vector<std::vector<float>>>(y, std::vector<std::vector<float>>(n_tokens, std::vector<float>(d_embed, 0.0f))));
    tokForBlock = mat(n_tokens, d_embed);

    params = (x * y * (b[0][0].params + (n_tokens * d_embed))) + (d_embed * n_tokens);
    long long int totalBlockSize = params * sizeof(float);

    // File handling logic
    bool open_for_read_write_existing = false;
    FILE* test_file = fopen(this->blockFilePath.c_str(), "rb");

    if (test_file) { // File exists
        #if defined(_WIN32) || defined(_WIN64)
            if (_fseeki64(test_file, 0LL, SEEK_END) == 0) {
                long long int existing_file_size = _ftelli64(test_file);
                if (existing_file_size == totalBlockSize) {
                    open_for_read_write_existing = true;
                }
            }
        #else // Assuming POSIX-like environment (Linux, macOS)
            if (fseeko64(test_file, 0LL, SEEK_END) == 0) {
                long long int existing_file_size = ftello64(test_file);
                if (existing_file_size == totalBlockSize) {
                    open_for_read_write_existing = true;
                }
            }
        #endif
        fclose(test_file);
    }

    if (open_for_read_write_existing) {
        blockFile = fopen(this->blockFilePath.c_str(), "rb+");
        if (!blockFile) {
            throw std::runtime_error("Could not open existing block file for read/write: " + this->blockFilePath);
        }
        std::cout << "BLOCK " << blockCount << " opened existing file: " << this->blockFilePath << std::endl;
    }
    else { // File does not exist, or exists but size mismatches
        blockFile = fopen(this->blockFilePath.c_str(), "wb+");
        if (!blockFile) {
            throw std::runtime_error("Could not create/truncate block file for writing: " + this->blockFilePath);
        }

        if (totalBlockSize > 0) {
        #if defined(_WIN32) || defined(_WIN64)
            if (_fseeki64(blockFile, totalBlockSize - 1, SEEK_SET) != 0) {
                fclose(blockFile);
                throw std::runtime_error("fseeko64/ _fseeki64 failed to seek in block file to preallocate: " + this->blockFilePath);
            }
        #else
            if (fseeko64(blockFile, totalBlockSize - 1, SEEK_SET) != 0) {
                fclose(blockFile);
                throw std::runtime_error("fseeko64/ _fseeki64 failed to seek in block file to preallocate: " + this->blockFilePath);
            }
        #endif

            if (fputc(0, blockFile) == EOF) {
                fclose(blockFile);
                throw std::runtime_error("fputc failed to write byte for preallocation: " + this->blockFilePath);
            }
        }

        if (fflush(blockFile) != 0) {
            fclose(blockFile);
            throw std::runtime_error("fflush failed after preallocation: " + this->blockFilePath);
        }
        
        if (test_file) { // File existed but size mismatched
            std::cout << "BLOCK " << blockCount << " truncated and recreated file due to size mismatch: " << this->blockFilePath << std::endl;
        }
        else { // File did not exist
            std::cout << "BLOCK " << blockCount << " created new file: " << this->blockFilePath << std::endl;
        }
    }

    rewind(blockFile);
    std::cout << "BLOCK " << blockCount << " file prepared. Block parameters: " << params << ". Size of File: " << static_cast<float>(params * sizeof(float)) / (1000 * 1000) << " MiBs (" 
                                        << static_cast<float>(params * sizeof(float)) / (1024 * 1024) << " MBs)" << std::endl;
}

#endif

// --- Common Member Functions ---

/**
 * @brief set vertical retention vectors of heads to blocks in single vector
 * @param EV_out shared space for vertical retention vectors of all heads of single block
 */
void block::setVerticalRetention(std::vector<std::vector<std::vector<std::vector<float>>>> &EV_out)
{
    size_t expected_tokens = (x > 0 && y > 0) ? b[0][0].EV.row : 0;
    size_t expected_embed_dim = (x > 0 && y > 0) ? b[0][0].EV.col : 0;

    if (EV_out.size() != x || (x > 0 && (EV_out[0].size() != y || (y > 0 && (EV_out[0][0].size() != expected_tokens || (expected_tokens > 0 && EV_out[0][0][0].size() != expected_embed_dim)))))) {
        EV_out.resize(x, std::vector<std::vector<std::vector<float>>>(y, \
            std::vector<std::vector<float>>(expected_tokens, std::vector<float>(expected_embed_dim, 0.0f))));
    }

    // complete block
    for(int i = 0; i < x; i++) {
        for(int j = 0; j < y; j++) {
            const mat& source_ev = b[i][j].EV; // Get reference to the mat object
            size_t num_tokens_in_head = source_ev.row;
            size_t embed_dim_in_head = source_ev.col;

            if (EV_out[i][j].size() != num_tokens_in_head || (num_tokens_in_head > 0 && EV_out[i][j][0].size() != embed_dim_in_head)) {
                 std::cerr << "Warning: Dimension mismatch for EV in block::setVerticalRetention at [" << i << "][" << j << "]. Skipping copy." << std::endl;
                 continue; // Skip this head
            }
            for(size_t k = 0; k < num_tokens_in_head; k++) {
                for(size_t col_idx = 0; col_idx < embed_dim_in_head; ++col_idx) {
                    EV_out[i][j][k][col_idx] = source_ev(k, col_idx);
                }
            }
        }
    }
}

// clear values of all the attention heads and EV
void block::clearValues() {
    if (tokForBlock.mapped_data && tokForBlock.mapped_size > 0)
        memset(tokForBlock.mapped_data, 0, tokForBlock.mapped_size);

    for (auto& dim1 : EV) {
        for (auto& dim2 : dim1) {
            for (auto& dim3 : dim2) {
                std::fill(dim3.begin(), dim3.end(), 0.0f);
            }
        }
    }

    for (auto& layer : b) {
        for (auto& head : layer) {
            head.clearValues(); // Call clearValues on each attention object
        }
    }
}
