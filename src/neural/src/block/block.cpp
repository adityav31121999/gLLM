// block.cpp: constructor for block class
#include "include/block.hpp" // Includes attention.hpp -> basic.hpp -> OpenCLContext definition if USE_OPENCL
#include <stdexcept>

// --- Non-OpenCL Constructors ---
#ifndef USE_OPENCL

/**
 * @brief Constructor for complete attention block (default: self attention, training) - NO OpenCL
 * @param x number of partial attentions (layers) in block
 * @param y number of attention heads in each partial attention
 * @param n number of tokens for each attention head (context window)
 * @param d dimension of each token (embedding dimension)
 * @param h height of MQ, MK matrices (internal dimension)
 * @param l layers of mlp within each attention head
 * @param vocab vocabulary size (Note: vocab parameter seems unused here, maybe intended for transformer?)
 */
block::block(int x_layers, int y_heads, int n_tokens, int d_embed, int h_internal, int l_mlp, int vocab) :
    x(x_layers),
    y(y_heads),
    error(0.0f), // Initialize error
    isSelfAttention(true),
    inTraining(true)
    // str is default initialized
    // b is initialized below
{
    if (x <= 0 || y <= 0 || n_tokens <= 0 || d_embed <= 0 || h_internal <= 0 || l_mlp <= 0) {
        throw std::invalid_argument("Block dimensions must be positive.");
    }
    // Initialize attention block (complete attention) using the appropriate attention constructor
    // Create one default attention object and let the vector copy it
    b.resize(x, std::vector<attention>(y, attention(n_tokens, d_embed, h_internal, l_mlp)));

    EH.resize(d_embed, 0.0f); // horizontal retention vector for the block
    // Collection of vertical retention vectors from all heads
    EV.resize(x, std::vector<std::vector<std::vector<float>>>(y, std::vector<std::vector<float>>(n_tokens, std::vector<float>(d_embed, 0.0f))));
    tokForBlock.resize(n_tokens, std::vector<float>(d_embed, 0.0f));
}

/**
 * @brief Constructor for complete attention block (default: training) - NO OpenCL
 * @param x number of partial attentions (layers) in block
 * @param y number of attention heads in each partial attention
 * @param n number of tokens for each attention head (context window)
 * @param d dimension of each token (embedding dimension)
 * @param h height of MQ, MK matrices (internal dimension)
 * @param l layers of mlp within each attention head
 * @param vocab vocabulary size (unused)
 * @param attentionType attention type of heads, true if self and false if cross
 */
block::block(int x_layers, int y_heads, int n_tokens, int d_embed, int h_internal, int l_mlp, int vocab, bool attentionType) :
    x(x_layers),
    y(y_heads),
    error(0.0f),
    isSelfAttention(attentionType),
    inTraining(true)
{
    if (x <= 0 || y <= 0 || n_tokens <= 0 || d_embed <= 0 || h_internal <= 0 || l_mlp <= 0) {
        throw std::invalid_argument("Block dimensions must be positive.");
    }
    // Initialize attention block with specified attention type
    b.resize(x, std::vector<attention>(y, attention(n_tokens, d_embed, h_internal, l_mlp, attentionType)));

    EH.resize(d_embed, 0.0f);
    EV.resize(x, std::vector<std::vector<std::vector<float>>>(y, std::vector<std::vector<float>>(n_tokens, std::vector<float>(d_embed, 0.0f))));
    tokForBlock.resize(n_tokens, std::vector<float>(d_embed, 0.0f));
}

/**
 * @brief Constructor for complete attention block - NO OpenCL
 * @param x number of partial attentions (layers) in block
 * @param y number of attention heads in each partial attention
 * @param n number of tokens for each attention head (context window)
 * @param d dimension of each token (embedding dimension)
 * @param h height of MQ, MK matrices (internal dimension)
 * @param l layers of mlp within each attention head
 * @param vocab vocabulary size (unused)
 * @param attentionType attention type of heads, true if self and false if cross
 * @param trainMode Training (true) or Inference (false)
 */
block::block(int x_layers, int y_heads, int n_tokens, int d_embed, int h_internal, int l_mlp, int vocab, bool attentionType, bool trainMode) :
    x(x_layers),
    y(y_heads),
    error(0.0f),
    isSelfAttention(attentionType),
    inTraining(trainMode)
{
     if (x <= 0 || y <= 0 || n_tokens <= 0 || d_embed <= 0 || h_internal <= 0 || l_mlp <= 0) {
        throw std::invalid_argument("Block dimensions must be positive.");
    }
    // Initialize attention block with specified type and training mode
    b.resize(x, std::vector<attention>(y, attention(n_tokens, d_embed, h_internal, l_mlp, attentionType, trainMode)));

    EH.resize(d_embed, 0.0f);
    // Only resize EV and tokForBlock if in training? Check logic. Assuming needed for both structure-wise.
    EV.resize(x, std::vector<std::vector<std::vector<float>>>(y, std::vector<std::vector<float>>(n_tokens, std::vector<float>(d_embed, 0.0f))));
    if (trainMode) {
        tokForBlock.resize(n_tokens, std::vector<float>(d_embed, 0.0f));
    }
    // If not training, tokForBlock might be handled differently (e.g., passed in during inference)
}

#else // USE_OPENCL is defined

// --- OpenCL Constructors ---

/**
 * @brief Constructor for complete attention block (default: self attention, training) - WITH OpenCL
 * @param context Reference to the shared OpenCL context.
 * @param x number of partial attentions (layers) in block
 * @param y number of attention heads in each partial attention
 * @param n number of tokens for each attention head (context window)
 * @param d dimension of each token (embedding dimension)
 * @param h height of MQ, MK matrices (internal dimension)
 * @param l layers of mlp within each attention head
 * @param vocab vocabulary size (unused)
 */
block::block(OpenCLContext& context, int x_layers, int y_heads, int n_tokens, int d_embed, int h_internal, int l_mlp, int vocab) :
    clcontext(context), // Initialize OpenCL context reference
    x(x_layers),
    y(y_heads),
    error(0.0f),
    isSelfAttention(true),
    inTraining(true)
{
    if (x <= 0 || y <= 0 || n_tokens <= 0 || d_embed <= 0 || h_internal <= 0 || l_mlp <= 0) {
        throw std::invalid_argument("Block dimensions must be positive.");
    }
    // Initialize attention block, passing context to attention constructor
    b.resize(x, std::vector<attention>(y, attention(context, n_tokens, d_embed, h_internal, l_mlp)));

    EH.resize(d_embed, 0.0f);
    EV.resize(x, std::vector<std::vector<std::vector<float>>>(y, std::vector<std::vector<float>>(n_tokens, std::vector<float>(d_embed, 0.0f))));
    tokForBlock.resize(n_tokens, std::vector<float>(d_embed, 0.0f));
    // Optional: Log OpenCL context usage
    // std::cout << "Block constructed with OpenCL context." << std::endl;
}

/**
 * @brief Constructor for complete attention block (default: training) - WITH OpenCL
 * @param context Reference to the shared OpenCL context.
 * @param x number of partial attentions (layers) in block
 * @param y number of attention heads in each partial attention
 * @param n number of tokens for each attention head (context window)
 * @param d dimension of each token (embedding dimension)
 * @param h height of MQ, MK matrices (internal dimension)
 * @param l layers of mlp within each attention head
 * @param vocab vocabulary size (unused)
 * @param attentionType attention type of heads, true if self and false if cross
 */
block::block(OpenCLContext& context, int x_layers, int y_heads, int n_tokens, int d_embed, int h_internal, int l_mlp, int vocab, bool attentionType) :
    clcontext(context),
    x(x_layers),
    y(y_heads),
    error(0.0f),
    isSelfAttention(attentionType),
    inTraining(true)
{
     if (x <= 0 || y <= 0 || n_tokens <= 0 || d_embed <= 0 || h_internal <= 0 || l_mlp <= 0) {
        throw std::invalid_argument("Block dimensions must be positive.");
    }
    // Initialize attention block with specified type, passing context
    b.resize(x, std::vector<attention>(y, attention(context, n_tokens, d_embed, h_internal, l_mlp, attentionType)));

    EH.resize(d_embed, 0.0f);
    EV.resize(x, std::vector<std::vector<std::vector<float>>>(y, std::vector<std::vector<float>>(n_tokens, std::vector<float>(d_embed, 0.0f))));
    tokForBlock.resize(n_tokens, std::vector<float>(d_embed, 0.0f));
}

/**
 * @brief Constructor for complete attention block - WITH OpenCL
 * @param context Reference to the shared OpenCL context.
 * @param x number of partial attentions (layers) in block
 * @param y number of attention heads in each partial attention
 * @param n number of tokens for each attention head (context window)
 * @param d dimension of each token (embedding dimension)
 * @param h height of MQ, MK matrices (internal dimension)
 * @param l layers of mlp within each attention head
 * @param vocab vocabulary size (unused)
 * @param attentionType attention type of heads, true if self and false if cross
 * @param trainMode Training (true) or Inference (false)
 */
block::block(OpenCLContext& context, int x_layers, int y_heads, int n_tokens, int d_embed, int h_internal, int l_mlp, int vocab, bool attentionType, bool trainMode) :
    clcontext(context),
    x(x_layers),
    y(y_heads),
    error(0.0f),
    isSelfAttention(attentionType),
    inTraining(trainMode)
{
     if (x <= 0 || y <= 0 || n_tokens <= 0 || d_embed <= 0 || h_internal <= 0 || l_mlp <= 0) {
        throw std::invalid_argument("Block dimensions must be positive.");
    }
    // Initialize attention block with specified type and mode, passing context
    b.resize(x, std::vector<attention>(y, attention(context, n_tokens, d_embed, h_internal, l_mlp, attentionType, trainMode)));

    EH.resize(d_embed, 0.0f);
    EV.resize(x, std::vector<std::vector<std::vector<float>>>(y, std::vector<std::vector<float>>(n_tokens, std::vector<float>(d_embed, 0.0f))));
    if (trainMode) {
        tokForBlock.resize(n_tokens, std::vector<float>(d_embed, 0.0f));
    }
}

#endif // USE_OPENCL

// --- Common Member Functions ---

/**
 * @brief set vertical retention vectors of heads to blocks in single vector
 * @param EV_out shared space for vertical retention vectors of all heads of single block
 */
void block::setVerticalRetention(std::vector<std::vector<std::vector<std::vector<float>>>> &EV_out)
{
    // Ensure the output vector has the correct dimensions
    if (EV_out.size() != x || (x > 0 && (EV_out[0].size() != y || (y > 0 && EV_out[0][0].size() != b[0][0].EV.size())))) {
         // Resize or throw error, resizing might be safer if dimensions can vary
         // For now, let's assume dimensions match the block's internal structure
         // Consider adding a check for b[0][0].EV dimensions if n_tokens can vary
         // EV_out.resize(x, std::vector<std::vector<std::vector<float>>>(y, std::vector<std::vector<float>>(n_tokens, std::vector<float>(d_embed, 0.0f))));
         // Or throw:
         // throw std::runtime_error("EV_out dimensions mismatch in setVerticalRetention");
         // Let's assume dimensions are pre-allocated correctly for now.
    }

    // complete block
    for(int i = 0; i < x; i++) {
        // layers of partial attention
        for(int j = 0; j < y; j++) {
            // attention heads of each partial attention
            // Check if EV size matches CONTEXT_WIN or n_tokens used in constructor
            size_t num_vectors = b[i][j].EV.size(); // Get actual size from attention head
            if (EV_out[i][j].size() != num_vectors) {
                 // Handle mismatch if necessary
                 // For now, assume they match or copy only up to the smaller size
                 num_vectors = std::min(num_vectors, EV_out[i][j].size());
            }
            for(size_t k = 0; k < num_vectors; k++) {
                // each vertical retention vector of head
                EV_out[i][j][k] = b[i][j].EV[k]; // Direct copy
            }
        }
    }
}

void block::clearValues() {
    // Clear 1D float vector EH
    std::fill(EH.begin(), EH.end(), 0.0f);

    // Clear 2D float vector tokForBlock
    for (auto& innerVec : tokForBlock) {
        std::fill(innerVec.begin(), innerVec.end(), 0.0f);
    }

    // Clear 4D float vector EV
    for (auto& dim1 : EV) {
        for (auto& dim2 : dim1) {
            for (auto& dim3 : dim2) {
                std::fill(dim3.begin(), dim3.end(), 0.0f);
            }
        }
    }

    // Clear each attention object within the block
    for (auto& layer : b) {
        for (auto& head : layer) {
            head.clearValues(); // Call clearValues on each attention object
        }
    }
}
