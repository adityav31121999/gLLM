// attention.cpp: constructor for incomplete attention
#include "include/attention.hpp" // Includes basic.hpp -> OpenCLContext definition if USE_OPENCL
#include "include/mat.hpp"
#include <numeric>
#include <stdexcept> // For potential exceptions

// --- Non-OpenCL Constructors ---
#ifndef USE_OPENCL

/**
 * @brief Constructor for incomplete attention (default: self attention, training) - NO OpenCL
 * @param n number of tokens for each attention head
 * @param d dimension of each token
 * @param h height of MQ, MK and columns of MV, MH
 * @param l layers of mlp
 */
attention::attention(int n, int d, int h, int l) :
    isSelfAttention(true),
    inTraining(true),
    tokenCount(0), // Initialize tokenCount
    ver(d, l, EPOCHS, LEARNING), // Initialize mlp members
    hor(d, l, EPOCHS, LEARNING),
    MQ(h, d),
    MK(h, d),
    MV(d, h),
    MH(d, h)
    // qkCache, qvCache, khCache are only used in inference mode (inTraining=false)
{
    if (n <= 0 || d <= 0 || h <= 0 || l <= 0) {
        throw std::invalid_argument("Attention dimensions must be positive.");
    }
    KdotQ.resize(n, std::vector<float>(n, 0.0f));
    K.resize(n, std::vector<float>(h, 0.0f));
    Q.resize(n, std::vector<float>(h, 0.0f));
    dh.resize(d, 0.0f);
    dv.resize(d, 0.0f);
    EH.resize(d, 0.0f);
    EV.resize(n, std::vector<float>(d, 0.0f)); // Use n, not CONTEXT_WIN here for consistency? Or always CONTEXT_WIN? Let's stick to n for now based on other constructors.
}

/**
 * @brief Constructor for incomplete attention (default: training) - NO OpenCL
 * @param n number of tokens for each attention head
 * @param d dimension of each token
 * @param h height of MQ, MK and columns of MV, MH
 * @param l layers of mlp
 * @param isSelf Self (true) or Cross (false) attention
 */
attention::attention(int n, int d, int h, int l, bool isSelf) :
    isSelfAttention(isSelf),
    inTraining(true),
    tokenCount(0),
    ver(d, l, EPOCHS, LEARNING),
    hor(d, l, EPOCHS, LEARNING),
    MQ(h, d),
    MK(h, d),
    MV(d, h),
    MH(d, h)
{
    if (n <= 0 || d <= 0 || h <= 0 || l <= 0) {
        throw std::invalid_argument("Attention dimensions must be positive.");
    }
    KdotQ.resize(n, std::vector<float>(n, 0.0f));
    K.resize(n, std::vector<float>(h, 0.0f));
    Q.resize(n, std::vector<float>(h, 0.0f));
    dh.resize(d, 0.0f);
    dv.resize(d, 0.0f);
    EH.resize(d, 0.0f);
    EV.resize(n, std::vector<float>(d, 0.0f));
}

/**
 * @brief Constructor for incomplete attention - NO OpenCL
 * @param n number of tokens for each attention head
 * @param d dimension of each token
 * @param h height of MQ, MK and columns of MV, MH
 * @param l layers of mlp
 * @param isSelf Self (true) or Cross (false) attention
 * @param trainMode Training (true) or Inference (false)
 */
attention::attention(int n, int d, int h, int l, bool isSelf, bool trainMode) :
    isSelfAttention(isSelf),
    inTraining(trainMode),
    tokenCount(0),
    ver(d, l, EPOCHS, LEARNING),
    hor(d, l, EPOCHS, LEARNING),
    // Conditionally initialize matrices based on training mode
    MQ(trainMode ? h : 0, trainMode ? d : 0), // Only needed in training? Check logic. Assuming yes for now.
    MK(trainMode ? h : 0, trainMode ? d : 0),
    MV(trainMode ? d : 0, trainMode ? h : 0),
    MH(trainMode ? d : 0, trainMode ? h : 0),
    qkCache(trainMode ? 0 : d, trainMode ? 0 : d), // Only needed in inference
    qvCache(trainMode ? 0 : d, trainMode ? 0 : d),
    khCache(trainMode ? 0 : d, trainMode ? 0 : d)
{
    if (n <= 0 || d <= 0 || h <= 0 || l <= 0) {
        throw std::invalid_argument("Attention dimensions must be positive.");
    }

    // Common initializations
    dh.resize(d, 0.0f);
    dv.resize(d, 0.0f);
    EH.resize(d, 0.0f);
    EV.resize(n, std::vector<float>(d, 0.0f));
    KdotQ.resize(n, std::vector<float>(n, 0.0f)); // Needed for both? Check usage. Assuming yes.

    if (trainMode) {
        // Training specific initializations
        K.resize(n, std::vector<float>(h, 0.0f));
        Q.resize(n, std::vector<float>(h, 0.0f));
        // Re-initialize matrices if needed (already done in initializer list if logic is correct)
        if (MQ.row == 0) MQ = mat(h, d);
        if (MK.row == 0) MK = mat(h, d);
        if (MV.row == 0) MV = mat(d, h);
        if (MH.row == 0) MH = mat(d, h);
    } else {
        // Inference specific initializations (Matrices initialized in initializer list)
        // K and Q might not be needed if using caches directly? Check logic.
        // Let's assume K/Q vectors are not stored persistently in inference.
    }
}

#else // USE_OPENCL is defined

// --- OpenCL Constructors ---

/**
 * @brief Constructor for incomplete attention (default: self attention, training) - WITH OpenCL
 * @param context Reference to the shared OpenCL context.
 * @param n number of tokens for each attention head
 * @param d dimension of each token
 * @param h height of MQ, MK and columns of MV, MH
 * @param l layers of mlp
 */
attention::attention(OpenCLContext& context, int n, int d, int h, int l) :
    clcontext(context), // Initialize OpenCL context reference
    isSelfAttention(true),
    inTraining(true),
    tokenCount(0),
    ver(context, d, l, EPOCHS, LEARNING), // Pass context to mlp constructor
    hor(context, d, l, EPOCHS, LEARNING), // Pass context to mlp constructor
    MQ(h, d),
    MK(h, d),
    MV(d, h),
    MH(d, h)
{
    if (n <= 0 || d <= 0 || h <= 0 || l <= 0) {
        throw std::invalid_argument("Attention dimensions must be positive.");
    }
    KdotQ.resize(n, std::vector<float>(n, 0.0f));
    K.resize(n, std::vector<float>(h, 0.0f));
    Q.resize(n, std::vector<float>(h, 0.0f));
    dh.resize(d, 0.0f);
    dv.resize(d, 0.0f);
    EH.resize(d, 0.0f);
    EV.resize(n, std::vector<float>(d, 0.0f));
}

/**
 * @brief Constructor for incomplete attention (default: training) - WITH OpenCL
 * @param context Reference to the shared OpenCL context.
 * @param n number of tokens for each attention head
 * @param d dimension of each token
 * @param h height of MQ, MK and columns of MV, MH
 * @param l layers of mlp
 * @param isSelf Self (true) or Cross (false) attention
 */
attention::attention(OpenCLContext& context, int n, int d, int h, int l, bool isSelf) :
    clcontext(context),
    isSelfAttention(isSelf),
    inTraining(true),
    tokenCount(0),
    ver(context, d, l, EPOCHS, LEARNING),
    hor(context, d, l, EPOCHS, LEARNING),
    MQ(h, d),
    MK(h, d),
    MV(d, h),
    MH(d, h)
{
    if (n <= 0 || d <= 0 || h <= 0 || l <= 0) {
        throw std::invalid_argument("Attention dimensions must be positive.");
    }
    KdotQ.resize(n, std::vector<float>(n, 0.0f));
    K.resize(n, std::vector<float>(h, 0.0f));
    Q.resize(n, std::vector<float>(h, 0.0f));
    dh.resize(d, 0.0f);
    dv.resize(d, 0.0f);
    EH.resize(d, 0.0f);
    EV.resize(n, std::vector<float>(d, 0.0f));
}

/**
 * @brief Constructor for incomplete attention - WITH OpenCL
 * @param context Reference to the shared OpenCL context.
 * @param n number of tokens for each attention head
 * @param d dimension of each token
 * @param h height of MQ, MK and columns of MV, MH
 * @param l layers of mlp
 * @param isSelf Self (true) or Cross (false) attention
 * @param trainMode Training (true) or Inference (false)
 */
attention::attention(OpenCLContext& context, int n, int d, int h, int l, bool isSelf, bool trainMode) :
    clcontext(context),
    isSelfAttention(isSelf),
    inTraining(trainMode),
    tokenCount(0),
    ver(context, d, l, EPOCHS, LEARNING),
    hor(context, d, l, EPOCHS, LEARNING),
    // Conditionally initialize matrices based on training mode
    MQ(trainMode ? h : 0, trainMode ? d : 0),
    MK(trainMode ? h : 0, trainMode ? d : 0),
    MV(trainMode ? d : 0, trainMode ? h : 0),
    MH(trainMode ? d : 0, trainMode ? h : 0),
    qkCache(trainMode ? 0 : d, trainMode ? 0 : d),
    qvCache(trainMode ? 0 : d, trainMode ? 0 : d),
    khCache(trainMode ? 0 : d, trainMode ? 0 : d)
{
    if (n <= 0 || d <= 0 || h <= 0 || l <= 0) {
        throw std::invalid_argument("Attention dimensions must be positive.");
    }

    // Common initializations
    dh.resize(d, 0.0f);
    dv.resize(d, 0.0f);
    EH.resize(d, 0.0f);
    EV.resize(n, std::vector<float>(d, 0.0f));
    KdotQ.resize(n, std::vector<float>(n, 0.0f));

    if (trainMode) {
        // Training specific initializations
        K.resize(n, std::vector<float>(h, 0.0f));
        Q.resize(n, std::vector<float>(h, 0.0f));
        if (MQ.row == 0) MQ = mat(h, d);
        if (MK.row == 0) MK = mat(h, d);
        if (MV.row == 0) MV = mat(d, h);
        if (MH.row == 0) MH = mat(d, h);
    } else {
        // Inference specific initializations (Matrices initialized in initializer list)
    }
     // Optional: Log OpenCL context usage
    // std::cout << "Attention head constructed with OpenCL context." << std::endl;
}

#endif // USE_OPENCL

// --- Common Member Functions ---

/**
 * @brief set attention type of model for cross and self attention
 * @param isSelforCross 1 for self attention and 0 for cross attention
 */
void attention::setAttentionType(bool isSelforCross) {
    isSelfAttention = isSelforCross;
}

// clear all the vectors of attention class
void attention::clearValues()
{
     // Clear 1D vectors by filling with 0.0f
    std::fill(EH.begin(), EH.end(), 0.0f);
    std::fill(dh.begin(), dh.end(), 0.0f);
    std::fill(dv.begin(), dv.end(), 0.0f);

    // Clear 2D vectors by filling inner vectors with 0.0f
    for (auto& innerVec : K) {
        std::fill(innerVec.begin(), innerVec.end(), 0.0f);
    }
    for (auto& innerVec : Q) {
        std::fill(innerVec.begin(), innerVec.end(), 0.0f);
    }
    for (auto& innerVec : KdotQ) {
        std::fill(innerVec.begin(), innerVec.end(), 0.0f);
    }
    for (auto& innerVec : EV) {
        std::fill(innerVec.begin(), innerVec.end(), 0.0f);
    }
    hor.clearValues();
    ver.clearValues();
}
