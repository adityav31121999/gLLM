#include "include/attention.hpp"
#include "include/mat.hpp"
#include <numeric>
#include <stdexcept>

// --- Non-OpenCL Constructors ---
#ifndef USE_OPENCL

/**
 * @brief Constructor for incomplete attention - NO OpenCL
 * @param n number of tokens for each attention head
 * @param d dimension of each token
 * @param h height of MQ, MK and columns of MV, MH
 * @param l layers of mlp
 * @param isSelf Self (true) or Cross (false) attention
 * @param trainMode Training (true) or Inference (false)
 */
attention::attention(int n, int d, int h, int l, bool isSelf, bool trainMode, float& learning, float lambda_L1, float lambda_L2) :
    isSelfAttention(isSelf), inTraining(trainMode), lambda_L1(lambda_L1), lambda_L2(lambda_L2),
    tokenCount(0), EV(n, d), KdotQ(n, n), dh(d, 0.0f), dv(d, 0.0f), EH(d, 0),
    ver(std::vector<unsigned int>(l, d), EPOCHS, learning, lambda_L1, lambda_L2),
    hor(std::vector<unsigned int>(l, d), EPOCHS, learning, lambda_L1, lambda_L2),
    MQ(trainMode ? h : 0, trainMode ? d : 0),
    MK(trainMode ? h : 0, trainMode ? d : 0),
    MV(trainMode ? d : 0, trainMode ? h : 0),
    MH(trainMode ? d : 0, trainMode ? h : 0),
    qkCache(trainMode ? 0 : d, trainMode ? 0 : d),
    qvCache(trainMode ? 0 : d, trainMode ? 0 : d),
    khCache(trainMode ? 0 : d, trainMode ? 0 : d),
    K(trainMode ? n : 0, trainMode ? h : 0),
    Q(trainMode ? n : 0, trainMode ? h : 0)
{
    // Initialize gradient and Adam moment matrices only if in training mode
    // These must be constructed with proper dimensions if trainMode is true
    /*    
    gMQ(trainMode ? h : 0, trainMode ? d : 0),
    gMK(trainMode ? h : 0, trainMode ? d : 0),
    gMV(trainMode ? d : 0, trainMode ? h : 0),
    gMH(trainMode ? d : 0, trainMode ? h : 0),
    m_MQ(trainMode ? h : 0, trainMode ? d : 0),
    m_MK(trainMode ? h : 0, trainMode ? d : 0),
    m_MV(trainMode ? d : 0, trainMode ? h : 0),
    m_MH(trainMode ? d : 0, trainMode ? h : 0),
    v_MQ(trainMode ? h : 0, trainMode ? d : 0),
    v_MK(trainMode ? h : 0, trainMode ? d : 0),
    v_MV(trainMode ? d : 0, trainMode ? h : 0),
    v_MH(trainMode ? d : 0, trainMode ? h : 0)
    */

    if (n <= 0 || d <= 0 || h <= 0 || l <= 0) {
        // Only throw if essential dimensions are invalid, as some can be 0 for specific modes.
        // For example, if n is 0, K and Q will be 0xH, KdotQ will be 0x0.
        // This check is good for overall sanity.
        throw std::invalid_argument("Attention dimensions (n, d, h, l) must be positive.");
    }

    if (trainMode == 1) {
        // No need for re-assignments like 'if (MQ.row == 0) MQ = mat(h, d);'
        // They are already correctly initialized in the member initializer list based on trainMode.
        // If trainMode is 1, they are (h,d) or (d,h).
        // If trainMode is 0, they are (0,0) and stay that way.
        
        initializeAdamMoments(); // Call this to zero out the moment matrices
        params = hor.params + ver.params + 
                 // Mat                       eh        Kdotq                       tokforblock                     
                 (4*static_cast<size_t>(h)*d) + d + (static_cast<size_t>(n)*n) + (static_cast<size_t>(n)*d) + (2*static_cast<size_t>(n)*h);
        attOffset = hor.params + ver.params + 4*h*d + n*n + 2*n*h * n*d + d;
    }
    else {
        // In inference mode, cache matrices are active, others are 0x0.
        // No explicit assignments needed here as they are done in initializer list.
        params = hor.params + ver.params + (3*static_cast<size_t>(d)*d) + d + (static_cast<size_t>(n)*n) + (static_cast<size_t>(n)*d);
    }
    std::cout << "ATTENTION constructed." << std::endl;
}

#else // USE_OPENCL is defined

#include <CL/cl.hpp>

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
attention::attention(OpenCLContext& context, int n, int d, int h, int l, bool isSelf, bool trainMode, float& learning, float lambda_L1, float lambda_L2) :
    clcontext(context), isSelfAttention(isSelf), inTraining(trainMode),
    tokenCount(0), EV(n, d), KdotQ(n, n), dh(d, 0.0f), dv(d, 0.0f), EH(d, 0),
    lambda_L1(lambda_L1), lambda_L2(lambda_L2),
    ver(context, std::vector<unsigned int>(l, d), EPOCHS, learning, lambda_L1, lambda_L2),
    hor(context, std::vector<unsigned int>(l, d), EPOCHS, learning, lambda_L1, lambda_L2),
    MQ(trainMode ? h : 0, trainMode ? d : 0),
    MK(trainMode ? h : 0, trainMode ? d : 0),
    MV(trainMode ? d : 0, trainMode ? h : 0),
    MH(trainMode ? d : 0, trainMode ? h : 0),
    qkCache(trainMode ? 0 : d, trainMode ? 0 : d),
    qvCache(trainMode ? 0 : d, trainMode ? 0 : d),
    khCache(trainMode ? 0 : d, trainMode ? 0 : d),
    K(trainMode ? n : 0, trainMode ? h : 0),
    Q(trainMode ? n : 0, trainMode ? h : 0)
{
    // Initialize gradient and Adam moment matrices only if in training mode
    /*    
    gMQ(trainMode ? h : 0, trainMode ? d : 0),
    gMK(trainMode ? h : 0, trainMode ? d : 0),
    gMV(trainMode ? d : 0, trainMode ? h : 0),
    gMH(trainMode ? d : 0, trainMode ? h : 0),
    m_MQ(trainMode ? h : 0, trainMode ? d : 0),
    m_MK(trainMode ? h : 0, trainMode ? d : 0),
    m_MV(trainMode ? d : 0, trainMode ? h : 0),
    m_MH(trainMode ? d : 0, trainMode ? h : 0),
    v_MQ(trainMode ? h : 0, trainMode ? d : 0),
    v_MK(trainMode ? h : 0, trainMode ? d : 0),
    v_MV(trainMode ? d : 0, trainMode ? h : 0),
    v_MH(trainMode ? d : 0, trainMode ? h : 0)
    */

    if (n <= 0 || d <= 0 || h <= 0 || l <= 0) {
        throw std::invalid_argument("Attention dimensions (n, d, h, l) must be positive.");
    }
    cl_int cl_err;
    size_t ev_buffer_size = static_cast<size_t>(CONTEXT_WIN) * d * sizeof(float);
    if (ev_buffer_size == 0) {
        throw std::runtime_error("Calculated EV buffer size is zero in attention constructor.");
    }
    // d_EV needs to be allocated here
    this->d_EV = cl::Buffer(this->clcontext.context, CL_MEM_READ_WRITE, ev_buffer_size, nullptr, &cl_err);
    CL_CHECK(cl_err); // Your CL_CHECK macro will handle the error

    if (trainMode == 1) {
        // initializeAdamMoments(); // Call this to zero out the moment matrices
        // hor + ver
        // weght matrices
        // kdotq
        // EV, Q, K
        params = hor.params + ver.params + 
                 // Mat                       eh        Kdotq                       tokforblock                     
                 (4*static_cast<size_t>(h)*d) + d + (static_cast<size_t>(n)*n) + (static_cast<size_t>(n)*d) + (2*static_cast<size_t>(n)*h);
        attOffset = hor.params + ver.params + 4*h*d + n*n + 2*n*h * n*d + d;
    }
    else {
        params = hor.params + ver.params + 
                 (3*static_cast<size_t>(d)*d) + d + (static_cast<size_t>(n)*n) + (static_cast<size_t>(n)*d);
    }
    std::cout << "ATTENTION constructed with OpenCL."<< std::endl;
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

void attention::initializeAdamMoments() {
    if (inTraining) { // Only initialize/zero if in training mode
        // Zero out moment and velocity matrices if they are mapped and have data
        // Corrected conditions to check the moment/velocity matrices themselves
    /*
        if (m_MQ.mapped_data && m_MQ.mapped_size > 0) {
            std::memset(m_MQ.mapped_data, 0, m_MQ.mapped_size);
        }
        if (v_MQ.mapped_data && v_MQ.mapped_size > 0) {
            std::memset(v_MQ.mapped_data, 0, v_MQ.mapped_size);
        }

        if (m_MK.mapped_data && m_MK.mapped_size > 0) {
            std::memset(m_MK.mapped_data, 0, m_MK.mapped_size);
        }
        if (v_MK.mapped_data && v_MK.mapped_size > 0) {
            std::memset(v_MK.mapped_data, 0, v_MK.mapped_size);
        }

        if (m_MV.mapped_data && m_MV.mapped_size > 0) {
            std::memset(m_MV.mapped_data, 0, m_MV.mapped_size);
        }
        if (v_MV.mapped_data && v_MV.mapped_size > 0) {
            std::memset(v_MV.mapped_data, 0, v_MV.mapped_size);
        }

        if (m_MH.mapped_data && m_MH.mapped_size > 0) {
            std::memset(m_MH.mapped_data, 0, m_MH.mapped_size);
        }
        if (v_MH.mapped_data && v_MH.mapped_size > 0) {
            std::memset(v_MH.mapped_data, 0, v_MH.mapped_size);
        }
    */
    }
    // Initialize moments for internal MLPs (they handle their own logic,
    // and should also be fixed with the same logic as mlp::initializeAdamMoments if not already)
    hor.initializeAdamMoments();
    ver.initializeAdamMoments();
}


// clear all the vectors of attention class
void attention::clearValues()
{
    std::fill(dh.begin(), dh.end(), 0.0f);
    std::fill(dv.begin(), dv.end(), 0.0f);

    // Clear mat objects using memset
    if (K.mapped_data && K.mapped_size > 0) memset(K.mapped_data, 0, K.mapped_size);
    if (Q.mapped_data && Q.mapped_size > 0) memset(Q.mapped_data, 0, Q.mapped_size);
    if (KdotQ.mapped_data && KdotQ.mapped_size > 0) memset(KdotQ.mapped_data, 0, KdotQ.mapped_size);
    if (EV.mapped_data && EV.mapped_size > 0) memset(EV.mapped_data, 0, EV.mapped_size);
    // Also clear cache matrices if they exist
    if (qkCache.mapped_data && qkCache.mapped_size > 0) memset(qkCache.mapped_data, 0, qkCache.mapped_size);
    if (qvCache.mapped_data && qvCache.mapped_size > 0) memset(qvCache.mapped_data, 0, qvCache.mapped_size);
    if (khCache.mapped_data && khCache.mapped_size > 0) memset(khCache.mapped_data, 0, khCache.mapped_size);

    hor.clearValues();
    ver.clearValues();
}

// serialise the attention class to train .bin file
void attention::serialiseattention4train(const std::string& locationWithFileName) {
    write2filefrommat(MQ, locationWithFileName);    // n x matheight
    write2filefrommat(MK, locationWithFileName);    // + n x matheight
    write2filefrommat(MV, locationWithFileName);    // + matheight x n
    write2filefrommat(MH, locationWithFileName);    // + matheight x n
    write2filefrommat(K, locationWithFileName);     // + n x d
    write2filefrommat(Q, locationWithFileName);     // + n x d
    write2filefrommat(KdotQ, locationWithFileName); // + n x n
    write2filefrommat(EV, locationWithFileName);    // + n x d
    hor.serialise4train(locationWithFileName);     // + (d x d x l) + (d x d x l)
    ver.serialise4train(locationWithFileName);     // + (d x d x l) + (d x d x l)
    // Write dimensions of EH
    size_t dim1_size = EH.size();
    std::ofstream outFile(locationWithFileName, std::ios::app | std::ios::binary | std::ios::app);
    if (!outFile.is_open()) {
        // + d
        std::cerr << "Error: Could not open file " << locationWithFileName << " for writing EH in attention::serialiseblock." << std::endl;
        return;
    }
    // Write data for EV
    outFile.write(reinterpret_cast<const char*>(EH.data()), EH.size() * sizeof(float));
    outFile.close();
}

// serialise the attention class to different .bin files from train .bin file
void attention::serialiseattention4use(const std::string& binDirectory, const std::string& locationWithFileName) {
    write2filefrommat(MQ, binDirectory + "/MQ.bin");
    write2filefrommat(MK, binDirectory + "/MK.bin");
    write2filefrommat(MV, binDirectory + "/MV.bin");
    write2filefrommat(MH, binDirectory + "/MH.bin");
    qkCache = MQ * MK.transpose();
    qvCache = MQ * MH;
    khCache = MK * MV;
    write2filefrommat(qkCache, binDirectory + "/QK.bin");
    write2filefrommat(qvCache, binDirectory + "/QV.bin");
    write2filefrommat(khCache, binDirectory + "/KH.bin");
    hor.serialise4train(binDirectory + "/HOR.bin");
    ver.serialise4train(binDirectory + "/VER.bin");
}
