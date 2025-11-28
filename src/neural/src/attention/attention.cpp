#include "include/attention.hpp"
#include "include/mat.hpp"
#include <numeric>
#include <stdexcept>

#ifndef USE_CL

/**
 * @brief Constructor for incomplete attention - NO OpenCL
 * @param n number of tokens for each attention head
 * @param d dimension of each token
 * @param h height of MQ, MK and columns of MV, MH
 * @param l layers of mlp
 * @param isSelf Self (true) or Cross (false) attention
 * @param trainMode Training (true) or Inference (false)
 */
attention::attention(int n, int d, int h, int l, bool isSelf, bool trainMode, float& learning) :
    isSelfAttention(isSelf), inTraining(trainMode),
    tokenCount(0), EV(n, d), KdotQ(n, n), h(d, 0.0f), v(d, 0.0f), EH(d, 0),
    ver(std::vector<unsigned int>(l, d), EPOCHS, learning),
    hor(std::vector<unsigned int>(l, d), EPOCHS, learning),
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

    if (trainMode == 1) {
        K = mat(n, h);
        Q = mat(n, h);
        if (MQ.row == 0) MQ = mat(h, d);        // as per maths this should be d x h
        if (MK.row == 0) MK = mat(h, d);        // as per maths this should be d x h
        if (MV.row == 0) MV = mat(d, h);        // as per maths this should be h x d
        if (MH.row == 0) MH = mat(d, h);        // as per maths this should be h x d
        params = hor.params + ver.params + 
                 // Mat                       eh        Kdotq                       tokforblock                     
                 (4*static_cast<size_t>(h)*d) + d + (static_cast<size_t>(n)*n) + (static_cast<size_t>(n)*d) + (2*static_cast<size_t>(n)*h);
    }
    else {
        if (qkCache.row == 0) qkCache = mat(d, d);      // as per maths this is MQ x MK^T, but here it would be MQ^T x MK
        if (khCache.row == 0) khCache = mat(d, d);      // as per maths this is MK x MH, but here it would be MK^T x MH^T
        if (qvCache.row == 0) qvCache = mat(d, d);      // as per maths this is MQ x MV, but here it would be MQ^T x MV^T
        params = hor.params + ver.params + (3*static_cast<size_t>(d)*d) + d + (static_cast<size_t>(n)*n) + (static_cast<size_t>(n)*d);
    }
    // std::cout << "ATTENTION constructed." << std::endl;
}

/**
 * @brief Constructor for incomplete attention from name - NO OpenCL
 * @param inAtt attention block name
 * @param n context window
 * @param d embedding dimension
 * @param h feature dimension = n
 * @param l layers of mlp
 * @param isSelf self attention = 1, else 0 for cross attention
 * @param trainMode in training = 1, else 0 for inference
 */
attention::attention(const std::string& inAtt, int n, int d, int h, int l, bool isSelf, bool trainMode, float& learning) :
    isSelfAttention(isSelf), inTraining(trainMode), tokenCount(0),
    hor(inAtt + "H", std::vector<unsigned int>(l, d), EPOCHS, learning),
    ver(inAtt + "V", std::vector<unsigned int>(l, d), EPOCHS, learning),
    h(d, 0.0f), v(d, 0.0f), EH(d, 0)
{
    if (n <= 0 || d <= 0 || h <= 0 || l <= 0) {
        throw std::invalid_argument("Attention dimensions must be positive.");
    }

    std::string ina = inAtt + "EV";         EV = mat(ina, n, d);
    ina = inAtt + "KdotQ";      KdotQ = mat(ina, n, n);
    if (trainMode == 1) {
        ina = inAtt + "K";      K = mat(ina, n, h);
        ina = inAtt + "Q";      Q = mat(ina, n, h);
        ina = inAtt + "MQ";     MQ = mat(ina, h, d);
        ina = inAtt + "MK";     MK = mat(ina, h, d);
        ina = inAtt + "MV";     MV = mat(ina, d, h);
        ina = inAtt + "MH";     MH = mat(ina, d, h);
        params = hor.params + ver.params + (4*static_cast<size_t>(h)*d) + d + (static_cast<size_t>(n)*n) + (static_cast<size_t>(n)*d) + (2*static_cast<size_t>(n)*h);
    }
    else {
        ina = inAtt + "qk";     qkCache = mat(ina, d, d);
        ina = inAtt + "kh";     khCache = mat(ina, d, d);
        ina = inAtt + "qv";     qvCache = mat(ina, d, d);
        params = hor.params + ver.params + (3*static_cast<size_t>(d)*d) + d + (static_cast<size_t>(n)*n) + (static_cast<size_t>(n)*d);
    }
    // std::cout << "ATTENTION with filename " << inAtt << " constructed." << std::endl;
}

#else // USE_CL is defined

#if defined(_WIN64)
    #include <CL/cl.hpp>
#elif defined(__linux__)
    #include <CL/opencl.hpp>
#endif

/**
 * @brief Constructor for incomplete attention - WITH OpenCL
 * @param context Reference to the shared OpenCL context.
 * @param n context window
 * @param d embedding dimension
 * @param h feature dimension = n
 * @param l layers of mlp
 * @param isSelf self attention = 1, else 0 for cross attention
 * @param trainMode in training = 1, else 0 for inference
 */
attention::attention(OpenCLContext& context, int n, int d, int h, int l, bool isSelf, bool trainMode, float& learning) :
    clcontext(context), isSelfAttention(isSelf), inTraining(trainMode),
    tokenCount(0), EV(n, d), KdotQ(n, n), h(d, 0.0f), v(d, 0.0f), EH(d, 0),
    ver(context, std::vector<unsigned int>(l, d), EPOCHS, learning),
    hor(context, std::vector<unsigned int>(l, d), EPOCHS, learning),
    MQ(trainMode ? h : 0, trainMode ? d : 0), MK(trainMode ? h : 0, trainMode ? d : 0),
    MV(trainMode ? d : 0, trainMode ? h : 0), MH(trainMode ? d : 0, trainMode ? h : 0),
    qkCache(trainMode ? 0 : d, trainMode ? 0 : d), qvCache(trainMode ? 0 : d, trainMode ? 0 : d),
    khCache(trainMode ? 0 : d, trainMode ? 0 : d)
{
    if (n <= 0 || d <= 0 || h <= 0 || l <= 0) {
        throw std::invalid_argument("Attention dimensions must be positive.");
    }
    cl_int cl_err;
    size_t ev_buffer_size = static_cast<size_t>(CONTEXT_WIN) * d * sizeof(float);
    if (ev_buffer_size == 0) {
        throw std::runtime_error("Calculated EV buffer size is zero in attention constructor.");
    }
    d_EV = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, ev_buffer_size, nullptr, &cl_err);
    CL_CHECK(cl_err); // Your CL_CHECK macro will handle the error

    if (trainMode == 1) {
        K = mat(n, h);
        Q = mat(n, h);
        if (MQ.row == 0) MQ = mat(h, d);        // as per maths this should be d x h
        if (MK.row == 0) MK = mat(h, d);        // as per maths this should be d x h
        if (MV.row == 0) MV = mat(d, h);        // as per maths this should be h x d
        if (MH.row == 0) MH = mat(d, h);        // as per maths this should be h x d
        params = hor.params + ver.params + 
                 (4*static_cast<size_t>(h)*d) + d + (static_cast<size_t>(n)*n) 
                 + (static_cast<size_t>(n)*d) + (2*static_cast<size_t>(n)*h);
    }
    else {
        if (qkCache.row == 0) qkCache = mat(d, d);      // as per maths this is MQ x MK^T, but here it would be MQ^T x MK
        if (khCache.row == 0) khCache = mat(d, d);      // as per maths this is MK x MH, but here it would be MK^T x MH^T
        if (qvCache.row == 0) qvCache = mat(d, d);      // as per maths this is MQ x MV, but here it would be MQ^T x MV^T
        params = hor.params + ver.params + (3*d*d) + d + (n*n) + (n*d);
    }
    // std::cout << "ATTENTION constructed with OpenCL -> " << params << std::endl;
}

/**
 * @brief Constructor for incomplete attention - WITH OpenCL
 * @param context Reference to the shared OpenCL context.
 * @param inAtt attention block name
 * @param n context window
 * @param d embedding dimension
 * @param h feature dimension = n
 * @param l layers of mlp
 * @param isSelf self attention = 1, else 0 for cross attention
 * @param trainMode in training = 1, else 0 for inference
 */
attention::attention(OpenCLContext& context, const std::string& inAtt, int n, int d, int h, int l, bool isSelf, bool trainMode, float& learning) :
    clcontext(context), isSelfAttention(isSelf), inTraining(trainMode), tokenCount(0),
    hor(context, inAtt + "H", std::vector<unsigned int>(l, d), EPOCHS, learning),
    ver(context, inAtt + "V", std::vector<unsigned int>(l, d), EPOCHS, learning),
    h(d, 0.0f), v(d, 0.0f), EH(d, 0)
{
    if (n <= 0 || d <= 0 || h <= 0 || l <= 0) {
        throw std::invalid_argument("Attention dimensions must be positive.");
    }
    cl_int cl_err;
    size_t ev_buffer_size = static_cast<size_t>(CONTEXT_WIN) * d * sizeof(float);
    if (ev_buffer_size == 0) {
        throw std::runtime_error("Calculated EV buffer size is zero in attention constructor.");
    }
    d_EV = cl::Buffer(clcontext.context, CL_MEM_READ_WRITE, ev_buffer_size, nullptr, &cl_err);
    CL_CHECK(cl_err);

    // modelName_blockName_A_i_j_ = ina
    std::string ina;
    ina = inAtt + "EV";         EV = mat(ina, n, d);
    ina = inAtt + "KdotQ";      KdotQ = mat(ina, n, n);
    if (trainMode == 1) {
        ina = inAtt + "K";      K = mat(ina, n, h);
        ina = inAtt + "Q";      Q = mat(ina, n, h);
        ina = inAtt + "MQ";      MQ = mat(ina, h, d);     // h row x d col
        ina = inAtt + "MK";      MK = mat(ina, h, d);     // h row x d col
        ina = inAtt + "MV";      MV = mat(ina, d, h);     // d row x h col
        ina = inAtt + "MH";      MH = mat(ina, d, h);     // d row x h col
        params = hor.params + ver.params
                 + (4*static_cast<size_t>(h)*d)
                 + d
                 + (static_cast<size_t>(n)*n)
                 + (static_cast<size_t>(n)*d)
                 + (2*static_cast<size_t>(n)*h);
    }
    else {
        ina = inAtt + "qk";      qkCache = mat(ina, d, d);
        ina = inAtt + "kh";      khCache = mat(ina, d, d);
        ina = inAtt + "qv";      qvCache = mat(ina, d, d);
        params = hor.params + ver.params
                 + (3*d*d) + d 
                 + (n*n) + (n*d);
    }
    // std::cout << "ATTENTION with filename " << inAtt << " constructed with OpenCL -> " << params << std::endl;
}

#endif // USE_CL

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
    std::fill(h.begin(), h.end(), 0.0f);
    std::fill(v.begin(), v.end(), 0.0f);

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
