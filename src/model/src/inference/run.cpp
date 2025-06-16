
// model.cpp: implementation of Model class
#include "include/model.hpp"
#include <iostream>
#include <stdexcept>
#include <filesystem>
#include <neural.hpp>

#ifdef USE_CUDA
    #define CUDA_CHECK(call) do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            throw std::runtime_error(cudaGetErrorString(err)); \
        } \
    } while (0)
#endif

/**
 * @brief run model for conversation
 */
void model::runModel(const std::string& binDirectory)
{
    std::cout << "You are now running the model " << info.modelName << std::endl;
    bool savechat;      // 1 to save chat
    bool newchat;       // 1 for new chat, 0 for endchat
    // take input
    T.currentTokenCount = 0;
    // Construct file paths once, as 'dir' is constant here.
    std::filesystem::path baseDirFs(binDirectory);
    std::string path_qk_bin = (baseDirFs / "QK.bin").string();
    std::string path_qv_bin = (baseDirFs / "QV.bin").string();
    std::string path_kh_bin = (baseDirFs / "KH.bin").string();
    std::string path_hor_bin = (baseDirFs / "HOR.bin").string();
    std::string path_ver_bin = (baseDirFs / "VER.bin").string();

    while(1) {
        while(T.currentTokenCount < FULL_CONTEXT) {
        std::cout << "Enter prompt: "; // Use 'total' member variable
        takeInput();
        if (this->chat != nullptr) {
            // Check if the file pointer is valid
            fprintf(this->chat, "Response:\n");
            fflush(this->chat); // Ensure it's written immediately (optional but good for logging)
        }
        std::vector<float> pValues(EMBEDDING, 0.0f);
        for(int i = 0; i < tinput.size(); i++) {
            // get embeddings for tokens
            T.getEmbedding(tinput[i], pValues);
            setRow(T.tokenEmbed, T.currentTokenCount+i, pValues);
        }
        #ifdef USE_CUDA
        // use cuda run function from transformer
            T.cuRun();
        #elif USE_OPENCL
        // use cl run function from transformer
            T.clRun();
        #elif USE_CPU
        // use cpp run function from transformer
            T.run();
        #endif
            if (this->chat != nullptr) {
                // Check if the file pointer is valid
                fprintf(this->chat, "\n");
                fflush(this->chat);
            }
        }
        // save model
        std::cout << "SAVE CHAT IN FILE (1 for save): ";
        std::cin >> savechat;
        if(savechat == 1) { saveChat(); }
        std::cout << "NEW CHAT (1) or END (0): ";
        std::cin >> newchat;
        // continue chatting or end chat
        if(newchat == 1) {
            newChat();
        }
        else {
            endChat();
            break;
        }
    }
}
