// model.cpp: implementation of Model class
#include "include/model.hpp"
#include <iostream>
#include <stdexcept>
#include <filesystem>
#include <neural.hpp>

/**
 * @brief run model for conversation
 * @param binDirectory path to binary files directory
 */
void model::runModel(const std::string& binDirectory)
{
    std::cout << "You are now running the model " << info.modelName << std::endl;
    bool savechat;      // 1 to save chat
    bool newchat;       // 1 for new chat, 0 for endchat
    bool contextChat;   // 0 for static and 1 for contextualised chat
    std::cout << "TYPE OF CHAT STATIC(0) or CONTEXT(1): ";
    std::cin >> contextChat;
    if (contextChat == 0) {
        std::cout << " -> Chat is static, using embedding matrix for predictions.";
        if(!T.deEmbeddings.mapped_data) {
            throw std::runtime_error ("Embedding matrix not available or loaded for static chat!");
        }
    }
    else {
        std::cout << " -> Chat is contextualised, using deEmbedding matrix for predictions.";
        if(!T.deEmbeddings.mapped_data) {
            std::cout << "De-embedding matrix not available or loaded for contextualised chat..." << std::endl;
            bool shift;
            std::cout << "Should we shift to static chat (0) or end the chat (1): ";
            std::cin >> shift;
            if (shift == 1) {
                std::cout << "Ending chat..." << std::endl;
                endChat();
                throw std::runtime_error("De-embedding matrix not loaded or available, hence ended chat, try again.");
            }
            else {
                std::cout << "Shifting to static chat " << std::endl;
                contextChat = 0;
            }
        }
    }

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
            std::cout << "Enter sequence1: ";
            takeInput();
            if (chat != nullptr) {
                // Check if the file pointer is valid
                fprintf(chat, "Sequence2:\n");
                fflush(chat); // Ensure it's written immediately (optional but good for logging)
            }
            std::vector<float> pValues(EMBEDDING, 0.0f);
            for(int i = 0; i < tinput.size(); i++) {
                // get embeddings for tokens
                pValues = TOK.getEmbeddingForToken(tinput[i]) + T.positionalEmbeddings(T.currentTokenCount+i, EMBEDDING);
                setRow(T.tokenEmbed, T.currentTokenCount+i, pValues);
            }
            T.sequence1Count = tinput.size();
            #ifdef USE_CU
            // use cuda run function from transformer
                T.cuRun(contextChat);
            #elif USE_CL
            // use cl run function from transformer
                T.clRun(contextChat);
            #elif USE_CPU
            // use cpp run function from transformer
                T.run(contextChat);
            #endif
            if (chat != nullptr) {
                // Check if the file pointer is valid
                fprintf(chat, "\n");
                fflush(chat);
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