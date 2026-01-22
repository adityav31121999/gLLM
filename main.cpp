// In main.cpp
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <iomanip>
#include "gllm.h"

// main function
int main(int argc, char *argv[])
{
    std::cout << "-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-" << std::endl;
    std::cout << "---------------------THIS IS gLLM PREVIEW: 0.1.5.0---------------------" << std::endl;
    std::cout << "-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-" << std::endl;

    try {
        std::cout << "Current working directory: " << std::filesystem::current_path() << std::endl;
    }
    catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error getting CWD: " << e.what() << std::endl;
    }

// set variables
    #ifdef __linux__
        std::string path2train = "/home/adi23444/code/train";
        std::string path2token = "/home/adi23444/code/train/token";
    #else
        std::string path2train = "D:/train";            // training directory
        std::string path2token = "D:/train/token";      // tokeniser data directory
    #endif

    int m = 2, x = NUMBER_OF_PA, y = NUMBER_OF_HEADS;
    int n = CONTEXT_WIN, d = EMBEDDING, matheight = CONTEXT_WIN;
    int layers = LAYERS_MLP;
    bool isSelf = 1, toTrain = 1, contextualise = 0, batchTraining = 0;
    float learning = 0.001f, lambda_l1 = 0.0001, lambda_l2 = 0.0016;

// train directory, create if not exists
    try {
        if (!std::filesystem::exists(path2train)) {
            std::cout << "Creating training directory: " << path2train << std::endl;
            std::filesystem::create_directories(path2train);
        }
        if (!std::filesystem::exists(path2token)) {
            std::cout << "Creating tokeniser data directory: " << path2token << std::endl;
            std::filesystem::create_directories(path2token);
        }
    }
    catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error creating directory: " << e.what() << std::endl;
        return 1;
    }

// training
    try {
    // file paths for
    #ifdef __linux__
        std::string path2Folder = "/home/adi23444/code/train";
    #else
        std::string path2Folder = "D:/train";
    #endif

    #ifdef USE_CL

        std::cout << "Using OpenCL for parallel execution" << std::endl;
        OpenCLContext clContext(kernelSourceFiles, kernelNames);
        model MODEL(clContext, path2train, path2token, m, x, y, n, d, matheight, layers, learning, 
                    lambda_l1, lambda_l2, isSelf, toTrain, contextualise, batchTraining);

    #elif USE_CU || USE_CPU

        #ifdef USE_CU
            std::cout << "Using CUDA for parallel execution" << std::endl;
        #elif USE_CPU
            std::cout << "Using CPU for Sequential/Multi-threaded Parallel execution" << std::endl;
        #endif
        model MODEL(path2train, path2token, m, x, y, n, d, matheight, layers, learning, 
                    lambda_l1, lambda_l2, isSelf, toTrain, contextualise, batchTraining);

    #endif

        // Initialize model weights
        float f1 = -0.02f, f2 = 0.02f;
        MODEL.T.blocks[0].randomValuesForBlock(f1, f2, 1);
        MODEL.setTokens2Transformer();
        std::cout << "-------------------- Training Model on Sequences ----------------------" << std::endl;
        std::cout << "Lambdas: L1 - " << lambda_l1 << ", L2 - " << lambda_l2 << ", Maximum Gradient Clip: " << MAX_GRAD_CLIP << std::endl;
        std::cout << "Learning Rate-> initial: " << learning << ", MAX: " << LEARNING_MAX << ", MIN: " << LEARNING_MIN << std::endl;
        std::cout << "Contextualised or Static training: " << contextualise << std::endl;
        MODEL.currentChatLogPath = path2train + "/chatlog.txt";
        std::string txtPath = path2train + "/txt" + "/group_100.txt";
        MODEL.trainSequence(txtPath, CONTEXT_WIN, contextualise);
        std::cout << "Clearing values for all blocks..." << std::endl;
        MODEL.T.clearValues();
    }
    catch (const std::exception& e) {
        std::cerr << "--------------------------------:ERROR:--------------------------------" << std::endl;
        std::cerr << "-> An ERROR occurred (MAIN):" << std::endl;
        std::cerr << "  - Exception type:\t" << typeid(e).name() << std::endl;
        std::cerr << "  - Error message:\t" << e.what() << std::endl;
        std::cerr << "-----------------------------------------------------------------------" << std::endl;
        return 1;
    }

    // END FOR TRAINING
    std::cout << "-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-" << std::endl;
    std::cout << "---------------------------------:END:---------------------------------" << std::endl;
    std::cout << "-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-" << std::endl;
    return 0;
}