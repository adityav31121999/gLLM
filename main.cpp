// In main.cpp
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include "gllm.h"

// main function
int main(int argc, char *argv[])
{
    std::cout << "-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-" << std::endl;
    std::cout << "---------------------THIS IS gLLM PREVIEW: 0.0.0.1---------------------" << std::endl;
    std::cout << "-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-" << std::endl;
    try {
        std::cout << "Current working directory: " << std::filesystem::current_path() << std::endl;
    }
    catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error getting CWD: " << e.what() << std::endl;
    }

    std::string path2train = "D:/train";
    int m = NUMBER_OF_BLOCKS, x = NUMBER_OF_PA, y = NUMBER_OF_PA;
    int n = CONTEXT_WIN, d = EMBEDDING, matheight = MATHEIGHTS;
    int layers = LAYERS_MLP;
    float learning = 0.001f, lambda_L1 = 0.0001f, lambda_L2 = 0.0025f;
    bool isSelf = true, toTrain = true;

    try {
        if (!std::filesystem::exists(path2train)) {
            std::cout << "Creating training directory: " << path2train << std::endl;
            std::filesystem::create_directories(path2train);
        }
    }
    catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error creating directory " << path2train << ": " << e.what() << std::endl;
        return 1;
    }

    try {
// MINIMUM VALUE FOR UNIFORM DISTRIBUTION OF RANDOM VALUES FOR WEIGHTS MATRICES
        float f = learning*EMBEDDING/MATHEIGHTS;
        std::cout << "fmin = " << -1 * f << ", fmax = " << f << std::endl;
        // create(path2train);             // create discrete bins

// MODEL DECLARATION
    #ifdef USE_OPENCL
        std::cout << "Using OpenCL for parallel execution" << std::endl;
        OpenCLContext clContext(kernelSourceFiles, kernelNames);
        tokeniser TOKENISER(path2train, clContext);
        model MODEL(clContext, path2train, m, x, y, n, d, matheight, layers, learning, lambda_L1, lambda_L2, TOKENISER.getVocabularySize(), isSelf, toTrain);
        MODEL.clcontext = clContext;
    #elif USE_CUDA || USE_CPU
        #ifdef USE_CUDA
            std::cout << "Using CUDA for parallel execution" << std::endl;
        #elif USE_CPU
            std::cout << "Using CPU for Sequential/Multi-threaded Parallel execution" << std::endl;
        #endif
        tokeniser TOKENISER(path2train);
        model MODEL(path2train, m, x, y, n, d, matheight, layers, learning, lambda_L1, lambda_L2, TOKENISER.getVocabularySize(), isSelf, toTrain);
    #endif
        MODEL.currentChatLogPath = path2train + "/chatlog.txt";
        MODEL.TOK = TOKENISER;

// TRAINING PROCESS
        std::cout << "Training first block: " << std::endl;
        MODEL.T.t[0].randomValuesForBlock(-1*f, f);
        std::string txtPath = path2train + "/txt" + "/dict.txt";
        MODEL.trainBlockSentence(txtPath);
        std::cout << "Clearing values for all blocks..." << std::endl;
        MODEL.T.clearValues();
    }
    catch (const std::exception& e) {
        std::cerr << "--------------------------------:ERROR:--------------------------------" << std::endl;
        std::cerr << "->An exception occurred during initialization or execution:" << std::endl;
        std::cerr << "Exception type: \n\t->" << typeid(e).name() << std::endl;
        std::cerr << "Error message: \n\t->" << e.what() << std::endl;
        std::cerr << "-----------------------------------------------------------------------" << std::endl;
        return 1; // Indicate error
    }

    // END FOR TRAINING
    std::cout << "-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-" << std::endl;
    std::cout << "---------------------------------:END:---------------------------------" << std::endl;
    std::cout << "-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-" << std::endl;
    return 0;
}


/*
// TOKENISER INFERENCE DEMO
        std::string test_sentence = "This is a test sentence for christianity and its international relationships to see the new tokenizer in action. Hence, need more words to see whether it will work or not, if not rework the code logic and try again. This tokeniser is (BPE) is supercalifragilisticexpialidocious at the ludicrous speed. Ludicrous speed can be given by higher multiple of light speed which is 2.9 * 10^8 m/s. Let's see if this works!";
        std::vector<std::string> tokenized_sentence;
        std::cout << "Original: \"" << test_sentence << "\"" << std::endl;
        TOKENISER.splitSentence(test_sentence, tokenized_sentence);
        std::cout << "Tokenized: { \n";
        for (const auto& token : tokenized_sentence) {
            std::cout << "'" << token << "' ";
        }
        std::cout << "\n}" << std::endl;
        std::cout << "Total tokens after tokenisation: " << tokenized_sentence.size() << std::endl;
        std::cout << "Tokenisation example done, moving to train block/model." << std::endl;

*/