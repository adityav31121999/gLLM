
#include "include/model.hpp"
#include <neural.hpp>
#include <fstream>
#include <iostream>
#include <cstdio>
#include <string>
#include <filesystem> // Required for std::filesystem::rename and path operations
#include <limits>     // Required for std::numeric_limits
#include <stdexcept>  // Required for std::runtime_error

// take prompt as input
void model::takeInput() {
    std::getline(std::cin, userPrompt);
    // tokenise sentence into words and punctuations
    
    this->T.promptCount = this->T.tokenise(userPrompt, this->T.mTokens, this->T.currentTokenCount) + 1;
    if(this->T.promptCount == 0) {
        std::cerr<< "Prompt cannot be empty" << std::endl;
        return;
    }

    // Write the user prompt to the chat log file
    if (this->chat != nullptr) {
        // Check if the file pointer is valid
        fprintf(this->chat, "Prompt: %s\n", userPrompt.c_str());
        fflush(this->chat); // Ensure it's written immediately (optional but good for logging)
    }
    else {
        // Optionally handle the case where the file isn't open
        std::cerr << "Warning: Chat log file is not open." << std::endl;
    }
}

// for new chat clear and set all to 0
void model::newChat() {
    for(int i = 0; i < x; i++) {
        for(int j = 0; j < y; j++) {
            this->T.t[0].b[i][j].clearValues();
        }
    }
    this->T.t[0].clearValues();

    // Define the standard path for the active chat log
    const std::string activeChatPath = "./active_chat.txt";

    // Close any existing chat file first
    if (this->chat != nullptr) {
        fclose(this->chat);
        this->chat = nullptr;
        // Don't clear currentChatLogPath here, as we might be overwriting the same file
    }

    // Set the current path
    this->currentChatLogPath = activeChatPath;

    // Open the new chat log file at the standard path, overwriting if it exists
    errno_t err = fopen_s(&this->chat, this->currentChatLogPath.c_str(), "w"); // "w" truncates/creates
    if (err != 0 || this->chat == nullptr) {
        std::cerr << "Error: Could not open active chat log file: " << this->currentChatLogPath << std::endl;
        this->currentChatLogPath.clear(); // Clear path if open failed
    } else {
        fprintf(this->chat, "--- New Chat Session Started ---\n");
        fflush(this->chat);
    }
    this->T.clearValues();
}

// end chat and clear all the values, exit transformer
void model::endChat() {
    for(int k = 0; k < m; k++) {
        for(int i = 0; i < x; i++) {
            for(int j = 0; j < y; j++) {
                this->T.t[k].b[i][j].clearValues();
            }
        }
        this->T.t[k].clearValues();
    }
    this->T.clearValues();
}

// save chat in txt file
void model::saveChat() {
    std::cout << "Provide location and name to save chat: " << std::endl;
    std::cout << "Name the file: ";
    std::string name;
    std::cin >> name;
    std::cout << "\nWhere to save: ";
    std::string location;
    std::cin >> location;
    std::cin.ignore((std::numeric_limits<std::streamsize>::max)(), '\n'); // Consume trailing newline

    if (this->chat == nullptr || this->currentChatLogPath.empty()) {
        std::cerr << "Error: No active chat log to save." << std::endl;
        return;
    }

    // Construct the destination path
    std::filesystem::path destDir = location;
    std::filesystem::path destFile = name;
    // Ensure the filename has a .txt extension
    if (!destFile.has_extension() || destFile.extension() != ".txt") {
        destFile.replace_extension(".txt");
    }
    std::filesystem::path destinationPath = destDir / destFile;

    // 1. Close the current chat file to flush buffers
    fclose(this->chat);
    this->chat = nullptr; // Set handle to null after closing

    // 2. Rename/Move the chat log file
    try {
        std::filesystem::create_directories(destDir); // Ensure destination directory exists
        std::filesystem::rename(this->currentChatLogPath, destinationPath);
        std::cout << "Chat saved successfully to: " << destinationPath.string() << std::endl;
        this->currentChatLogPath.clear(); // Clear the path as the file has been moved
    }
    catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error saving chat: Failed to move file from '"
                  << this->currentChatLogPath << "' to '" << destinationPath.string()
                  << "'. Reason: " << e.what() << std::endl;
        // Attempt to reopen the original file? Or leave it closed?
        // For now, we leave chat and currentChatLogPath cleared/null.
    }
}
