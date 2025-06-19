
#include "include/model.hpp"
#include <neural.hpp>
#include <fstream>
#include <iostream>
#include <cstdio>
#include <string>
#include <filesystem>
#include <limits>
#include <stdexcept>

// take prompt as input
void model::takeInput() {
    std::getline(std::cin, userPrompt);
    // tokenise sentence into words and punctuations
    std::vector<std::vector<float>> promptEmbed;
    
    tokenize_with_numbers(userPrompt, tinput);
    T.promptCount = tinput.size();

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
#if defined(_WIN64)
    errno_t err = fopen_s(&this->chat, this->currentChatLogPath.c_str(), "w"); // "w" truncates/creates
    if (err != 0 || this->chat == nullptr) {
        // Consider logging the error code 'err' for Windows
        throw std::runtime_error("Failed to open new chat log file: " + this->currentChatLogPath + " (Error: " + std::to_string(err) + ")");
    }
#else // POSIX systems (Linux, macOS, etc.)
    this->chat = fopen(this->currentChatLogPath.c_str(), "w"); // "w" truncates/creates
    if (this->chat == nullptr) {
        // On POSIX, errno is set by fopen on failure.
        // #include <cstring> for strerror
        // #include <cerrno> for errno
        throw std::runtime_error("Failed to open new chat log file: " + this->currentChatLogPath + " (Error: " + strerror(errno) + ")");
    }
#endif
    else {
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
    }
}
