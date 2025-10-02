#include "include/model.hpp"
#include <neural.hpp>
#include <fstream>
#include <iostream>
#include <cstdio>
#include <string>
#include <filesystem>
#include <limits>
#include <stdexcept>

// take sequence1 as input
void model::takeInput() {
    bool sortIt = 0;
    std::getline(std::cin, userSequence1);
    // tokenise sentence into words and punctuations
    std::vector<std::vector<float>> sequence1Embed;
    
    tokenize_with_numbers(userSequence1, tinput, sortIt);
    T.sequence1Count = tinput.size();

    // Write the user sequence1 to the chat log file
    if (chat != nullptr) {
        // Check if the file pointer is valid
        fprintf(chat, "Sequence1: %s\n", userSequence1.c_str());
        fflush(chat);
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
            T.blocks[0].b[i][j].clearValues();
        }
    }
    T.blocks[0].clearValues();

    // Define the standard path for the active chat log
    const std::string activeChatPath = "./active_chat.txt";

    // Close any existing chat file first
    if (chat != nullptr) {
        fclose(chat);
        chat = nullptr;
        // Don't clear currentChatLogPath here, as we might be overwriting the same file
    }

    // Set the current path
    currentChatLogPath = activeChatPath;

    // Open the new chat log file at the standard path, overwriting if it exists
#if defined(_WIN64)
    errno_t err = fopen_s(&chat, currentChatLogPath.c_str(), "w"); // "w" truncates/creates
    if (err != 0 || chat == nullptr) {
        // Consider logging the error code 'err' for Windows
        throw std::runtime_error("Failed to open new chat log file: " + currentChatLogPath + " (Error: " + std::to_string(err) + ")");
    }
#else // POSIX systems (Linux, macOS, etc.)
    chat = fopen(currentChatLogPath.c_str(), "w"); // "w" truncates/creates
    if (chat == nullptr) {
        // On POSIX, errno is set by fopen on failure.
        // #include <cstring> for strerror
        // #include <cerrno> for errno
        throw std::runtime_error("Failed to open new chat log file: " + currentChatLogPath + " (Error: " + strerror(errno) + ")");
    }
#endif
    else {
        fprintf(chat, "--- New Chat Session Started ---\n");
        fflush(chat);
    }
    T.clearValues();
}


// end chat and clear all the values, exit transformer
void model::endChat() {
    for(int k = 0; k < m; k++) {
        for(int i = 0; i < x; i++) {
            for(int j = 0; j < y; j++) {
                T.blocks[k].b[i][j].clearValues();
            }
        }
        T.blocks[k].clearValues();
    }
    T.clearValues();
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

    if (chat == nullptr || currentChatLogPath.empty()) {
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
    fclose(chat);
    chat = nullptr; // Set handle to null after closing

    // 2. Rename/Move the chat log file
    try {
        std::filesystem::create_directories(destDir); // Ensure destination directory exists
        std::filesystem::rename(currentChatLogPath, destinationPath);
        std::cout << "Chat saved successfully to: " << destinationPath.string() << std::endl;
        currentChatLogPath.clear(); // Clear the path as the file has been moved
    }
    catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error saving chat: Failed to move file from '"
                  << currentChatLogPath << "' to '" << destinationPath.string()
                  << "'. Reason: " << e.what() << std::endl;
    }
}
