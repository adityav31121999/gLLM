
#include "include/model.hpp"
#include "include/model_fs.hpp"

#include <neural.hpp>
#include <iostream>
#include <cstdio>
#include <string>

// take prompt as input
void model::takeInput() {
    std::cout << "Enter prompt (Maximum Allowable tokens " << total/4 << "): "; // Use 'total' member variable
    std::getline(std::cin, userPrompt);
    // tokenise sentence into words and punctuations
    this->T.promptCount = this->T.tokenise(userPrompt, this->T.mTokens, this->T.currentTokenCount) + 1;
    if(this->T.promptCount == 0) {
        std::cerr<< "Prompt cannot be empty" << std::endl;
        return;
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
    // 
}

// load chat from txt file
void model::loadChat() {
    // 
}
