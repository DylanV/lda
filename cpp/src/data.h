//
// Created by Dylan on 14/04/2016.
//

#ifndef LDA_DATA_H
#define LDA_DATA_H

#include "lda.h"
#include <string>

doc_corpus load_corpus(std::string file_path);
std::vector<std::string> split(std::string const& str, char delim);
std::vector<std::string> load_vocab(std::string file_path);

#endif //LDA_DATA_H
