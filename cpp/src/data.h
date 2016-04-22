//
// Created by Dylan on 14/04/2016.
//

/*!
 \file data.h
 */

#ifndef LDA_DATA_H
#define LDA_DATA_H

#include "lda.h"
#include <string>

//! load a corpus from a file
doc_corpus load_corpus(std::string file_path);
//! split a string with the given delimiter
std::vector<std::string> split(std::string const& str, char delim);
//! load vocab from file
std::vector<std::string> load_vocab(std::string file_path);

#endif //LDA_DATA_H
