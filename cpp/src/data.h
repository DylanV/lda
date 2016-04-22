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

struct lda_settings {
    double converged_threshold = 1e-4;
    int min_iterations = 2;
    int max_iterations = 100;
    double inference_converged_threshold = 1e-6;
    int inference_max_iterations = 20;
};

struct alpha_settings {
    bool singular = true;
    double newton_threshold = 1e-5;
    int max_iterations = 1000;
};

struct file_settings {
    char param_sep = ' ';
    char param_nl = '\n';
};

//! load a corpus from a file
doc_corpus load_corpus(std::string file_path);
//! split a string with the given delimiter
std::vector<std::string> split(std::string const& str, char delim);
//! load vocab from file
std::vector<std::string> load_vocab(std::string file_path);

#endif //LDA_DATA_H
