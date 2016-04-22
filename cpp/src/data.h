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

//! settings struct for the lda model
struct lda_settings {
    double converged_threshold = 1e-4;      /*!< The convergance threshold used in training */
    int min_iterations = 2;                 /*!< Minimum number of iterations to train for */
    int max_iterations = 100;               /*!< Maximum number of iterations to train for */
    double inf_converged_threshold = 1e-6;  /*!< Document inference convergance threshold*/
    int inf_max_iterations = 20;            /*!< Document inference max iterations*/
};

//! settings struct for alpha updates
struct alpha_settings {
    bool singular = true;           /*!< whether alpha is singuler or a vector*/
    double newton_threshold = 1e-5; /*!< threshold for newtons method*/
    int max_iterations = 1000;      /*!< Maximum number of iterations for alpha update*/
};

//! settings struct for file writing and reading for params mostly
struct file_settings {
    char param_sep = ' ';   /*!< Item seperator on line*/
    char param_nl = '\n';   /*!< Line seperator*/
};

//! load a corpus from a file
doc_corpus load_corpus(std::string file_path);
//! split a string with the given delimiter
std::vector<std::string> split(std::string const& str, char delim);
//! load vocab from file
std::vector<std::string> load_vocab(std::string file_path);

#endif //LDA_DATA_H
