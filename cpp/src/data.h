//
// Created by Dylan on 14/04/2016.
//

/*!
 \file data.h
 */

#ifndef LDA_DATA_H
#define LDA_DATA_H

#include <string>
#include <map>
#include <vector>

//! A document struct
/*!
    Represents a document in bag of words style.
 */
struct document {
    std::map<int,int> wordCounts;   /*!< Map for the word counts. Maps id to count. */
    int count;                      /*!< The total number of words in the document. */
    int uniqueCount;                /*!< The number of unique words in the document. */
};

//! A corpus struct
/*!
    Represents a corpus of documents.
    \sa document
 */
struct doc_corpus {
    std::vector<document> docs; /*! vector of document structs. \sa document */
    int numTerms;               /*! the total number of unique terms in the corpus. */
    int numDocs;                /*! the total number of documents. */
};

//! settings struct for the lda model
struct lda_settings {
    lda_settings() : converged_threshold(1e-4), min_iterations(2),
                     max_iterations(100), inf_converged_threshold(1e-6), inf_max_iterations(20){}

    double converged_threshold;     /*!< The convergance threshold used in training */
    int min_iterations;             /*!< Minimum number of iterations to train for */
    int max_iterations;             /*!< Maximum number of iterations to train for */
    double inf_converged_threshold; /*!< Document inference convergance threshold*/
    int inf_max_iterations;         /*!< Document inference max iterations*/
};

//! settings struct for alpha updates
struct alpha_settings {
    alpha_settings() : estimate_alpha(true), concentration(true), newton_threshold(1e-5),
                       max_iterations(1000), init_prec(1), init_s(100) {}

    bool estimate_alpha;     /*!< Whether to estimate alpha*/
    bool concentration;      /*!< Whether alpha should be the concentration parameter or the dirichlet mean*/
    double newton_threshold; /*!< threshold for newtons method*/
    int max_iterations;      /*!< Maximum number of iterations for alpha update*/
    double init_prec;        /*!< Initial value for the concentration parameter*/
    int init_s;              /*!< Initial value for the conc coeff when estimating*/
};

//! load a corpus from a file
doc_corpus load_corpus(std::string file_path);
//! split a string with the given delimiter
std::vector<std::string> split(std::string const& str, char delim);
//! load vocab from file
std::vector<std::string> load_vocab(std::string file_path);
//! load settings from file
void load_settings(std::string file_path, alpha_settings& alpha, lda_settings& lda);

#endif //LDA_DATA_H
