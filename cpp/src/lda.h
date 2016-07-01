//
// Created by Dylan on 13/04/2016.
//

/*!
 * \file lda.h
 */

#ifndef LDA_LDA_H
#define LDA_LDA_H

#include <map>
#include <vector>
#include <set>
#include <string>
#include "dirichlet.h"

//! A document struct
/*!
    Represents a document in bag of words style.
 */
struct document {
    std::map<int,int> wordCounts;   /*!< Map for the word counts. Maps id to count. */
    size_t count;                      /*!< The total number of words in the document. */
    size_t uniqueCount;                /*!< The number of unique words in the document. */
};

//! A corpus struct
/*!
    Represents a corpus of documents.
    \sa document
 */
struct doc_corpus {
    std::vector<document> docs; /*! vector of document structs. \sa document */
    size_t numTerms;               /*! the total number of unique terms in the corpus. */
    size_t numDocs;                /*! the total number of documents. */
};

//! settings struct for the lda model
struct lda_settings {
    lda_settings() : converged_threshold(1e-6), min_iterations(2),
                     max_iterations(100), inf_converged_threshold(1e-6),
                     inf_max_iterations(20), estimate_alpha(false),
                     alpha_update_interval(1){}

    double converged_threshold;     /*!< The convergence threshold used in training */
    int min_iterations;             /*!< Minimum number of iterations to train for */
    int max_iterations;             /*!< Maximum number of iterations to train for */
    double inf_converged_threshold; /*!< Document inference convergence threshold*/
    int inf_max_iterations;         /*!< Document inference max iterations*/
    bool estimate_alpha;            /*!< Whether to estimate alpha*/
    int alpha_update_interval;      /*!< interval to update alpha on*/
};

//! A sufficient statistics struct
/*!
    Convenience struct to hold the sufficient statistics to update alpha and beta (varGamma)
    \sa lda
 */
struct suff_stats {

    std::vector<std::vector<double>> classWord; /*!< 2d vector for the topic-word sufficient stats for beta.*/
    std::vector<double> classTotal;             /*!< vector (K) for the topic sufficient stats for  beta.*/

    std::vector<double> alpha_ss; /*!< Sufficient stats for alpha */
    size_t numDocs;    /*!< the number of documents include in the suff stats so far */
};

//! A latent dirichlet allocation model class
/*!
    A class for a latent dirichlet allocation model. Holds the model parameters.
    Can be trained on a corpus given a number of topics or loaded from a parameter file.
 */
class lda {

public:
    //! lda constructor
    lda(doc_corpus& corp, lda_settings settings);

    doc_corpus corpus;  /*!< document corpus for the lda */

    size_t numTopics;      /*!< number of topics */
    size_t numDocs;        /*!< total number of documents in the corpus. */
    size_t numTerms;       /*!< total number of terms(words) in the corpus. */

    // model parameters
    std::vector<std::vector<double>> logProbW;  /*!< the topic-word log prob (unnormalised p(w|z)) */
    dirichlet alpha;   /*!< the lda alpha parameter */

    // variational parameters
    std::vector<std::vector<double>> varGamma;          /*!< gamma latent dirichlet parameter */
    std::vector<std::vector<std::vector<double>>> phi;  /*!< phi latent dirichlet parameter */

    double likelihood;  /*!< the total log-likelihood for the corpus (I think - might be the lower bound) */

    //! Train the lda on the corpus given the number of topics.
    void train(int num_topics, alpha_settings a_settings);

private:
    //Settings
    //Training Settings
    double CONV_THRESHHOLD;     /*!< The convergence threshold used in training */
    int MIN_ITER;               /*!< Minimum number of iterations to train for */
    int MAX_ITER;               /*!< Maximum number of iterations to train for */
    //Document E-step Inference Settings
    double INF_CONV_THRESH;     /*!< Document inference convergence threshold*/
    int INF_MAX_ITER;           /*!< Document inference max iterations*/
    //Settings regarding alpha
    bool EST_ALPHA;
    int UPDATE_INTERVAL;

    //! randomly initialise the given sufficient statistics
    void randomSSInit(suff_stats& ss);
    //! zero initialise the given sufficient statistics
    void zeroSSInit(suff_stats& ss);

    //! Sets up alpha given the settings
    dirichlet setup_alpha(alpha_settings settings);

    //! get the maximum likelihood model from the sufficient statistics
    void mle(suff_stats& ss, bool optAlpha);

    //! perform the e-step of the EM algo on the given document
    double doc_e_step(document const& doc, suff_stats& ss, std::vector<double>& var_gamma,
                      std::vector<std::vector<double>>& phi);

    //! performs inference on the current document
    double inference(document const& doc, std::vector<double>& var_gamma,
                     std::vector<std::vector<double>>& phi);

    //! calculates the log likelihood for the current document
    double compute_likelihood(document const& doc, std::vector<double>& var_gamma,
                              std::vector<std::vector<double>>& phi);

};


#endif //LDA_LDA_H
