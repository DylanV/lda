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
#include "data.h"


//! A sufficient statistics struct
/*!
    Convenience struct to hold the sufficient statistics to update alpha and beta (varGamma)
    \sa lda
 */
struct suff_stats {

    std::vector<std::vector<double>> classWord; /*!< 2d vector for the topic-word sufficient stats for beta.*/
    std::vector<double> classTotal;             /*!< vector (K) for the topic sufficient stats for  beta.*/

    double alphaSS; /*!< the (singular) alpha sufficient statistics */
    int numDocs;    /*!< the number of documents include in the suff stats so far */
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

    int numTopics;      /*!< number of topics */
    int numDocs;        /*!< total number of documents in the corpus. */
    int numTerms;       /*!< total number of terms(words) in the corpus. */

    std::vector<std::vector<double>> logProbW;  /*!< the topic-word log prob (unnormalised beta) */
    dirichlet alpha;   /*!< the lda alpha parameter */

    double likelihood;  /*!< the total log-likelihood for the corpus */

    //! Train the lda on the corpus given the number of topics.
    void train(int num_topics, alpha_settings a_settings);

    //! Write the dirichlet parameters to files in the given folder
    void writeParams(std::string folder_path);

    //! load the lda model parameters from the parameter files in the given folder.
    void loadFromParams(std::string folder_path);

private:
    //Training Settings
    double CONV_THRESHHOLD;      /*!< The convergance threshold used in training */
    int MIN_ITER;                 /*!< Minimum number of iterations to train for */
    int MAX_ITER;               /*!< Maximum number of iterations to train for */
    //Document E-step Inference Settings
    double INF_CONV_THRESH;  /*!< Document inference convergance threshold*/
    int INF_MAX_ITER;            /*!< Document inference max iterations*/
    //Settings regarding alpha
    bool EST_ALPHA;
    bool FULL_ALPHA;

    std::vector<double> alpha_ss_vec;

    std::vector<std::vector<double>> varGamma;          /*!< gamma latent dirichlet parameter */
    std::vector<std::vector<std::vector<double>>> phi;  /*!< phi latent dirichlet parameter */

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

    //! normalise logProbW and writes to file
    void writeBetaToFile(std::string folder_path);
    //! writes alpha to file
    void writeAlphaToFile(std::string folder_path);
    //! write gamma to file
    void writeGammaToFile(std::string folder_path);

    //! loads beta to logProbW from file
    std::vector<std::vector<double>> loadBetaFromFile(std::string file_path);
};


#endif //LDA_LDA_H
