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

//! A sufficient statistics struct
/*!
    Convenience struct to hold the sufficient statistics to update alpha and beta (varGamma)
    \sa lda
 */
struct suff_stats {

    std::vector<std::vector<double>> classWord; /*!< 2d vector for the topic-word sufficient stats for beta.*/
    std::vector<double> classTotal;             /*!< vector (K) for the topic sufficient stats for  beta.*/

    std::vector<double> alphaSS; /*!< the alpha sufficient statistics */
    int numDocs;    /*!< the number of documents include in the suff stats so far */
};

//! A latent dirichlet allocation model class
/*!
    A class for a latent dirichlet allocation model. Holds the model parameters.
    Can be trained on a corpus given a number of topics or loaded from a parameter file.
    http://videolectures.net/mlss09uk_blei_tm/
 */
class lda {

public:
    //! lda constructor
    lda(doc_corpus& corp);

    doc_corpus corpus;  /*!< document corpus for the lda */

    int numTopics;      /*!< number of topics */
    int numDocs;        /*!< total number of documents in the corpus. */
    int numTerms;       /*!< total number of terms(words) in the corpus. */

    std::vector<std::vector<double>> logProbW;  /*!< the topic-word log prob */
    std::vector<double> alpha;   /*!< the lda alpha parameter */

    double likelihood;  /*!< the total log-likelihood for the corpus */

    //! Train the lda on the corpus given the number of topics.
    void train(int num_topics);

    //! Write the dirichlet parameters to files in the given folder
    void writeParams(std::string folder_path);

    //! load the lda model parameters from the parameter files in the given folder.
    void loadFromParams(std::string folder_path);

private:
    std::vector<std::vector<double>> varGamma;          /*!< gamma latent dirichlet parameter */
    std::vector<std::vector<std::vector<double>>> phi;  /*!< phi latent dirichlet parameter */

    //! randomly initialise the given sufficient statistics
    void randomSSInit(suff_stats& ss);
    //! zero initialise the given sufficient statistics
    void zeroSSInit(suff_stats& ss);

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
