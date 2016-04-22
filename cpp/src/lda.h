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

//! A sufficient statistices struct
/*!
    Convenience struct to hold the sufficient statistics to update alpha and beta (varGamma)
    \sa lda
 */
struct suff_stats {
    /*! 2d vector (KxW) for the topic-word sufficient stats for beta.*/
    std::vector<std::vector<double>> classWord;
    /*! vector (K) for the topic sufficient stats for  beta.*/
    std::vector<double> classTotal;

    double alphaSS; /*!< the (singular) alpha sufficient statistics */
    int numDocs; /*!< the number of documents include in the suff stats so far */
};

//! A latent dirichlet allocation model class
/*!
    A class for a latent dirchlet allocation model. Holds the model parameters.
    Can be trained on a corpus given a number of topics or loaded from a parameter file.
 */

class lda {

public:
    //! lda constructor
    /*!
     The constructor for the lda class. Accepts a reference to the document corpus,
     \param corp a reference to the document corpus for the lda.
     */
    lda(doc_corpus& corp);


    doc_corpus corpus; /*!< document corpus for the lda */

    int numTopics;  /*!< number of topics */
    int numDocs;    /*!< total number of documents in the corpus. */
    int numTerms;   /*!< total number of terms(words) in the corpus. */

    std::vector<std::vector<double>> logProbW;  /*!< the topic-word log prob (unnormalised beta) */
    double alpha;   /*!< the lda alpha parameter */

    double likelihood; /*!< the total log-likelihood for the corpus

    //! Train the lda on the corpus given the number of topics.
    /*!
        Perform variational inference on the corpus to get the dirichlet parameters
        (logProbW->beta and alpha) for the corpus.
        \param num_topics an int for the number of topics to train on
     */
    void train(int num_topics);

    //! Write the dirichlet parameters to files in the given folder
    /*!
        Writes the lda alpha, beta and gamma dirchlet parameters to files.
        File contains the shape of the parameter in the first line. Followed by the parameter line seperated
        by vector indice if 2d vector and space seperated for 1d vector. So a 2d beta parameter will have each
        space seperated topic distribtion on a new line.
        /param folder_path the path to the folder where the parameter files will be written.
     */
    void writeParams(std::string folder_path);

    //! load the lda model parameters from the parameter files in the given folder.
    /*!
        /param folder_path the path to the folder containing the parameter files
        /sa writeParams()
     */
    void loadFromParams(std::string folder_path);

private:
    std::vector<std::vector<double>> varGamma; /*!< gamma latent dirichet parameter */
    std::vector<std::vector<std::vector<double>>> phi; /*!< phi latent dirichlet parameter */

    //! randomly initiliase the given sufficient statistics
    void randomSSInit(suff_stats& ss);
    //! zero initiliase the given sufficient statistics
    void zeroSSInit(suff_stats& ss);

    //! get the maximum likelihood model from the sufficient statistics
    /*!
        Calculates a maximum likelihood model for the lda given the sufficient stats.
        Newton-Raphson update on alpha is optional. optAlpha default is true.
        \param ss a reference to the sufficeint statistics struct for the lda
        \param optAlpha set to true if alpha should be inferreded
        \sa suff_stats
     */
    void mle(suff_stats& ss, bool optAlpha);

    //! perform the e-step of the EM algo on the given document
    /*!
        Calls inference to update the latent parameters (gamma, phi) for the current document \sa inference().
        Then updates the sufficient statistics for the lda.
        \param doc reference to current document \sa document
        \param ss reference to the model sufficient statistics \sa suff_stats
        \param var_gamma reference to the current document topic distribtion parameter \sa varGamma
        \param phi reference to the current document topic-word distribtion \sa phi
        \return the log likelihood for the document
     */
    double doc_e_step(document const& doc, suff_stats& ss, std::vector<double>& var_gamma,
                      std::vector<std::vector<double>>& phi);

    //! performs inference on the current document
    /*!
        Performs inference to update the latent dirchlet factors gamma and phi for the given document.
        \param doc reference to current document \sa document
        \param ss reference to the model sufficient statistics \sa suff_stats
        \param var_gamma reference to the current document topic distribtion parameter \sa varGamma
        \param phi reference to the current document topic-word distribtion \sa phi
        \return the log likelihood for the document
        \sa doc_e_step(), compute_likelihood()
     */
    double inference(document const& doc, std::vector<double>& var_gamma,
                     std::vector<std::vector<double>>& phi);

    //! calculates the log likelihood for the current document
    /*!
        \param doc reference to current document \sa document
        \param var_gamma reference to the current document topic distribtion parameter \sa varGamma
        \param phi reference to the current document topic-word distribtion \sa phi
        \return the log likelihood for the document
        \sa doc_e_step(), compute_likelihood()
     */
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
