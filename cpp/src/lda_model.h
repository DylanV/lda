/*!
 * \file lda_model.h
 */

#ifndef LDA_LDA_MODEL_H
#define LDA_LDA_MODEL_H

#include <vector>
#include <map>
#include <string>

/*! Represents a document bag of words style.
 * A document is a collection of words. Word order does not matter only word frequency is recoreded.
 */
struct document {
    std::map<int,int> wordCounts;      /*!< Map of word id to word counts. */
    size_t count;                      /*!< Total number of words in the document. */
    size_t uniqueCount;                /*!< Number of unique words (tokens) in the document. */
};

/*! Represents a corpus of documents.
 * @sa document
 */
struct doc_corpus {
    std::vector<document> docs;    /*!< Vector of document structs. \sa document */
    size_t numTerms;               /*!< Total number of unique terms in the corpus. */
    size_t numDocs;                /*!< Total number of documents. */
};

/*! Holds all of the settings for the lda model.
 * @sa gibbs @sa var_bayes
 */
struct lda_settings {
    lda_settings() : converged_threshold(1e-6), min_iterations(2),
                     max_iterations(100), inf_converged_threshold(1e-6),
                     inf_max_iterations(20), estimate_alpha(false),
                     alpha_update_interval(1){}

    double converged_threshold;     /*!< The convergence threshold used in training. */
    int min_iterations;             /*!< Minimum number of iterations to train for. */
    int max_iterations;             /*!< Maximum number of iterations to train for. */
    double inf_converged_threshold; /*!< Document inference convergence threshold. */
    int inf_max_iterations;         /*!< Document inference max iterations. */
    bool estimate_alpha;            /*!< Whether to estimate alpha. */
    int alpha_update_interval;      /*!< Interval to update alpha on. */
};

/*! Represetation of latent dirichlet allocation model.
 * Each of the inference methods inherits from this class so that there is a
 * common interface between them.
 */
class lda_model {
public:
    /*! Train the model
     * Trains the model on the corpus given the number of topics
     * @param [in] numTopics Number of topics to train with/
     */
    virtual void train(size_t numTopics) = 0;
    /*! Save the model parameters to files.
     * Saves the model parameters to files in the given folder. Previous files are overwritten.
     * @param [in] file_dir Folder directory to write the parameter files.
     */
    virtual void save_parameters(std::string file_dir) = 0;
};

#endif //LDA_LDA_MODEL_H
