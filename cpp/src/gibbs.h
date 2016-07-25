/*!
 \file gibbs.h
 */

#ifndef LDA_GIBBS_H
#define LDA_GIBBS_H

#include "lda_model.h"
#include <vector>
#include <random>

class gibbs : public lda_model{

public:
    //! The constructor.
    gibbs(doc_corpus& corp);
    //! Train a lda model with the given number of topics.
    void train(size_t numTopics);
    //! Save the model parameters to the given folder.
    void save_parameters(std::string file_dir);

private:

    doc_corpus corpus;                      /*! The corpus */
    std::default_random_engine generator;   /*! A RNG for sampling*/

    // Model parameters
    // ================
    double alpha = 0.1;                     /*! Symmetrc alpha. Topic distribution prior. */
    double beta = 0.001;                    /*! Symmetric beta. Topic word distribution prior. */
    std::vector<std::vector<double>> phi;   /*! Topic word distribution. */
    std::vector<std::vector<double>> theta; /*! Document topic proportions. */

    // Convenience constants
    // =====================
    size_t numTopics;      /*!< number of topics */
    size_t numDocs;        /*!< total number of documents in the corpus. */
    size_t numTerms;       /*!< total number of terms(words) in the corpus. */

    // Inference variables
    // ===================
    std::vector<std::vector<int>> n_dk; /*! Number of words assigned to topic k from from doc d. */
    std::vector<std::vector<int>> n_kw; /*! Number of times word w is assigned topic k. */
    std::vector<int> n_k;               /*! Number of times topic k is assigned. */
    std::vector<int> n_d;               /*! Nuber of words assigned in doc d. Technically constant. */
    std::vector<std::vector<int>> topic_assignments;    /*! The topic assigned to each word */

    // Functions
    // =========
    //! Initialises the inference variables.
    void zero_init_counts();
    //! Randomly assigns a topic to each word in the corpus.
    void random_assign_topics();
    //! Get the conditional topic distribution for a word given all the assignments of every other word.
    std::vector<double> get_pz(int d, int w);
    //! Obtain an integer sample from a multinomial with the given probabily density.
    int sample_multinomial(std::vector<double> probabilities);
    //! Estimate and update the model parameters phi and theta from the current inference variables.
    void estimate_parameters();
};


#endif //LDA_GIBBS_H
