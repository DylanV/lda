/*!
 \file gibbs.h
 */

#ifndef LDA_GIBBS_H
#define LDA_GIBBS_H

#include "lda_model.h"
#include <vector>
#include <random>

struct gibbs_settings : lda_settings {
    gibbs_settings(std::map<std::string, std::string> raw_settings){

    }
};


class gibbs : public lda_model{

public:
    //! The constructor.
    gibbs(const doc_corpus &corp, const gibbs_settings &settings);
    //! Train a lda model with the given number of topics.
    void train(size_t numTopics);
    //! Save the model parameters to the given folder.
    void save_parameters(std::string file_dir);

private:

    doc_corpus corpus;                      /*!< The corpus */
    std::default_random_engine generator;   /*!< A RNG for sampling*/

    // Model parameters
    // ================
    double alpha = 0.01;            /*!< Symmetrc dirichlet alpha parameter. Topic distribution prior. */
    double beta = 1.0;               /*!< Symmetric dirichlet beta parameter. Topic word distribution prior. */
    std::vector<std::vector<double>> phi;   /*!< Topic word distribution. */
    std::vector<std::vector<double>> theta; /*!< Document topic proportions. */
    double MAX_ITER = 200.0;

    // Convenience constants
    // =====================
    size_t numTopics;      /*!< number of topics */
    size_t numDocs;        /*!< total number of documents in the corpus. */
    size_t numTerms;       /*!< total number of terms(words) in the corpus. */

    // Inference count variables
    // =========================
    std::vector<std::vector<int>> n_dk; /*!< Number of words assigned to topic k from from doc d. */
    std::vector<std::vector<int>> n_kw; /*!< Number of times word w is assigned topic k. */
    std::vector<int> n_k;               /*!< Number of times topic k is assigned. */
    std::vector<int> n_d;               /*!< Nuber of words assigned in doc d. Technically constant. */

    std::vector<std::vector<int>> topic_assignments;    /*!< The topic assigned to each word */

    // Functions
    // =========
    /*! Initialises the inference variables.
     * All the vectors are initialised here and will stay fixed size for the rest of training.
     */
    void zero_init_counts();

    /*! Randomly initialises the topic assignments.
     * Each word in the corpus is assigned a random topic. The counts are updated approriately.
     * Words can be repeated in a document but each instance of a repeated word is sampled seperately.
     * We do not want all instances of a repeated word to be assigned the same topic.
     */
    void random_assign_topics();

    /*! Get the conditional topic distribution for a word given all the assignments of every other word.
     * Gets the conditional topic distribution of a given word in a document. Returns a vector of K probabilities
     * This is a multinomial distribution denoting the probabily that each of the K topics generated the word.
     * @param [in] d Current document id
     * @param [in] w Current word in the document
     * @return Vector multinomial probability parameter
     */
    std::vector<double> get_pz(const int d, const int w);

    /*! Sample from a multinomial with the given probabilites
     * @param [in] probabilities Mass function for the multinomial
     * @return The sampled category
     */
    int sample_multinomial(const std::vector<double> probabilities);

    /*! Estimate and update the model parameters phi and theta from the current inference variables.
     * Updates phi and theta given the current model inference parameters (counts).
     * Slightly overcomplicated as aplha and beta and symmetric so are not strictly necessary.
     * The parameters could be generated by just normalising the counts.
     */
    void estimate_parameters();
};


#endif //LDA_GIBBS_H
