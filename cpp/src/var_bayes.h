/*!
 \file var_bayes.h
 */

#ifndef LDA_VAR_BAYES_H
#define LDA_VAR_BAYES_H

#include "lda_model.h"
#include "dirichlet.h"

/*! Sufficient statistics for all the model paramters.
 * Sufficient statistics for alpha and beta.
 */
struct suff_stats {
    // beta
    std::vector<std::vector<double>> classWord; /*!< 2d vector for the topic-word sufficient stats for beta.*/
    std::vector<double> classTotal;             /*!< Vector (K) for the topic sufficient stats for  beta.*/
    // alpha
    std::vector<double> alpha_ss; /*!< Sufficient stats for alpha .*/
    size_t numDocs;    /*!< Number of documents included in the suff stats. */
};

class var_bayes : public lda_model{
public:
    var_bayes(doc_corpus& corp, lda_settings settings, alpha_settings a_settings);

    void train(size_t numTopics);
    void save_parameters(std::string file_dir);

private:
    doc_corpus corpus;  /*!< Document corpus for the lda. */
    double likelihood;  /*!< Total log-likelihood for the corpus. */

    // Convenience constants
    // =====================
    size_t numTopics;      /*!< Number of topics. */
    size_t numDocs;        /*!< Total number of documents in the corpus. */
    size_t numTerms;       /*!< Total number of terms(words) in the corpus. */

    // Model parameters
    // ================
    std::vector<std::vector<double>> logProbW;  /*!< Topic-word log prob (unnormalised beta) */
    dirichlet alpha;   /*!< Alpha parameter :  topic distribution */

    // Variational parameters
    // ======================
    std::vector<std::vector<double>> varGamma;          /*!< Gamma : per document topic distribution */
    std::vector<std::vector<std::vector<double>>> phi;  /*!< Phi : per document word topic assignments */

    // Settings
    // ========
    // Alpha
    alpha_settings ALPHA_SETTINGS;
    // Training Settings
    double CONV_THRESHHOLD;     /*!< The convergence threshold used in training */
    int MIN_ITER;               /*!< Minimum number of iterations to train for */
    int MAX_ITER;               /*!< Maximum number of iterations to train for */
    // Document E-step Inference Settings
    double INF_CONV_THRESH;     /*!< Document inference convergence threshold*/
    int INF_MAX_ITER;           /*!< Document inference max iterations*/
    // Settings regarding alpha
    bool EST_ALPHA;             /*!< Whether alpha should be estimated */
    int UPDATE_INTERVAL;        /*!< The iteration interval on which alpha is estimated */

    // Functions
    // =========
    /*! Train the lda on the corpus with the number of topics supplied
     * @param [in] num_topics Number of topics for the model
     * @param [in] a_settings Settings for the alpha dirichlet prior. @sa dirichlet
     */
    void train(int num_topics, alpha_settings a_settings);

    // Inference helper functions
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


#endif //LDA_VAR_BAYES_H
