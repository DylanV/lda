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

struct var_bayes_settings : lda_settings {

    var_bayes_settings(std::map<std::string, std::string> raw_settings) : CONV_THRESH(1e-6),
                           INF_CONV_THRESH(1e-6),
                           INF_MAX_ITER(20)
    {
        if(raw_settings.size() > 0){
            lda_settings::set_values(raw_settings);

            for(const auto &pair : raw_settings){
                std::string key = pair.first;
                std::transform(key.begin(), key.end(), key.begin(), ::toupper);

                if(key == "CONV_THRESH"){
                    std::string val = pair.second;
                    CONV_THRESH = stod(val);
                } else if(key == "INF_CONV_THRESH"){
                    std::string val = pair.second;
                    INF_CONV_THRESH = stod(val);
                } else if(key == "INF_MAX_ITER"){
                    std::string val = pair.second;
                    INF_MAX_ITER = stoi(val);
                }

            }
        }
    }

    double CONV_THRESH;     /*!< The convergence threshold used in training. */
    double INF_CONV_THRESH; /*!< Document inference convergence threshold. */
    int INF_MAX_ITER;         /*!< Document inference max iterations. */
};

class var_bayes : public lda_model{
public:
    /*! Constructor
     * @sa lda_model
     * @param corp Document corpus. @sa doc_corpus
     * @param settings LDA settings. @sa lda_settings
     * @return var_bayes lda model
     */
    var_bayes(const doc_corpus &corp, const var_bayes_settings &settings);

    /*! Train on the given corpus with variational inference.
     * @param [in] numTopics Number of topics to train model with.
     */
    void train(size_t numTopics);

    /*! Write the model paramters to files in the given folder.
     * @param [in] file_dir Folder directory.
     */
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
    std::vector<std::vector<double>> gamma;          /*!< Gamma : per document topic distribution */
    std::vector<std::vector<std::vector<double>>> phi;  /*!< Phi : per document word topic assignments */

    // Settings
    // ========
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


    // Inference helper functions
    // ==========================

    /*! Randomly initialise the given sufficient statistics.
     *  Randomly sets the sufficient statistics for alpha and beta. Note that it is
     *  assumed that the sufficient stat vectors already exist with the correct dimensions.
     *  So zeroSSInit should be called before this function. @sa zeroSSInit
     * @param [in,out] ss The sufficient statistic struct @sa suff_stats .
     */
    void randomSSInit(suff_stats& ss);
    /*! Initiliase the sufficient statistics objects for alpha and beta.
     * Initialises the sufficient statistics objects (vectors) and fills them with zeros.
     * Vectors should now have the correct dimensions and never need to be moved or re-initialised
     * only written to.
     * @param [out] ss The sufficient statistic struct @sa suff_stats .
     */
    void zeroSSInit(suff_stats& ss);

    /*! Sets up the alpha dirichlet with the given settings
     * @sa dirichlet
     * @param [in] settings the settings struct. @sa alpha_settings
     * @return The alpha dirichlet.
     */
    dirichlet setup_alpha();

    /*! Calculate a maximum lokelihood estimate of the global model parameters
     * Estimates the dirichlet priors alpha and beta given the variational parameters.
     * @param [in] ss The sufficient statistics. @sa suff_stats
     * @param [in] optAlpha Whether alpha should be estimated. Default true.
     */
    void mle(suff_stats& ss, const bool optAlpha);

    /*! E-step. Estimate the variational parameters.
     * Estimate gamma and phi with a document with alpha and beta fixed. Updates the sufficient statistics.
     * @param [in] doc The current document
     * @param [out] ss The sufficient statistics for alpha and beta
     * @param [in,out] var_gamma The topic distribution for the document
     * @param [in,out] phi The word topic assignments for the document
     * @return The (lower bound) log-likelihood of the document
     */
    double doc_e_step(const document &doc, suff_stats& ss, std::vector<double>& var_gamma,
                      std::vector<std::vector<double>>& phi);

    /*! The actual inference for the e step
     * Updates gamma and phi. @sa doc_e_step
     * @param [in] doc The current document
     * @param [in,out] var_gamma Gamma for the current document
     * @param [in,out] phi Phi for the current document.
     * @return The log-likelihood for this document.
     */
    double inference(const document &doc, std::vector<double>& var_gamma,
                     std::vector<std::vector<double>>& phi);

    /*! Calculate the log-likelihood for the given document.
     * @param [in] doc The document in question.
     * @param [in] var_gamma Gamma for the document.
     * @param [in] phi Phi for the document.
     * @return The log-likelihood
     */
    double compute_likelihood(const document &doc, const std::vector<double> &var_gamma,
                              const std::vector<std::vector<double>> &phi);
};


#endif //LDA_VAR_BAYES_H
