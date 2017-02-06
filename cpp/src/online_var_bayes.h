
#ifndef LDA_ONLINE_VAR_BAYES_H
#define LDA_ONLINE_VAR_BAYES_H

#include "lda_model.h"
#include "dirichlet.h"

struct online_var_bayes_settings : lda_settings {

    online_var_bayes_settings(std::map<std::string, std::string> raw_settings)
    {

    }

};

class online_var_bayes : public lda_model{
public:
    //! The constructor.
    online_var_bayes(const doc_corpus &corp, const online_var_bayes_settings &settings);
    //! Train a lda model with the given number of topics.
    void train(size_t numTopics);
    //! Save the model parameters to the given folder.
    void save_parameters(std::string file_dir);
private:
    doc_corpus corpus;                      /*!< The corpus */

    // The parameters on the priors
    double ALPHA;
    double ETA;

    // Convenience
    size_t numDocs;
    size_t numTopics;
    size_t numTerms;

    // Learning parameters
    double tau;
    double kappa;
    int updateCount;
    int batchSize;
    int numRuns;

    // Variational
    std::vector<std::vector<double>> lambda;
    std::vector<std::vector<double>> Elogbeta;
    std::vector<std::vector<double>> Elogtheta;

    // RNG
    std::default_random_engine generator;
};


#endif //LDA_ONLINE_VAR_BAYES_H
