
#ifndef LDA_EXPECTATION_PROP_H
#define LDA_EXPECTATION_PROP_H


#include "lda_model.h"

class expectation_prop : public lda_model {
public:
    expectation_prop(doc_corpus& corpus);

    void train(size_t numTopics);

    void save_parameters(std::string file_dir);

private:
    doc_corpus corpus;  /*!< Document corpus for the lda. */

    const int MAX_ITERATION = 200;
    const double CONV_THRESHHOLD = 1e-5;

    // Convenience constants
    // =====================
    size_t numTopics;      /*!< Number of topics. */
    size_t numDocs;        /*!< Total number of documents in the corpus. */
    size_t numTerms;       /*!< Total number of terms(words) in the corpus. */
    int numSkipped;

    std::vector<double> alpha;
    std::vector<std::vector<double>> Pword;

    std::vector<std::vector<double>> s;
    std::vector<std::vector<std::vector<double>>> beta;
    std::vector<std::vector<double>> gamma;

    void setup_parameters();
    double doc_e_step(int d);
    void m_step();
};


#endif //LDA_EXPECTATION_PROP_H
