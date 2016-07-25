//
// Created by Dylan Verrezen on 2016/07/25.
//

#ifndef LDA_GIBBS_H
#define LDA_GIBBS_H

#include "lda_model.h"
#include <vector>

class gibbs : public lda_model{

public:
    // Constructor
    gibbs(doc_corpus& corp);

    void train(size_t numTopics);
    void save_parameters(std::string file_dir);

private:

    doc_corpus corpus;

    // Model parameters
    // ================
    double alpha = 0.1;
    double beta = 0.001;
    std::vector<std::vector<double>> phi;
    std::vector<std::vector<double>> theta;


    // Convenience constants
    // =====================
    size_t numTopics;      /*!< number of topics */
    size_t numDocs;        /*!< total number of documents in the corpus. */
    size_t numTerms;       /*!< total number of terms(words) in the corpus. */

    // Inference variables
    // ===================
    std::vector<std::vector<int>> n_dk;
    std::vector<std::vector<int>> n_kw;
    std::vector<int> n_k;
    std::vector<int> n_d;
    std::vector<std::vector<int>> topic_assignments;

    // Functions
    // =========
    void zero_init_counts();
    void random_assign_topics();
    std::vector<double> get_pz(int d, int w);
    int sample_multinomial(std::vector<double> probabilities);
    void estimate_parameters();
};


#endif //LDA_GIBBS_H
