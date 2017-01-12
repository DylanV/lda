
#ifndef LDA_EXPECTATION_PROP_H
#define LDA_EXPECTATION_PROP_H

#include "lda_model.h"

struct ep_settings : lda_settings {

    ep_settings(std::map<std::string, std::string> raw_settings) : E_MAX_ITERATIONS(200),
                                                                   CONV_THRESH(1e-6),
                                                                   ALPHA_INIT(0.1)
    {
        if(raw_settings.size() > 0){
            lda_settings::set_values(raw_settings);

            for(const auto &pair : raw_settings){
                std::string key = pair.first;
                std::transform(key.begin(), key.end(), key.begin(), ::toupper);

                if(key == "E_MAX_ITERATIONS"){
                    std::string val = pair.second;
                    E_MAX_ITERATIONS = stoi(val);
                } else if(key == "CONV_THRESH"){
                    std::string val = pair.second;
                    CONV_THRESH = stod(val);
                } else if(key == "ALPHA_INIT"){
                    std::string val = pair.second;
                    ALPHA_INIT = stod(val);
                }

            }
        }
    }

    int E_MAX_ITERATIONS;         /*!< The convergence threshold used in training. */
    double CONV_THRESH;           /*!< Document inference convergence threshold. */
    double ALPHA_INIT;            /*!< Document inference max iterations. */
};

class expectation_prop : public lda_model {
    public:
    expectation_prop(const doc_corpus &corpus, const ep_settings &settings);

    void train(size_t numTopics);

    void save_parameters(std::string file_dir);

private:
    doc_corpus corpus;  /*!< Document corpus for the lda. */

    int MAX_ITERATION = 25;
    int E_MAX_ITERATIONS = 200;
    double CONV_THRESHHOLD = 1e-5;
    double ALPHA_INIT = 0.1;

    // Convenience constants
    // =====================
    size_t numTopics;      /*!< Number of topics. */
    size_t numDocs;        /*!< Total number of documents in the corpus. */
    size_t numTerms;       /*!< Total number of terms(words) in the corpus. */
    bool first = true;

    std::vector<double> alpha;
    std::vector<std::vector<double>> Pword;

    std::vector<std::vector<std::vector<double>>> beta;
    std::vector<std::vector<double>> gamma;

    void setup_parameters();
    double doc_e_step(int d);
    void m_step();
};


#endif //LDA_EXPECTATION_PROP_H
