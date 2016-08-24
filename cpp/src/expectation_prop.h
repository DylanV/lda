
#ifndef LDA_EXPECTATION_PROP_H
#define LDA_EXPECTATION_PROP_H

#include "lda_model.h"

class expectation_prop : public lda_model {
public:
    //! The constructor.
    expectation_prop(doc_corpus& corp);
    //! Train a lda model with the given number of topics.
    void train(size_t numTopics);
    //! Save the model parameters to the given folder.
    void save_parameters(std::string file_dir);

private:
    doc_corpus corpus;        /*!< The corpus */

};


#endif //LDA_EXPECTATION_PROP_H
