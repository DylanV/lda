/*!
 * \file lda_model.h
 */

#ifndef LDA_LDA_MODEL_H
#define LDA_LDA_MODEL_H

#include <vector>
#include <map>
#include <string>
#include <algorithm>

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

    lda_settings() : EMPIRICAL_BAYES(false),
                     ALPHA_UPDATE_INTERVAL(1),
                     MIN_ITER(1),
                     MAX_ITER(100) {}

    bool EMPIRICAL_BAYES;
    int ALPHA_UPDATE_INTERVAL;
    int MAX_ITER;
    int MIN_ITER;

    void set_values(std::map<std::string, std::string> raw_settings){
        for(const auto &pair : raw_settings){
            std::string key = pair.first;
            std::transform(key.begin(), key.end(), key.begin(), ::toupper);

            if(key == "EMPIRICAL_BAYES"){
                std::string val = pair.second;
                std::transform(val.begin(), val.end(), val.begin(), ::tolower);
                if(val == "1" || val == "true"){
                    EMPIRICAL_BAYES = true;
                }
            } else if(key == "ALPHA_UPDATE_INTERVAL"){
                std::string val = pair.second;
                ALPHA_UPDATE_INTERVAL = stoi(val);
            } else if(key == "MAX_ITER"){
                std::string val = pair.second;
                MAX_ITER = stoi(val);
            } else if(key == "MIN_ITER"){
                std::string val = pair.second;
                MIN_ITER = stoi(val);
            }

        }
    }

};

/*! Represetation of latent dirichlet allocation model.
 * Each of the inference methods inherits from this class so that there is a
 * common interface between them.
 */
class lda_model {
public:
    /*! Train the model on the corpus given the number of topics
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
