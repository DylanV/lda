//
// Created by Dylan Verrezen on 2016/07/25.
//

#include "gibbs.h"
#include "util.h"
#include <random>

gibbs::gibbs(doc_corpus& corp){
    corpus = corp;
    numDocs = corpus.numDocs;
    numTerms = corpus.numTerms;

};

void gibbs::train(size_t numTopics) {
    this->numTopics = numTopics;

    zero_init_counts();
    random_assign_topics();

    for(int iter=0; iter<100; ++iter){
        for(int d=0; d<numDocs; ++d){

            document curr_doc = corpus.docs[d];
            int doc_word_index = 0;

            for(auto const& word_count : curr_doc.wordCounts){

                int w = word_count.first;
                int count = word_count.second;

                for(int i=0; i<count; ++i){
                    // get current assignment
                    int z = topic_assignments[d][doc_word_index];
                    // decrement counts
                    n_dk[d][z] -= 1;
                    n_kw[z][w] -= 1;
                    n_k[z] -= 1;
                    n_d[d] -= 1;
                    // get conditional distribution
                    std::vector<double> pz = get_pz(d, w);
                    // update assignment
                    z = sample_multinomial(pz);
                    topic_assignments[d][doc_word_index] = z;
                    doc_word_index++;
                    // update counts
                    n_dk[d][z] += 1;
                    n_kw[z][w] += 1;
                    n_k[z] += 1;
                    n_d[d] += 1;
                }
            }
        }
    }
}

void gibbs::save_parameters(std::string file_dir) {

}

void gibbs::zero_init_counts() {
    n_dk = std::vector<std::vector<int>>(numDocs, std::vector<int>(numTopics, 0));
    n_kw = std::vector<std::vector<int>>(numTopics, std::vector<int>(numTerms, 0));
    n_k = std::vector<int>(numTopics);
    n_d = std::vector<int>(numDocs);
    topic_assignments = std::vector<std::vector<int>>(numDocs);

    for(int d=0; d<numDocs; ++d){
        topic_assignments[d] = std::vector<int>(corpus.docs[d].count), 0;
    }
}

void gibbs::random_assign_topics() {

    std::default_random_engine generator;
    std::uniform_int_distribution<int> uniform(0,numTopics);

    for(int d=0; d<numDocs; d++){
        document curr_doc = corpus.docs[d];
        int doc_word_index = 0;
        for(auto const& word_count : curr_doc.wordCounts){
            int w = word_count.first;
            int count = word_count.second;
            for(int i=0; i<count; ++i){
                // randomly pick a topic
                int z = uniform(generator);
                // update assignment
                topic_assignments[d][doc_word_index] = z;
                doc_word_index++; // because words can appear multiple times in a document
                // update counts
                n_dk[d][z] += 1;
                n_kw[z][w] += 1;
                n_k[z] += 1;
                n_d[d] += 1;
            }
        }
    }
}

std::vector<double> gibbs::get_pz(int d, int w) {
    std::vector<double> pz = std::vector<double>(numTopics);

    for(int k=0; k<numTopics; ++k){
        pz[k] = (n_kw[k][w] + beta) / (n_k[k] + beta*numTerms);
        pz[k] *= (n_dk[d][k] + alpha) / (n_d[d] + alpha*numTopics);
    }

    norm(pz);
    return pz;
}

int gibbs::sample_multinomial(std::vector<double> probabilities) {
    std::default_random_engine generator;
    std::discrete_distribution<int> multi (probabilities.begin(), probabilities.end());
    return multi(generator);
}