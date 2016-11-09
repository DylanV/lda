/*!
 \file gibbs.cpp
 */

#include "gibbs.h"
#include "util.h"
#include "data.h"
#include <iostream>
#include <sstream>
#include <chrono>

gibbs::gibbs(doc_corpus& corp){

    corpus = corp;
    numDocs = corpus.numDocs;
    numTerms = corpus.numTerms;
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    generator =  std::default_random_engine(seed);
};

void gibbs::train(size_t numTopics) {

    this->numTopics = numTopics;

    zero_init_counts();
    random_assign_topics();

//    this->beta = 1/numTerms;

    double max_iter = 100.0;

    for(int iter=0; iter<max_iter; ++iter){
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

        std::cout << "<";
        int count = 0;
        int length = 50;
        for(int x=0; x<(iter/max_iter)*length; x++){
            std::cout << "=";
            count++;
        }
        for(int x=0; x<length-count; x++){
            std::cout << "-";
        }
        std::cout << "> " << iter+1 << "/" << max_iter << '\r' << std::flush;
    }
    // update phi and theta
    estimate_parameters();
}

void gibbs::save_parameters(std::string file_dir) {

    std::fstream fs; //get the filestream

    //write phi
    fs.open(file_dir+"phi.dat", std::fstream::out | std::fstream::trunc);
    write_2d_vector_to_fs(fs, phi);
    fs.close();

    //write theta
    fs.open(file_dir+"theta.dat", std::fstream::out | std::fstream::trunc);
    write_2d_vector_to_fs(fs, theta);
    fs.close();
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

    std::uniform_int_distribution<int> uniform(0,numTopics-1);

    for(int d=0; d<numDocs; d++){
        document curr_doc = corpus.docs[d];
        int doc_word_index = 0;

        bool god_sample = false;
        int true_topic = 0;
        int ran = uniform(generator);
        int thresh = 8;
        if(ran < thresh){
            if(d < 60000){
                god_sample = true;
            }
            if(d < 5923 ){
                true_topic = 0;
            } else if( d < 12665 ){
                true_topic = 1;
            } else if( d < 18623 ){
                true_topic = 2;
            } else if( d < 24754 ){
                true_topic = 3;
            } else if ( d < 30596) {
                true_topic = 4;
            } else if ( d < 36017) {
                true_topic = 5;
            } else if ( d < 41935) {
                true_topic = 6;
            } else if ( d < 48200) {
                true_topic = 7;
            } else if ( d < 54051) {
                true_topic = 8;
            } else if ( d < 60000) {
                true_topic = 9;
            }
        }

        for(auto const& word_count : curr_doc.wordCounts){
            int w = word_count.first;
            int count = word_count.second;
            for(int i=0; i<count; ++i){
                int z;
                if(god_sample){
                    z = true_topic;
                }
                else {
                    // randomly pick a topic
                    z = uniform(generator);
                }
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

std::vector<double> gibbs::get_pz(const int d, const int w) {

    std::vector<double> pz = std::vector<double>(numTopics);

    for(int k=0; k<numTopics; ++k){
        pz[k] = (n_kw[k][w] + beta) / (n_k[k] + beta*numTerms);
        pz[k] *= (n_dk[d][k] + alpha) / (n_d[d] + alpha*numTopics);
    }

    norm(pz);
    return pz;
}

int gibbs::sample_multinomial(const std::vector<double> probabilities) {

    std::discrete_distribution<int> multi (probabilities.begin(), probabilities.end());
    return multi(generator);
}

void gibbs::estimate_parameters() {

    // calculate phi
    phi = std::vector<std::vector<double>>(numTopics, std::vector<double>(numTerms, 0));
    double total = 0;
    for(int k=0; k<numTopics; ++k){
        for(int w=0; w<numTerms; ++w){
            phi[k][w] = n_kw[k][w] + beta;
            total += n_kw[k][w];
        }
    }
    total += numTerms*beta;
    for(int k=0; k<numTopics; ++k){
        norm(phi[k]);
    }

    // calculate theta
    theta = std::vector<std::vector<double>>(numDocs, std::vector<double>(numTopics, 0));
    total = 0;
    for(int d=0; d<numDocs; ++d){
        for(int k=0; k<numTopics; ++k){
            theta[d][k] = n_dk[d][k] + alpha;
            total += n_dk[d][k];
        }
    }
    total += numTopics*alpha;
    for(int d=0; d<numDocs; ++d){
        for(int k=0; k<numTopics; ++k){
            theta[d][k] /= total;
        }
    }
}