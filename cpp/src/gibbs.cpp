/*!
 \file gibbs.cpp
 */

#include "gibbs.h"
#include "util.h"
#include "data.h"
#include <iostream>
#include <sstream>

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

    for(int iter=0; iter<200; ++iter){
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
        for(int x=0; x<(iter/200.0)*length; x++){
            std::cout << "=";
            count++;
        }
        for(int x=0; x<length-count; x++){
            std::cout << "-";
        }
        std::cout << "> " << iter+1 << "/200" << '\r' << std::flush;
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
        for(int w=0; w<numTerms; ++w){
            phi[k][w] /= total;
        }
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