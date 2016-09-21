
#include "expectation_prop.h"
#include "util.h"
#include "data.h"
#include <math.h>
#include <cstdlib>
#include <iostream>

expectation_prop::expectation_prop(doc_corpus &corpus) {
    this->corpus = corpus;
    numDocs = corpus.numDocs;
    numTerms = corpus.numTerms;
}

void expectation_prop::save_parameters(std::string file_dir) {
    std::fstream fs; //get the filestream
    fs.open(file_dir+"beta.dat", std::fstream::out | std::fstream::trunc);
    write_2d_vector_to_fs(fs, Pword);
    fs.close();
}

void expectation_prop::train(size_t numTopics) {
    this->numTopics = numTopics;

    setup_parameters();
    double likelihood;
    double old_likelihood = 0.0;
    int iteration = 0;
    bool converged = false;
    while(!converged && iteration < MAX_ITERATION){
        iteration++;
        likelihood = 0.0;
        for(int d=0; d<numDocs; ++d){
            likelihood += doc_e_step(d);
        }
        m_step();

        double conv = fabs((old_likelihood - likelihood)/old_likelihood);
        old_likelihood = likelihood;
        if(conv < CONV_THRESHHOLD){
            converged = true;
        }

        std::cout << "Iteration " << iteration << ": with likelihood: " << likelihood << std::endl;
    }
}

void expectation_prop::setup_parameters() {
    alpha = std::vector<double>(numTopics, 1.0/numTopics);
    Pword = std::vector<std::vector<double>>(numTopics, std::vector<double>(numTerms));

    std::vector<std::vector<double>> classWord
            = std::vector<std::vector<double>>(numTopics, std::vector<double>(numTerms, 1.0/numTerms));
    std::vector<double> classTotal = std::vector<double>(numTopics, 0.0);

    for(int k=0; k<numTopics; k++){
        for(int n=0; n<numTerms; n++){
            classWord[k][n] += rand();
            classTotal[k] += classWord[k][n];
        }
    }

    for(int k=0; k<numTopics;k++){
        for(int w=0; w<numTerms; w++){
            if(classWord[k][w] > 0){
                Pword[k][w] = log(classWord[k][w]) - log(classTotal[k]);
            } else {
                Pword[k][w] = -100.0;
            }
            Pword[k][w] = exp(Pword[k][w]);
        }
    }

    s = std::vector<std::vector<double>>(numDocs);
    gamma = std::vector<std::vector<double>>(numDocs);
    beta = std::vector<std::vector<std::vector<double>>>(numDocs);

    for(int d=0; d<numDocs; ++d){
        size_t W = corpus.docs[d].uniqueCount;
        s[d] = std::vector<double>(W, 1); // all s init to 1
        beta[d] = std::vector<std::vector<double>>(W, std::vector<double>(numTopics, 0));
        gamma[d] = std::vector<double>(numTopics);
        for(int k=0; k<numTopics; ++k){
            gamma[d][k] = alpha[k];
        }
    }
}

double expectation_prop::doc_e_step(int d) {
    document doc = corpus.docs[d];

    int w=0;
    for(auto const& word_count : doc.wordCounts){
        //deletion
        std::vector<double> old_posterior = std::vector<double>(numTopics);
        bool skipWord = false;
        double old_posterior_total = 0;
        for(int k=0; k<numTopics; ++k){
            old_posterior[k] = gamma[d][k] - beta[d][w][k];
            old_posterior_total += old_posterior[k];
            if(old_posterior[k] < 0){
                skipWord = true;
            }
        }

        // skip words where any of the 'old' posterior < 0
        if(!skipWord){

            // calculate Z_w needed for moment matching and update steps
            double Z_w = 0.0;
            double prob_total = 0.0;
            for(int k=0; k<numTopics; ++k){
                prob_total += Pword[k][word_count.first] * old_posterior[k];
            }
            Z_w = prob_total/old_posterior_total;

            // moment matching
            double m_a;
            double m2_a;
            std::vector<double> gamma_prime = std::vector<double>(numTopics);
            double numer_sum = 0;
            double denom_sum = 0;

            for(int k=0; k<numTopics; ++k){
                m_a = (1.0/Z_w)*(old_posterior[k]/old_posterior_total)*
                        ((Pword[k][word_count.first]+prob_total)/(1.0+old_posterior_total));
                m2_a = (1.0/Z_w)*(old_posterior[k]/old_posterior_total)*
                        ((old_posterior[k]+1)/(old_posterior_total+1.0));
                m2_a *= ((2.0*Pword[k][word_count.first] + prob_total)/(2.0 + old_posterior_total));

                gamma_prime[k] = m_a;
                numer_sum += (m_a - m2_a);
                denom_sum += (m2_a - m_a*m_a);
            }
            for(int k=0; k<numTopics; ++k){
                gamma_prime[k] *= (numer_sum/denom_sum);
            }

            // update
            double step_size = 1.0/numTerms;
            std::vector<double> beta_new = std::vector<double>(numTopics);
            bool make_changes = true;
            for(int k=0; k<numTopics; ++k){
                beta_new[k] = (1.0-step_size)*beta[d][w][k];
                beta_new[k] += step_size*(gamma_prime[k] - old_posterior[k]);
            }
            // inclusion
            std::vector<double> gamma_new = std::vector<double>(numTopics);
            for(int k=0; k<numTopics; ++k){
                gamma_new[k] = gamma[d][k] + (word_count.second)*(beta[d][w][k] - beta_new[k]);
                if(gamma_new[k] < 0){
                    make_changes = false;
                }
            }
            if(make_changes){
                // inclusion continued
                for(int k=0; k<numTopics; ++k){
                    gamma[d][k] = gamma_new[k];
                }
                // update continued
                for(int k=0; k<numTopics; ++k){
                    beta[d][w][k] = beta_new[k];
                }
                double prod_gamma_prime = 1.0;
                double prod_gamma_old = 1.0;
                double gamma_prime_sum = 0;
                for(int k=0; k<numTopics; ++k){
                    prod_gamma_prime *= digamma(gamma_prime[k]);
                    prod_gamma_old *= digamma(old_posterior[k]);
                    gamma_prime_sum += gamma_prime[k];
                }

                s[d][w] = Z_w;
                s[d][w] *= (digamma(gamma_prime_sum) / prod_gamma_prime);
                s[d][w] *= (prod_gamma_old / digamma(old_posterior_total));
            }
        }

        w++;
    }

    // calculate document likelihood
    double gamma_sum = 0;
    double gamma_prod = 1.0;
    double alpha_sum = 0;
    double alpha_prod = 1.0;
    for(int k=0; k<numTopics; k++){
        gamma_sum += gamma[d][k];
        alpha_sum += alpha[k];
        gamma_prod *= digamma(gamma[d][k]);
        alpha_prod *= digamma(alpha[k]);
    }
    double likelihood = (gamma_prod/digamma(gamma_sum)) * (digamma(alpha_sum)/alpha_prod);
    w=0;
    for(auto const& word_count : doc.wordCounts) {
        likelihood *= pow(s[d][w],word_count.second);
        w++;
    }

    return likelihood;
}

void expectation_prop::m_step() {
    std::vector<std::vector<double>> word_prob_new
            = std::vector<std::vector<double>>(numTopics, std::vector<double>(numTerms, 0));
    for(int d=0; d<numDocs; ++d){

        document doc = corpus.docs[d];

        double doc_gamma_sum = 0;
        for(int k=0; k<numTopics; ++k){
            doc_gamma_sum += gamma[d][k];
        }

        for(int k_a=0; k_a<numTopics; ++k_a){
            int w=0;
            for(auto const& word_count : doc.wordCounts) {

                double prob_new = word_count.second * Pword[k_a][word_count.first];
                prob_new *= gamma[d][k_a];
                prob_new /= doc_gamma_sum;

                double denom_sum = 0;
                for(int k_b=0; k_b<numTopics; ++k_b){
                    double m_iab = gamma[d][k_b];
                    if(k_a == k_b){
                        m_iab += 1.0;
                    }
                    m_iab /= (doc_gamma_sum + 1);
                    denom_sum += Pword[k_b][word_count.first]*m_iab;
                }
                prob_new /= denom_sum;

                double Sa = -1.0;
                double enum_sum = 0;
                double denom = 0;
                for(int k_b=0; k_b<numTopics; ++k_b){
                    double m_iab = gamma[d][k_b];
                    if(k_a == k_b){
                        m_iab += 1.0;
                    }
                    m_iab /= (doc_gamma_sum + 1.0);
                    enum_sum += Pword[k_b][word_count.first]*Pword[k_b][word_count.first]*m_iab;
                    denom += Pword[k_b][word_count.first]*m_iab;
                }
                Sa += enum_sum/(denom*denom);
                prob_new *= (1.0 + (Sa)/(2.0+doc_gamma_sum) );
                word_prob_new[k_a][word_count.first] += prob_new;
                w++;
            }
        }
    }

    for(int k=0; k<numTopics; ++k){
        norm(word_prob_new[k]);
    }
    Pword = word_prob_new;
}
