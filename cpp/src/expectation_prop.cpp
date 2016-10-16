
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
        numSkipped = 0;
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

        std::cout << "Iteration " << iteration << ": with likelihood: " << likelihood <<  " " << numSkipped <<std::endl;
    }
}

void expectation_prop::setup_parameters() {
    alpha = std::vector<double>(numTopics, ALPHA_INIT);
    Pword = std::vector<std::vector<double>>(numTopics, std::vector<double>(numTerms, 0.0));

    std::vector<std::vector<double>> classWord
            = std::vector<std::vector<double>>(numTopics, std::vector<double>(numTerms, 1.0/numTerms));
    std::vector<double> classTotal = std::vector<double>(numTopics, 0.0);

    srand (time(NULL));

    for(int k=0; k<numTopics; k++){
        for(int n=0; n<numTerms; n++){
            classWord[k][n] += (rand());
            classTotal[k] += classWord[k][n];
        }
    }

    for(int k=0; k<numTopics;k++){
        for(int w=0; w<numTerms; w++){
            Pword[k][w] = log(classWord[k][w]) - log(classTotal[k]);
            Pword[k][w] = exp(Pword[k][w]);
        }
    }

//    Pword[0][0] = 0.2;
//    Pword[0][1] = 0.2;
//    Pword[0][2] = 0.5;
//    Pword[0][9] = 0.1;
//
//    Pword[1][3] = 0.3;
//    Pword[1][4] = 0.3;
//    Pword[1][5] = 0.3;
//    Pword[1][9] = 0.1;
//
//    Pword[2][6] = 0.2;
//    Pword[2][7] = 0.3;
//    Pword[2][8] = 0.4;
//    Pword[2][9] = 0.1;

    s = std::vector<std::vector<double>>(numDocs);
    beta = std::vector<std::vector<std::vector<double>>>(numDocs);
    for(int d=0; d<numDocs; ++d){
        s[d] = std::vector<double>(corpus.docs[d].count);
        beta[d] = std::vector<std::vector<double>>(corpus.docs[d].count, std::vector<double>(numTopics, 0));
    }
    gamma = std::vector<std::vector<double>>(numDocs, std::vector<double>(numTopics, ALPHA_INIT));
}

double expectation_prop::doc_e_step(int d) {
    document doc = corpus.docs[d];
    std::map<int, int> local_counts = doc.wordCounts;
    int all_zero = 0;

    int w=0;
    while(all_zero == 0){
        all_zero = 1;
        for(auto &word_count : local_counts){
            int type = word_count.first;
            double token_count = word_count.second;

            //deletion
            std::vector<double> old_posterior = std::vector<double>(numTopics);
            bool skipWord = false;
            double old_posterior_total = 0.0;
            for (int k = 0; k < numTopics; ++k) {
                old_posterior[k] = gamma[d][k] - beta[d][w][k];
                old_posterior_total += old_posterior[k];
                if (old_posterior[k] < 0) {
                    skipWord = true;
                    numSkipped++;
                    break;
                }
            }
            // skip words where any of the 'old' posterior < 0
            if (!skipWord) {

                // calculate Z_w needed for moment matching and update steps
                double Z_w = 0.0;
                double prob_total = 0.0;
                for (int k = 0; k < numTopics; ++k) {
                    prob_total += Pword[k][type] * old_posterior[k];
                }
                Z_w = prob_total / old_posterior_total;

                // moment matching
                double m_a;
                double m2_a;
                std::vector<double> gamma_prime = std::vector<double>(numTopics);
                double numer_sum = 0;
                double denom_sum = 0;

                for (int k = 0; k < numTopics; ++k) {
                    m_a = (1.0 / Z_w);
                    m_a *= (old_posterior[k] / old_posterior_total);
                    m_a *= ((Pword[k][type] + prob_total) / (1.0 + old_posterior_total));

                    m2_a = (1.0 / Z_w);
                    m2_a *= (old_posterior[k] / old_posterior_total);
                    m2_a *= ((old_posterior[k] + 1) / (old_posterior_total + 1.0));
                    m2_a *= ((2.0 * Pword[k][word_count.first] + prob_total) / (2.0 + old_posterior_total));

                    gamma_prime[k] = m_a;
                    numer_sum += (m_a - m2_a);
                    denom_sum += (m2_a - m_a * m_a);
                }
                for (int k = 0; k < numTopics; ++k) {
                    gamma_prime[k] *= (numer_sum / denom_sum);
                }

                // update
                double step_size = 1.0/token_count;
                std::vector<double> beta_new = std::vector<double>(numTopics);
                bool make_changes = true;
                for (int k = 0; k < numTopics; ++k) {
                    beta_new[k] = (1.0-step_size)*beta[d][w][k];
                    beta_new[k] += step_size*(gamma_prime[k] - old_posterior[k]);
                }
                // inclusion
                std::vector<double> gamma_new = std::vector<double>(numTopics);
                for (int k = 0; k < numTopics; ++k) {
                    gamma_new[k] = old_posterior[k] + (token_count) * (beta_new[k] - beta[d][w][k]);
                    if (gamma_new[k] < 0) {
                        make_changes = false;
                        numSkipped++;
                    }
                }
                if (make_changes) {
                    // inclusion continued
                    for (int k = 0; k < numTopics; ++k) {
                        gamma[d][k] = gamma_new[k];
                    }
                    // update continued
                    for (int k = 0; k < numTopics; ++k) {
                        beta[d][w][k] = beta_new[k];
                    }
                    double prod_gamma_prime = 1.0;
                    double prod_gamma_old = 1.0;
                    double gamma_prime_sum = 0;
                    for (int k = 0; k < numTopics; ++k) {
                        prod_gamma_prime *= tgamma(gamma_prime[k]);
                        prod_gamma_old *= tgamma(old_posterior[k]);
                        gamma_prime_sum += gamma_prime[k];
                    }

                    s[d][w] = Z_w;
                    s[d][w] *= (tgamma(gamma_prime_sum) / prod_gamma_prime);
                    s[d][w] *= (prod_gamma_old / tgamma(old_posterior_total));
                }
            }
            word_count.second = word_count.second - 1;
            if(word_count.second != 0){
                all_zero *= 0;
            }
            ++w;
        }
    }

    // calculate document likelihood
    double gamma_sum = 0;
    double gamma_prod = 1.0;
    double alpha_sum = 0;
    double alpha_prod = 1.0;
    for(int k=0; k<numTopics; k++){
        gamma_sum += gamma[d][k];
        alpha_sum += alpha[k];
        gamma_prod *= tgamma(gamma[d][k]);
        alpha_prod *= tgamma(alpha[k]);
    }
    double likelihood = (gamma_prod/tgamma(gamma_sum)) * (tgamma(alpha_sum)/alpha_prod);
    for(int w=0; w<doc.count; ++d) {
        likelihood *= s[d][w];
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
            }
        }
    }

    for(int k=0; k<numTopics; ++k){
        norm(word_prob_new[k]);
    }
    Pword = word_prob_new;
}
