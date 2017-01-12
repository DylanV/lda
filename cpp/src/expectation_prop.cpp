
#include "expectation_prop.h"
#include "util.h"
#include "data.h"
#include <math.h>
#include <cstdlib>
#include <iostream>

expectation_prop::expectation_prop(const doc_corpus &corpus, const ep_settings &settings) {
    this->corpus = corpus;
    numDocs = corpus.numDocs;
    numTerms = corpus.numTerms;

    MAX_ITERATION = settings.MAX_ITER;
    E_MAX_ITERATIONS = settings.E_MAX_ITERATIONS;
    CONV_THRESHHOLD = settings.CONV_THRESH;
    ALPHA_INIT = settings.ALPHA_INIT;
}

void expectation_prop::save_parameters(std::string file_dir) {
    std::fstream fs; //get the filestream

    // Save beta p(w|z)
    fs.open(file_dir+"beta.dat", std::fstream::out | std::fstream::trunc);
    write_2d_vector_to_fs(fs, Pword);
    fs.close();

    // Save gamma
    fs.open(file_dir+"gamma.dat", std::fstream::out | std::fstream::trunc);
    write_2d_vector_to_fs(fs, gamma);
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
        first = false;

        std::cout << "Iteration " << iteration << ": with likelihood: " << likelihood <<std::endl;
    }
}

void expectation_prop::setup_parameters() {
    alpha = std::vector<double>(numTopics, ALPHA_INIT);

    dirichlet dir_beta = dirichlet(numTerms, 1.0);
    Pword = dir_beta.sample(numTopics);

    beta = std::vector<std::vector<std::vector<double>>>(numDocs);
    for(int d=0; d<numDocs; ++d){
        beta[d] = std::vector<std::vector<double>>(corpus.docs[d].uniqueCount, std::vector<double>(numTopics, 0));
    }
    gamma = std::vector<std::vector<double>>(numDocs, std::vector<double>(numTopics, ALPHA_INIT));
}

double expectation_prop::doc_e_step(int d) {

    // Document variables
    document doc = corpus.docs[d];
    std::vector<double> doc_gamma = std::vector<double>(numTopics,ALPHA_INIT);
    std::vector<std::vector<double>> doc_beta = beta[d];
    std::vector<double> s = std::vector<double>(doc.uniqueCount, 1);
    int w=0; // used to index beta and s. Not actually the type w
    int converged = 1;
    int skipped = 0;
    double beta_change = 0;
    double log_likelihood = 0.0;
    int max_iter = E_MAX_ITERATIONS;
    if(!first){
        max_iter /= 2;
    }
    double s_total = 0;

    // these vectors are initialised here to save having to re-init them every iteration
    std::vector<double> gamma_not_w = std::vector<double>(numTopics);
    std::vector<double> new_beta = std::vector<double>(numTopics);
    std::vector<double> new_gamma = std::vector<double>(numTopics);

    // Calculate gamma
    for(int k=0; k<numTopics; ++k){
        w=0;
        for(auto const& type_count : doc.wordCounts){
            doc_gamma[k] += doc_beta[w][k] * type_count.second;
            ++w;
        }
    }

    for(int iter=0; iter<max_iter; ++ iter){
        converged = 1;
        skipped = 0;
        w=0;
        beta_change = 0;
        s_total = 0;
        for(auto const& type_count : doc.wordCounts){

            int type = type_count.first;
            int n_w = type_count.second;

            // Get the 'old' posterior
            bool any_gamma_not_w_neg = false;
            for(int k=0; k<numTopics; ++k){
                gamma_not_w[k] = doc_gamma[k] - doc_beta[w][k];
                if(gamma_not_w[k] < 0){
                    any_gamma_not_w_neg = true;
                }
            }
            // If any of the old posterior are negative skip this word for this iteration
            if(any_gamma_not_w_neg){
                ++skipped;
                ++w;
                continue;
            }

            // For convenience calculate the sum of the old posterior just once
            double sum_gamma_not_w = sum(gamma_not_w);

            // Calculate the new beta for this word
            // First calculate Z_w needed for moment matching and update steps
            double Z_w = 0.0;
            double prob_total = 0.0;
            for (int k = 0; k < numTopics; ++k) {
                prob_total += Pword[k][type] * gamma_not_w[k];
            }
            Z_w = prob_total / sum_gamma_not_w;

            // Use moment matching to calculate gamma'
            double m_a;
            double m2_a;
            std::vector<double> gamma_prime = std::vector<double>(numTopics);
            double numer_sum = 0;
            double denom_sum = 0;

            for (int k = 0; k < numTopics; ++k) {
                m_a = (1.0 / Z_w);
                m_a *= (gamma_not_w[k] / sum_gamma_not_w);
                m_a *= ((Pword[k][type] + prob_total) / (1.0 + sum_gamma_not_w));

                m2_a = (1.0 / Z_w);
                m2_a *= (gamma_not_w[k] / sum_gamma_not_w);
                m2_a *= ((gamma_not_w[k] + 1) / (sum_gamma_not_w + 1.0));
                m2_a *= ((2.0 * Pword[k][type] + prob_total) / (2.0 + sum_gamma_not_w));

                gamma_prime[k] = m_a;
                numer_sum += (m_a - m2_a);
                denom_sum += (m2_a - m_a * m_a);
            }
            for (int k = 0; k < numTopics; ++k) {
                gamma_prime[k] *= (numer_sum / denom_sum);
                // Get the new beta
                new_beta[k] = gamma_prime[k] - gamma_not_w[k];
            }

            // Damp the changes in beta and update gamma
            bool any_gamma_neg = false;
            double step_size = 1.0/n_w;
            for(int k=0; k<numTopics; ++k){
                new_beta[k] = step_size*new_beta[k] + (1-step_size)*doc_beta[w][k];
                new_gamma[k] = doc_gamma[k] + n_w * (new_beta[k] - doc_beta[w][k]);
                if(new_gamma[k] < 0){
                    any_gamma_neg = true;
                }
            }
            // If any of the new gamma are negative skip this word
            if(any_gamma_neg){
                ++skipped;
                ++w;
                continue;
            }

            // Update
            doc_gamma = new_gamma;
            //Calculate the largest change in beta
            for(int k=0; k<numTopics; ++k){
                double change = fabs(new_beta[k] - doc_beta[w][k]);
                if(change > beta_change){
                    beta_change = change;
                }
            }
            doc_beta[w] = new_beta;
            // If the change in beta for this word is too great we are still not converged
            if(beta_change > 1e-5){
                converged *= 0;
            }

            // Calculate s for this word in log space
            std::vector<double> gam_beta = std::vector<double>(numTopics);
            bool any_neg_gam_beta = false;
            for(int k=0; k<numTopics; ++k){
                gam_beta[k] = doc_gamma[k] + doc_beta[w][k];
                if(gam_beta[k] < 0){
                    any_neg_gam_beta = true;
                }
            }

            if((!any_neg_gam_beta and (skipped == 0 and converged == 1)) or iter == max_iter-1){
                s[w] = log(Z_w);
                for(int k=0; k<numTopics; ++k){
                    s[w] += lgamma(doc_gamma[k]);
                    s[w] -= lgamma(gam_beta[k]);
                }
                s[w] -= lgamma(sum(doc_gamma));
                s[w] += lgamma(sum(gam_beta));
                s[w] *= n_w;
            }
            s_total += s[w];

            // Next word
            ++w;
        }

        if(converged == 1 and skipped == 0){
            break;
        }
    }

    gamma[d] = doc_gamma;
    beta[d] = doc_beta;

    log_likelihood = numTopics*lgamma(ALPHA_INIT) - lgamma(ALPHA_INIT*numTopics);
    for(int k=0; k<numTopics; ++k){
        log_likelihood += lgamma(gamma[d][k]);
    }
    log_likelihood += lgamma(sum(gamma[d]));
    log_likelihood += s_total;

    return log_likelihood;
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
