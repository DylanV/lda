//
// Created by Dylan on 13/04/2016.
//

#include "lda.h"
#include "alpha.h"
#include "util.h"
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <math.h>
#include <iostream>

lda::lda(doc_corpus &corp, int num_topics) {
    corpus = corp;
    numDocs = corpus.numDocs;
    numTerms = corpus.numTerms;
    numTopics = num_topics;
    logProbW = std::vector<std::vector<double>>(numTopics, std::vector<double>(numTerms, 0));
    alpha = 1;
}

void lda::train(){
    suff_stats ss;
    randomSSInit(ss);
    mle(ss, false);

    alpha = 1.0/numTopics;

    varGamma = std::vector<std::vector<double>>(numDocs, std::vector<double>(numTopics, 0));
    phi = std::vector<std::vector<std::vector<double>>>(numDocs);
    for(int d=0; d<corpus.numDocs; d++){
        phi[d] = std::vector<std::vector<double>>(corpus.docs[d].uniqueCount, std::vector<double>((numTopics)));
    }

    int iteration = 0;
    double likelihood;
    double old_likelihood = 0;
    double converged = 1;

    while ( ( (converged <0) || (converged>1e-4) ) && ( (iteration > 2) || (iteration < 100) ) ) {
        iteration++;
        likelihood = 0;

        zeroSSInit(ss);
        std::cout << "iteration" << iteration << std::endl;

        for(int d=0; d<numDocs; d++){
            document doc = corpus.docs[d];
            std::vector<double> var_gamma = varGamma[d];
            std::vector<std::vector<double>> doc_phi = phi[d];
            likelihood += doc_e_step(doc, ss, var_gamma, doc_phi);
        }
        std::cout << "likelihood: " << likelihood << std::endl;
        mle(ss, true);

        converged = (old_likelihood - likelihood)/old_likelihood;
        old_likelihood = likelihood;

    }
}

double lda::doc_e_step(document const& doc, suff_stats &ss, std::vector<double>& var_gamma, std::vector<std::vector<double>>& phi) {

    double likelihood = inference(doc, var_gamma, phi);

    double gamma_sum = 0;
    for(int k=0; k<numTopics; k++){
        gamma_sum += var_gamma[k];
        ss.alphaSS += digamma(var_gamma[k]);
    }
    ss.alphaSS -= numTopics * digamma(gamma_sum);

    int n=0;
    for(std::pair<int,int> const& word_count : doc.wordCounts){
        for(int k=0; k<numTopics; k++){
            ss.classWord[k][word_count.first] += word_count.second * phi[n][k];
            ss.classTotal[k] += word_count.second * phi[n][k];
        }
        n++;
    }

    ss.numDocs++;

    return likelihood;
}

double lda::inference(document const& doc, std::vector<double>& var_gamma, std::vector<std::vector<double>>& phi){
    std::vector<double> old_phi(numTopics);
    std::vector<double> digamma_gam(numTopics);

    for(int k=0; k<numTopics; k++){
        var_gamma[k] = alpha + doc.count/numTopics;
        digamma_gam[k] = digamma(var_gamma[k]);
        for(int n=0; n<doc.uniqueCount; n++){
            phi[n][k] = 1/numTopics;
        }
    }

    int iteration = 0;
    double converged = 1;
    double phisum;
    double likelihood = 0;
    double old_likelihood = 0;

    while((converged > 1e-6) && (iteration < 20)){
        iteration++;
        int n=0;

        for(std::pair<int, int> const& word_count : doc.wordCounts){
            phisum = 0;
            for(int k=0; k<numTopics; k++){
                old_phi[k] = phi[n][k];
                phi[n][k] = digamma_gam[k] + logProbW[k][word_count.first];

                if(k>0){
                    phisum = log_sum(phisum, phi[n][k]);
                } else {
                    phisum = phi[n][k];
                }
            }

            for(int k=0; k<numTopics; k++){
                phi[n][k] = exp(phi[n][k] - phisum);
                var_gamma[k] = var_gamma[k] + word_count.second*(phi[n][k] - old_phi[k]);
                digamma_gam[k] = digamma(var_gamma[k]);
            }
            n++;
        }

        likelihood = compute_likelihood(doc, var_gamma, phi);
        converged = (old_likelihood - likelihood)/old_likelihood;
        old_likelihood = likelihood;
    }

    return likelihood;
}

double lda::compute_likelihood(document const &doc, std::vector<double>& var_gamma, std::vector<std::vector<double>>& phi) {

    double likelihood = 0;
    double var_gamma_sum = 0;
    std::vector<double> dig(numTopics);

    for(int k=0; k< numTopics; k++){
        dig[k] = digamma(var_gamma[k]);
        var_gamma_sum += var_gamma[k];
    }
    double digsum = digamma(var_gamma_sum);

    likelihood = lgamma(alpha * numTopics);
    likelihood -= numTopics * lgamma(alpha);
    likelihood -= lgamma(var_gamma_sum);

    for(int k=0; k<numTopics; k++){
        likelihood += (alpha - 1)*(dig[k] - digsum);
        likelihood += lgamma(var_gamma[k]);
        likelihood -= (var_gamma[k] - 1)*(dig[k] - digsum);

        int n=0;
        for(std::pair<int,int> const& word_count: doc.wordCounts){
            if(phi[n][k] > 0){
                likelihood += word_count.second * ( phi[n][k] * ( (dig[k] - digsum)-log(phi[n][k])+logProbW[k][word_count.first]));
            }
            n++;
        }
    }
    return likelihood;
}

void lda::mle(suff_stats &ss, bool optAlpha=true) {

    for(int k=0; k<numTopics;k++){
        for(int w=0; w<numTerms; w++){
            if(ss.classWord[k][w] > 0){
                logProbW[k][w] = log(ss.classWord[k][w])- log(ss.classTotal[k]);
            } else {
                logProbW[k][w] = -100.0;
            }
        }
    }
    if(optAlpha){
        alpha = opt_alpha(ss.alphaSS, ss.numDocs, numTopics);
    }
}

void lda::randomSSInit(suff_stats& ss) {
    ss.classWord = std::vector<std::vector<double>>(numTopics, std::vector<double>(numTerms, 1.0/numTerms));
    ss.classTotal = std::vector<double>(numTopics, 0.0);

    srand(time(NULL));

    for(int k=0; k<numTopics; k++){
        for(int n=0; n<numTerms; n++){
            ss.classWord[k][n] += rand();
            ss.classTotal[k] += ss.classWord[k][n];
        }
    }
}

void lda::zeroSSInit(suff_stats& ss) {
    ss.classWord = std::vector<std::vector<double>>(numTopics, std::vector<double>(numTerms, 0.0));
    ss.classTotal = std::vector<double>(numTopics, 0.0);
    ss.numDocs = 0;
    ss.alphaSS = 0;
}





