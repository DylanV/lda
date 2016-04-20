//
// Created by Dylan on 13/04/2016.
//

#include "lda.h"
#include "alpha.h"
#include "util.h"
#include "data.h"
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <math.h>
#include <iostream>
#include <fstream>
#include <assert.h>

lda::lda(doc_corpus &corp)
{
    corpus = corp;
    numDocs = corpus.numDocs;
    numTerms = corpus.numTerms;
}

void lda::train(int num_topics)
{

    numTopics = num_topics;
    logProbW = std::vector<std::vector<double>>(numTopics, std::vector<double>(numTerms, 0));

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
    double likelihood  = 0;
    double old_likelihood = 0;
    double converged = 1;

    while ( ( (converged <0) || (converged>1e-4) ) && ( (iteration <= 2) || (iteration <= 100) ) ) {
        iteration++;
        likelihood = 0;
        clock_t start = clock();
        zeroSSInit(ss);

        for(int d=0; d<numDocs; d++){
            document doc = corpus.docs[d];
            std::vector<double>& var_gamma = varGamma[d];
            std::vector<std::vector<double>>& doc_phi = phi[d];
            likelihood += doc_e_step(doc, ss, var_gamma, doc_phi);
        }
        mle(ss, true);

//        std::cout << "Iteration: " << iteration << " with likelihood: " << likelihood
//            << " in " << double(clock() - start)/CLOCKS_PER_SEC << " seconds." << std::endl;

        converged = (old_likelihood - likelihood)/old_likelihood;
        old_likelihood = likelihood;

    }
    std::cout << "Converged in " << iteration << " iterations with likelihood of " << likelihood << std::endl;
    lda::likelihood = likelihood;
}

double lda::doc_e_step(document const& doc, suff_stats &ss, std::vector<double>& var_gamma,
                       std::vector<std::vector<double>>& phi)
{
    double likelihood = inference(doc, var_gamma, phi);

    double gamma_sum = 0;
    for(int k=0; k<numTopics; k++){
        gamma_sum += var_gamma[k];
        ss.alphaSS += digamma(var_gamma[k]);
    }
    ss.alphaSS -= numTopics * digamma(gamma_sum);

    int n=0;
    for(auto const& word_count : doc.wordCounts){
        for(int k=0; k<numTopics; k++){
            ss.classWord[k][word_count.first] += word_count.second * phi[n][k];
            ss.classTotal[k] += word_count.second * phi[n][k];
        }
        n++;
    }

    ss.numDocs++;

    return likelihood;
}

double lda::inference(document const& doc, std::vector<double>& var_gamma, std::vector<std::vector<double>>& phi)
{
    std::vector<double> old_phi(numTopics);
    std::vector<double> digamma_gam(numTopics);

    phi = std::vector<std::vector<double>>(doc.uniqueCount, std::vector<double>(numTopics, 1/numTopics));
    var_gamma = std::vector<double>(numTopics, alpha + doc.count/numTopics);

    for(int k=0; k<numTopics; k++){
        digamma_gam[k] = digamma(var_gamma[k]);
    }

    int iteration = 0;
    double converged = 1;
    double phisum;
    double likelihood = 0;
    double old_likelihood = 0;

    while((converged > 1e-6) && (iteration < 20)){
        iteration++;
        int n=0;

        for(auto const& word_count : doc.wordCounts){
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

double lda::compute_likelihood(document const &doc, std::vector<double>& var_gamma,
                               std::vector<std::vector<double>>& phi)
{

    double likelihood = 0;
    double var_gamma_sum = 0;
    std::vector<double> dig(numTopics);

    for(int k=0; k< numTopics; k++){
        dig[k] = digamma(var_gamma[k]);
        var_gamma_sum += var_gamma[k];
    }
    double digsum = digamma(var_gamma_sum);

    likelihood = lgamma(alpha * numTopics) - numTopics * lgamma(alpha) - lgamma(var_gamma_sum);

    for(int k=0; k<numTopics; k++){
        likelihood += (alpha - 1)*(dig[k] - digsum) + lgamma(var_gamma[k]) - (var_gamma[k] - 1)*(dig[k] - digsum);

        int n=0;
        for(auto const& word_count: doc.wordCounts){
            if(phi[n][k] > 0){
                likelihood += word_count.second *
                        ( phi[n][k] * ( (dig[k] - digsum)-log(phi[n][k])+logProbW[k][word_count.first]));
            }
            n++;
        }
    }
    return likelihood;
}

void lda::mle(suff_stats &ss, bool optAlpha=true)
{

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

void lda::randomSSInit(suff_stats& ss)
{
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

void lda::zeroSSInit(suff_stats& ss)
{
    ss.classWord = std::vector<std::vector<double>>(numTopics, std::vector<double>(numTerms, 0.0));
    ss.classTotal = std::vector<double>(numTopics, 0.0);
    ss.numDocs = 0;
    ss.alphaSS = 0;
}

void lda::writeBetaToFile(std::string folder_path)
{
    char sep = ' ';
    char nl = '\n';
    std::fstream beta_fs;
    beta_fs.open(folder_path+"beta.dat", std::fstream::out | std::fstream::trunc);
    if(beta_fs.is_open()){
        beta_fs << numTopics << sep << corpus.numTerms << nl;
        for(int k=0; k<numTopics; k++){
            for(int n=0; n<corpus.numTerms; n++){
                beta_fs << exp(logProbW[k][n]) << sep;
            }
            beta_fs << nl;
        }
    }
}

void lda::writeAlphaToFile(std::string folder_path)
{
    char nl = '\n';
    std::fstream fs;
    fs.open(folder_path+"alpha.dat", std::fstream::out | std::fstream::trunc);
    if(fs.is_open()){
        fs << alpha << nl;
    }
}

void lda::writeGammaToFile(std::string folder_path)
{
    char sep = ' ';
    char nl = '\n';
    std::fstream fs;
    fs.open(folder_path+"gamma.dat", std::fstream::out | std::fstream::trunc);
    if(fs.is_open()){
        std::vector<double> gammaSum(numDocs,0);
        for(int d=0; d<numDocs; d++){
            for(int k=0; k<numTopics; k++){
                gammaSum[d] += varGamma[d][k];
            }
        }
        fs << numDocs << sep << numTopics << nl;
        for(int d=0; d<numDocs; d++){
            for(int k=0; k<numTopics; k++){
                fs << varGamma[d][k]/gammaSum[d] << sep;
            }
            fs << nl;
        }
    }
}

void lda::writeParams(std::string folder_path)
{
    writeBetaToFile(folder_path);
    writeGammaToFile(folder_path);
    writeAlphaToFile(folder_path);
}

void lda::loadFromParams(std::string folder_path)
{
    char sep = ' ';
    logProbW = loadBetaFromFile(folder_path + "beta.dat");
}

std::vector<std::vector<double>> lda::loadBetaFromFile(std::string file_path) {

    std::vector<std::vector<double>> beta;
    char sep = ' ';

    std::ifstream fs(file_path);
    if(fs.is_open()){
        std::string line;
        bool readFirst = false;

        while(!fs.eof()){
            getline(fs, line);
            std::vector<std::string> items = split(line, sep);

            if(!readFirst){
                assert(items.size() == 2);
                readFirst = true;
                numTopics = stoi(items[0]);
                numTerms = stoi(items[1]);
                assert(numTerms == corpus.numTerms);
            } else{
                if(line != ""){
                    assert(items.size() == numTerms);
                    std::vector<double> top_probs;
                    for(auto const& item : items){
                        top_probs.push_back(stod(item));
                    }
                    beta.push_back(top_probs);
                }
            }
        }
    }

    return beta;
}



