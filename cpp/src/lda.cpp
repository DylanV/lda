//
// Created by Dylan on 13/04/2016.
//

#include "lda.h"
#include "alpha.h"
#include "util.h"
#include "data.h"
#include <math.h>
#include <iostream>
#include <fstream>
#include <assert.h>

lda::lda(doc_corpus &corp, lda_settings settings)
{
/*!
     The constructor for the lda class. Accepts a reference to the document corpus,
     \param corp a reference to the document corpus for the lda.
 */
    corpus = corp;
    numDocs = corpus.numDocs;
    numTerms = corpus.numTerms;

    CONV_THRESHHOLD = settings.converged_threshold;
    MIN_ITER = settings.min_iterations;
    MAX_ITER = settings.max_iterations;
    INF_CONV_THRESH = settings.inf_converged_threshold;
    INF_MAX_ITER = settings.inf_max_iterations;
}

void lda::train(int num_topics, alpha_settings a_settings)
{
/*!
    Perform variational bayes on the corpus to get the dirichlet parameters
    (logProbW->beta and alpha) for the corpus.
    \param num_topics an int for the number of topics to train on
*/

    numTopics = num_topics;
    logProbW = std::vector<std::vector<double>>(numTopics, std::vector<double>(numTerms, 0));

    EST_ALPHA = a_settings.estimate_alpha;
    alpha = setup_alpha(a_settings);

    suff_stats ss;
    randomSSInit(ss);
    mle(ss, false);

    varGamma = std::vector<std::vector<double>>(numDocs, std::vector<double>(numTopics, 0));
    phi = std::vector<std::vector<std::vector<double>>>(numDocs);
    for(int d=0; d<corpus.numDocs; d++){  
        phi[d] = std::vector<std::vector<double>>(corpus.docs[d].uniqueCount, std::vector<double>((numTopics)));
    }

    int iteration = 0;
    double likelihood  = 0;
    double old_likelihood = 0;
    double converged = 1;

    while ( ( (converged <0) || (converged>CONV_THRESHHOLD) ) && ( (iteration <= MIN_ITER) || (iteration <= MAX_ITER) ) ) {
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
        mle(ss, EST_ALPHA);

        std::cout << "Iteration " << iteration << ": with likelihood: " << likelihood
            << " in " << double(clock() - start)/CLOCKS_PER_SEC << " seconds." << std::endl;

        converged = (old_likelihood - likelihood)/old_likelihood;
        old_likelihood = likelihood;

    }
    std::cout << "Converged in " << iteration << " iterations with likelihood of " << likelihood << std::endl;
    lda::likelihood = likelihood;
}

double lda::doc_e_step(document const& doc, suff_stats &ss, std::vector<double>& var_gamma,
                       std::vector<std::vector<double>>& phi)
{
/*!
    Calls inference to update the latent parameters (gamma, phi) for the current document \sa inference().
    Then updates the sufficient statistics for the lda.
    \param doc reference to current document \sa document
    \param ss reference to the model sufficient statistics \sa suff_stats
    \param var_gamma reference to the current document topic distribution parameter \sa varGamma
    \param phi reference to the current document topic-word distributionn \sa phi
    \return the log likelihood for the document
 */
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
/*!
    Performs inference to update the latent dirichlet factors gamma and phi for the given document.
    \param doc reference to current document \sa document
    \param ss reference to the model sufficient statistics \sa suff_stats
    \param var_gamma reference to the current document topic distribution parameter \sa varGamma
    \param phi reference to the current document topic-word distribution \sa phi
    \return the log likelihood for the document
    \sa doc_e_step(), compute_likelihood()
 */
    std::vector<double> old_phi(numTopics);
    std::vector<double> digamma_gam(numTopics);

//    phi = std::vector<std::vector<double>>(doc.uniqueCount, std::vector<double>(numTopics, 1/numTopics));
//    var_gamma = std::vector<double>(numTopics, alpha + doc.count/numTopics);

    for(int w=0; w<doc.uniqueCount; w++){
        for(int k=0; k<numTopics; k++){
            phi[w][k] = 1/numTopics;
        }
    }

    double init = alpha.s + doc.count/numTopics;
    for(int k=0; k<numTopics; k++){
        var_gamma[k] = init;
    }

    for(int k=0; k<numTopics; k++){
        digamma_gam[k] = digamma(var_gamma[k]);
    }

    int iteration = 0;
    double converged = 1;
    double phisum;
    double likelihood = 0;
    double old_likelihood = 0;

    while((converged > INF_CONV_THRESH) && (iteration < INF_MAX_ITER)){
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
/*!
    \param doc reference to current document \sa document
    \param var_gamma reference to the current document topic distribution parameter \sa varGamma
    \param phi reference to the current document topic-word distribution  \sa phi
    \return the log likelihood for the document
    \sa doc_e_step(), compute_likelihood()
 */
    double likelihood = 0;
    double var_gamma_sum = 0;
    std::vector<double> dig(numTopics);

    for(int k=0; k< numTopics; k++){
        dig[k] = digamma(var_gamma[k]);
        var_gamma_sum += var_gamma[k];
    }
    double digsum = digamma(var_gamma_sum);

    likelihood = lgamma(alpha.s * numTopics) - numTopics * lgamma(alpha.s) - lgamma(var_gamma_sum);

    for(int k=0; k<numTopics; k++){
        likelihood += (alpha.s - 1)*(dig[k] - digsum) + lgamma(var_gamma[k]) - (var_gamma[k] - 1)*(dig[k] - digsum);

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
/*!
    Calculates a maximum likelihood model for the lda given the sufficient stats.
    Newton-Raphson update on alpha is optional. optAlpha default is true.
    \param ss a reference to the sufficient statistics struct for the lda
    \param optAlpha set to true if alpha should be inferred
    \sa suff_stats
 */
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
        alpha.estimate_precision(ss.alphaSS, ss.numDocs);
        std::cout << alpha.s << std::endl;
    }
}

dirichlet lda::setup_alpha(alpha_settings settings){
    return dirichlet(numTopics, settings);
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
//    ss.classWord = std::vector<std::vector<double>>(numTopics, std::vector<double>(numTerms, 0.0));
//    ss.classTotal = std::vector<double>(numTopics, 0.0);

    for(int k=0; k<numTopics; k++){
        for(int w=0; w<numTerms; w++){
            ss.classWord[k][w] = 0;
        }
        ss.classTotal[k] = 0;
    }

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
                beta_fs << exp(logProbW[k][n]);
                if(n != corpus.numTerms-1){
                    beta_fs << sep;
                }
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
        fs << alpha.s << nl;
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
                fs << varGamma[d][k]/gammaSum[d];
                if(k != numTopics - 1){
                    fs << sep;
                }
            }
            fs << nl;
        }
    }
}

void lda::writeParams(std::string folder_path)
{
/*!
    Writes the lda alpha, beta and gamma dirchlet parameters to files.
    File contains the shape of the parameter in the first line. Followed by the parameter line seperated
    by vector indix if 2d vector and space separated for 1d vector. So a 2d beta parameter will have each
    space separated topic distribtion on a new line.
    /param folder_path the path to the folder where the parameter files will be written.
 */
    writeBetaToFile(folder_path);
    writeGammaToFile(folder_path);
    writeAlphaToFile(folder_path);
}

void lda::loadFromParams(std::string folder_path)
{
/*!
    /param folder_path the path to the folder containing the parameter files
    /sa writeParams()
 */
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



