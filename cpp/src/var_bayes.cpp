//
// Created by Dylan Verrezen on 2016/07/25.
//

#include "var_bayes.h"
#include "util.h"
#include <math.h>
#include "data.h"
#include <iostream>
#include <fstream>

var_bayes::var_bayes(doc_corpus &corp, lda_settings settings, alpha_settings a_settings)
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
    EST_ALPHA = settings.estimate_alpha;
    UPDATE_INTERVAL = settings.alpha_update_interval;

    ALPHA_SETTINGS = a_settings;
}

void var_bayes::train(size_t numTopics) {
    train(numTopics, ALPHA_SETTINGS);
}

void var_bayes::save_parameters(std::string file_dir) {
    std::fstream fs; //get the filestream

    //get beta from logProbW
    std::vector<std::vector<double> > beta
            = std::vector<std::vector<double> >(numTopics, std::vector<double>(numTerms, 0));
    for(int k=0; k<numTopics; k++){
        for(int n=0; n<corpus.numTerms; n++){
            beta[k][n] = exp(logProbW[k][n]);
        }
    }
    //write beta
    fs.open(file_dir+"beta.dat", std::fstream::out | std::fstream::trunc);
    write_2d_vector_to_fs(fs, beta);
    fs.close();

    //write gamma
    fs.open(file_dir+"gamma.dat", std::fstream::out | std::fstream::trunc);
    write_2d_vector_to_fs(fs, varGamma);
    fs.close();

    //write alpha
    fs.open(file_dir+"alpha.dat", std::fstream::out | std::fstream::trunc);
    write_vector_to_fs(fs, alpha.alpha);
    fs.close();
}

void var_bayes::train(int num_topics, alpha_settings a_settings)
{
/*!
    Perform variational bayes on the corpus to get the dirichlet hyperparameters
    (logProbW->beta and alpha) for the corpus.
    \param num_topics an int for the number of topics to train on
*/

    numTopics = size_t(num_topics);
    logProbW = std::vector<std::vector<double>>(numTopics, std::vector<double>(numTerms, 0));

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
    bool update_alpha;

    while ( ( (converged>CONV_THRESHHOLD) ) and ( (iteration <= MIN_ITER) or (iteration <= MAX_ITER) ) ) {
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
        update_alpha = ((iteration % UPDATE_INTERVAL == 0) and (EST_ALPHA));
        mle(ss, update_alpha);

        converged = fabs((old_likelihood - likelihood)/old_likelihood);
        old_likelihood = likelihood;

        std::cout << "Iteration " << iteration << ": with likelihood: " << likelihood
                  << " in " << double(clock() - start)/CLOCKS_PER_SEC << " seconds. (" << converged << ")";
        if(update_alpha){
            std::cout << " (alpha update)";
        }
        std::cout << std::endl;
    }
    std::cout << "Converged in " << iteration << " iterations with likelihood of " << likelihood << std::endl;
    var_bayes::likelihood = likelihood;
}

double var_bayes::doc_e_step(document const& doc, suff_stats &ss, std::vector<double>& var_gamma,
                       std::vector<std::vector<double>>& phi)
{
/*!
    Calls inference method to update the latent parameters (gamma, phi) for the current document \sa inference().
    Then updates the sufficient statistics for the lda.
    \param doc reference to current document \sa document
    \param ss reference to the model sufficient statistics \sa suff_stats
    \param var_gamma reference to the current document topic distribution parameter \sa varGamma
    \param phi reference to the current document topic-word distributionn \sa phi
    \return the log likelihood for the document
 */
    double likelihood = inference(doc, var_gamma, phi);

    int topic=0;
    for(const double& val: dirichlet_expectation(var_gamma)){
        ss.alpha_ss[topic] += val;
        topic++;
    }

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

double var_bayes::inference(document const& doc, std::vector<double>& var_gamma, std::vector<std::vector<double>>& phi)
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
    std::vector<double> digamma_gam(numTopics);

    for(int k=0; k<numTopics; k++){
        var_gamma[k] = alpha.alpha[k] + doc.count/numTopics;
    }

    int iteration = 0;
    double converged = 1;
    double phisum;
    std::vector<double> prev_gamma = std::vector<double>(numTopics);

    while((converged > INF_CONV_THRESH) and (iteration < INF_MAX_ITER)){
        iteration++;

        for(int k=0; k<numTopics; k++){
            digamma_gam[k] = digamma(var_gamma[k]);
            prev_gamma[k] = var_gamma[k];
            var_gamma[k] = alpha.alpha[k];
        }

        int n=0;
        for(auto const& word_count : doc.wordCounts){
            phisum = 0;
            for(int k=0; k<numTopics; k++){
                phi[n][k] = digamma_gam[k] + logProbW[k][word_count.first];

                if(k>0){
                    phisum = log_sum(phisum, phi[n][k]);
                } else {
                    phisum = phi[n][k];
                }
            }

            for(int k=0; k<numTopics; k++){
                phi[n][k] = exp(phi[n][k] - phisum);
                var_gamma[k] = var_gamma[k] + word_count.second*(phi[n][k]);
            }
            n++;
        }

        converged = 0;
        for(int k=0; k<numTopics; ++k){
            converged += fabs(prev_gamma[k] - var_gamma[k]);
        }
        converged /= numTopics;
    }

    return compute_likelihood(doc, var_gamma, phi);;
}

double var_bayes::compute_likelihood(document const &doc, std::vector<double>& var_gamma,
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

    for(int k=0; k<numTopics; k++){
        likelihood += lgamma(alpha.alpha[k] * numTopics);
        likelihood -= numTopics * lgamma(alpha.alpha[k]) - lgamma(var_gamma_sum);
        likelihood += (alpha.alpha[k] - 1)*(dig[k] - digsum);
        likelihood += lgamma(var_gamma[k]) - (var_gamma[k] - 1)*(dig[k] - digsum);

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

void var_bayes::mle(suff_stats &ss, bool optAlpha=true)
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
        alpha.update(ss.alpha_ss, ss.numDocs);
    }
}

dirichlet var_bayes::setup_alpha(alpha_settings settings){
    return dirichlet(numTopics, settings);
}

void var_bayes::randomSSInit(suff_stats& ss)
{
    ss.classWord = std::vector<std::vector<double>>(numTopics, std::vector<double>(numTerms, 1.0/numTerms));
    ss.classTotal = std::vector<double>(numTopics, 0.0);
    ss.alpha_ss = std::vector<double>(numTopics, 0);

    srand(time(NULL));

    for(int k=0; k<numTopics; k++){
        for(int n=0; n<numTerms; n++){
            ss.classWord[k][n] += rand();
            ss.classTotal[k] += ss.classWord[k][n];
        }
    }
}

void var_bayes::zeroSSInit(suff_stats& ss)
{

    for(int k=0; k<numTopics; k++){
        for(int w=0; w<numTerms; w++){
            ss.classWord[k][w] = 0;
        }
        ss.classTotal[k] = 0;
        ss.alpha_ss[k] = 0;
    }

    ss.numDocs = 0;
}