#include "online_var_bayes.h"
#include "data.h"
#include "util.h"
#include <fstream>

using namespace std;

online_var_bayes::online_var_bayes(const doc_corpus &corp, const online_var_bayes_settings &settings){
    ALPHA = 0.1;
    ETA = 0.01;
    tau = 2;
    kappa = 0.75;

    corpus = corp;
    numDocs = corp.numDocs;
    numTerms = corp.numTerms;
    batchSize = 5;
}

void online_var_bayes::save_parameters(std::string file_dir) {
    fstream fs; //get the filestream
    //write beta
    fs.open(file_dir+"beta.dat", fstream::out | fstream::trunc);
    write_2d_vector_to_fs(fs, Elogbeta);
    fs.close();
}

void online_var_bayes::train(size_t numTopics) {
    updateCount = 0;
    this->numTopics = numTopics;

    int batchStartId, batchEndID = 0;
    int iter = 0;

    default_random_engine generator;

    // Get the topic distribution
    lambda = vector<vector<double>>(numTopics, vector<double>(numTerms));
    Elogbeta = vector<vector<double>>(numTopics);
    gamma_distribution<double> gam(2.0, 1.0);
    for(int i=0; i<numTopics; ++i){
        for(int w=0; w<numTerms; ++w){
            lambda[i][w] = gam(generator);
        }
        Elogbeta[i] = dirichlet_expectation(lambda[i]);
    }

    // Loop through corpus in batches
    while(batchEndID < numDocs-1){
        //Get the batch start and end ids
        batchStartId = iter*batchSize;
        batchEndID = (iter + 1)*batchSize;
        if(batchEndID > numDocs){
            batchEndID=numDocs;
        }
        int numDocsBatch = batchEndID - batchStartId;

        //new lambda
        vector<vector<double>> lambda_new = vector<vector<double>>(numTopics, vector<double>(numTerms, 0));

        //E-STEP
        // get the batch variational distributions
        vector<vector<double>> gamma = vector<vector<double>>(numDocsBatch, vector<double>(numTopics));
        vector<vector<double>> Elogtheta = vector<vector<double>>(numDocsBatch);

        gamma_distribution<double> gam(2.0, 1.0);
        for(int i=0; i<numDocsBatch; ++i){
            for(int k=0; k<numTopics; ++k){
                gamma[i][k] = gam(generator);
            }
            Elogtheta[i] = dirichlet_expectation(gamma[i]);
        }

        vector<vector<vector<double>>> phi = vector<vector<vector<double>>>(numDocsBatch);

        // work through batch and for each document update gamma and phi
        int batchIndex = 0; // doc index in batch
        for(int d = batchStartId; d<batchEndID; ++d){
            document doc = corpus.docs[d];
            phi[batchIndex] = vector<vector<double>>(doc.uniqueCount, vector<double>(numTopics, 0));
            bool converged = false;
            while(!converged){
                int w=0;
                vector<double> gamma_new = vector<double>(numTopics, ALPHA);

                for(const auto & word_pair : doc.wordCounts){
                    int word_id = word_pair.first;
                    int word_count = word_pair.second;
                    // calculate phi
                    double phisum = 0;
                    for(int k=0; k<numTopics; ++k){
                        phi[batchIndex][w][k] = exp(Elogtheta[batchIndex][k] + Elogbeta[k][word_id]) ;
                        phisum += phi[batchIndex][w][k];
                    }
                    if(phisum==0){phisum += 1e-100;}
                    // normalise phi and update gamma
                    for(int k=0; k<numTopics; ++k){
                        phi[batchIndex][w][k] /= phisum;
                        gamma_new[k] += word_count * phi[batchIndex][w][k];
                    }
                    w++;
                }
                // get the change in gamma
                double change = 0;
                for(int k=0; k<numTopics; ++k){
                    change += fabs(gamma[batchIndex][k] - gamma_new[k]);
                }
                change /= numTopics;
                // check for convergance
                if(change<0.00001){
                    converged = true;
                }

                //save changes in gamma
                gamma[batchIndex] = gamma_new;
                Elogtheta[batchIndex] = dirichlet_expectation(gamma[batchIndex]);
            }
            // gamma has converged for this doc. Add its contribution to the new lambda
            int w=0;
            for(const auto & word_pair : doc.wordCounts){
                int word_id = word_pair.first;
                int word_count = word_pair.second;
                for(int k=0; k<numTopics; ++k){
                    lambda_new[k][word_id] += word_count * phi[batchIndex][w][k];
                }
                w++;
            }
            // next doc in batch
            batchIndex++;
        }

        // M-STEP
        double learning_rate = pow((tau+ iter), -1*kappa);
        // update lambda new
        for(int k=0; k<numTopics; ++k){
            for(int w=0; w<numTerms; ++w){
                lambda_new[k][w] *= numDocs/batchSize;
                lambda_new[k][w] += ETA;
            }
        }
        // set lamda to the lambda new
        for(int k=0; k<numTopics; ++k){
            for(int w=0; w<numTerms; ++w){
                lambda[k][w] = (1-learning_rate)*lambda[k][w] + (learning_rate)*lambda_new[k][w];
            }
            Elogbeta[k] = dirichlet_expectation(lambda[k]);
        }
        // next batch
        iter++;
    }
}
