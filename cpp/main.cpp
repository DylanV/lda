/*!
 \file main.cpp
 */

#include <iostream>
#include <expectation_prop.h>
#include "data.h"
#include "var_bayes.h"
#include "gibbs.h"

using namespace std;

int main(int argc, char* argv[]) {

    bool ratings = false;

    if (argc < 7) {
        std::cerr << "Insufficient arguments" << endl;
        std::cerr << "Standard usage is --corpus <infile> --output <outdir> --topics <numtopics>" << endl;
        std::cerr << "Settings can be loaded from file with the argument: --setting <infile>" << endl;
        std::cerr << "Inference method can be chosen with the argument: --inference <1/2/3>" << endl;
        std::cerr << "Where <1/2/3> corresponds to <variational inference/collapsed gibbs/expectation propagtion> respectively." << endl;
        std::cin.get();
        exit(0);

    } else {
        string corpus_path = "", output_dir="", settings_path="";
        int inference_method = 1;
        int numTopics = 0;
        bool settings_path_passed = false;

        for(int i=1; i<argc-1; i++){

            string argument = argv[i];

            if(argument == "--corpus_path" || argument == "-c"){
                corpus_path.assign(argv[i+1]);
            }
            else if(argument == "--topics" || argument == "-t"){
                numTopics = stoi(argv[i+1]);
                if(numTopics<= 0){
                    std::cerr << "Number of topics should be positive and non-zero.";
                    std::cin.get();
                    exit(0);
                }
            }
            else if(argument == "--output" || argument == "-o"){
                output_dir.assign(argv[i+1]);
                if(output_dir.find_last_of("\\/") != output_dir.size()-1){
                    output_dir.append("/"); // If the directory does not end in a slash add one
                }
            }
            else if(argument == "--setting_path" || argument == "-s"){
                settings_path.assign(argv[i+1]);
                settings_path_passed = true;
            }
            else if(argument == "--inference" || argument == "-i"){
                inference_method = std::stoi(argv[i+1]);
            }
            else if(argument == "-r"){
                ratings = true;
            }
        }

        cout << "Loading corpus from "<< corpus_path << endl;
        doc_corpus corpus = load_corpus(corpus_path);
        cout << "Corpus loaded with " << corpus.numDocs << " documents.\n" << endl;

        if(ratings){
            corpus.numDocs -= 1;
            corpus.docs.erase(corpus.docs.begin());
        }

        lda_settings l;
        lda_model * model;

        if(inference_method == 1){
            var_bayes bayes_model = var_bayes(corpus, l);
            model = &bayes_model;
        }else if(inference_method == 2){
            gibbs gibbs_model = gibbs(corpus);
            model = &gibbs_model;
        }else{
            expectation_prop ep_model = expectation_prop(corpus);
            model = &ep_model;
        }

        cout << "Training lda with " << numTopics << " topics:" << endl;

        clock_t start = clock();
        model->train(numTopics);
        cout << "\nTrained in " << double(clock() - start)/CLOCKS_PER_SEC
        << " seconds. \n" << endl;

        cout << "Writing dirichlet parameters to files in "<< output_dir << endl;
        model->save_parameters(output_dir);
    }

    return 0;
}