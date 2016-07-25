#include <iostream>
#include "data.h"
#include "var_bayes.h"

using namespace std;

int main(int argc, char* argv[]) {

    const bool ratings = true;

    if (argc < 7) {
        std::cerr << "Insufficient arguments" << endl;
        std::cerr << "Standard usage is -corpus <infile> -param <outdir> -topics <numtopics>" << endl;
        std::cerr << "Optionally settings can also be loaded -corpus <infile> "
                             "-param <outdir> -topics <numtopics> -setting <infile>" << endl;
        std::cin.get();
        exit(0);

    } else {
        string corpus_path = "", param_dir="", settings_path="";
        int numTopics = 0;
        bool settings_path_passed = false;
        for(int i=1; i<argc-1; i++){

            string argument = argv[i];

            if(argument == "-corpus"){
                corpus_path.assign(argv[i+1]);
            }
            else if(argument == "-topics"){
                numTopics = stoi(argv[i+1]);
                if(numTopics<= 0){
                    std::cerr << "Number of topics should be positive and non-zero.";
                    std::cin.get();
                    exit(0);
                }
            }
            else if(argument == "-param"){
                param_dir.assign(argv[i+1]);
                if(param_dir.find_last_of("\\/") != param_dir.size()-1){
                    param_dir.append("/"); // If the directory does not end in a slash add one
                }
            }
            else if(argument == "-setting"){
                settings_path.assign(argv[i+1]);
                settings_path_passed = true;
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
        alpha_settings a;
        if(settings_path_passed){
            cout << "Loading settings from "<< settings_path << endl;
            load_settings("../settings.txt", a, l);
        } else {
            cout << "Using default inference settings\n" << endl;
        }

        var_bayes inference_model = var_bayes(corpus, l, a);

        lda_model * model = &inference_model;
        cout << "Training lda with " << numTopics << " topics:" << endl;
        clock_t start = clock();
        model->train(numTopics);
        cout << "\nTrained in " << double(clock() - start)/CLOCKS_PER_SEC
        << " seconds. \n" << endl;

        cout << "Writing dirichlet parameters to files in "<< param_dir << endl;
        model->save_parameters(param_dir);
    }

    return 0;
}