#include <iostream>
#include "data.h"
#include "lda.h"
#include <math.h>
#include <time.h>
#include <algorithm>

using namespace std;

void check_ap(void){
    cout << "Loading corpus" << endl;
    doc_corpus corpus = load_corpus("../datasets/ap/ap.dat");
    lda vb(corpus);

    cout << "loading vocabulary" << endl;
    vector<string> vocab = load_vocab("../datasets/ap/vocab.txt");

    cout << "Training lda:" << endl;
    clock_t start = clock();
    vb.train(150);
    cout << "Trained in " << double(clock() - start)/CLOCKS_PER_SEC << " seconds." << std::endl;
//    vb.writeParams("../params/ap150topic/");

//    vb.loadFromParams("../params/");

    for(int k=0; k<vb.numTopics; k++){
        vector<pair<double, string>> word_probs;
        for(int w=0; w<vb.numTerms; w++){
            double prob = exp(vb.logProbW[k][w]);
//            double prob = vb.logProbW[k][w];
            if(prob > 1e-3){
                word_probs.push_back(pair<double, string>(1-prob, vocab[w]));
            }
        }
        sort(begin(word_probs), end(word_probs));
        for(int i=0; i<20; i++){
            cout << word_probs[i].second << " ";
        }
        cout << endl;
    }
}

void check_dummy(void){
    cout << "Loading corpus" << endl;
    doc_corpus corpus = load_corpus("../datasets/dummy2.dat");
    lda vb(corpus);

    double best_likelihood = -0;
    int best_topics = 3;
    for(int i=3; i<20; i++){
        cout << i << ": ";
        vb.train(i);
        if(vb.likelihood > best_likelihood){
            best_likelihood = vb.likelihood;
            best_topics = i;
        }
    }

    cout << "Best number of topics is: " << best_topics << endl;
}

void check_big_dummy(void){
    cout << "Loading corpus" << endl;
    doc_corpus corpus = load_corpus("../datasets/dummy2.dat");
    lda vb(corpus);

    cout << "Training lda" << endl;
    clock_t start = clock();
    vb.train(10);
    cout << "Trained in " << double(clock() - start)/CLOCKS_PER_SEC << " seconds." << endl;

    vb.writeParams("../params/dat/");
}

void run_ratings_lda(void){
    cout << "Loading corpus" << endl;
    doc_corpus corpus = load_corpus("../datasets/rating.dat");
    corpus.numDocs -= 1;
    corpus.docs.erase(corpus.docs.begin());
    lda vb(corpus);
    int topics = 50;
    cout << "Training lda:" << endl;
    clock_t start = clock();
    vb.train(topics);
    cout << "Trained with "<< topics << " topics in "
        << double(clock() - start)/CLOCKS_PER_SEC << " seconds." << endl;
    vb.writeParams("../params/ratings/50/");
}

int main(int argc, char* argv[]) {
//    check_ap();
//    check_big_dummy();
//    run_ratings_lda();

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
                    param_dir.append("/");
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

        lda_settings l;
        alpha_settings a;
        if(settings_path_passed){
            cout << "Loading settings from "<< settings_path << endl;
            load_settings("../settings.txt", a, l);
        } else {
            cout << "Using default inference settings\n" << endl;
        }

        lda vb(corpus);
        cout << "Training lda with " << numTopics << " topics:" << endl;
        clock_t start = clock();
        vb.train(numTopics);
        cout << "\nTrained in " << double(clock() - start)/CLOCKS_PER_SEC << " seconds.\n" << endl;

        cout << "Writing dirichlet parameters to files in "<< param_dir << endl;
        vb.writeParams(param_dir);
    }

    return 0;
}