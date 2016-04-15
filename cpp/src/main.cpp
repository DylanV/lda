#include <iostream>
#include "data.h"
#include "lda.h"
#include <math.h>
#include <time.h>

using namespace std;

int main() {
    cout << "Loading corpus" << endl;
    doc_corpus corpus = load_corpus("../datasets/ap/ap.dat");
    lda vb(corpus);

    cout << "loading vocabulary" << endl;
    vector<string> vocab = load_vocab("../datasets/ap/vocab.txt");

    cout << "Training lda:" << endl;
    clock_t start = clock();
    vb.train(150);
    cout << "Trained in " << double(clock() - start)/CLOCKS_PER_SEC << " seconds." << std::endl;
    vb.writeParams("../params/ap150topic/");

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

    return 0;
}