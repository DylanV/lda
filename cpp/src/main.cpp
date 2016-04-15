#include <iostream>
#include "data.h"
#include "lda.h"
#include <math.h>

using namespace std;

int main() {
    cout << "Loading corpus:" << endl;
    doc_corpus corpus = load_corpus("../datasets/dummy.dat");
    lda vb(corpus);
    cout << "Training lda:" << endl;
    vb.train(4);
    cout << vb.alpha << endl;

    for(int k=0; k<4; k++){
        cout << "topic: " << k << endl;
        for(int w=0; w<corpus.numTerms; w++){
            double prob = exp(vb.logProbW[k][w]);
            if(prob > 0.05){
                cout << w << " " << prob << " ";
            }
        }
        cout << endl;
    }

    double min = -9999999999999;
    int best = 2;
    for(int t=2; t<20; t++){
        vb = lda(corpus);
        vb.train(t);
        cout << t << " " << vb.likelihood << endl;
        if(vb.likelihood > min){
            min = vb.likelihood;
            best = t;
        }
    }
    cout << best << endl;

    return 0;
}