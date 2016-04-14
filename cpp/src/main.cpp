#include <iostream>
#include "data.h"
#include "lda.h"
#include <math.h>

using namespace std;

int main() {
    cout << "Loading corpus:" << endl;
    doc_corpus corpus = load_corpus("../datasets/dummy.dat");
    lda vb(corpus, 4);
    cout << "Training lda:" << endl;
    vb.train();
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
    return 0;
}