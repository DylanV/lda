//
// Created by Dylan on 18/05/2016.
//

#include "dirichlet.h"

dirichlet::dirichlet(double init_prec, std::vector<double> init_mean)
{
    s = init_prec;
    mean = init_mean;
    K = int(mean.size());
    alpha = std::vector<double>(K);
    for(int i=0; i<K; i++){
        alpha[i] = mean[i]*s;
    }
}

dirichlet::dirichlet(double init_prec, int K)
{
    s = init_prec;
    dirichlet::K = K;
    mean = std::vector<double>(K, 1.0/K);
    for(int i=0; i<K; i++){
        alpha[i] = mean[i]*s;
    }
}
