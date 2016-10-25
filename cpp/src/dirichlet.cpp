/*!
 \file dirichlet.cpp
 */

#include "dirichlet.h"
#include "util.h"
#include <math.h>

dirichlet::dirichlet(){

}

dirichlet::dirichlet(size_t K, double alpha) {
    this->K = K;
    this->alpha = std::vector<double>(K, alpha);
    dirichlet::dirichlet(K, this->alpha);
}

dirichlet::dirichlet(size_t K, std::vector<double> alpha) {
    this->K = K;
    this->alpha = alpha;
    // Calculate the precision
    this->s = sum(this->alpha);
    this->mean = alpha;
    double prev_val = alpha[0];
    symmetric = true;
    for(int k=0; k<this->K; ++k){
        // Calculate the mean
        this->mean[k] = this->mean[k] / this->s;
        // Check if alpha is symmetric
        if(this->alpha[k] == prev_val){
            symmetric = false;
        }
        prev_val = this->alpha[k];
    }
}

void dirichlet::update(std::vector<double> ss, size_t D) {
//    if(SYMMETRIC){
//        double total_ss = 0;
//        for(const double& val : ss){
//            total_ss += val;
//        }
//        symmetric_update(total_ss, D);
//    }
//    else{
//        asymmetric_update(ss, D);
//    }
}


void dirichlet::symmetric_update(double ss, size_t D) {

    double a, log_a, init_a = INIT_A;
    double df, d2f;
    int iter = 0;

    log_a = log(init_a);

    do{
        iter++;
        a = exp(log_a);
        if (isnan(a)){
            init_a = init_a * 10;
            a = init_a;
            log_a = log(a);
        }
        df = D * (K * digamma(K * a) - K * digamma(a)) + ss;
        d2f = D * (K * K * trigamma(K * a) - K * trigamma(a));
        log_a = log_a - df/(d2f * a + df);
    } while ((fabs(df) > NEWTON_THRESH) and (iter < MAX_ALPHA_ITER));

    if(!isnan(exp(log_a))){
        a = exp(log_a);
        this->s = K*a;
        for(int i=0; i<K; i++){
            alpha[i] = a;
        }
    }
}

void dirichlet::asymmetric_update(std::vector<double> ss, size_t D) {

    int iteration = 0;
    double thresh = 0;
    double alpha_sum = 0;
    std::vector<double> gradient(K);
    std::vector<double> hessian(K);
    double sum_g_h, sum_1_h;
    double z, c;
    int decay = 0;

    do {
        iteration++;
        alpha_sum = sum(alpha);

        z = D*trigamma(alpha_sum);
        sum_g_h = 0, sum_1_h = 0;

        for(int k=0; k<K; k++){
            gradient[k] = D * (digamma(alpha_sum) - digamma(alpha[k])) + ss[k];
            hessian[k] = -1.0*(D * trigamma(alpha[k]));
            sum_g_h += gradient[k] / hessian[k];
            sum_1_h += 1.0/hessian[k];
        }

        c = sum_g_h / ((1.0 / z) + sum_1_h);

        std::vector<double> alpha_new(this->alpha);
        std::vector<double> step_size(K);

        thresh = 0;

        while(true){
            bool singular_hessian = false;
            for(int k=0; k<K; k++){
                step_size[k] = pow(0.9,decay)*(gradient[k] - c) / hessian[k];
                if(step_size[k] >= this->alpha[k]){
                    singular_hessian = true;
                }
            }
            if(singular_hessian){
                decay++;
                if(decay > 10){
                    break;
                }
            }else{
                for(int k=0; k<K; k++){
                    alpha_new[k] -= step_size[k];
                    thresh += fabs(step_size[k]);
                }
                break;
            }
        }
        alpha = alpha_new;
    }while(iteration < MAX_ALPHA_ITER and thresh > NEWTON_THRESH);
    // update class members
    alpha_sum = sum(alpha);
    s = alpha_sum;
    for(int k=0; k<K; k++){
        mean[k] = alpha[k] / alpha_sum;
    }
}