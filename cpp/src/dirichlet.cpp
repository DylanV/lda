/*!
 \file dirichlet.cpp
 */

#include "dirichlet.h"
#include "util.h"
#include <math.h>

dirichlet::dirichlet(size_t K, double alpha) {
    this->K = K;
    this->alpha = std::vector<double>(K, alpha);
    calculate_properties();
}

dirichlet::dirichlet(size_t K, std::vector<double> alpha) {
    this->K = K;
    this->alpha = alpha;
    calculate_properties();
}

void dirichlet::calculate_properties(){
    s = sum(alpha);
    mean = alpha;
    double prev_val = alpha[0];
    symmetric = true;
    for(int k=0; k<this->K; ++k){
        // Calculate the mean
        mean[k] = mean[k] / s;
        // Check if alpha is symmetric
        if(alpha[k] != prev_val){
            symmetric = false;
        }
        prev_val = this->alpha[k];
    }
}

std::vector<double> dirichlet::sample(){
    std::vector<double>  sample = std::vector<double>(K);
    double sum = 0.0;

    for(int k=0; k<K; ++k){
        std::gamma_distribution<double> gam(alpha[k], 1.0);
        double sample_k = gam(generator);
        sum += sample_k;
        sample[k] = sample_k;
    }

    for(int k=0; k<K; ++k){
        sample[k] /= sum;
    }

    return sample;
}

std::vector<std::vector<double>> dirichlet::sample(int N){
    std::vector<std::vector<double>> samples;
    for(int n=0; n<N; n++){
        samples.insert(samples.end(), sample());
    }
    return samples;
}

void dirichlet::estimate_mean(dirichlet_suff_stats ss){
    std::vector<double> new_mean = std::vector<double>(K);
    std::vector<double> old_mean = mean;

    for(int k=0; k<K; ++k){
        old_mean[k] = exp(ss.logp[k]/ss.N);
    }

    double alpha_sum;
    double max_change;
    bool conv = false;
    int iter = 0;

    while(!conv and iter<10){
        alpha_sum = 0;
        max_change = 0;
        for(int k=0; k<K; ++k){
            // Calculate digamma(alpha_k)
            double digamme_alpha_k = (ss.logp[k]/ss.N);
            for(int j=0; j<K; ++j){
                digamme_alpha_k -= old_mean[j] * ((ss.logp[j]/ss.N) - digamma(s*old_mean[j]));
            }
            // Get alpha by inverting the digamma function
            double alpha_k = inv_digamma(digamme_alpha_k);
            alpha_sum += alpha_k;
            new_mean[k] = alpha_k;
        }

        for(int k=0; k<K; ++k){
            new_mean[k] /= alpha_sum;
            if(fabs(new_mean[k] - old_mean[k]) > max_change){
                max_change = fabs(new_mean[k] - old_mean[k]);
            }
        }

        if(max_change < 1e-6){
            conv = true;
        }

        old_mean = new_mean;
        ++iter;
    }

    mean = new_mean;
}

void dirichlet::symmetric_update(double ss, size_t D) {

    double a, log_a, init_a = INIT_A;
    double df, d2f;
    int iter = 0;

    log_a = log(init_a);

    do{
        iter++;
        a = exp(log_a);
//        if (isnan(a)){
//            init_a = init_a * 10;
//            a = init_a;
//            log_a = log(a);
//        }
        df = D * (K * digamma(K * a) - K * digamma(a)) + ss;
        d2f = D * (K * K * trigamma(K * a) - K * trigamma(a));
        log_a = log_a - df/(d2f * a + df);
    } while ((fabs(df) > NEWTON_THRESH) and (iter < MAX_ALPHA_ITER));

//    if(!isnan(exp(log_a))){
//        a = exp(log_a);
//        this->s = K*a;
//        for(int i=0; i<K; i++){
//            alpha[i] = a;
//        }
//    }
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