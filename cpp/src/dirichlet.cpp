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

bool dirichlet::estimate_mean(dirichlet_suff_stats ss){
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

    if(!is_on_simplex(new_mean)){
        return false;
    }

    mean = new_mean;
    calculate_alpha();

    return true;
}

bool dirichlet::estimate_precision(dirichlet_suff_stats ss) {

    double s_old = s;
    double s_like_deriv;
    double s_like_sec_deriv;
    double s_inv;
    double s_new = 0.0;

    bool conv = false;
    int iter = 0;

    while(!conv and iter<100){

        s_like_deriv = (ss.N * digamma(s_old));
        s_like_sec_deriv = (ss.N * trigamma(s_old));

        for(int k=0; k<K;++k){
            s_like_deriv -= ss.N*mean[k]*digamma(s_old*mean[k]);
            s_like_deriv += mean[k]*ss.logp[k];

            s_like_sec_deriv -= ss.N * mean[k] * mean[k] * trigamma(s_old*mean[k]);
        }

        s_inv = 1.0/s_old + (1.0/s_old)*(1.0/s_old)*(1.0/s_like_sec_deriv)*(s_like_deriv);
        s_new = 1.0/s_inv;

        conv = fabs(s_new - s_old) < 1e-6;
        s_old = s_new;
        ++iter;
    }
    if(s < 0){
        return false;
    }

    s = s_new;
    calculate_alpha();

    return true;
}

void dirichlet::calculate_alpha() {
    double prev_alpha = alpha[0];
    bool is_symm = true;
    for(int k=0; k<K; ++k){
        alpha[k] = mean[k]*s;
        if(alpha[k] != prev_alpha){
            is_symm = false;
        }
        prev_alpha = alpha[k];
    }
    symmetric = is_symm;
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

bool dirichlet::estimate(dirichlet_suff_stats ss) {
    std::vector<double> new_alpha = std::vector<double>(K);
    std::vector<double> old_alpha = alpha;
    double alpha_sum;

    int iter = 0;
    bool conv = false;
    double max_change;

    while(!conv and iter<100){

        alpha_sum = 0;
        for(int k=0; k<K; ++k){
            alpha_sum += old_alpha[k];
        }
        max_change = 0;
        for(int k=0; k<K; ++k){
            new_alpha[k] = inv_digamma(digamma(alpha_sum) + ss.logp[k]/ss.N);
            double change = fabs(new_alpha[k]- old_alpha[k]);
            if(change > max_change){
                max_change = change;
            }
        }

        if(max_change < 1e-6){
            conv = true;
        }

        old_alpha = new_alpha;
        ++iter;
    }

    for(int k=0; k<K; ++k){
        if(new_alpha[k] < 0){
            return false;
        }
    }

    alpha = new_alpha;
    calculate_properties();

    return true;
}
