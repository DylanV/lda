//
// Created by Dylan on 18/05/2016.
//

#include "dirichlet.h"
#include "util.h"
#include <math.h>

dirichlet::dirichlet()
{
    s = 0;
    K = 0;
}

dirichlet::dirichlet(int K, alpha_settings settings)
{
    double a = settings.init;
    dirichlet::K = K;
    if(a == 0){
        a = 1.0/K;
    }

    mean = std::vector<double>(K, 1.0/K);
    alpha = std::vector<double>(K, 1.0);
    s = 0;
    for(int i=0; i<K; i++){
        alpha[i] = alpha[i]*a;
        s += alpha[i];
    }

    INIT_A = settings.init_s;
    NEWTON_THRESH = settings.newton_threshold;
    MAX_ALPHA_ITER = settings.max_iterations;
    SYMMETRIC = settings.symmetric;
}

void dirichlet::update(std::vector<double> ss, int D) {
    if(SYMMETRIC){
        double total_ss = 0;
        for(const double& val : ss){
            total_ss += val;
        }
        symmetric_update(total_ss, D);
    }
    else{
        asymmetric_update(ss, D);
    }
}

void dirichlet::symmetric_update(double ss, int D)
{
/*!
Update alpha of the dirichlet given the sufficient statistics. Alpha is updated as symmetric.
 The sufficient stats are the sum of the observed samples of the diriclet.
The mean remains unchanged.
\param ss the sufficient statistics
\param D the number of observed samples for the sufficient statistics
*/
    double a, log_a, init_a = INIT_A;
    double f, df, d2f;
    int iter = 0;

    log_a = log(init_a);
    double prev_a = a;
    do{
        iter++;
        a = exp(log_a);
        if (isnan(a)){
            init_a = init_a * 10;
            a = init_a;
            log_a = log(a);
        }
        f = D * (lgamma(K * a) - K * lgamma(a) + (a - 1) * ss);
        df = D * (K * digamma(K * a) - K * digamma(a)) + ss;
        d2f = D * (K * K * trigamma(K * a) - K * trigamma(a));
        log_a = log_a - df/(d2f * a + df);
    } while ((fabs(df) > NEWTON_THRESH) && (iter < MAX_ALPHA_ITER));

    if(!isnan(exp(log_a))){
        a = exp(log_a);
        this->s = K*a;
        for(int i=0; i<K; i++){
            alpha[i] = a;
        }
    }
}

void dirichlet::asymmetric_update(std::vector<double> ss, int D)
{
/*!
Performs Newton-Raphson for dirichlet with special hessian. Linear time
Fully updates the dirichlet from the given sufficient statistics. Mean and precision are estimated.
Sufficient statistics are the observed samples of the dirichlet where for one sample and dimension k:
ss[k] += digamma(sample[k]) - digamma(sum(samples))
/param ss the sufficient statistics in a K dimension vector
/param D the number of samples in the sufficient statistics
*/
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
                step_size[k] = pow(0.8,decay)*(gradient[k] - c) / hessian[k];
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