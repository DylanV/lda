/*!
 \file dirichlet.h
 */

#ifndef LDA_DIRICHLET_H
#define LDA_DIRICHLET_H


#include <vector>
#include <stdlib.h>
#include <random>

/*! The sufficient statistics for estimating a dirichlet
 * Given a set of observed data D= {p_1,...p_N} the sufficient statistic is a
 * fixed dimension function of the data calculated as 1/N * sum(log(p_i)) from 1 to N.
 * This can only be calculated when we have all the data we want to use. The log sum and total are
 * therefore tracked seperately.
 */
struct dirichlet_suff_stats {
    std::vector<double> logp;      /*!< Sum of the log of the observed multinomial samples*/
    size_t N;                      /*!< Number of observed samples. */
};

//! Represents a K dimension dirichlet distribution with special hessian
/*!
    Keeps track of the three dirichlet parameters.
    Alpha, mean and s (precision, concentration coefficient)
    Can update the dirichlet from observed samples.
    Class is limited in the assumption that is works with LDA. Assumes that the diriclet has a hessian
    in the form that allows for linear time update.
 */
class dirichlet {
public:

    // Constructors
    // ============
    //! Default constructor
    dirichlet(){};

    /*!
     * @param K Dimension of the dirichlet
     * @param alpha value for alpha for all K
     * @return diriclet
     */
    dirichlet(size_t K, double alpha);

    /*!
     * @param K Dimension of the dirichlet
     * @param alpha Parameter vector
     * @return diriclet
     */
    dirichlet(size_t K, std::vector<double> alpha);

    // Attributes
    // ==========

    double s;                   /*!< The precision*/
    size_t K;                   /*!< Dimension  */
    bool symmetric;             /*!< Is the dirichlet symmetric */
    std::vector<double> alpha;  /*!< The alpha (psuedo count) parameter*/
    std::vector<double> mean;   /*!< The mean, which is simply alpha / precision*/

    // Sampling
    // ========
    /*!
     * Gets a single sample from the dirichlet
     * @return A k dimension multinomial on the K-1 simplex
     */
    std::vector<double> sample();

    /*!
     * Samples the dirichlet N times
     * @param N The number of samples
     * @return A vector of N K dimension multinomial samples
     */
    std::vector<std::vector<double>> sample(int N);

    // MLE estimators
    // ==============
    /*! MLE estimate of the mean
     * Gets a maximum likelihood estimate of the mean from the given
     * sufficient statistics and updatest the mean and alpha of the dirichlet.
     * @param ss The sufficient statistics
     * @return boolean indicating whether the update was succesful
     */
    bool estimate_mean(dirichlet_suff_stats ss);

    /*! MLE estimate of the precision
     * Gets a maximum likelihood estimate of the precision from the given
     * sufficient statistics and updatest the precision and alpha of the dirichlet.
     * @param ss The sufficient statistics
     * @return boolean indicating whether the update was succesful
     */
    bool estimate_precision(dirichlet_suff_stats ss);

    /*! MLE estimate of the alpha parameter
     * Uses the inverting digamma method from Minka to get a maximum likelihood estimate
     * of the alpha parameter for the dirichlet.
     * @param ss The sufficient statistics
     * @return boolean indicating whether the update was succesful
     */
    bool estimate(dirichlet_suff_stats ss);


private:

    /*! RNG */
    std::default_random_engine generator;

    /*! From alpha and K calculates the properties of the dirichlet. */
    void calculate_properties();

    /*! From the mean and precision recalculates tha alpha parameter and checks for symmetry */
    void calculate_alpha();

};



#endif //LDA_DIRICHLET_H
