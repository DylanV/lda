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
    //! Default constructor
    dirichlet(){};

    /*!
     * @param K
     * @param alpha
     * @return
     */
    dirichlet(size_t K, double alpha);

    /*!
     * @param K
     * @param alpha
     * @return
     */
    dirichlet(size_t K, std::vector<double> alpha);

    double s;                   /*!< The concentration coefficient or precision of the dirichlet.*/
    size_t K;                   /*!< The number of dimensions*/
    bool symmetric;             /*!< Is the dirichlet symmetric */
    std::vector<double> alpha;  /*!< The alpha psuedo count parameter*/
    std::vector<double> mean;   /*!< The mean of the dirichlet which is simply alpha / precision*/

    std::default_random_engine generator;
    std::vector<double> sample();
    std::vector<std::vector<double>> sample(int N);
private:
    // Update settings
    int INIT_A = 100;               /*!< The initial precision for the precision update */
    double NEWTON_THRESH = 1e-5;    /*!< The threshold for netwon-raphson update convergance */
    int MAX_ALPHA_ITER = 1000;      /*!< Max number of iterations for newton-raphson */

    /*! From alpha and K calculates the properties of the dirichlet.
     */
    void calculate_properties();

    /*! Update the alpha of the dirichlet when it is symmetric.
     * The sufficient stats are the sum of the observed samples of the diriclet.
     * @param [in] ss Sufficient statistics. Sum of the dirichlet expectation of samples from the dirichlet.
     * @param [in] D Number of samples in the sufficient statistics.
     */
    void symmetric_update(double ss, size_t D);

    /*! Update the alpha of an assymetric alpha
     * @param [in] ss Sufficient statistics. Sum of the dirichlet expectation of samples from the dirichlet.
     * @param [in] D Number of samples in the sufficient statistics.
     */
    void asymmetric_update(std::vector<double> ss, size_t D);
};



#endif //LDA_DIRICHLET_H
