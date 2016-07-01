//
// Created by Dylan on 18/05/2016.
//

#ifndef LDA_DIRICHLET_H
#define LDA_DIRICHLET_H


#include <vector>

//! settings struct for alpha updates
struct alpha_settings {
    alpha_settings() : symmetric(true), newton_threshold(1e-5),
                       max_iterations(1000), init(0), init_s(100) {}

    bool symmetric;      /*!< Whether alpha should be the symmetric*/
    double newton_threshold; /*!< threshold for newtons method*/
    int max_iterations;      /*!< Maximum number of iterations for alpha update*/
    double init;        /*!< Initial value for the concentration parameter*/
    int init_s;              /*!< Initial value for the conc coeff when estimating*/
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
    dirichlet();
    //! Constructor to start with symmetric dirichlet
    dirichlet(size_t K, alpha_settings settings);

    double s;                   /*!< The concentration coefficient or precision of the dirichlet.*/
    size_t K;                      /*!< The number of dimensions*/
    std::vector<double> alpha;  /*!< The alpha psuedo count parameter*/
    std::vector<double> mean;   /*!< The mean of the dirichlet which is simply alpha / precision*/

    void update(std::vector<double> ss, size_t D);

private:
    // Update settings
    int INIT_A = 100;               /*! The initial precision for the precision update */
    double NEWTON_THRESH = 1e-5;    /*! The threshold for netwon-raphson update convergance */
    int MAX_ALPHA_ITER = 1000;      /*! Max number of iterations for newton-raphson */
    bool SYMMETRIC = false;

    //! Update the dirichlet alpha assuming alhpa is symmetric
    void symmetric_update(double ss, size_t D);
    //! Fully update the dirichlet alpha
    void asymmetric_update(std::vector<double> ss, size_t D);
};



#endif //LDA_DIRICHLET_H
