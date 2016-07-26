/*!
 * File:   adaptive_sampling.h
 * Author: David Eriksson, dme65@cornell.edu
 *
 * Created on 7/18/16.
 */

#ifndef SOT_COMMON_H
#define SOT_COMMON_H

#include <armadillo>
#include <random>

//!SOT namespace
namespace sot {
    typedef arma::mat mat; /**< Default matrix class */
    typedef arma::vec vec; /**< Default (column) vector class */
    typedef arma::rowvec rowvec; /**< Default (row) vector class */
    typedef arma::fmat fmat; /**< Default single precision matrix class */
    typedef arma::fvec fvec; /**< Default single precision (column) vector class */
    typedef arma::Col<int> ivec; /**< Default index (column) vector class */
    typedef arma::uvec uvec; /**< Default unsigned (column) vector class */
    
    //! Namespace for the random number generator 
    namespace rng {
        std::mt19937 mt(0); /**< The global random number generator */                                
    }
}

#endif
