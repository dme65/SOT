//
//  common.h
//  Surrogate Optimization
//
//  Created by David Eriksson on 2/2/16.
//  Copyright © 2016 David Eriksson. All rights reserved.
//

#ifndef __Surrogate_Optimization__Common__
#define __Surrogate_Optimization__Common__

#include <armadillo>
#include <random>

namespace sot {
    
    #define ARMA_USE_BLAS
    #define __pi__ 3.14159265358979323846

    typedef arma::mat mat;
    typedef arma::vec vec;
    typedef arma::rowvec rowvec;
    typedef arma::fmat fmat;
    typedef arma::fvec fvec;
    typedef arma::Col<int> ivec;
    typedef arma::uvec uvec;
    
    // Set seeds
    namespace rng {
        std::random_device rd;
        std::mt19937 mt(rd());
    }
}

#endif /* common_h */
