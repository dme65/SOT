//
//  utils.h
//  Surrogate Optimization
//
//  Created by David Eriksson on 7/30/15.
//  Copyright (c) 2015 David Eriksson. All rights reserved.
//

#ifndef __Surrogate_Optimization__utils__
#define __Surrogate_Optimization__utils__

#include <cassert>
#include <thread>         // std::this_thread::sleep_for
#include <chrono>         // std::chrono::seconds
#include <iostream>
#include "common.h"

namespace sot {
    
    // Inputs:
    //  - x: Vector of size d x 1
    //  - Y: Matrix of size d x n
    //
    // Output:
    //  - Vector of size 1 x n
    //
    // Note: If we are unlucky we might get a roundoff error and get a number that is actually negative
    //       on the magnitude of machine precision. This can give terrible effects later so this is why
    //       we take the absolute value of the result.
    //
    
    template <class MatType, class VecType>
    inline VecType SquaredPointSetDistance(const VecType& x, const MatType& Y) {
        return arma::abs(arma::repmat(arma::sum(x % x,0),Y.n_cols,1) + arma::sum(Y % Y,0).t() - 2*Y.t()*x);
    };
    
    // Inputs:
    //  - X: Matrix of size d x m
    //  - Y: Matrix of size d x n
    //
    // Output:
    //  - Matrix of size m x n
    //
    // Note: If we are unlucky we might get a roundoff error and get a number that is actually negative
    //       on the magnitude of machine precision. This can give terrible effects later so this is why
    //       we take the absolute value of the result.
    //
    template <class MatType>
    inline MatType SquaredPairwiseDistance(const MatType& X, const MatType& Y) {
        //return arma::abs((arma::repmat(arma::sum(X % X, 0).t(),1,Y.n_cols) + arma::repmat(arma::sum(Y % Y,0), X.n_cols, 1) - 2*(X.t()*Y)));
        MatType dists = - 2*(X.t()*Y);
        dists.each_row() += arma::sum(Y % Y, 0);
        dists.each_col() += arma::sum(X % X, 0).t();
        dists = arma::abs(dists);
        return dists;
    };
    
    template <class MatType>
    inline void SquaredPairwiseDistance2(const MatType& X, const MatType& Y, MatType &dists) {
        //return arma::abs((arma::repmat(arma::sum(X % X, 0).t(),1,Y.n_cols) + arma::repmat(arma::sum(Y % Y,0), X.n_cols, 1) - 2*(X.t()*Y)));
        dists = - 2*(X.t()*Y);
        dists.each_row() += arma::sum(Y % Y, 0);
        dists.each_col() += arma::sum(X % X, 0).t();
        dists = arma::abs(dists);
    };
    
    // Inputs:
    //  - X: Matrix of size d x n
    //  - xlow: Vector of size d x 1
    //  - xup: Vector of size d x 1
    //
    // Output:
    //  - Matrix of size d x n
    //
    
    inline mat ToUnitBox(const mat& X, const vec& xlow, const vec& xup) {
        return (X - arma::repmat(xlow, 1, X.n_cols))/arma::repmat(xup - xlow, 1, X.n_cols);
    };
    
    // Inputs:
    //  - X: Vector of size d x 1
    //  - xlow: Vector of size d x 1
    //  - xup: Vector of size d x 1
    //
    // Output:
    //  - Vector of size d x 1
    //
    inline vec ToUnitBox(const vec& X, const vec& xlow, const vec& xup) {
        return (X - xlow)/(xup - xlow);
    };
    
    // Inputs:
    //  - X: Matrix of size d x n
    //  - xlow: Vector of size d x 1
    //  - xup: Vector of size d x 1
    //
    // Output:
    //  - Matrix of size d x n
    //
    inline mat FromUnitBox(const mat& X, const vec& xlow, const vec& xup) {
        return arma::repmat(xlow, 1, X.n_cols) + arma::repmat(xup - xlow, 1, X.n_cols) % X;
    };
    
    // Inputs:
    //  - X: Vector of size d x 1
    //  - xlow: Vector of size d x 1
    //  - xup: Vector of size d x 1
    //
    // Output:
    //  - Vector of size d x 1
    //
    inline vec FromUnitBox(const vec& X, const vec& xlow, const vec& xup) {
        return xlow + (xup - xlow) % X;
    };
    
    // Unit rescale
    template <class VecType>
    inline VecType UnitRescale(const VecType& x) {
        double xmin = arma::min(x);
        double xmax = arma::max(x);
        if( xmin == xmax ) {
            return arma::ones<VecType>(x.n_elem);
        }
        return (x - xmin)/(xmax - xmin);
    };
    
    // Keeps track of function values and results
    class Result {
    public:
        int dim;
        int maxeval;
        int exp_des_eval;
        vec fx;
        mat x;
        double fbest;
        vec xbest;
        Result(int maxeval, int exp_des_eval, int dim) {
            this->maxeval = maxeval;
            this->exp_des_eval = exp_des_eval;
            this->dim = dim;
            fx = arma::datum::inf * arma::ones<vec>(maxeval);
            fbest = arma::datum::inf;
            x = arma::datum::inf * arma::ones<mat>(dim, maxeval);
            xbest = arma::datum::inf * arma::ones<mat>(dim);
        }
        void reset() {
            x = arma::datum::inf * arma::ones<mat>(dim, maxeval);
            xbest = arma::datum::inf * arma::ones<mat>(dim);
            fx = arma::datum::inf * arma::ones<vec>(maxeval);
            fbest = arma::datum::inf;
        }
    };
    
    // Computes the Pareto front of x, y
    inline uvec pareto_front(const vec &x, const vec &y) {
        assert(x.n_rows == y.n_rows);
        double tol = 1e-10;
        uvec isort = sort_index(x);
        vec x2 = x(isort);
        vec y2 = y(isort);
        uvec indvec = arma::ones<uvec>(x.n_rows);
        indvec(0) = isort(0);
        int indcur = 1;
        double ycur = y2(0);
        
        for(int i=1; i < x.n_rows; i++) {
            if (y2(i) <= ycur + tol) {
                indvec(indcur) = isort(i);
                ycur = y2(i);
                indcur++;
            }
        }
        indvec = indvec.head(indcur);
        return indvec;
    };
    
    // Cumulative minimum
    template <class VecType>
    inline VecType cummin(const VecType& x) {
        VecType out(x.n_elem);
        auto minval = x(0);
        out(0) = minval;
        for(int i=1; i < x.n_elem; i++) {
            if (x(i) < minval) {
                minval = x(i);
            }
            out(i) = minval;
        }
        return out;
    };
    
    class StopWatch {
    private:
        std::chrono::time_point<std::chrono::system_clock> starttime, endtime;
        bool started;
    public:
        StopWatch() {
            this->started = false;
        }
        void start() {
            assert(!this->started);
            this->starttime = std::chrono::system_clock::now();
            this->started = true;
        }
        double stop() {
            assert(this->started);
            this->endtime = std::chrono::system_clock::now();
            this->started = false;
            std::chrono::duration<double> elapsed_seconds = 
                this->endtime - this->starttime;
            return elapsed_seconds.count();
        }
    };
    
    // Random integer in interval [0, i-1]
    inline double randi(int i) { 
        std::uniform_int_distribution<int> randi(0, i-1);
        return randi(rng::mt);
    }
    
    inline double randn() {
        std::normal_distribution<double> randn(0.0, 1.0);
        return randn(rng::mt);
    }
    
    inline double rand() {
        std::uniform_real_distribution<double> rand(0, 1);
        return rand(rng::mt);
    }
}

#endif /* defined(__Surrogate_Optimization__utils__) */