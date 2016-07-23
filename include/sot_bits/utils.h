//
//  utils.h
//  Surrogate Optimization
//
//  Created by David Eriksson on 7/30/15.
//  Copyright (c) 2015 David Eriksson. All rights reserved.
//

#ifndef Surrogate_Optimization_utils_h
#define Surrogate_Optimization_utils_h

#include <thread>         // std::this_thread::sleep_for
#include <chrono>         // std::chrono::seconds
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
    inline VecType squaredPointSetDistance(const VecType& x, const MatType& Y) {
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
    inline MatType squaredPairwiseDistance(const MatType& X, const MatType& Y) {
        MatType dists = - 2*(X.t()*Y);
        dists.each_row() += arma::sum(Y % Y, 0);
        dists.each_col() += arma::sum(X % X, 0).t();
        dists = arma::abs(dists);
        return dists;
    };
    
    template <class MatType>
    inline void squaredPairwiseDistance2(const MatType& X, const MatType& Y, MatType &dists) {
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
    
    inline mat toUnitBox(const mat& X, const vec& xlow, const vec& xup) {
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
    inline vec toUnitBox(const vec& X, const vec& xlow, const vec& xup) {
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
    inline mat fromUnitBox(const mat& X, const vec& xlow, const vec& xup) {
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
    inline vec fromUnitBox(const vec& X, const vec& xlow, const vec& xup) {
        return xlow + (xup - xlow) % X;
    };
    
    // Unit rescale
    template <class VecType>
    inline VecType unitRescale(const VecType& x) {
        double xmin = arma::min(x);
        double xmax = arma::max(x);
        if( xmin == xmax ) {
            return arma::ones<VecType>(x.n_elem);
        }
        return (x - xmin)/(xmax - xmin);
    };
    
    // Keeps track of function values and results
    class Result {
    protected:
        int mNumEvals = 0;
        int mDim;
        int mMaxEvals;
        vec mfX;
        mat mX;
        double mfBest;
        vec mxBest;
    public:
        Result(int maxEvals, int dim) {
            mMaxEvals = maxEvals;
            mNumEvals = 0;
            mDim = dim;
            mfX = std::numeric_limits<double>::max() * arma::ones<vec>(mMaxEvals);
            mfBest = std::numeric_limits<double>::max();
            mX = std::numeric_limits<double>::max() * arma::ones<mat>(dim, maxEvals);
            mxBest = std::numeric_limits<double>::max() * arma::ones<mat>(dim);
        }
        int dim() const { return mDim; }
        int numEvals() const { return mNumEvals; }
        vec fX() const { return mfX.rows(0, mNumEvals-1); }
        mat X() const { return mX.cols(0, mNumEvals-1); }
        vec xBest() const { return mxBest; }
        double fBest() const { return mfBest; }
        void addEval(vec &x, double fVal) { 
            mX.col(mNumEvals) = x;
            mfX(mNumEvals) = fVal;
            if (fVal < mfBest) {
                mfBest = fVal;
                mxBest = x;
            }
            mNumEvals++;
        }
        void reset() {
            mNumEvals = 0;
            mX = std::numeric_limits<double>::max() * arma::ones<mat>(mDim, mMaxEvals);
            mxBest = std::numeric_limits<double>::max() * arma::ones<mat>(mDim);
            mfX = std::numeric_limits<double>::max() * arma::ones<vec>(mMaxEvals);
            mfBest = std::numeric_limits<double>::max();
        }
    };
    
    // Computes the Pareto front of x, y
    inline uvec paretoFront(const vec &x, const vec &y) {
        if(x.n_rows != y.n_rows) { throw std::logic_error("paretoFront: x and y need to have the same length"); }
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
    inline VecType cumMin(const VecType& x) {
        VecType out(x.n_elem);
        auto minVal = x(0);
        out(0) = minVal;
        for(int i=1; i < x.n_elem; i++) {
            if (x(i) < minVal) {
                minVal = x(i);
            }
            out(i) = minVal;
        }
        return out;
    };
    
    class StopWatch {
    private:
        std::chrono::time_point<std::chrono::system_clock> mStartTime, mEndTime;
        bool mStarted;
    public:
        StopWatch() {
            this->mStarted = false;
        }
        void start() {
            if(mStarted) { throw std::logic_error("StopWatch: The StopWatch is already running, so can't start!"); }
            this->mStartTime = std::chrono::system_clock::now();
            this->mStarted = true;
        }
        double stop() {
            if(mStarted) { throw std::logic_error("StopWatch: The StopWatch is not running, so can't stop!"); }
            this->mEndTime = std::chrono::system_clock::now();
            this->mStarted = false;
            std::chrono::duration<double> elapsedSeconds = 
                this->mEndTime - this->mStartTime;
            return elapsedSeconds.count();
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

#endif