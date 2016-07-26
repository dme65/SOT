/*!
 * File:   utils.h
 * Author: David Eriksson, dme65@cornell.edu
 *
 * Created on 7/18/16.
 */


#ifndef SOT_UTILS_H
#define SOT_UTILS_H

#include <thread>         // std::this_thread::sleep_for
#include <chrono>         // std::chrono::seconds
#include "common.h"

//!SOT namespace
namespace sot {
    
    //! Fast level-2 distance computation between one point and a set of points
    /*!
     * 
     * \param x The vector
     * \param Y The matrix
     * \returns The vector of distances between x and the columns of Y
     * 
     * \tparam MatType Matrix class
     * \tparam VecType Vector class
     * 
     * \author David Eriksson, dme65@cornell.edu
     */
    template <class MatType = mat, class VecType = vec>
    inline VecType squaredPointSetDistance(const VecType& x, const MatType& Y) {
        return arma::abs(arma::repmat(arma::sum(x % x,0), Y.n_cols,1) + arma::sum(Y % Y,0).t() - 2*Y.t()*x);
    };
    
    //! Fast level-3 distance computation between two sets of points
    /*!
     * 
     * \param X The first matrix
     * \param Y The second matrix
     * \returns The matrix of distances between the columns of X and the columns of Y
     * 
     * \tparam MatType Matrix class
     * 
     * \author David Eriksson, dme65@cornell.edu
     */
    template <class MatType = mat>
    inline MatType squaredPairwiseDistance(const MatType& X, const MatType& Y) {
        MatType dists = - 2*(X.t()*Y);
        dists.each_row() += arma::sum(Y % Y, 0);
        dists.each_col() += arma::sum(X % X, 0).t();
        dists = arma::abs(dists);
        return dists;
    };
    
    //! Map one point to the unit box
    /*!
     * 
     * \param x Point
     * \param xLow Lower variable bounds
     * \param xUp Upper variable bounds
     * \returns x mapped to the unit box
     *
     * \author David Eriksson, dme65@cornell.edu
     */
    inline vec toUnitBox(const vec& x, const vec& xLow, const vec& xUp) {
        return (x - xLow)/(xUp - xLow);
    };
    
    //! Map multiple points to the unit box
    /*!
     * 
     * \param X Points
     * \param xLow Lower variable bounds
     * \param xUp Upper variable bounds
     * \returns Point in X mapped to the unit box
     * 
     * \author David Eriksson, dme65@cornell.edu
     */
    inline mat toUnitBox(const mat& X, const vec& xLow, const vec& xUp) {
        return (X - arma::repmat(xLow, 1, X.n_cols))/arma::repmat(xUp - xLow, 1, X.n_cols);
    };
    
    //! Map one point from the unit box to another hypercube
    /*!
     * 
     * \param x Point
     * \param xLow Lower variable bounds
     * \param xUp Upper variable bounds
     * \returns x mapped from the unit box
     *
     * \author David Eriksson, dme65@cornell.edu
     */
    inline vec fromUnitBox(const vec& x, const vec& xLow, const vec& xUp) {
        return xLow + (xUp - xLow) % x;
    };
    
    //! Map multiple points from the unit box to another hypercube
    /*!
     * 
     * \param X Points
     * \param xLow Lower variable bounds
     * \param xUp Upper variable bounds
     * \returns Points in X mapped from the unit box
     *
     * \author David Eriksson, dme65@cornell.edu
     */
    inline mat fromUnitBox(const mat& X, const vec& xLow, const vec& xUp) {
        return arma::repmat(xLow, 1, X.n_cols) + arma::repmat(xUp - xLow, 1, X.n_cols) % X;
    };
    
    //! Map a vector of values to the range [0, 1] 
    /*!
     * \param x Vector of values
     * \returns Values in x mapped to the range [0, 1]
     *
     * \author David Eriksson, dme65@cornell.edu
     */
    inline vec unitRescale(const vec& x) {
        double xMin = arma::min(x);
        double xMax = arma::max(x);
        if( xMin == xMax ) {
            return arma::ones(x.n_elem);
        }
        return (x - xMin)/(xMax - xMin);
    };
    
    //! Optimization result class
    /*!
     * This is a class that stores the result from the optimization runs and
     * is returned from all of the strategies. It stores the values from all
     * of the evaluations and the best solution and function value.
     * 
     * \author David Eriksson, dme65@cornell.edu
     */
    class Result {
    protected:
        int mNumEvals = 0; /*!< Number of evaluations */
        int mDim; /*!< Number of dimensions */
        int mMaxEvals; /*!< Evaluation budget */
        vec mfX; /*!< Function values */
        mat mX; /*!< Evaluated points */
        double mfBest; /*!< Best function value */
        vec mxBest; /*!< Best solution found */
    public:
        //! Constructor
        /*!
         * \param maxEvals Evaluation budget
         * \param dim Number of dimensions 
         */
        Result(int maxEvals, int dim) {
            mMaxEvals = maxEvals;
            mNumEvals = 0;
            mDim = dim;
            mfX = std::numeric_limits<double>::max() * arma::ones<vec>(mMaxEvals);
            mfBest = std::numeric_limits<double>::max();
            mX = std::numeric_limits<double>::max() * arma::ones<mat>(dim, maxEvals);
            mxBest = std::numeric_limits<double>::max() * arma::ones<mat>(dim);
        }
        //! Method for getting the number of dimensions
        int dim() const { return mDim; } /*!< \returns Number of dimensions */
        //! Method for getting the number of finished evaluations
        int numEvals() const { return mNumEvals; } /*!< \returns Number of finished evaluations */
        //! Method for getting the values of the finished evaluations
        vec fX() const { return mfX.rows(0, mNumEvals-1); } /*!< \returns Values of finished evluations */
        //! Method for getting the evaluated points
        mat X() const { return mX.cols(0, mNumEvals-1); } /*!< \returns Evaluated points */
        //! Method for getting the best solution found so far
        vec xBest() const { return mxBest; } /*!< \returns best solution found so far */
        //! Method for getting the value of the best solution found so far
        double fBest() const { return mfBest; } /*!< \returns Value of best solution found so far */
        //! Method for adding a finished evaluation
        /*!
         * \param x Evaluated point
         * \param funVal Value of the evaluated point
         */
        void addEval(vec &x, double funVal) { 
            mX.col(mNumEvals) = x;
            mfX(mNumEvals) = funVal;
            if (funVal < mfBest) {
                mfBest = funVal;
                mxBest = x;
            }
            mNumEvals++;
        }
        //! Method for resetting the object
        void reset() {
            mNumEvals = 0;
            mX = std::numeric_limits<double>::max() * arma::ones<mat>(mDim, mMaxEvals);
            mxBest = std::numeric_limits<double>::max() * arma::ones<mat>(mDim);
            mfX = std::numeric_limits<double>::max() * arma::ones<vec>(mMaxEvals);
            mfBest = std::numeric_limits<double>::max();
        }
    };
    
    //! Computes the Pareto front
    /*!
     * \param x Vector of values
     * \param y Vector of values
     * \returns Indices of the points on the Pareto front
     * 
     * \throws std::logic_error If x and y are not of the same length
     * 
     * \author David Eriksson, dme65@cornell.edu
     */    
    inline uvec paretoFront(const vec &x, const vec &y) {
        if(x.n_rows != y.n_rows) { 
            throw std::logic_error("paretoFront: x and y need to have the same length"); 
        }
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
    
    //! Computes the cumulative minimum
    /*!
     * \param x Vector of values
     * \returns Cumulative minimum of x
     * 
     * \author David Eriksson, dme65@cornell.edu
     */     
    inline vec cumMin(const vec& x) {
        vec out(x.n_elem);
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
    
    //! Stop watch class
    /*!
     * This class can be used to measure the elapsed time (in seconds) for different strategies
     * or other functions. It is a wrapper around std::chrono.
     * 
     * \author David Eriksson, dme65@cornell.edu
     */    
    class StopWatch {
    private:
        std::chrono::time_point<std::chrono::system_clock> mStartTime; /*!< Time when the watch was started */
        std::chrono::time_point<std::chrono::system_clock> mEndTime; /*!< Time when the watch was stopped */
        bool mStarted; /*!< True if the watch has been started */
    public:
        //! Constructor
        /*!
         * Initializes a watch that isn't started
         */
        StopWatch() {
            this->mStarted = false;
        }
        //! Starts the watch
        /*!
         * \throws std::logic_error If the watch has already been started
         */
        void start() {
            if(mStarted) { throw std::logic_error("StopWatch: The StopWatch is already running, so can't start!"); }
            this->mStartTime = std::chrono::system_clock::now();
            this->mStarted = true;
        }
        //! Stops the watch and returns the time elapsed
        /*!
         * \returns Time elapsed (seconds) since the watch was started
         * \throws std::logic_error If the watch hasn't already been started
         */
        double stop() {
            if(mStarted) { throw std::logic_error("StopWatch: The StopWatch is not running, so can't stop!"); }
            this->mEndTime = std::chrono::system_clock::now();
            this->mStarted = false;
            std::chrono::duration<double> elapsedSeconds = 
                this->mEndTime - this->mStartTime;
            return elapsedSeconds.count();
        }
    };
    
    
    //! Generate a random integer
    /*!
     * \param i Specifies the upper range (i is not included)
     * \returns Random integer in the range [0, i-1]
     * 
     * \author David Eriksson, dme65@cornell.edu
     */       
    inline double randi(int i) { 
        std::uniform_int_distribution<int> randi(0, i-1);
        return randi(rng::mt);
    }
    
    //! Generate a N(0,1) realization
    /*!
     * \returns Realization drawn from a N(0,1) distribution
     * 
     * \author David Eriksson, dme65@cornell.edu
     */
    inline double randn() {
        std::normal_distribution<double> randn(0.0, 1.0);
        return randn(rng::mt);
    }
    
    //! Generate a U[0,1] realization
    /*!
     * \returns Realization drawn from a U[0,1] distribution
     * 
     * \author David Eriksson, dme65@cornell.edu
     */
    inline double rand() {
        std::uniform_real_distribution<double> rand(0, 1);
        return rand(rng::mt);
    }

    //! Set the seed to a random seed
    /*!
     * Uses the high_resolution_clock in chrono to create a random seed for SOT
     * and Armadillo. By default SOT and Armadillo use seed 0 each time for reproducibility.
     * 
     * \author David Eriksson, dme65@cornell.edu
     */
    inline void setSeedRandom() {
        // Set the armadillo seed randomly
        arma::arma_rng::set_seed_random();
        
        // Set the SOT seed using chrono
        typedef std::chrono::high_resolution_clock myClock;
        myClock::time_point beginning = myClock::now();
        myClock::duration d = myClock::now() - beginning;
        unsigned newSeed = d.count();
                
        rng::mt.seed(newSeed);
    }
    
    //! Set the seed to a given seed
    /*!
     * Sets the seed of both SOT and Armadillo to the given seed.
     * 
     * \author David Eriksson, dme65@cornell.edu
     */
    inline void setSeed(unsigned seed) {
        arma::arma_rng::set_seed(seed);
        rng::mt.seed(seed);
    }
}

#endif