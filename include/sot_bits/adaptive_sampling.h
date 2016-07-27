/*!
 * File:   adaptive_sampling.h
 * Author: David Eriksson, dme65@cornell.edu
 *
 * Created on 7/18/16.
 */

#ifndef SOT_ADAPTIVE_SAMPLING_H
#define SOT_ADAPTIVE_SAMPLING_H

#include "common.h"
#include "utils.h"
#include "problem.h"

//!SOT namespace
namespace sot {
    
    //!  Abstract class for a SOT adaptive sampling class
    /*!
     * This is the abstract class that should be used as a Base class for all
     * sampling objects in SOT. The sampling object is used to propose new 
     * evaluations after the initial experimental design has been evaluated.
     * 
     * \author David Eriksson, dme65@cornell.edu
     */
    
    class Sampling {
    public:
        //! Virtual method for reseting the object
        /*!
         * \param budget The remaining evaluation budget
         */
        virtual void reset(int budget) = 0;
        
        //! Virtual method for proposing new evaluations
        /*!
         * \param xBest The best solution found so far
         * \param points Previously evaluated points
         * \param sigma The sampling radius
         * \param newPoints Number of new evaluations to be generated
         * \return The proposed points
         */
        virtual mat makePoints(const vec &xBest, const mat &points, double sigma, int newPoints) = 0;
    };
    
    //! Stochastic RBF
    /*!
     * This is an implementation of the SRBF method that generates the candidate
     * points by perturbing each variable by a normally distrubuted realization. 
     * 
     * \tparam MeritFunction The merit function is used to pick the most promising out of 
     * the generated candidate points.
     * 
     * \author David Eriksson, dme65@cornell.edu
     */
    template<class MeritFunction = MeritWeightedDistance>
    class SRBF : public Sampling {
    protected:
        std::shared_ptr<Problem> mData; /*!< A shared pointer to the optimization problem */
        std::shared_ptr<Surrogate> mSurf; /*!< A shared pointer to the surrogate model */
        int mNumCand; /*!< Number of candidate points that are generated in makePoints */
        int mDim; /*!< Number of dimensions (extracted from mData) */
        vec mxLow; /*!< Lower variable bounds (extracted from mData) */
        vec mxUp; /*!< Upper variable bounds (extracted from mData) */
        double mDistTol; /*!< Distance tolerance */
        int mNumEvals = 0; /*!< Current evaluation count */
        int mBudget; /*!< Evaluation budget for the adaptive sampling phase */
        MeritFunction mMerit; /*!< Merit function that is used for picking candidate points */
    public:
        //! Constructor
        /*!
         * \param data A shared pointer to the optimization problem
         * \param surf A shared pointer to the surrogate model
         * \param numCand Number of candidate points that are generated in makePoints
         * \param budget Evaluation budget for the adaptive sampling phase
         */
        SRBF(const std::shared_ptr<Problem>& data, const std::shared_ptr<Surrogate>& surf, int numCand, int budget) {
            mData = std::shared_ptr<Problem>(data);
            mSurf = std::shared_ptr<Surrogate>(surf);
            mBudget = budget;
            mNumCand = numCand;
            mDim = data->dim();
            mxLow = data->lBounds();
            mxUp = data->uBounds();
            mDistTol = 1e-3*sqrt(arma::sum(arma::square(mxUp - mxLow)));
        }
        
        //! Resets the object for a new budget (useful if a strategy restarts)
        /*!
         * \param budget New evaluation budget
         */
        void reset(int budget) {
            mBudget = budget;
            mNumEvals = 0;
        }
        
        //! Proposes new evaluations
        /*!
         * \param xBest The best solution found so far
         * \param points Previously evaluated points
         * \param sigma The sampling radius
         * \param newPoints Number of new evaluations to be generated
         * \return The proposed points
         */
        mat makePoints(const vec &xBest, const mat &points, double sigma, int newPoints) {

            mat cand = arma::repmat(xBest, 1, mNumCand);

            for(int i=0; i < mNumCand; i++) {
                for(int j=0; j < mDim; j++) {
                    cand(j, i) += sigma * randn();
                    if(cand(j, i) > mxUp(j)) { 
                        cand(j, i) = fmax(2*mxUp(j) - cand(j, i), mxLow(j)); 
                    }
                    else if(cand(j, i) < mxLow(j)) { 
                        cand(j, i) = fmin(2*mxLow(j) - cand(j, i), mxUp(j)); 
                    }
                }
            }
            
            // Update counter
            mNumEvals += newPoints;
            
            return mMerit.pickPoints(cand, mSurf, points, newPoints, mDistTol);
        }
    };
    
    //! DYnamic COordinate search using Response Surface models
    /*!
     * This is an implementation of the DYCORS method that perturbs fewer and
     * fewer variables as the optimization proceeds. The candidate points
     * are generated by perturbing each variable using the probability proposed
     * by DYCORS. 
     * 
     * \tparam MeritFunction The merit function is used to pick the most promising out of 
     * the generated candidate points.
     * 
     * \todo Should use SRBF as a Base class
     * 
     * \author David Eriksson, dme65@cornell.edu
     */
    
    template<class MeritFunction = MeritWeightedDistance>
    class DYCORS : public Sampling {
    protected:
        std::shared_ptr<Problem> mData; /*!< A shared pointer to the optimization problem */
        std::shared_ptr<Surrogate> mSurf; /*!< A shared pointer to the surrogate model */
        int mNumCand; /*!< Number of candidate points that are generated in makePoints */
        int mDim; /*!< Number of dimensions (extracted from mData) */
        vec mxLow; /*!< Lower variable bounds (extracted from mData) */
        vec mxUp;  /*!< Upper variable bounds (extracted from mData) */
        double mDistTol; /*!< Distance tolerance */
        int mNumEvals = 0; /*!< Current evaluation count */
        int mBudget; /*!< Evaluation budget for the adaptive sampling phase */
        MeritFunction mMerit; /*!< Merit function that is used for picking candidate points */
    public:
        //! Constructor
        /*!
         * \param data A shared pointer to the optimization problem
         * \param surf A shared pointer to the surrogate model
         * \param numCand Number of candidate points that are generated in makePoints
         * \param budget Evaluation budget for the adaptive sampling phase
         */
        DYCORS(const std::shared_ptr<Problem>& data, const std::shared_ptr<Surrogate>& surf, 
                int numCand, int budget) {
            mData = std::shared_ptr<Problem>(data);
            mSurf = std::shared_ptr<Surrogate>(surf);
            mBudget = budget;
            mNumCand = numCand;
            mDim = data->dim();
            mxLow = data->lBounds();
            mxUp = data->uBounds();
            mDistTol = 1e-3*sqrt(arma::sum(arma::square(mxUp - mxLow)));
        }
        
        //! Resets the object for a new budget (useful if a strategy restarts)
        /*!
         * \param budget New evaluation budget
         */
        void reset(int budget) {
            mBudget = budget;
            mNumEvals = 0;
        }
        
        //! Proposes new evaluations
        /*!
         * \param xBest The best solution found so far
         * \param points Previously evaluated points
         * \param sigma The sampling radius
         * \param newPoints Number of new evaluations to be generated
         * \return The proposed points
         */
        mat makePoints(const vec &xBest, const mat &points, double sigma, int newPoints) {                
            double dds_prob = std::min(20.0/mDim, 1.0) * 
                (1.0 - (std::log(mNumEvals + 1.0) / std::log(double(mBudget))));
            mat cand = arma::repmat(xBest, 1, mNumCand);
            for(int i=0; i < mNumCand; i++) {

                int count = 0;
                for(int j=0; j < mDim; j++) {
                    if(rand() < dds_prob) {
                        count++;
                        cand(j, i) += sigma*randn();
                    }
                }
                // If no index was perturbed we force one
                if(count == 0) {
                    int ind = randi(mDim);
                    cand(ind, i) += sigma*randn();
                }

                // Make sure we are still in the domain
                for(int j=0; j < mDim; j++) {
                    if(cand(j, i) > mxUp(j)) { 
                        cand(j, i) = fmax(2*mxUp(j) - cand(j, i), mxLow(j)); 
                    }
                    else if(cand(j, i) < mxLow(j)) { 
                        cand(j, i) = fmin(2*mxLow(j) - cand(j, i), mxUp(j)); 
                    }
                }
            }
            
            // Update counter
            mNumEvals += newPoints;
            
            return mMerit.pickPoints(cand, mSurf, points, newPoints, mDistTol);
        }
    };
    
    //! Uniformly chosen candidate points
    /*!
     * This method generates each candidate points as a uniformly chosen point from
     * the domain.
     * 
     * \tparam MeritFunction The merit function is used to pick the most promising out of 
     * the generated candidate points.
     * 
     * \todo Should use SRBF as a Base class
     * 
     * \author David Eriksson, dme65@cornell.edu
     */    
    template<class MeritFunction = MeritWeightedDistance>
    class Uniform : public Sampling {
    protected:
        std::shared_ptr<Problem> mData; /*!< A shared pointer to the optimization problem */
        std::shared_ptr<Surrogate> mSurf; /*!< A shared pointer to the surrogate model */
        int mNumCand; /*!< Number of candidate points that are generated in makePoints */
        int mDim; /*!< Number of dimensions (extracted from mData) */
        vec mxLow; /*!< Lower variable bounds (extracted from mData) */
        vec mxUp;  /*!< Upper variable bounds (extracted from mData) */
        double mDistTol; /*!< Distance tolerance */
        int mNumEvals = 0; /*!< Current evaluation count */
        int mBudget; /*!< Evaluation budget for the adaptive sampling phase */
        MeritFunction mMerit; /*!< Merit function that is used for picking candidate points */
    public:
        //! Constructor
        /*!
         * \param data A shared pointer to the optimization problem
         * \param surf A shared pointer to the surrogate model
         * \param numCand Number of candidate points that are generated in makePoints
         * \param budget Evaluation budget for the adaptive sampling phase
         */
        Uniform(const std::shared_ptr<Problem>& data, const std::shared_ptr<Surrogate>& surf, int numCand, int budget) {
            mData = std::shared_ptr<Problem>(data);
            mSurf = std::shared_ptr<Surrogate>(surf);
            mBudget = budget;
            mNumCand = numCand;
            mDim = data->dim();
            mxLow = data->lBounds();
            mxUp = data->uBounds();
            mDistTol = 1e-3*sqrt(arma::sum(arma::square(mxUp - mxLow)));
        }
        
        //! Resets the object for a new budget (useful if a strategy restarts)
        /*!
         * \param budget New evaluation budget
         */
        void reset(int budget) {
            mBudget = budget;
            mNumEvals = 0;
        }
        
        //! Proposes new evaluations
        /*!
         * \param xBest The best solution found so far
         * \param points Previously evaluated points
         * \param sigma The sampling radius
         * \param newPoints Number of new evaluations to be generated
         * \return The proposed points
         */
        mat makePoints(const vec &xBest, const mat &points, double sigma, int newPoints) {
         
            mat cand = arma::randu<mat>(mDim, mNumCand);
            for(int j=0; j < mDim; j++) {
                cand.row(j) = mxLow(j) + (mxUp(j) - mxLow(j)) * cand.row(j);
            }
        
            // Update counter
            mNumEvals += newPoints;
            
            return mMerit.pickPoints(cand, mSurf, points, newPoints, mDistTol);
        }
    };
}

#endif
