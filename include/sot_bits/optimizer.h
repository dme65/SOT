/*!
 * File:   optimizer.h
 * Author: David Eriksson, dme65@cornell.edu
 *
 * Created on 7/18/16.
 */

#ifndef SOT_OPTIMIZER_H
#define SOT_OPTIMIZER_H

#include "common.h"
#include "utils.h"

//!SOT namespace
namespace sot {  
    
    //!  The surrogate optimization algorithm
    /*!
     * This is a general surrogate optimization algorithm that uses the supplied
     * experimental design, surrogate model, and adaptive sampling to minimize
     * the given optimization problem. The optimization algorithm starts by
     * searching far from previously generated points and start searching closer
     * and closer to the best solution found so far when not enough improvement
     * is made. The algorithm restarts with a new experimental design when the
     * search radius gets too small.
     * 
     * \author David Eriksson, dme65@cornell.edu
     */
    
    class Optimizer {
    protected:
        std::shared_ptr<Problem> mData; /*!< A shared pointer to the optimization problem */
        std::shared_ptr<ExpDesign> mExpDes;  /*!< A shared pointer to the experimental design */
        std::shared_ptr<Surrogate> mSurf; /*!< A shared pointer to the surrogate model */
        std::shared_ptr<Sampling> mSampling;  /*!< A shared pointer to the adpative sampling */
        const double mSigmaMax = 0.2; /*!< Largest search radius w.r.t. the unix box */
        const double mSigmaMin = 0.005; /*!< Smallest search radius w.r.t. the unix box */
        int mSuccTol = 3; /*!< After mSuccTol consecutive improvements the search radius is increased */
        int mFailTol; /*!< After mSuccTol consecutive iteration without improvement the search radius is decreased */
        int mMaxEvals; /*!< Evaluation budget */
        int mNumEvals; /*!< Evaluations carried out */
        int mInitPoints; /*!< Points in the initial design (extracted from mExpDes) */
        int mDim; /*!< Number of dimensions (extracted from mData) */
        vec mxLow; /*!< Lower variable bounds (extracted from mData) */
        vec mxUp; /*!< Upper variable bounds (extracted from mData) */
        std::string mName = "Surrogate Optimizer"; /*!< Strategy name */
    public:
        //! Constructor
        /*!
         * \param data A shared pointer to the optimization problem
         * \param expDes A shared pointer to the experimental design
         * \param surf A shared pointer to the surrogate model
         * \param sampling A shared pointer to the adaptive sampling
         * \param maxEvals Evaluation budget
         * \throws std::logic_error If size of experimental design exceeds the evaluation budget
         */
        Optimizer(std::shared_ptr<Problem>& data, std::shared_ptr<ExpDesign>& expDes, 
                std::shared_ptr<Surrogate>& surf, std::shared_ptr<Sampling>& sampling, 
                int maxEvals) {
            mData = std::shared_ptr<Problem>(data);
            mExpDes = std::shared_ptr<ExpDesign>(expDes);
            mSurf = std::shared_ptr<Surrogate>(surf);
            mSampling = std::shared_ptr<Sampling>(sampling);
            mMaxEvals = maxEvals;
            mNumEvals = 0;
            mInitPoints = mExpDes->numPoints();
            mDim = data->dim();
            mxLow = data->lBounds();
            mxUp = data->uBounds();
            mFailTol = data->dim();
            
            if(mMaxEvals < mInitPoints) { 
                throw std::logic_error("Experimental design larger than evaluation budget"); 
            }
        }
        
        //! Runs the optimization algorithm
        /*!
         * \return A Result object with the results from the run
         */
        Result run() {   
            Result res(mMaxEvals, mDim);
            mNumEvals = 0;
            
            double fBestLoc = std::numeric_limits<double>::max();
            vec xBestLoc;
                        
        start:
            double sigma = mSigmaMax;
            int fail = 0;
            int succ = 0;
            
            mat initDes = fromUnitBox(mExpDes->generatePoints(), mxLow, mxUp);
            
            ////////////////////////////// Evaluate the initial design //////////////////////////////
            int iStart = mNumEvals;
            int iEnd = fmin(mNumEvals + mInitPoints - 1, mMaxEvals - 1);
            for(int i=mNumEvals; i <= iEnd ; i++) {
                vec x = initDes.col(i - iStart);
                double fx = mData->eval(x);
                res.addEval(x, fx);
                if(fx < fBestLoc) {
                    xBestLoc = x;
                    fBestLoc = fx;
                }
                mNumEvals++;
            }
            
            ////////////////////////////// Add points to the rbf //////////////////////////////
            if(iStart < iEnd) {
                mSurf->addPoints(res.X().cols(iStart, iEnd), res.fX().rows(iStart, iEnd));
            }
            ////////////////////////////// The fun starts now! //////////////////////////////////////
            while (mNumEvals < mMaxEvals) {
                // Fit the RBF
                mSurf->fit();
                
                // Find new points to evaluate
                mat X = res.X().cols(iStart, mNumEvals - 1);
                vec newx = mSampling->makePoints(xBestLoc, X, sigma*(mxUp(0) - mxLow(0)), 1);
                                
                // Evaluate
                double fVal = mData->eval(newx);
                res.addEval(newx, fVal);
                mNumEvals++;
              
                // Process evaluation
                if(fVal < fBestLoc) {
                    if(fVal < fBestLoc - 1e-3 * fabs(fBestLoc)) {
                        fail = 0;
                        succ++;
                    }
                    else {
                        fail++;
                        succ = 0;
                    }
                    fBestLoc= fVal;
                    xBestLoc = newx;
                }
                else {
                    fail++;
                    succ = 0;
                }
                
                // Update sigma if necessary
                if(fail == mFailTol) {
                    fail = 0;
                    succ = 0;
                    sigma /= 2.0;
                    int budget = mMaxEvals - mNumEvals - 1;
                    // Restart if sigma is too small and the budget 
                    // is larger than the initial design
                    if (sigma < mSigmaMin and budget > mInitPoints) {
                        fBestLoc = std::numeric_limits<double>::max();
                        mSurf->reset();
                        mSampling->reset(mMaxEvals - mNumEvals - mInitPoints);
                        goto start;
                    }
                }
                if(succ == mSuccTol) {
                    fail = 0;
                    succ = 0;
                    sigma = fmin(sigma * 2.0, mSigmaMax);
                }
                                
                // Add to surface
                mSurf->addPoint(newx, fVal);
            }
            
            return res;
        }
    };
}

#endif
