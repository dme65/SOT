
//
//  optimizer_dycors.h
//  Surrogate Optimization
//
//  Created by David Eriksson on 8/12/15.
//  Copyright (c) 2015 David Eriksson. All rights reserved.
//

#ifndef Surrogate_Optimization_optimizer_h
#define Surrogate_Optimization_optimizer_h

#include <cassert>
#include <iostream>
#include "common.h"
#include "utils.h"

namespace sot {  
    class Optimizer {
    protected:
        std::shared_ptr<Problem> mData;
        std::shared_ptr<ExpDesign> mExpDes;
        std::shared_ptr<Surrogate> mSurf;
        std::shared_ptr<Sampling> mSampling;
        const double mSigmaMax = 0.2;
        const double mSigmaMin = 0.005;
        int mFailTol, mSuccTol;
        int mMaxEvals;
        int mNumEvals;
        int mInitPoints;
        int mDim;
        vec mxLow;
        vec mxUp;
        std::string mName;
    public:
        Optimizer(std::shared_ptr<Problem>& data, std::shared_ptr<ExpDesign>& expDes, 
                std::shared_ptr<Surrogate>& surf, std::shared_ptr<Sampling>& sampling, 
                int maxevals) {
            mData = std::shared_ptr<Problem>(data);
            mExpDes = std::shared_ptr<ExpDesign>(expDes);
            mSurf = std::shared_ptr<Surrogate>(surf);
            mSampling = std::shared_ptr<Sampling>(sampling);
            mMaxEvals = maxevals;
            mNumEvals = 0;
            mInitPoints = mExpDes->numPoints();
            mDim = data->dim();
            mxLow = data->lBounds();
            mxUp = data->uBounds();
            mFailTol = data->dim();
            mSuccTol = 3;
            mName = "DYCORS";
            
            assert(mMaxEvals > mInitPoints);
        }
        
        /*
        Optimizer(std::shared_ptr<Problem>& data, std::shared_ptr<ExpDesign>& exp_des, 
                std::shared_ptr<Surrogate>& surf, std::shared_ptr<Sampling>& sampling, 
                int maxeval, int numthreads) {
            this->numthreads = numthreads;
            Optimizer(data, exp_des, surf, sampling, maxeval);
        }
        */
        
        Result run() {   
            arma::arma_rng::set_seed_random();
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
