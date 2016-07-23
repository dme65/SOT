//
//  dds.h
//  Surrogate Optimization
//
//  Created by David Eriksson on 8/22/15.
//  Copyright (c) 2015 David Eriksson. All rights reserved.
//

#ifndef Surrogate_Optimization_dds_h
#define Surrogate_Optimization_dds_h

#include <iostream>
#include "common.h"
#include "utils.h"

namespace sot {

    class DDS {
    protected:
        std::shared_ptr<Problem> mData;
        std::shared_ptr<ExpDesign> mExpDes;
        int mMaxEvals;
        int mNumEvals;
        int mInitPoints;
        int mDim;
        vec mxLow;
        vec mxUp;
        std::string mName;
    public:
        DDS(std::shared_ptr<Problem>& data, std::shared_ptr<ExpDesign>& expDes, int maxevals) {
            mData = std::shared_ptr<Problem>(data);
            mExpDes = std::shared_ptr<ExpDesign>(expDes);
            mMaxEvals = maxevals;
            mNumEvals = 0;
            mInitPoints = expDes->numPoints();
            mDim = data->dim();
            mxLow = data->lBounds();
            mxUp = data->uBounds();
            mName = "DDS";
            if(mMaxEvals < mInitPoints) { throw std::logic_error("Experimental design larger than evaluation budget"); }
        }
        
        Result run() {
            arma::arma_rng::set_seed_random();
            Result res(mMaxEvals, mDim);
            mNumEvals = 0;
            
            double sigma = 0.2*(mxUp(0) - mxLow(0));
            mat init_des = fromUnitBox(mExpDes->generatePoints(), mxLow, mxUp);
            
            for(int i=0; i < mInitPoints ; i++) {
                vec x = init_des.col(i);
                res.addEval(x, mData->eval(x));
                mNumEvals++;
            }
            
            while (mNumEvals < mMaxEvals) {
                
                ////////////////////////// Select a new evaluations //////////////////////////
                double ddsProb = 1 - log(mNumEvals - mInitPoints)/log(mMaxEvals - mInitPoints);
                ddsProb = fmax(ddsProb, 1.0/mDim);
                
                vec cand = res.xBest();
                int count = 0;
                for(int j=0; j < mDim; j++) {
                    if(rand() < ddsProb) {
                        count++;
                        cand(j) += sigma * randn();
                        if(cand(j) > mxUp(j)) { 
                            cand(j) = fmax(2*mxUp(j) - cand(j), mxLow(j)); 
                        }
                        else if(cand(j) < mxLow(j)) { 
                            cand(j) = fmin(2*mxLow(j) - cand(j), mxUp(j)); 
                        }
                    }
                }
                // If no index was perturbed we force one
                if(count == 0) {
                    int ind = randi(mDim);
                    cand(ind) += sigma * randn();
                    if(cand(ind) > mxUp(ind)) { 
                        cand(ind) = fmax(2*mxUp(ind) - cand(ind), mxLow(ind)); 
                    }
                    else if(cand(ind) < mxLow(ind)) { 
                        cand(ind) = fmin(2*mxLow(ind) - cand(ind), mxUp(ind)); 
                    }
                }
                
                /////////////////////// Evaluate ///////////////////////
                res.addEval(cand, mData->eval(cand));
                
                mNumEvals++;
            }
                                
            return res;
        }
    };
}

#endif
