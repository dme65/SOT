/*!
 * File:   dds.h
 * Author: David Eriksson, dme65@cornell.edu
 *
 * Created on 7/18/16.
 */

#ifndef SOT_DDS_H
#define SOT_DDS_H

#include <iostream>
#include "common.h"
#include "utils.h"
#include <thread>
#include <mutex>

//!SOT namespace
namespace sot {

    //!  The Dynamically Dimensioned Search optimization algorithm
    /*!
     * This is an implementation of the DDS algorithm. DDS generates a new point
     * to evaluate by perturbing the best solution found so far. Each variable
     * is perturbed with a probability that decreases with the number of 
     * evaluations carried out, which means that fewer and fewer variables are 
     * perturbed throughout the optimization process.
     *
     * \class DDS
     * \author David Eriksson, dme65@cornell.edu
     */
    
    class DDS {
    protected:
        std::shared_ptr<Problem> mData; /*!< A shared pointer to the optimization problem */
        std::shared_ptr<ExpDesign> mExpDes; /*!< A shared pointer to the experimental design */
        int mMaxEvals; /*!< Evaluation budget */
        int mNumEvals; /*!< Evaluation counter */
        int mInitPoints; /*!< Number of points in the experimental design (extracted from mExpDes) */
        int mDim; /*!< Number of dimensions (extracted from mData) */
        vec mxLow; /*!< Lower variable bounds (extracted from mData) */
        vec mxUp; /*!< Upper variable bounds (extracted from mData) */
        std::string mName = "DDS"; /*!< Strategy name */
        int mNumThreads; /*!< Number of threads */
        int mEvalCount = 0; /*!< Evaluation counter for evalauting batches */
        std::mutex mMutex; /*!< Mutex for assigning evaluations to the threads */
        
        //! Evalaute a batch of points in parallel
        /*!
         * \param batch Batch of points to be evaluated
         * \param funVals Vector to write the function values to
         */        
        void evalBatch(const mat &batch, vec &funVals) {
            mMutex.lock();
            int myEval = mEvalCount;
            mEvalCount++;
            mMutex.unlock();
            
            while(myEval < batch.n_cols) {
                vec x = batch.col(myEval);
                funVals[myEval] = mData->eval(x);
                
                mMutex.lock();
                myEval = mEvalCount;
                mEvalCount++;
                mMutex.unlock();
            }
        }
    public:
        //! Constructor
        /*!
         * \param data A shared pointer to the optimization problem
         * \param expDes A shared pointer to the experimental design
         * \param maxEvals Evaluation budget
         */
        DDS(std::shared_ptr<Problem>& data, std::shared_ptr<ExpDesign>& expDes, int maxEvals) {
            mData = std::shared_ptr<Problem>(data);
            mExpDes = std::shared_ptr<ExpDesign>(expDes);
            mMaxEvals = maxEvals;
            mNumEvals = 0;
            mInitPoints = expDes->numPoints();
            mDim = data->dim();
            mxLow = data->lBounds();
            mxUp = data->uBounds();
            mName = "DDS";
            mNumThreads = 1;
            if(mMaxEvals < mInitPoints) { 
                throw std::logic_error("Experimental design larger than evaluation budget"); 
            }
        }
        //! Constructor
        /*!
         * \param data A shared pointer to the optimization problem
         * \param expDes A shared pointer to the experimental design
         * \param maxEvals Evaluation budget
         * \param numThreads Number of threads
         */
        DDS(std::shared_ptr<Problem>& data, std::shared_ptr<ExpDesign>& expDes, int maxEvals, int numThreads) 
            : DDS(data, expDes, maxEvals) {
            mNumThreads = numThreads;
        }
        
        //! Runs the optimization algorithm
        /*!
         * \return A Result object with the results from the run
         */
        Result run() {
            std::vector<std::thread> threads(mNumThreads);
            Result res(mMaxEvals, mDim);
            mNumEvals = 0;
            
            vec sigma = 0.2*(mxUp - mxLow);
            mat initDes = fromUnitBox(mExpDes->generatePoints(), mxLow, mxUp);
            vec initFunVal = arma::zeros(mInitPoints);

            if(mNumThreads > 1) { // Evaluate in synchronous parallel
                mEvalCount = 0;            
                for(int i=0; i < mNumThreads; i++) {
                    threads[i] = std::thread(&sot::DDS::evalBatch, this, 
                            std::ref(initDes), std::ref(initFunVal));
                }

                for(int i=0; i < mNumThreads; i++) {
                    threads[i].join();
                }
            }
            else { // Evaluate in serial
                for(int i=0; i < mInitPoints; i++) {
                    vec x = initDes.col(i);
                    initFunVal(i) = mData->eval(x);
                }
            }
            
            res.addEvals(initDes, initFunVal);
            mNumEvals += mInitPoints;
            
            while (mNumEvals < mMaxEvals) {
                
                ////////////////////////// Select a new evaluations //////////////////////////
                double ddsProb = 1 - std::log((double)mNumEvals - mInitPoints)/
                    std::log((double)mMaxEvals - mInitPoints);
                ddsProb = fmax(ddsProb, 1.0/mDim);
                
                int newEvals = std::min<int>(mNumThreads, mMaxEvals - mNumEvals);
                mat batch = arma::zeros<mat>(mDim, newEvals);
                vec batchVals = arma::zeros(newEvals);  

                for(int i=0; i < newEvals; i++) {
                    vec cand = res.xBest();
                    int count = 0;
                    for(int j=0; j < mDim; j++) {
                        if(rand() < ddsProb) {
                            count++;
                            cand(j) += sigma(j) * randn();
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
                        int ind = randi(mDim-1);
                        cand(ind) += sigma(ind) * randn();
                        if(cand(ind) > mxUp(ind)) { 
                            cand(ind) = fmax(2*mxUp(ind) - cand(ind), mxLow(ind)); 
                        }
                        else if(cand(ind) < mxLow(ind)) { 
                            cand(ind) = fmin(2*mxLow(ind) - cand(ind), mxUp(ind)); 
                        }
                    }
                    batch.col(i) = cand;
                }
                
                               
                if(newEvals > 1) { // Evaluate in synchronous parallel
                    mEvalCount = 0;
                    for(int i=0; i < newEvals; i++) {
                        threads[i] = std::thread(&sot::DDS::evalBatch, this, 
                                std::ref(batch), std::ref(batchVals));
                    }

                    for(int i=0; i < newEvals; i++) {
                        threads[i].join();
                    }
                }
                else { // Evaluate in serial
                    batchVals(0) = mData->eval((vec)batch);
                }
                      
                // Update evaluation counter
                mNumEvals += newEvals;                
                res.addEvals(batch, batchVals);                
            }
                                
            return res;
        }
    };
}

#endif
