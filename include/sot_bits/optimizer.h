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
#include <thread>
#include <mutex>

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
            mNumEvals = 0;
            mMaxEvals = maxEvals;
            mInitPoints = mExpDes->numPoints();
            mDim = data->dim();
            mxLow = data->lBounds();
            mxUp = data->uBounds();
            mFailTol = data->dim();
            mNumThreads = 1;

            if(mMaxEvals < mInitPoints) { 
                throw std::logic_error("Experimental design larger than evaluation budget"); 
            }
        }
        //! Constructor
        /*!
         * \param data A shared pointer to the optimization problem
         * \param expDes A shared pointer to the experimental design
         * \param surf A shared pointer to the surrogate model
         * \param sampling A shared pointer to the adaptive sampling
         * \param maxEvals Evaluation budget
         * \param numThreads Number of threads to use for parallel evaluations
         * 
         * \throws std::logic_error If size of experimental design exceeds the evaluation budget
         */
        Optimizer(std::shared_ptr<Problem>& data, std::shared_ptr<ExpDesign>& expDes, 
                std::shared_ptr<Surrogate>& surf, std::shared_ptr<Sampling>& sampling, 
                int maxEvals, int numThreads) : Optimizer(data, expDes, surf, sampling, maxEvals) 
        {
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
            
            double fBestLoc = std::numeric_limits<double>::max();
            vec xBestLoc;
                        
        start:
            double sigma = mSigmaMax;
            int fail = 0;
            int succ = 0;
            
            mat initDes = fromUnitBox(mExpDes->generatePoints(), mxLow, mxUp);
            vec initFunVal = arma::zeros(mInitPoints);
            
            ////////////////////////////// Evaluate the initial design //////////////////////////////\

            if(mNumThreads > 1) { // Evaluate in synchronous parallel
                mEvalCount = 0;            
                for(int i=0; i < mNumThreads; i++) {
                    threads[i] = std::thread(&sot::Optimizer::evalBatch, this, 
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

            int iStart = mNumEvals;
            int iEnd = std::min<int>(mNumEvals + mInitPoints - 1, mMaxEvals - 1);
            for(int i=mNumEvals; i <= iEnd ; i++) {
                vec x = initDes.col(i - iStart);
                double fx = initFunVal[i - iStart];
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
                int newEvals = std::min<int>(mNumThreads, mMaxEvals - mNumEvals);
                mat batch = mSampling->makePoints(xBestLoc, X, sigma*(mxUp(0) - mxLow(0)), newEvals);
                vec batchVals = arma::zeros(newEvals);  
                               
                if(newEvals > 1) { // Evaluate in synchronous parallel
                    mEvalCount = 0;
                    for(int i=0; i < newEvals; i++) {
                        threads[i] = std::thread(&sot::Optimizer::evalBatch, this, 
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

                // Add to results
                for(int i=0; i < newEvals; i++) {
                    vec x = batch.col(i);
                    res.addEval(x, batchVals(i));
                }
              
                // Process evaluations
                for(int i=0; i < newEvals; i++) {
                    vec newx = batch.col(i);
                    double fVal = batchVals(i);
                    
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
                }
                                
                // Add to surface
                if (batch.n_cols > 1) {
                    mSurf->addPoints(batch, batchVals);
                } 
                else {
                    mSurf->addPoint((vec)batch, batchVals(0));
                }
            }
            
            return res;
        }
    };
}

#endif

