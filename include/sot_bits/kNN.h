/*!
 * File:   kNN.h
 * Author: David Eriksson, dme65@cornell.edu
 *
 * Created on 7/18/16.
 */


#ifndef SOT_kNN_H
#define SOT_kNN_H

#include "common.h"
#include "utils.h"
#include "surrogate.h"

//!SOT namespace
namespace sot {
    
    //!  k-nearest neighbors
    /*!
     * The kNN method is a robust regression method that approximates the value
     * at a given point as the average of the k closest points.
     *
     * \class kNN
     *
     * \author David Eriksson, dme65@cornell.edu
     */
    class kNN : public Surrogate {
    protected:
        int mDim; /*!< Number of dimensions */
        int mMaxPoints; /*!< Capacity */
        int mNumPoints; /*!< Current number of points */
        int mk; /*!< k (number of neighbors used in averaging) */
        mat mX; /*!< Current points */
        vec mfX; /*!< Current point values */
    public:
        //! Constructor
        /*!
         * \param maxPoints Capacity
         * \param dim Number of dimensions
         * \param k (number of neighbors used in averaging)
         */
        kNN(int maxPoints, int dim, int k) {
            mDim = dim;
            mNumPoints = 0;
            mMaxPoints = maxPoints;
            mk = k;
            mX.resize(dim, maxPoints);
            mfX.resize(maxPoints);
        }

        int dim() const {
            return mDim;
        }
        int numPoints() const {
            return mNumPoints;
        }
        mat X() const {
            return mX.cols(0, mNumPoints-1);
        }
        vec X(int i) const {
            return mX.col(i);
        }
        vec fX() const {
            return mfX.rows(0, mNumPoints-1);
        }
        double fX(int i) const {
            return mfX(i);
        }
        void addPoint(const vec &point, double funVal) {
            mX.col(mNumPoints) = point;
            mfX(mNumPoints) = funVal;
            mNumPoints++;
        }
        void addPoints(const mat &points, const vec &funVals) {
            int n = points.n_cols;
            mX.cols(mNumPoints, mNumPoints + n - 1) = points;
            mfX.rows(mNumPoints, mNumPoints + n - 1) = funVals;
            mNumPoints += n;
        }
        double eval(const vec &point) const {
            vec dists = squaredPointSetDistance(point, X());
            uvec indices = sort_index(dists);
            return arma::mean(mfX(indices.rows(0, mk - 1)));
        }
        double eval(const vec &point, const vec &dists) const {
            return eval(point);
        }
        vec evals(const mat &points) const {
            vec vals = arma::zeros<vec>(points.n_cols);
            for(int i=0; i < points.n_cols; i++) {
                vals(i) = eval(points.col(i));
            }
            return vals;
        }
        vec evals(const mat &points, const mat &dists) const {
            return evals(points);
        }
        //! Method for evaluating the kNN derivative at one point (not implemented)
        /*!
         * \throws std::logic_error Not available for kNN
         */
        vec deriv(const vec& point) const {
            throw std::logic_error("No derivatives for kNN");
        }        
        void reset() {
            mNumPoints = 0;
        }
        //! Fits kNN (does nothing)
        void fit() {
            return;
        }
    };
}

#endif
