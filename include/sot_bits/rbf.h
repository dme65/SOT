//
//  rbf.h
//  Surrogate Optimization
//
//  Created by David Eriksson on 9/14/15.
//  Copyright (c) 2015 David Eriksson. All rights reserved.
//
//
// Internally works on the unit hypercube
//

#ifndef Surrogate_Optimization_rbf_h
#define Surrogate_Optimization_rbf_h

#include <iostream>
#include "common.h"
#include "utils.h"
#include "surrogate.h"

namespace sot {

    class CubicKernel {
    private:
        int mPhiZero = 0;
        int mOrder = 2;
    public:
        inline int order() const { return mOrder; }
        inline int phiZero() const { return mPhiZero; }
        inline double eval(double dist) const { return dist * dist * dist; }
        inline double deriv(double dist) const { return 3 * dist * dist; }
        inline mat eval(const mat &dists) const { return dists % dists % dists; }
        inline mat deriv(const mat &dists) const { return 3 * dists % dists; }
    };
    
    class ThinPlateKernel {
    private:
        int mPhiZero = 0;
        int mOrder = 2;
    public:
        inline int order() const { return mOrder; }
        inline int phiZero() const { return mPhiZero; }
        inline double eval(double dist) const { return dist * dist * log(dist + 1e-10);}
        inline double deriv(double dist) const { return dist * (1.0 + 2.0 * log(dist + 1e-10)); }
        inline mat eval(const mat &dists) const { return dists % dists % arma::log(dists + 1e-10); }
        inline mat deriv(const mat &dists) const { return dists % (1 + 2.0 * arma::log(dists + 1e-10)); }
    };
    
    class LinearKernel {
    private:
        int mPhiZero = 0;
        int mOrder = 1;
    public:
        inline int order() const { return mOrder; }
        inline int phiZero() const { return mPhiZero; }
        inline double eval(double dist) const { return dist; }
        inline double deriv(double dist) const { return 1.0; }
        inline mat eval(const mat &dists) const { return dists; }
        inline mat deriv(const mat &dists) const { return arma::ones<mat>(dists.n_rows, dists.n_cols); }
    };
    
    class LinearTail {
    private:
        int mDegree = 1;
    public:
        inline int degree() const { return mDegree; }
        inline mat eval(const mat &x) const { return arma::join_vert(arma::ones<mat>(1, x.n_cols), x); }
        inline vec eval(const vec &x) const {
            vec tail = arma::zeros<vec>(x.n_rows + 1);
            tail(0) = 1;
            tail.tail(x.n_rows) = x;
            return tail;
        }
        inline mat deriv(const mat &x) const { return arma::join_vert(arma::zeros<mat>(1, x.n_rows), arma::eye<mat>(x.n_rows, x.n_rows)); }
        inline int nTail(int d) const { return 1 + d; }
    };
    
    class ConstantTail {
    private:
        int mDegree = 0;
    public:
        inline int degree() const { return mDegree; }
        inline mat eval(const mat &x) const { return arma::ones<mat>(x.n_rows, 1); }
        inline vec eval(const vec &x) const { return arma::ones<mat>(1, 1); }
        inline mat deriv(const mat &x) const { return arma::zeros<mat>(x.n_rows, 1); }
        inline int nTail(int d) const { return 1; }
    };
    
    template<class Kernel, class Tail>
    class RBFInterpolant : public Surrogate {
        
    protected:
        Kernel mKernel;
        Tail mTail;
        mat mL, mU;
        uvec mp;
        vec mCoeffs;
        vec mfX;
        mat mCenters;
        int mMaxPoints;
        int mnTail;
        int mNumPoints;
        bool mDirty;
        vec mxLow, mxUp;
        double mEta = 1e-6;
        int mDim;
        
    public:
        
        RBFInterpolant(int max_points, int d, vec xlow, vec xup, double eta) :
            RBFInterpolant(max_points, d, xlow, xup) {
            this->mEta = eta;
        }
        
        RBFInterpolant(int maxPoints, int dim, vec xLow, vec xUp) {
            this->mMaxPoints = maxPoints;
            this->mDim = dim;
            this->mnTail = mTail.nTail(dim);
            this->mNumPoints = 0;
            this->mCenters = arma::zeros<mat>(dim, maxPoints);
            this->mL = arma::zeros<mat>(maxPoints + mnTail, maxPoints + mnTail);
            this->mU = arma::zeros<mat>(maxPoints + mnTail, maxPoints + mnTail);
            this->mp = arma::zeros<uvec>(maxPoints + mnTail);
            this->mfX = arma::zeros<vec>(maxPoints + mnTail);
            this->mCoeffs = vec(maxPoints + mnTail);
            this->mDirty = false;
            this->mxLow = xLow;
            this->mxUp = xUp;
            
            if (not (mKernel.order() - 1 <= mTail.degree())) {
                throw std::logic_error("Kernel and tail mismatch");
            }
        }
        
        int dim() const {
            return mDim;
        }
        
        // Reset RBF
        void reset() {
            mNumPoints = 0;
        }
        
        // Number of points
        int numPoints() const {
            return mNumPoints;
        }
        
        // Return points
        mat X() const {
            return fromUnitBox((mat)mCenters.cols(0, mNumPoints - 1), mxLow, mxUp);
        }
        
        // Return point
        vec X(int i) const {
            return fromUnitBox((mat)mCenters.col(i), mxLow, mxUp);
        }
        
        // Return function values
        vec fX() const {
            return mfX.rows(mnTail, mnTail + mNumPoints - 1);
        }
        
        // Return function value
        double fX(int i) const {
            return mfX(mnTail + i);
        }
        
        vec coeffs() {
            if(mDirty) { throw std::logic_error("RBF not updated"); }
            return mCoeffs;
        }
       
        // Fit the RBF
        void fit() {
            if(mNumPoints < mnTail) { throw std::logic_error("Not enough points"); }       
            if (mDirty) {
                int n = mNumPoints + mnTail;
                mCoeffs = arma::solve(arma::trimatl(mL(arma::span(0, n - 1), arma::span(0, n - 1))), mfX(mp(arma::span(0, n - 1))));
                mCoeffs = arma::solve(arma::trimatu(mU(arma::span(0, n - 1), arma::span(0, n - 1))), mCoeffs);
                mDirty = false;
            }
        }
        
        // Set points
        void setPoints(const mat &ppoints, const vec &funVals) {
            
            // Map point to be in the unit box
            mat points = toUnitBox(ppoints, mxLow, mxUp);
            
            mNumPoints = (int)points.n_cols;
            int n = mNumPoints + mnTail;
            if(mNumPoints < mnTail) { 
                throw std::logic_error("Not enough points"); 
            }
            mat px = mTail.eval(points);
            mat phi = mKernel.eval(arma::sqrt(squaredPairwiseDistance(points, points)));
                    
            mat A = arma::zeros<mat>(n, n);
            A(arma::span(mnTail, n - 1), arma::span(0, mnTail - 1)) = px.t();
            A(arma::span(0, mnTail - 1), arma::span(mnTail, n - 1)) = px;
            A(arma::span(mnTail, n - 1), arma::span(mnTail, n - 1)) = phi;
            mfX.rows(mnTail, n - 1) = funVals;
            
            // REGULARIZATION
            A += mEta*arma::eye(n, n);
            
            // Compute the initial LU factorization of A
            mat L, U, P;
            arma::lu(L, U, P, A);
            
            mL(arma::span(0, n - 1), arma::span(0, n - 1)) = L;
            mU(arma::span(0, n - 1), arma::span(0, n - 1)) = U;
            
            // Convert P to a permutation vector
            for(int i = 0; i < n; i++) {
                uvec temp = find(P.row(i) > 0.5);
                mp(i) = temp(0);
            }
            
            mCenters.cols(0, mNumPoints - 1) = points;
            mDirty = true;
        }
        
        // Add one point
        void addPoint(const vec &ppoint, double funVal) {
            if(mNumPoints == 0) {
                std::cout << "You need to use set_points first" << std::endl;
                abort();
            }

            // Map point to be in the unit box
            vec point = toUnitBox(ppoint, mxLow, mxUp);
            
            int nAct = mnTail + mNumPoints;
            if(mNumPoints + 1 > mMaxPoints) { 
                throw std::logic_error("Capacity exceeded"); 
            }
      
            vec vx = arma::join_vert(mTail.eval(point), 
                    mKernel.eval(arma::sqrt(squaredPointSetDistance(point, mCenters.cols(0, mNumPoints - 1)))));
            vec u12 = arma::solve(arma::trimatl(mL(arma::span(0, nAct - 1), arma::span(0, nAct - 1))), vx.rows(mp.head(nAct)));
            vec l21 = (arma::solve(arma::trimatl(mU(arma::span(0, nAct - 1), arma::span(0, nAct - 1)).t()), vx));
            double u22 = mKernel.phiZero() + mEta - arma::dot(u12, l21);

            mL(nAct, arma::span(0, nAct - 1)) = l21.t();
            mL(nAct, nAct) = 1;
            mU(arma::span(0, nAct - 1), nAct) = u12;
            mU(nAct, nAct) = u22;
            mp(nAct) = nAct;

            // Update F and add the centers
            mfX(nAct) = funVal;
            mCenters.col(mNumPoints) = point;
            mNumPoints++;
            
            mDirty = true;
        }
        
        void addPoints(const mat &ppoints, const vec &funVals) {
            if(mNumPoints == 0) {
                return setPoints(ppoints, funVals);
            }
            
            // Map point to be in the unit box
            mat points = toUnitBox(ppoints, mxLow, mxUp);
            
            int nAct = mnTail + mNumPoints;
            int n = (int)funVals.n_rows;
            if(n < 2) { throw std::logic_error("Use add_point instead"); }
            if(mNumPoints + n > mMaxPoints) { throw std::logic_error("Capacity exceeded"); }
            
            auto px = mTail.eval(points);
            mat B = arma::zeros(nAct, n);
            B.rows(arma::span(0, mDim)) = px;
            B.rows(mnTail, nAct - 1) = mKernel.eval(arma::sqrt(squaredPairwiseDistance<mat>(mCenters.cols(0, mNumPoints - 1), points)));
            mat K = mKernel.eval(arma::sqrt(squaredPairwiseDistance<mat>(points, points)));
            
            // REGULARIZATION
            K += mEta*arma::eye(n, n);

            // Update the LU factorization
            mat U12 = arma::solve(arma::trimatl(mL(arma::span(0, nAct - 1), arma::span(0, nAct - 1))), B.rows(mp.head(nAct)));
            mat L21 = (arma::solve(arma::trimatl(mU(arma::span(0, nAct - 1), arma::span(0, nAct - 1)).t()), B)).t();
            mat C;
            try {
                C = arma::chol(K - L21*U12, "lower");
            }
            catch (std::runtime_error) {
                std::cout << "Warning: Cholesky factorization failed, computing new LU from scratch..." << std::endl;
                // Add new points
                mfX.rows(nAct, nAct + n - 1) = funVals;
                mCenters.cols(mNumPoints, mNumPoints + n - 1) = points;
                mNumPoints += n;
                // Build LU from scratch
                setPoints(X(), mfX.rows(mnTail, nAct + n - 1));
                return;
            }
            mL(arma::span(nAct, nAct + n - 1), arma::span(0, nAct - 1)) = L21;
            mL(arma::span(nAct, nAct + n - 1), arma::span(nAct, nAct + n - 1)) = C;
            mU(arma::span(0, nAct - 1), arma::span(nAct, nAct + n - 1)) = U12;
            mU(arma::span(nAct, nAct + n - 1), arma::span(nAct, nAct + n - 1)) = C.t();
            mp.rows(arma::span(nAct, nAct + n - 1)) = arma::linspace<uvec>(nAct, nAct + n - 1, n);
            
            // Update F and add the centers
            mfX.rows(nAct, nAct + n - 1) = funVals;
            mCenters.cols(mNumPoints, mNumPoints + n - 1) = points;
            mNumPoints += n;
            
            mDirty = true;
        }
        
        // Evaluate one point without provided distance
        double eval(const vec &ppoint) const {
            if(mDirty) { throw std::logic_error("RBF not updated"); }       

            // Map point to be in the unit box
            vec point = toUnitBox(ppoint, mxLow, mxUp);
            
            vec px = mTail.eval(point);
            vec phi = mKernel.eval(arma::sqrt(squaredPointSetDistance(point, (mat)mCenters.cols(0, mNumPoints - 1))));
            vec c = mCoeffs.head(mNumPoints + mnTail);
            return arma::dot(c.head(mnTail), px) + arma::dot(c.tail(mNumPoints), phi);
        }
        
        // Evaluate one point with provided distance
        /*
        double eval(const vec &point, const vec &dists) const {
            if(mDirty) { throw std::logic_error("RBF not updated"); }       

            // Map point to be in the unit box
            point = toUnitBox(point, mxLow, mxUp);
            
            vec px = mTail.eval(point);
            vec phi = mKernel.eval(dists);
            vec c = mCoeffs.head(mNumPoints + mnTail);
            return arma::dot(c.head(mnTail), px) + arma::dot(c.tail(mNumPoints), phi);
        }
        */
        
        // Evaluate multiple points
        vec evals(const mat &ppoints) const {
            if(mDirty) { throw std::logic_error("RBF not updated"); }       

            // Map point to be in the unit box
            mat points = toUnitBox(ppoints, mxLow, mxUp);
            
            mat px = mTail.eval(points);
            mat phi = mKernel.eval(arma::sqrt(squaredPairwiseDistance((mat)mCenters.cols(0, mNumPoints - 1), points)));
            vec c = mCoeffs.head(mNumPoints + mnTail);
            return px.t() * c.head(mnTail) + phi.t() * c.tail(mNumPoints);
        }
        
        // Evaluate multiple points with provided distances
        /*
        vec evals(const mat &points, const mat &dists) const {
            if(mDirty) { throw std::logic_error("RBF not updated"); }       

            // Map point to be in the unit box
            points = toUnitBox(points, mxLow, mxUp);
            
            mat px = mTail.eval(points);
            mat phi = mKernel.eval(dists);
            vec c = mCoeffs.head(mNumPoints + mnTail);
            return px.t() * c.head(mnTail) + phi.t() * c.tail(mNumPoints);
        }
        */
        
        // Gradient of the surface at the current point
        vec deriv(const vec &ppoint) const {
            if(mDirty) { throw std::logic_error("RBF not updated"); }       

            // Map point to be in the unit box
            vec point = toUnitBox(ppoint, mxLow, mxUp);
            
            mat dpx = mTail.deriv(point);
            vec c = mCoeffs.head(mNumPoints + mnTail);
            vec dists = arma::sqrt(squaredPairwiseDistance<mat>(mCenters.cols(0, mNumPoints - 1), point));
            dists.elem(arma::find(dists < 1e-10)).fill(1e-10); // Better safe than sorry
            mat dsx = - mCenters.cols(0, mNumPoints - 1);
            dsx.each_col() += point;
            dsx.each_row() %= (mKernel.deriv(dists) % c.tail(mNumPoints)/dists).t();
            return arma::sum(dsx, 1) + dpx.t() * c.head(mnTail);
        }
    };

    template<class Kernel, class Tail>
    class RBFInterpolantCap : public RBFInterpolant<Kernel,Tail> {
        public:
        RBFInterpolantCap(Kernel kernel,Tail tail,int max_points, int d, vec xlow, vec xup) :
            RBFInterpolant<Kernel,Tail>(kernel, tail, max_points, d, xlow, xup) {}
        void fit() {
            if (this->dirty) {
                int nact = this->num_points + this->ntail;
                vec ff = this->F.head(nact);
                double medf = arma::median(ff.rows(this->ntail, nact-1));
                for(int i=this->ntail;i<nact;i++) {
                    if (ff(i) > medf) { ff(i) = medf; }
                }
                this->coeffs = arma::solve(arma::trimatl(this->L(arma::span(0, nact - 1), arma::span(0, nact - 1))), ff(this->p));
                this->coeffs = arma::solve(arma::trimatu(this->U(arma::span(0, nact - 1), arma::span(0, nact - 1))), this->coeffs);
                this->dirty = false;
            }
        }
    };
}

#endif
