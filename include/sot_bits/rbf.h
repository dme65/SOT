/*!
 * File:   rbf.h
 * Author: David Eriksson, dme65@cornell.edu
 *
 * Created on 7/18/16.
 */


#ifndef SOT_RBF_H
#define SOT_RBF_H

#include "common.h"
#include "utils.h"
#include "surrogate.h"

//!SOT namespace
namespace sot {

    //! Abstract class for a radial kernel
    /*!
     * This is the abstract class that should be used as a Base class for all
     * RBF kernels
     * 
     * \author David Eriksson, dme65@cornell.edu
     */
    class Kernel {
        //! Method for getting the order of the kernel
        virtual inline int order() const = 0; /*!< \returns Order of the kernel */    
        //! Method for getting the value of the kernel at 0
        virtual inline int phiZero() const = 0; /*!< \returns Value of kernel at 0 */ 
        //! Method for evaluating the kernel for a given distance
        /*!
         * \param dist Distance for which to evaluate the kernel
         * \returns Value of kernel at dist
         */
        virtual inline double eval(double dist) const = 0;
        //! Method for evaluating the derivative of the kernel for a given distance
        /*!
         * \param dist Distance for which to evaluate the derivative of the kernel
         * \returns Derivative of kernel at dist
         */
        virtual inline double deriv(double dist) const = 0;
        //! Method for evaluating the kernel for a matrix of distances
        /*!
         * \param dists Distances for which to evaluate the derivative of the kernel
         * \returns Values of kernel at dists
         */
        virtual inline mat eval(const mat &dists) const = 0;
        //! Method for evaluating the derivative of the kernel for a matrix of distances
        /*!
         * \param dists Distances for which to evaluate the derivative of the kernel
         * \returns Values of the derivative of the kernel at dists
         */
        virtual inline mat deriv(const mat &dists) const = 0;
    };
    
    //! Cubic kernel
    /*!
     * This is an implementation of the popular cubic kernel 
     * \f$\varphi(r)=r^3\f$ which is of order 2.
     * 
     * \author David Eriksson, dme65@cornell.edu
     */
    class CubicKernel : public Kernel {
    private:
        int mPhiZero = 0; /*!< Value of the cubic kernel at 0 */
        int mOrder = 2; /*!< Order of the cubic kernel */
    public:
        inline int order() const { return mOrder; }  
        inline int phiZero() const { return mPhiZero; } 
        inline double eval(double dist) const { return dist * dist * dist; }
        inline double deriv(double dist) const { return 3 * dist * dist; }
        inline mat eval(const mat &dists) const { return dists % dists % dists; }
        inline mat deriv(const mat &dists) const { return 3 * dists % dists; }
    };
    
    //! TPS kernel
    /*!
     * This is an implementation of the popular thin-plate spline kernel 
     * \f$\varphi(r)=r^2\,\log(r)\f$ which is of order 2.
     * 
     * \author David Eriksson, dme65@cornell.edu
     */
    class ThinPlateKernel : public Kernel {
    private:
        int mPhiZero = 0; /*!< Value of the kernel at 0 */
        int mOrder = 2; /*!< Order of the kernel */
    public:
        inline int order() const { return mOrder; } 
        inline int phiZero() const { return mPhiZero; }
        inline double eval(double dist) const { return dist * dist * log(dist + 1e-10);}
        inline double deriv(double dist) const { return dist * (1.0 + 2.0 * log(dist + 1e-10)); }
        inline mat eval(const mat &dists) const { return dists % dists % arma::log(dists + 1e-10); }
        inline mat deriv(const mat &dists) const { return dists % (1 + 2.0 * arma::log(dists + 1e-10)); }
    };
    
    //! Linear kernel
    /*!
     * This is an implementation of the linear kernel 
     * \f$\varphi(r)=r\,\log(r)\f$ which is of order 1.
     * 
     * \author David Eriksson, dme65@cornell.edu
     */
    class LinearKernel : public Kernel {
    private:
        int mPhiZero = 0; /*!< Value of the linear kernel at 0 */
        int mOrder = 1; /*!< Order of the linear kernel */
    public:
        inline int order() const { return mOrder; } 
        inline int phiZero() const { return mPhiZero; } 
        inline double eval(double dist) const { return dist; }
        inline double deriv(double dist) const { return 1.0; }
        inline mat eval(const mat &dists) const { return dists; }
        inline mat deriv(const mat &dists) const { 
            return arma::ones<mat>(dists.n_rows, dists.n_cols); 
        }
    };
    
    //! Abstract class for a polynomial tail
    /*!
     * This is the abstract class that should be used as a Base class for all
     * Polynomial tails 
     * 
     * \author David Eriksson, dme65@cornell.edu
     */
    class Tail {
        //! Method for getting the degree of the tail
        virtual inline int degree() const = 0;  /*!< \returns Degree of the tail */  
        //! Method for the dimensionality of the polynomial space
        /*!
         * \param dim Dimensionality of the input
         * \returns Dimensionality of the polynomial space (number of basis functions)
         */
        virtual inline int dimTail(int dim) const = 0;
        //! Method for evaluating the monomial basis function for a given point
        /*!
         * \param point Point for which to evaluate the monomial basis function of the tail
         * \returns Values of the monomial basis functions at the point
         */
        virtual inline vec eval(const vec &point) const = 0;
        //! Method for evaluating the monomial basis function for multiple points
        /*!
         * \param points Points for which to evaluate the monomial basis function of the tail
         * \returns Values of the monomial basis functions at the points
         */
        virtual inline mat eval(const mat &points) const = 0;
        //! Method for evaluating the derivative of the monomial basis function for multiple points
        /*!
         * \param points Points for which to evaluate the derivative of the monomial basis function of the tail
         * \returns Values of the derivative of the monomial basis functions at the points
         */
        virtual inline mat deriv(const mat &points) const = 0;
    };
    
   //! Linear polynomial tail
    /*!
     * This is an implementation of the linear polynomial tail with basis 
     * \f$\{1,x_1,x_2,\dots,x_d\}\f$ of degree 1. Popular to use with the
     * Cubic or the TPS kernel.
     * 
     * \author David Eriksson, dme65@cornell.edu
     */
    class LinearTail : public Tail {
    private:
        int mDegree = 1; /*!< Degree of the polynomial tail */
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
        inline int dimTail(int dim) const { return 1 + dim; }
    };
    
    //! Constant polynomial tail
    /*!
     * This is an implementation of the constant polynomial tail with basis 
     * \f$\{1\}\f$ of degree 0. Popular to use with the linear kernel.
     * 
     * \author David Eriksson, dme65@cornell.edu
     */
    class ConstantTail : public Tail {
    private:
        int mDegree = 0; /*!< Degree of the polynomial tail */
    public:
        inline int degree() const { return mDegree; }
        inline mat eval(const mat &x) const { return arma::ones<mat>(x.n_rows, 1); }
        inline vec eval(const vec &x) const { return arma::ones<mat>(1, 1); }
        inline mat deriv(const mat &x) const { return arma::zeros<mat>(x.n_rows, 1); }
        inline int dimTail(int dim) const { return 1; }
    };
    
    //! Radial basis function
    /*!
     * A radial basis function (RBF) interpolant is a weighted sum of radial 
     * basis functions. It is common to add a polynomial tail as well to
     * assure that the interpolant can exactly reproduce polynomial of that degree.
     * This leads to an interpolant of the form
     * 
     * \f$ s(y) = \displaystyle\sum_{i=1}^n \lambda_i \varphi(\|y-x_i\|) + 
     *            \displaystyle\sum_{i=1}^m c_i \pi_i(y)
     * \f$
     * 
     * where \f$ y_i\f$ are the n centers and \f$ \{\pi_i\}_{i=1}^m\f$ is a basis
     * of the polynomial space of the tail. Given a set of points 
     * \f$ X=\{x_1,\ldots,x_n\}\f$ with values \f$ f_X = \{f(x_1),\ldots,f(x_n)\}\f$
     * the interpolation conditions are
     * 
     * \f$ s(x_i) = f(x_i)\f$ for \f$ i=1,\ldots,n\f$
     * 
     * and in order to get a unique interpolant one usually adds the conditions
     * 
     * \f$ \displaystyle\sum_{i=1}^n \lambda_i \pi_j(x_i) = 0\f$ for \f$ j=1,\ldots,m\f$
     * 
     * which leads to a system of equations of size \f$(m+n) x (m+n)\f$. With the 
     * notation \f$ \Phi_{i,j}=\varphi(\|x_i-x_j\|)\f$ and \f$ P_{ij} = \pi_j(x_i)\f$
     * we can write in a more compact form
     * 
     * \f$ \left(\begin{array}{cc} 0 & P^T \\ P & \Phi \end{array}\right)
     *     \left(\begin{array}{c} c \\ \lambda \end{array}\right) = 
     *     \left(\begin{array}{c} 0 \\ f_X \end{array}\right)
     * \f$
     * 
     * We can see that adding one more point corresponds to adding a column and
     * a row to this matrix which is why this ordering of \f$\Phi\f$ and \f$P\f$
     * is convenient. We store the LU decomposition of this matrix and use the
     * fact that the Schur complement is symmetric and positive definite so we
     * can update the LU-decomposition by computing the Cholesky decomposition
     * of the Schur complement. We can hence add \f$k\f$ new points using
     * roughly \f$2n^2k\f$ flops under the assumption that \f$n\gg k\f$. Solving
     * for the new coefficients is then a matter of back and forward subsitution
     * which will take roughly \f$2(m+n)^2\f$ flops. This is better than solving
     * the system from scratch which takes roughly \f$(2/3)(m+n)^3\f$ flops. 
     * 
     * In order to compute the initial decomposition we need at least m initial points 
     * that serve as "outer" points that are needed to uniquely fit the polynomial
     * tail and to construct the initial LU-decomposition with pivoting. No pivoting
     * is needed from that point.
     * 
     * The domain is automatically scaled to the unit box to avoid scaling
     * issues since the kernel and polynomial tail scale differently.
     * 
     * \tparam RBFKernel Radial kernel (Cubic is default)
     * \tparam PolyTail Polynomial tail (Linear is default)
     * 
     * \author David Eriksson, dme65@cornell.edu
     */
    template<class RBFKernel = CubicKernel, class PolyTail = LinearTail>
    class RBFInterpolant : public Surrogate {
        
    protected:
        RBFKernel mKernel; /*!< The radial kernel */
        PolyTail mTail; /*!< The polynomial tail */
        mat mL; /*!< Lower triangular part of the LU decomposition */
        mat mU; /*!< Upper triangular part of the LU decomposition */
        uvec mp; /*!< Permutation vector */
        vec mCoeffs; /*!< Coefficient vector */
        vec mfX; /*!< Function values */
        mat mCenters; /*!< Interpolation nodes (centers) */
        int mMaxPoints; /*!< Capacity */
        int mDimTail; /*!< Dimensionality of the polynomial space */
        int mNumPoints; /*!< Current number of points */
        bool mDirty; /*!< True if the coefficients need to be recomputed */
        vec mxLow; /*!< Lower variable bounds */
        vec mxUp; /*!< Upper variable bounds */
        double mEta = 1e-6; /*!< Damping added to the kernel to avoid ill-conditioning */
        int mDim; /*!< Number of dimensions */
        
        //! Computes the initial LU decomposition
        /*!
         * \param ppoints Initial points
         * \param funVals Values at the initial points
         * 
         * \throws std::logic_error if the number of points are less than the dimension of the polynomial space
         */
        void setPoints(const mat &ppoints, const vec &funVals) {
            
            // Map point to be in the unit box
            mat points = toUnitBox(ppoints, mxLow, mxUp);
            
            mNumPoints = (int)points.n_cols;
            int n = mNumPoints + mDimTail;
            if(mNumPoints < mDimTail) { 
                throw std::logic_error("Not enough points"); 
            }
            mat px = mTail.eval(points);
            mat phi = mKernel.eval(arma::sqrt(squaredPairwiseDistance(points, points)));
                    
            mat A = arma::zeros<mat>(n, n);
            A(arma::span(mDimTail, n - 1), arma::span(0, mDimTail - 1)) = px.t();
            A(arma::span(0, mDimTail - 1), arma::span(mDimTail, n - 1)) = px;
            A(arma::span(mDimTail, n - 1), arma::span(mDimTail, n - 1)) = phi;
            mfX.rows(mDimTail, n - 1) = funVals;
            
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
        
    public:
         //! Constructor
        /*!
         * \param maxPoints Capacity
         * \param dim Number of dimensions
         * \param xLow Lower variable bounds
         * \param xUp Upper variable bounds
         * \param eta Damping coefficient (non-negative)
         */
        RBFInterpolant(int maxPoints, int dim, vec xLow, vec xUp, double eta) :
            RBFInterpolant(maxPoints, dim, xLow, xUp) {
            mEta = eta;
        }
        //! Constructor with default eta
        /*!
         * \param maxPoints Capacity
         * \param dim Number of dimensions
         * \param xLow Lower variable bounds
         * \param xUp Upper variable bounds
         * 
         * \throws std::logic_error If the polynomial tail degree is too low
         */
        RBFInterpolant(int maxPoints, int dim, vec xLow, vec xUp) {
            mMaxPoints = maxPoints;
            mDim = dim;
            mDimTail = mTail.dimTail(dim);
            mNumPoints = 0;
            mCenters = arma::zeros<mat>(dim, maxPoints);
            mL = arma::zeros<mat>(maxPoints + mDimTail, maxPoints + mDimTail);
            mU = arma::zeros<mat>(maxPoints + mDimTail, maxPoints + mDimTail);
            mp = arma::zeros<uvec>(maxPoints + mDimTail);
            mfX = arma::zeros<vec>(maxPoints + mDimTail);
            mCoeffs = vec(maxPoints + mDimTail);
            mDirty = false;
            mxLow = xLow;
            mxUp = xUp;
            
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
            return mfX.rows(mDimTail, mDimTail + mNumPoints - 1);
        }
        
        // Return function value
        double fX(int i) const {
            return mfX(mDimTail + i);
        }
        
        //! Method for getting the RBF interpolation coefficients
        /*!
         * \returns RBF interpolation coefficients.
         * 
         * \throws std::logic_error If the polynomial tail degree is too low
         */
        vec coeffs() {
            if(mDirty) { throw std::logic_error("RBF not updated"); }
            return mCoeffs;
        }
       
        // Fit the RBF
        void fit() {
            if(mNumPoints < mDimTail) { throw std::logic_error("Not enough points"); }       
            if (mDirty) {
                int n = mNumPoints + mDimTail;
                mCoeffs = arma::solve(arma::trimatl(mL(arma::span(0, n - 1), arma::span(0, n - 1))), mfX(mp(arma::span(0, n - 1))));
                mCoeffs = arma::solve(arma::trimatu(mU(arma::span(0, n - 1), arma::span(0, n - 1))), mCoeffs);
                mDirty = false;
            }
        }
        
        //! Method for adding one points with known value
        /*!
         * \param ppoint Point to be added
         * \param funVal Function value at the point
         * 
         * \throws std::logic_error if capacity is exceeded
         */        
        void addPoint(const vec &ppoint, double funVal) {
            if(mNumPoints == 0) {
                vec fVal = {funVal};
                mat point = (mat)ppoint;
                return setPoints(point, fVal);
            }

            // Map point to be in the unit box
            vec point = toUnitBox(ppoint, mxLow, mxUp);
            
            int nAct = mDimTail + mNumPoints;
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
        
        //! Method for adding multiple points with known values
        /*!
         * \param ppoints Points to be added
         * \param funVals Function values at the points
         * 
         * \throws std::logic_error if one point is supplied or if capacity is exceeded
         */
        void addPoints(const mat &ppoints, const vec &funVals) {
            if(mNumPoints == 0) {
                return setPoints(ppoints, funVals);
            }
            
            // Map point to be in the unit box
            mat points = toUnitBox(ppoints, mxLow, mxUp);
            
            int nAct = mDimTail + mNumPoints;
            int n = (int)funVals.n_rows;
            if(n < 2) { throw std::logic_error("Use add_point instead"); }
            if(mNumPoints + n > mMaxPoints) { throw std::logic_error("Capacity exceeded"); }
            
            auto px = mTail.eval(points);
            mat B = arma::zeros(nAct, n);
            B.rows(arma::span(0, mDim)) = px;
            B.rows(mDimTail, nAct - 1) = mKernel.eval(arma::sqrt(squaredPairwiseDistance<mat>(mCenters.cols(0, mNumPoints - 1), points)));
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
                setPoints(X(), mfX.rows(mDimTail, nAct + n - 1));
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
        
        //! Method for evaluating the surrogate model at a point
        /*!
         * \param ppoint Point for which to evaluate the surrogate model
         * \returns Value of the surrogate model at the point
         * 
         * \throws std::logic_error if coefficients aren't updated
         */
        double eval(const vec &ppoint) const {
            if(mDirty) { throw std::logic_error("RBF not updated. You need to call fit() first"); }       

            // Map point to be in the unit box
            vec point = toUnitBox(ppoint, mxLow, mxUp);
            
            vec px = mTail.eval(point);
            vec phi = mKernel.eval(arma::sqrt(squaredPointSetDistance(point, (mat)mCenters.cols(0, mNumPoints - 1))));
            vec c = mCoeffs.head(mNumPoints + mDimTail);
            return arma::dot(c.head(mDimTail), px) + arma::dot(c.tail(mNumPoints), phi);
        }
        
        //! Method for evaluating the surrogate model at multiple points
        /*!
         * \param ppoints Points for which to evaluate the surrogate model
         * \returns Values of the surrogate model at the points
         * 
         * \throws std::logic_error if coefficients aren't updated
         */        
        vec evals(const mat &ppoints) const {
            if(mDirty) { throw std::logic_error("RBF not updated. You need to call fit() first"); }       

            // Map point to be in the unit box
            mat points = toUnitBox(ppoints, mxLow, mxUp);
            
            mat px = mTail.eval(points);
            mat phi = mKernel.eval(arma::sqrt(squaredPairwiseDistance(
                    (mat)mCenters.cols(0, mNumPoints - 1), points)));
            vec c = mCoeffs.head(mNumPoints + mDimTail);
            return px.t() * c.head(mDimTail) + phi.t() * c.tail(mNumPoints);
        }
        
        //! Method for evaluating the derivative of the surrogate model at a point
        /*!
         * \param ppoint Point for which to evaluate the derivative of the surrogate model
         * \returns Value of the derivative of the surrogate model at the point
         * 
         * \throws std::logic_error if coefficients aren't updated
         */
        vec deriv(const vec &ppoint) const {
            if(mDirty) { throw std::logic_error("RBF not updated. You need to call fit() first"); }       

            // Map point to be in the unit box
            vec point = toUnitBox(ppoint, mxLow, mxUp);
            
            mat dpx = mTail.deriv(point);
            vec c = mCoeffs.head(mNumPoints + mDimTail);
            vec dists = arma::sqrt(squaredPairwiseDistance<mat>(
                    mCenters.cols(0, mNumPoints - 1), point));
            dists.elem(arma::find(dists < 1e-10)).fill(1e-10); // Better safe than sorry
            mat dsx = - mCenters.cols(0, mNumPoints - 1);
            dsx.each_col() += point;
            dsx.each_row() %= (mKernel.deriv(dists) % c.tail(mNumPoints)/dists).t();
            return arma::sum(dsx, 1) + dpx.t() * c.head(mDimTail);
        }
    };

    //! Capped RBF interpolant
    /*!
     * This is a capped version of the RBF interpolant that is useful in cases
     * where there are large function values. This version replaces all of the
     * function values that are above the median of the function values by
     * the value of the median.
     * 
     * \tparam RBFKernel Radial kernel (Cubic is default)
     * \tparam PolyTail Polynomial tail (Linear is default)
     * 
     * \author David Eriksson, dme65@cornell.edu
     */
    template<class RBFKernel, class PolyTail>
    class RBFInterpolantCap : public RBFInterpolant<RBFKernel,PolyTail> {
    public:
        //! Constructor
        /*!
         * \param maxPoints Capacity
         * \param dim Number of dimensions
         * \param xLow Lower variable bounds
         * \param xUp Upper variable bounds
         * \param eta Damping coefficient (non-negative)
         */
        RBFInterpolantCap(int maxPoints, int dim, vec xLow, vec xUp, double eta) :
            RBFInterpolant<Kernel,Tail>(maxPoints, dim, xLow, xUp, eta) {}
        //! Constructor with default eta
        /*!
         * \param maxPoints Capacity
         * \param dim Number of dimensions
         * \param xLow Lower variable bounds
         * \param xUp Upper variable bounds
         */
        RBFInterpolantCap(int maxPoints, int dim, vec xLow, vec xUp) :
            RBFInterpolant<Kernel,Tail>(maxPoints, dim, xLow, xUp) {}
        void fit() {
            if(this->mNumPoints < this->mDimTail) { throw std::logic_error("Not enough points"); }
            if (this->mDirty) {
                int n = this->mNumPoints + this->mDimTail;
                vec ff = this->fX();
                double medf = arma::median(ff.rows(this->mDimTail, n-1)); // Computes the median
                for(int i=this->mDimTail; i<n; i++) { // Apply the capping
                    if (ff(i) > medf) { ff(i) = medf; }
                }
                
                // Solve for the coefficients
                this->coeffs = arma::solve(arma::trimatl(
                        this->mL(arma::span(0, n - 1), arma::span(0, n - 1))), ff(this->mp));
                this->coeffs = arma::solve(arma::trimatu(
                        this->mU(arma::span(0, n - 1), arma::span(0, n - 1))), this->coeffs);
                this->mDirty = false;
            }
        }
    };      
}

#endif
