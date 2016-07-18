//
//  utils.h
//  Surrogate Optimization
//
//  Created by David Eriksson on 7/30/15.
//  Copyright (c) 2015 David Eriksson. All rights reserved.
//

#ifndef __Surrogate_Optimization__utils__
#define __Surrogate_Optimization__utils__

#include <stdio.h>
#include <armadillo>
#include <assert.h>
#include <thread>         // std::this_thread::sleep_for
#include <chrono>         // std::chrono::seconds
#include <iomanip>
#include <iostream>
#include "common.h"

namespace sot {
    
    // Inputs:
    //  - x: Vector of size d x 1
    //  - Y: Matrix of size d x n
    //
    // Output:
    //  - Vector of size 1 x n
    //
    // Note: If we are unlucky we might get a roundoff error and get a number that is actually negative
    //       on the magnitude of machine precision. This can give terrible effects later so this is why
    //       we take the absolute value of the result.
    //
    
    template <class MatType, class VecType>
    inline VecType SquaredPointSetDistance(const VecType& x, const MatType& Y) {
        return arma::abs(arma::repmat(arma::sum(x % x,0),Y.n_cols,1) + arma::sum(Y % Y,0).t() - 2*Y.t()*x);
    };
    
    // Inputs:
    //  - X: Matrix of size d x m
    //  - Y: Matrix of size d x n
    //
    // Output:
    //  - Matrix of size m x n
    //
    // Note: If we are unlucky we might get a roundoff error and get a number that is actually negative
    //       on the magnitude of machine precision. This can give terrible effects later so this is why
    //       we take the absolute value of the result.
    //
    template <class MatType>
    inline MatType SquaredPairwiseDistance(const MatType& X, const MatType& Y) {
        //return arma::abs((arma::repmat(arma::sum(X % X, 0).t(),1,Y.n_cols) + arma::repmat(arma::sum(Y % Y,0), X.n_cols, 1) - 2*(X.t()*Y)));
        MatType dists = - 2*(X.t()*Y);
        dists.each_row() += arma::sum(Y % Y, 0);
        dists.each_col() += arma::sum(X % X, 0).t();
        dists = arma::abs(dists);
        return dists;
    };
    
    template <class MatType>
    inline void SquaredPairwiseDistance2(const MatType& X, const MatType& Y, MatType &dists) {
        //return arma::abs((arma::repmat(arma::sum(X % X, 0).t(),1,Y.n_cols) + arma::repmat(arma::sum(Y % Y,0), X.n_cols, 1) - 2*(X.t()*Y)));
        dists = - 2*(X.t()*Y);
        dists.each_row() += arma::sum(Y % Y, 0);
        dists.each_col() += arma::sum(X % X, 0).t();
        dists = arma::abs(dists);
    };
    
    // Inputs:
    //  - X: Matrix of size d x n
    //  - xlow: Vector of size d x 1
    //  - xup: Vector of size d x 1
    //
    // Output:
    //  - Matrix of size d x n
    //
    
    inline mat ToUnitBox(const mat& X, const vec& xlow, const vec& xup) {
        return (X - arma::repmat(xlow, 1, X.n_cols))/arma::repmat(xup - xlow, 1, X.n_cols);
    };
    
    // Inputs:
    //  - X: Vector of size d x 1
    //  - xlow: Vector of size d x 1
    //  - xup: Vector of size d x 1
    //
    // Output:
    //  - Vector of size d x 1
    //
    inline vec ToUnitBox(const vec& X, const vec& xlow, const vec& xup) {
        return (X - xlow)/(xup - xlow);
    };
    
    // Inputs:
    //  - X: Matrix of size d x n
    //  - xlow: Vector of size d x 1
    //  - xup: Vector of size d x 1
    //
    // Output:
    //  - Matrix of size d x n
    //
    inline mat FromUnitBox(const mat& X, const vec& xlow, const vec& xup) {
        return arma::repmat(xlow, 1, X.n_cols) + arma::repmat(xup - xlow, 1, X.n_cols) % X;
    };
    
    // Inputs:
    //  - X: Vector of size d x 1
    //  - xlow: Vector of size d x 1
    //  - xup: Vector of size d x 1
    //
    // Output:
    //  - Vector of size d x 1
    //
    inline vec FromUnitBox(const vec& X, const vec& xlow, const vec& xup) {
        return xlow + (xup - xlow) % X;
    };
    
    // Unit rescale
    template <class VecType>
    inline VecType UnitRescale(const VecType& x) {
        double xmin = arma::min(x);
        double xmax = arma::max(x);
        if( xmin == xmax ) {
            return arma::ones<VecType>(x.n_elem);
        }
        return (x - xmin)/(xmax - xmin);
    };
    
    // Keeps track of function values and results
    class Result {
    public:
        int dim;
        int maxeval;
        int exp_des_eval;
        vec fx;
        mat x;
        double fbest;
        vec xbest;
        Result(int maxeval, int exp_des_eval, int dim) {
            this->maxeval = maxeval;
            this->exp_des_eval = exp_des_eval;
            this->dim = dim;
            fx = arma::datum::inf * arma::ones<vec>(maxeval);
            fbest = arma::datum::inf;
            x = arma::datum::inf * arma::ones<mat>(dim, maxeval);
            xbest = arma::datum::inf * arma::ones<mat>(dim);
        }
        void reset() {
            x = arma::datum::inf * arma::ones<mat>(dim, maxeval);
            xbest = arma::datum::inf * arma::ones<mat>(dim);
            fx = arma::datum::inf * arma::ones<vec>(maxeval);
            fbest = arma::datum::inf;
        }
    };
    
    inline void readmat(mat &out, std::string fname, int n_rows, int n_cols) {
        if(out.n_rows != n_rows or out.n_cols != n_cols) {
            out = arma::zeros<mat>(n_rows, n_cols);
        }
        
        std::ifstream in(fname);
        std::string line;
        for(int i=0; i < n_rows; i++) {
            std::getline(in, line);
            double value;
            std::stringstream ss(line);
            for(int j=0; j < n_cols; j++) {
                ss >> value;
                out(i,j) = value;
            }
        }
    };
    
    // Computes the Pareto front of x, y
    inline uvec pareto_front(const vec &x, const vec &y) {
        assert(x.n_rows == y.n_rows);
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
    
    class QRSystem {
    private:
        mat Qmat;
        mat Rmat;
        int m;
        int n;
    public:
        QRSystem(int m_max, int n_max) {
            this->Qmat = arma::zeros(m_max, m_max);
            this->Rmat = arma::zeros(m_max, n_max);
            this->m = 0;
            this->n = 0;
        }
        mat Q() {
            return Qmat(arma::span(0, m-1), arma::span(0, m-1));
        }
        mat Q_econ() {
            return Qmat(arma::span(0, m-1), arma::span(0, n-1));
        }
        mat R() {
            return Rmat(arma::span(0, m-1), arma::span(0, n-1));
        }
        mat R_econ() {
            return Rmat(arma::span(0, n-1), arma::span(0, n-1));
        }
        void set_points(mat A) {
            assert(A.n_rows <= this->Qmat.n_rows && A.n_cols <= this->Rmat.n_cols);
            assert(A.n_cols <= A.n_rows);
            Qmat *= 0;
            Rmat *= 0;
            mat Q1, R1;
            qr(Q1, R1, A);

            this->Qmat(arma::span(0, A.n_rows-1), arma::span(0, A.n_rows-1)) = Q1;
            this->Rmat(arma::span(0, A.n_rows-1), arma::span(0, A.n_cols-1)) = R1;
            this->m = A.n_rows;
            this->n = A.n_cols;
        }
        void set_points(mat Q1, mat R1) {
            assert(Q1.n_rows == Q1.n_cols && Q1.n_rows == R1.n_rows && R1.n_rows == R1.n_cols);
            assert(Q1.n_rows <= this->Qmat.n_rows && R1.n_cols <= this->Rmat.n_cols);
            assert(R1.n_cols <= R1.n_rows);
            this->Qmat *= 0;
            this->Rmat *= 0;
            this->Qmat(arma::span(0, Q1.n_rows-1), arma::span(0, Q1.n_cols-1)) = Q1;
            this->Rmat(arma::span(0, R1.n_rows-1), arma::span(0, R1.n_cols-1)) = R1;
            this->m = Q1.n_rows;
            this->n = R1.n_cols;
        }
        // Add a row to A
        void add_row(const vec &a) {
            assert(m+1 <= Qmat.n_rows);
            assert(a.n_elem == n);
            Qmat(m, m) = 1;
            Rmat(m ,arma::span(0,n-1)) = a.t();
            
            for (int i=0; i < n; i++) { // Each iteration is O(m)
                // Zero out elements
                double r = sqrt(Rmat(i,i)*Rmat(i,i) + Rmat(m,i)*Rmat(m,i));
                double c = Rmat(i,i)/r;
                double s = -Rmat(m,i)/r;
                
                // Avoid making the matrix multiplications since things are sparse
                vec Rij = Rmat(i, arma::span(0, n-1)).t();
                Rmat(i, arma::span(0, n-1)) = c*Rmat(i, arma::span(0, n-1)) - s*Rmat(m, arma::span(0, n-1));
                Rmat(m, arma::span(0, n-1)) = s*Rij.t() + c*Rmat(m, arma::span(0, n-1));
                
                vec Qji = Qmat(arma::span(0, m), i);
                Qmat(arma::span(0, m), i) = c*Qmat(arma::span(0, m), i) - s*Qmat(arma::span(0, m), m);
                Qmat(arma::span(0, m), m) = s*Qji + c*Qmat(arma::span(0, m), m);
            }
            m++;
        }
        // Add a column to A
        void add_col(const vec &a) {
            assert(n+1 <= Rmat.n_cols);
            assert(n < m);
            Rmat(arma::span(0, m-1), n) = Qmat(arma::span(0, m-1), arma::span(0, m-1)).t() * a;
            n++;
        }
    };
    
    // Cumulative minimum
    template <class VecType>
    inline VecType cummin(const VecType& x) {
        VecType out(x.n_elem);
        auto minval = x(0);
        out(0) = minval;
        for(int i=1; i < x.n_elem; i++) {
            if (x(i) < minval) {
                minval = x(i);
            }
            out(i) = minval;
        }
        return out;
    };
    
    class StopWatch {
    private:
        std::chrono::time_point<std::chrono::system_clock> starttime, endtime;
        bool started;
    public:
        StopWatch() {
            this->started = false;
        }
        void start() {
            assert(!this->started);
            this->starttime = std::chrono::system_clock::now();
            this->started = true;
        }
        double stop() {
            assert(this->started);
            this->endtime = std::chrono::system_clock::now();
            this->started = false;
            std::chrono::duration<double> elapsed_seconds = 
                this->endtime - this->starttime;
            return elapsed_seconds.count();
        }
    };
    
    vec gaussian_smoothing(mat X, vec f, double sigma) {
        mat W = arma::exp(-SquaredPairwiseDistance(X, X)/(2.0*sigma*sigma));
        return (W * f) / arma::sum(W, 2);
    }
}

#endif /* defined(__Surrogate_Optimization__utils__) */