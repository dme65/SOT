/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   approximant.h
 * Author: davideriksson
 *
 * Created on May 2, 2016, 12:13 PM
 */

#ifndef APPROXIMANT_H
#define APPROXIMANT_H

#include <armadillo>
#include "common.h"

namespace sot {
    
    // Template
    class Approximant {
    protected:
        int max_points;
        int num_points;
        mat Xmat;
        vec fXvec;
        int d;
    public:
        int npts() { return num_points; }
        int dim() const { return d; }
        virtual void reset() = 0;
        virtual vec X(int i) const = 0;
        virtual mat X() const = 0;
        virtual double fX(int) const = 0;
        virtual vec fX() const = 0;
        virtual void add_point(const vec&, double) = 0;
        virtual void add_points(const mat&, const vec&) = 0;
        virtual double eval(const vec&) const = 0;
        virtual vec evals(const mat&) const = 0;
        virtual vec deriv(const vec&) const = 0;
        virtual void fit() = 0;
    };
    
    class CubicKernel {
    public:
        int phi_zero = 0;
        inline double eval(double dist) const { return dist * dist * dist; }
        inline double deriv(double dist) const { return 3 * dist * dist; }
        inline mat eval(const mat &dists) const { return dists % dists % dists; }
        inline mat deriv(const mat &dists) const { return 3 * dists % dists; }
    };
    
    class ThinPlateKernel {
    public:
        int phi_zero = 0;
        inline double eval(double dist) const { return dist * dist * log(dist + 1e-10);}
        inline double deriv(double dist) const { return dist * (1.0 + 2.0 * log(dist + 1e-10)); }
        inline mat eval(const mat &dists) const { return dists % dists % arma::log(dists + 1e-10); }
        inline mat deriv(const mat &dists) const { return dists % (1 + 2.0 * arma::log(dists + 1e-10)); }
    };
    
    class LinearKernel {
    public:
        int phi_zero = 0;
        inline double eval(double dist) const { return dist; }
        inline double deriv(double dist) const { return 1.0; }
        inline mat eval(const mat &dists) const { return dists; }
        inline mat deriv(const mat &dists) const { return arma::ones<mat>(dists.n_rows, dists.n_cols); }
    };
    
    class LinearTail {
    public:
        inline mat eval(const mat &x) const { return arma::join_vert(arma::ones<mat>(1, x.n_cols), x); }
        inline vec eval(const vec &x) const {
            vec tail = arma::zeros<vec>(x.n_rows + 1);
            tail(0) = 1;
            tail.tail(x.n_rows) = x;
            return tail;
        }
        inline mat deriv(const mat &x) const { return arma::join_vert(arma::zeros<mat>(1, x.n_rows), arma::eye<mat>(x.n_rows, x.n_rows)); }
        inline int n_tail(int d) const { return 1 + d; }
    };
    
    class ConstantTail {
    public:
        inline mat eval(const mat &x) const { return arma::ones<mat>(x.n_rows, 1); }
        inline vec eval(const vec &x) const { return arma::ones<mat>(1, 1); }
        inline mat deriv(const mat &x) const { return arma::zeros<mat>(x.n_rows, 1); }
        inline int n_tail(int d) const { return 1; }
    };
}


#endif /* APPROXIMANT_H */

