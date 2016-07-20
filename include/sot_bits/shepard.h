//
//  shepard.h
//  Surrogate Optimization
//
//  Created by David Eriksson on 2/4/16.
//  Copyright © 2016 David Eriksson. All rights reserved.
//

#ifndef shepard_h
#define shepard_h

#include "surrogate.h"
#include <armadillo>
#include "utils.h"
#include "common.h"

namespace sot {
    
    class Shepard : Approximant {
    protected:
        double p;
    public:
        Shepard(int max_points, int d, double p) {
            this->num_points = 0;
            this->max_points = max_points;
            this->p = p;
            this->Xmat.resize(d, max_points);
            this->fXvec.resize(max_points);
        }
        int npts() {
            return num_points;
        }
        vec X(int i) const {
            return Xmat.col(i);
        }
        mat X() const {
            return Xmat.cols(0, num_points-1);
        }
        double fX(int i) const {
            return fXvec(i);
        }
        vec fX() const {
            return fXvec.rows(0, num_points-1);
        }
        vec compute_weights(const vec &x) const {
            return 1/arma::pow(SquaredPointSetDistance<mat,vec>(x, X()), p/2);
        }
        void add_point(const vec &point, double fun_val) {
            Xmat.col(num_points) = point;
            fXvec(num_points) = fun_val;
            num_points++;
        }
        void add_points(const mat &points, const vec &fun_vals) {
            int npts = points.n_cols;
            Xmat.cols(num_points, num_points + npts - 1) = points;
            fXvec.rows(num_points, num_points + npts - 1) = fun_vals;
            num_points += npts;
        }
        double eval(const vec &point) const {
            vec weights = compute_weights(point);
            return arma::dot(compute_weights(point), this->fX())/arma::sum(weights);
        }
        vec evals(const mat &points, const mat &dists) const {
            vec vals = arma::zeros<vec>(points.n_cols);
            for(int i=0; i < points.n_cols; i++) {
                vals(i) = eval(points.col(i));
            }
            return vals;
        }
        vec evals(const mat &points) const {
            vec vals = arma::zeros<vec>(points.n_cols);
            for(int i=0; i < points.n_cols; i++) {
                vals(i) = eval(points.col(i));
            }
            return vals;
        }
        vec deriv(const vec &point) const {
            abort();
        }
        
        void reset() {
            this->num_points = 0;
        }
        
        void fit() {
            return;
        }
    };
}

#endif /* shepard_h */
