//
//  kNN.h
//  Surrogate Optimization
//
//  Created by David Eriksson on 2/2/16.
//  Copyright © 2016 David Eriksson. All rights reserved.
//

#ifndef Surrogate_Optimization_kNN_h
#define Surrogate_Optimization_kNN_h

#include "common.h"
#include "utils.h"
#include "surrogate.h"

namespace sot {
    
    class kNN : public Surrogate {
    protected:
        int max_points;
        int num_points;
        int k;
        mat Xmat;
        vec fX;
    public:
        int dim;
        kNN(int max_points, int dim, int k) {
            this->num_points = 0;
            this->max_points = max_points;
            this->k = k;
            this->Xmat.resize(dim, max_points);
            this->fX.resize(max_points);
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
        void add_point(const vec &point, double fun_val) {
            Xmat.col(num_points) = point;
            fX(num_points) = fun_val;
            num_points++;
        }
        void add_points(const mat &points, const vec &fun_vals) {
            int npts = points.n_cols;
            Xmat.cols(num_points, num_points+npts-1) = points;
            fX.rows(num_points, num_points+npts-1) = fun_vals;
            num_points += npts;
        }
        double eval(const vec &point) const {
            vec dists = squaredPointSetDistance(point, X());
            uvec indices = sort_index(dists);
            return arma::mean(fX(indices.rows(0, k-1)));
        }
        vec evals(const mat &points, const mat &dists) const {
            vec vals = arma::zeros<vec>(points.n_cols);
            for(int i=0; i < points.n_cols; i++) {
                vals(i) = eval(points.col(i));
            }
            return vals;
        }
    };
}


#endif
