/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   newsimpletest.cpp
 * Author: davideriksson
 *
 * Created on July 19, 2016, 12:44 PM
 */

#include <sot.h>
#include <armadillo>
#include <stdlib.h>
#include <iostream>
#include <cassert>

using namespace sot;

int test_rbf() {
    int dim = 2;
    int n = 500;
    int m = 10;
    mat x = arma::randu<mat>(dim, n);
    vec fx = arma::zeros<vec>(n);
    for(int i=0; i<n; i++) {
        fx(i) = x(1, i) * sin(x(0, i)) + x(0, i) * cos(x(1, i));
    }
    mat y = arma::randu<mat>(dim, m);;
    mat dists = arma::sqrt(SquaredPairwiseDistance(x, y));

    CubicRBF rbf(n, dim, arma::zeros(dim), arma::ones(dim), 0.0);
    rbf.set_points(x, fx);
    rbf.fit();

    // Evaluate at the center to see that we are interpolating
    vec vals = rbf.evals(x);
    for(int i=0; i < x.n_cols; i++) {
        if (fabs(vals(i) - fx(i)) >= 1e-10) {
            return (EXIT_FAILURE);
        }
    }
    
    // Evaluate at some other points
    vals = rbf.evals(y);
    for(int i=0; i < y.n_cols; i++) {
        double fval = y(1, i) * sin(y(0, i)) + y(0, i) * cos(y(1, i));
        if (fabs(vals(i) - fval) >= 1e-3) {
            return (EXIT_FAILURE);
        }
    }
    
    // Look at derivatives
    vec pred = arma::zeros<vec>(2);
    for(int i=0; i < y.n_cols; i++) {
        vec deriv = rbf.deriv(y.col(i));
        pred(0) = y(1, i) * cos(y(0, i)) + cos(y(1,i));
        pred(1) = sin(y(0, i)) - y(0, i) * sin(y(1, i));
        if (arma::norm(deriv - pred) >= 1e-2) {
            return (EXIT_FAILURE);
        }
    }
    
    return (EXIT_SUCCESS);
}

int main(int argc, char** argv) {
    return test_rbf();
}