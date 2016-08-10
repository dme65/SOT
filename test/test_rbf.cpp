/* 
 * File:   test_rbf.cpp
 * Author: David Eriksson
 *
 * Created on July 19, 2016
 */

#include <sot.h>
using namespace sot;

int test_rbf() {
    int dim = 2;
    int n = 500;
    int m = 10;
    mat x = arma::randu<mat>(dim, n);
    vec fx = (x.row(1) % arma::sin(x.row(0)) + x.row(0) % arma::cos(x.row(1))).t();
    mat y = arma::randu<mat>(dim, m);;
    vec fy = (y.row(1) % arma::sin(y.row(0)) + y.row(0) % arma::cos(y.row(1))).t();

    CubicRBF rbf(n, dim, arma::zeros(dim), arma::ones(dim), 0.0);
    rbf.addPoints(x, fx);
    rbf.fit();

    // Evaluate at the center to see that we are interpolating
    vec vals = rbf.evals(x);
    if (arma::max(arma::abs(vals - fx)) >= 1e-10) { // LCOV_EXCL_LINE
        return (EXIT_FAILURE); // LCOV_EXCL_LINE
    }
    
    // Evaluate at some other points
    vals = rbf.evals(y);
    if (arma::max(arma::abs(vals - fy)) >= 1e-3) { // LCOV_EXCL_LINE
        return (EXIT_FAILURE); // LCOV_EXCL_LINE
    }
    
    // Look at derivatives
    vec pred = arma::zeros<vec>(2);
    for(int i=0; i < y.n_cols; i++) {
        vec deriv = rbf.deriv(y.col(i));
        pred(0) = y(1, i) * cos(y(0, i)) + cos(y(1,i));
        pred(1) = sin(y(0, i)) - y(0, i) * sin(y(1, i));
        if (arma::norm(deriv - pred) >= 1e-2) { // LCOV_EXCL_LINE
            return (EXIT_FAILURE); // LCOV_EXCL_LINE
        }
    }
    
    return (EXIT_SUCCESS);
}

int main(int argc, char** argv) {
    return test_rbf();
}