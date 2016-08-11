/* 
 * File:   test_kNN.cpp
 * Author: David Eriksson
 *
 * Created on July 19, 2016
 */

#include <sot.h>
using namespace sot;

int test_kNN() {
    int dim = 2;
    int n = 500;
    int m = 10;
    mat X = arma::randu<mat>(dim, n);
    vec fX = (X.row(1) % arma::sin(X.row(0)) + X.row(0) % arma::cos(X.row(1))).t();
    mat Y = arma::randu<mat>(dim, m);;
    vec fY = (Y.row(1) % arma::sin(Y.row(0)) + Y.row(0) % arma::cos(Y.row(1))).t();

    kNN surf(n, dim, 3);
    surf.addPoints(X, fX);
    surf.fit();

    // Evaluate at the center to see that we are interpolating
    vec vals = surf.evals(X);
    if (arma::max(arma::abs(vals - fX)) >= 1e-1) { // LCOV_EXCL_LINE
        return (EXIT_FAILURE); // LCOV_EXCL_LINE
    }
    
    // Evaluate at some other points
    vals = surf.evals(Y);
    if (arma::max(arma::abs(vals - fY)) >= 1e-1) { // LCOV_EXCL_LINE
        return (EXIT_FAILURE); // LCOV_EXCL_LINE
    }
    mat dists = arma::sqrt(squaredPairwiseDistance(Y, X));
    if(arma::norm(vals - surf.evals(Y, dists)) > 1e-10) { // LCOV_EXCL_LINE
        return (EXIT_FAILURE); // LCOV_EXCL_LINE
    }

    // Check that all of the getters are correct
    if(dim != surf.dim() || n != surf.numPoints()) {
        return (EXIT_FAILURE); // LCOV_EXCL_LINE
    }
    if (arma::norm(fX - surf.fX()) > 1e-10) { // LCOV_EXCL_LINE
        return (EXIT_FAILURE); // LCOV_EXCL_LINE
    }
    if (arma::norm(X - surf.X()) > 1e-10) { // LCOV_EXCL_LINE
        return (EXIT_FAILURE); // LCOV_EXCL_LINE
    }
    for(int i=0; i < n; i++) {
        if (std::abs(fX(i) - surf.fX(i)) > 1e-10) { // LCOV_EXCL_LINE
            return (EXIT_FAILURE); // LCOV_EXCL_LINE
        }
        if (arma::norm(X.col(i) - surf.X(i)) > 1e-10) { // LCOV_EXCL_LINE
            return (EXIT_FAILURE); // LCOV_EXCL_LINE
        }
    }

    // Reset the surrogate
    surf.reset();

    // Add all points again
    for(int i=0; i < n; i++) {
        surf.addPoint(X.col(i), fX(i));
    }
    surf.fit();

    // Check all of the evaluation methods
    vals = arma::zeros(n);
    for(int i=0; i < n; i++) {
        vals(i) = surf.eval(X.col(i));
    }
    if (arma::max(arma::abs(vals - fX)) >= 1e-1) { // LCOV_EXCL_LINE
        return (EXIT_FAILURE); // LCOV_EXCL_LINE
    }

    // Evaluate at some other points
    vals = arma::zeros(m);
    for(int i=0; i < m; i++) {
        vec y = Y.col(i);
        vals(i) = surf.eval(y);
        vec dists = arma::sqrt(squaredPointSetDistance(y, X));
        if (std::abs(vals(i) - surf.eval(y, dists)) > 1e-10) { // LCOV_EXCL_LINE
            return (EXIT_FAILURE); // LCOV_EXCL_LINE
        }
    }
    if (arma::max(arma::abs(vals - fY)) >= 1e-1) { // LCOV_EXCL_LINE
        return (EXIT_FAILURE); // LCOV_EXCL_LINE
    }

    // Finally check the derivative, which isn't supported
    bool exceptionThrown = false;
    try {
        vec y = surf.deriv(Y.col(0));
    }
    catch (const std::logic_error& e) {
        exceptionThrown = true;
    }

    if(not exceptionThrown) { return EXIT_FAILURE; } // LCOV_EXCL_LINE

    return (EXIT_SUCCESS);
}

int main(int argc, char** argv) {
    return test_kNN();
}