/*
 * File:   test_adaptive.cpp
 * Author: David Eriksson
 *
 * Created on Aug 10, 2016
 */

#include <sot.h>
using namespace sot;

int test_adaptive() {

    int dim = 10;
    int maxEvals = 500;

    std::shared_ptr<Problem> data(std::make_shared<UnitBoxProblem<Ackley>>(dim));
    std::shared_ptr<ExpDesign> slhd(std::make_shared<SLHD>(2*(dim+1), dim));
    std::shared_ptr<Surrogate> rbf(std::make_shared<CubicRBF>(maxEvals, dim));

    std::vector<std::shared_ptr<Sampling>> adaptive_sampling;
    adaptive_sampling.push_back(std::make_shared<DYCORS<>>(data, rbf, 100*dim, maxEvals - slhd->numPoints()));
    adaptive_sampling.push_back(std::make_shared<SRBF<>>(data, rbf, 100*dim, maxEvals - slhd->numPoints()));
    adaptive_sampling.push_back(std::make_shared<Uniform<>>(data, rbf, 100*dim, maxEvals - slhd->numPoints()));
    adaptive_sampling.push_back(std::make_shared<GASampling>(data, rbf, 50, 100));

    for(int i=0; i < adaptive_sampling.size(); i++) {
        mat X = fromUnitBox(slhd->generatePoints(), data->lBounds(), data->uBounds());
        vec fX = data->evals(X);

        rbf->addPoints(X, fX);
        rbf->fit();

        arma::uword indBest;
        double fBest = fX.min(indBest);
        vec xBest = X.col(indBest);

        vec sigma = 0.2 * (data->uBounds() - data->lBounds());
        vec newX = adaptive_sampling[i]->makePoints(xBest, X, sigma, 1);

        // Check that the new point is in the domain
        if (not arma::all(newX <= data->uBounds())) { // LCOV_EXCL_LINE
            return (EXIT_FAILURE); // LCOV_EXCL_LINE
        }
        if (not arma::all(newX >= data->lBounds())) { // LCOV_EXCL_LINE
            return (EXIT_FAILURE); // LCOV_EXCL_LINE
        }

        // Generate 2 points
        int n = 2;
        mat newXmat = adaptive_sampling[i]->makePoints(xBest, X, sigma, n);

        for(int j=0; j < n; j++) {
            if (not arma::all(newXmat.col(j) <= data->uBounds())) { // LCOV_EXCL_LINE
                return (EXIT_FAILURE); // LCOV_EXCL_LINE
            }
            if (not arma::all(newXmat.col(j) >= data->lBounds())) { // LCOV_EXCL_LINE
                return (EXIT_FAILURE); // LCOV_EXCL_LINE
            }
        }

        // Reset the object
        adaptive_sampling[i]->reset(10);
        rbf->reset();
    }

    return EXIT_SUCCESS;
}

int main(int argc, char** argv) {
    return test_adaptive();
}