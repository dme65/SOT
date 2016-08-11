/*
 * File:   test_expdes.cpp
 * Author: David Eriksson
 *
 * Created on Aug 11, 2016
 */

#include <sot.h>
using namespace sot;

int test_expdes() {

    int dim = 10;
    std::vector<std::shared_ptr<ExpDesign>> experimentalDesign;

    // Number of points in each design
    std::vector<int> numPoints = {50, 50, 51, 50,
                                  static_cast<int>(std::pow(2, dim)),
                                  static_cast<int>(1+std::pow(2, dim))};

    mat F = arma::randu(dim, numPoints[0]);
    experimentalDesign.push_back(std::make_shared<FixedDesign>(F));
    experimentalDesign.push_back(std::make_shared<SLHD>(numPoints[1], dim));
    experimentalDesign.push_back(std::make_shared<SLHD>(numPoints[2], dim));
    experimentalDesign.push_back(std::make_shared<LHD>(numPoints[3], dim));
    experimentalDesign.push_back(std::make_shared<TwoFactorial>(dim));
    experimentalDesign.push_back(std::make_shared<CornersMid>(dim));

    for(int i=0; i < experimentalDesign.size(); i++) {

        // Check that the dimensions are correct
        mat X = experimentalDesign[i]->generatePoints();
        if (X.n_cols != numPoints[i] || X.n_rows != dim) { // LCOV_EXCL_LINE
            return (EXIT_FAILURE); // LCOV_EXCL_LINE
        }
        if (experimentalDesign[i]->numPoints() != numPoints[i]
            || experimentalDesign[i]->dim() != dim) { // LCOV_EXCL_LINE
            return (EXIT_FAILURE); // LCOV_EXCL_LINE
        }

        // Check that the points are in the domain
        if(arma::max(arma::vectorise(X)) > 1) { // LCOV_EXCL_LINE
            return (EXIT_FAILURE); // LCOV_EXCL_LINE
        }
        if(arma::min(arma::vectorise(X)) < 0) { // LCOV_EXCL_LINE
            return (EXIT_FAILURE); // LCOV_EXCL_LINE
        }
    }

    return EXIT_SUCCESS;
}

int main(int argc, char** argv) {
    return test_expdes();
}