/* 
 * File:   test_dycors.cpp
 * Author: David Eriksson
 *
 * Created on July 19, 2016, 12:44 PM
 */

#include <sot.h>

using namespace sot;

int test_dycors() {

    int dim = 10;
    int maxeval = 500;
    
    std::shared_ptr<Problem> data(std::make_shared<Ackley>(dim));
    std::shared_ptr<ExpDesign> slhd(std::make_shared<SLHD>(2*(dim+1), dim));
    std::shared_ptr<Surrogate> rbf(std::make_shared<CubicRBF>(maxeval, dim, data->lBounds(), data->uBounds()));
    std::shared_ptr<Sampling> dycors(std::make_shared<DYCORS<>>(data, rbf, 100*dim, maxeval - slhd->numPoints()));
    
    Optimizer opt(data, slhd, rbf, dycors, maxeval);
    Result res = opt.run();    
    
    std::cout << res.fBest() << std::endl;
    // Check that we made enough progress and that we are feasible
    if (res.fBest() > -20.0) {
        return (EXIT_FAILURE);
    }
    if (not arma::all(res.xBest() >= data->lBounds())) {
        return (EXIT_FAILURE);
    }    
    if (not arma::all(res.xBest() <= data->uBounds())) {
        return (EXIT_FAILURE);
    }
    
    return EXIT_SUCCESS;
}

int main(int argc, char** argv) {
    return test_dycors();
}

