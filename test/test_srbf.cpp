/* 
 * File:   test_srbf.cpp
 * Author: David Eriksson
 *
 * Created on July 19, 2016, 12:44 PM
 */

#include <sot.h>
#include <armadillo>
#include <stdlib.h>
#include <iostream>

using namespace sot;

int test_srbf() {

    int dim = 10;
    int maxeval = 500;
    
    std::shared_ptr<Problem> data(new Ackley(dim));
    std::shared_ptr<ExpDesign> slhd(new SLHD(2*(dim+1), dim));
    std::shared_ptr<Surrogate> rbf(new TPSRBF(maxeval, dim, data->lbound(), data->rbound()));
    std::shared_ptr<Sampling> dycors(new SRBF<>(data, rbf, 100*dim, maxeval - slhd->npts()));
    
    Optimizer opt(data, slhd, rbf, dycors, maxeval);
    Result res = opt.run();
    
    std::cout << res.fbest << std::endl;
    // Check that we made enough progress and that we are feasible
    if (res.fbest > -20.0) {
        return (EXIT_FAILURE);
    }
    if (not arma::all(res.xbest >= data->lbound())) {
        return (EXIT_FAILURE);
    }    
    if (not arma::all(res.xbest <= data->rbound())) {
        return (EXIT_FAILURE);
    }
    
    return EXIT_SUCCESS;
}

int main(int argc, char** argv) {
    return test_srbf();
}

