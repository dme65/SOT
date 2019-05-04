//
// Created by David Eriksson on 8/11/16.
//

/*
 * File:   test_optprobs.cpp
 * Author: David Eriksson
 *
 * Created on Aug 11, 2016
 */

#include <sot.h>
using namespace sot;

int test_optprobs() {

    int dim = 10;
    std::vector<std::shared_ptr<Problem>> problems;
    problems.push_back(std::make_shared<RotatedProblem<Ackley>>(dim));
    problems.push_back(std::make_shared<UnitBoxProblem<Ackley>>(dim));
    problems.push_back(std::make_shared<Sphere>(dim));
    problems.push_back(std::make_shared<SumSquares>(dim));
    problems.push_back(std::make_shared<Schwefel22>(dim));
    problems.push_back(std::make_shared<Exponential>(dim));
    problems.push_back(std::make_shared<Tablet>(dim));
    problems.push_back(std::make_shared<Step>(dim));
    problems.push_back(std::make_shared<Zakharov>(dim));
    problems.push_back(std::make_shared<Rosenbrock>(dim));
    problems.push_back(std::make_shared<Griewank>(dim));
    problems.push_back(std::make_shared<Schaffer2>(dim));
    problems.push_back(std::make_shared<Schwefel26>(dim));
    problems.push_back(std::make_shared<Himmelblau>(dim));
    problems.push_back(std::make_shared<Ackley>(dim));
    problems.push_back(std::make_shared<Rastrigin>(dim));
    problems.push_back(std::make_shared<Michalewicz>(dim));
    problems.push_back(std::make_shared<Keane>(dim));
    problems.push_back(std::make_shared<Levy>(dim));
    problems.push_back(std::make_shared<Salomon>(dim));
    problems.push_back(std::make_shared<Schubert3>(dim));
    problems.push_back(std::make_shared<SineEnvelope>(dim));
    problems.push_back(std::make_shared<CosineMixture>(dim));
    problems.push_back(std::make_shared<Schoen>(dim, 100));

    for(int i=0; i < problems.size(); i++) {

        std::cout << "Testing " << problems[i]->name() << std::endl;

        vec lBounds = problems[i]->lBounds();
        vec uBounds = problems[i]->uBounds();

        // Check dimensionality
        if(problems[i]->dim() != dim) { // LCOV_EXCL_LINE
            return (EXIT_FAILURE); // LCOV_EXCL_LINE
        }

        if(!arma::all(lBounds < uBounds)) { // LCOV_EXCL_LINE
            return (EXIT_FAILURE); // LCOV_EXCL_LINE
        }

        // Check the optimum of evaluate a random points
        vec opt = problems[i]->optimum();
        if(opt.n_elem == dim) { // Check if the optimum is known
            if(std::abs(problems[i]->min() - problems[i]->eval(opt)) > 1e-3) { // LCOV_EXCL_LINE
                return (EXIT_FAILURE); // LCOV_EXCL_LINE
            }
        }
        else {
            vec x = lBounds + (uBounds - lBounds) % arma::randu(dim);
            double val = problems[i]->eval(x);
            double val2 = problems[i]->min();
        }

        // Evaluate multiple points
        int numPoints = 10;
        mat X = arma::randu(dim, numPoints);
        for(int i=0; i < numPoints; i++) {
            X.col(i) = lBounds + (uBounds - lBounds) % X.col(i);
        }
        vec vals = problems[i]->evals(X);

        if(vals.n_elem != numPoints) { // LCOV_EXCL_LINE
            return (EXIT_FAILURE); // LCOV_EXCL_LINE
        }

        // Check the rotation and translation
        if(i == 0) {
            // Need to make a dynamic cast to get the derived methods
            mat rotation = std::dynamic_pointer_cast<RotatedProblem<Ackley>>(problems[i])->rotation();
            vec translation = std::dynamic_pointer_cast<RotatedProblem<Ackley>>(problems[i])->translation();
            // Check that the rotation is orthogonal
            if (arma::norm(arma::eye(dim, dim) - rotation.t() * rotation) > 1e-10) { // LCOV_EXCL_LINE
                return (EXIT_FAILURE); // LCOV_EXCL_LINE
            }
            // Check that the translation is in the domain
            if (!arma::all(translation < uBounds)) { // LCOV_EXCL_LINE
                return (EXIT_FAILURE); // LCOV_EXCL_LINE
            }
            if (!arma::all(translation > lBounds)) { // LCOV_EXCL_LINE
                return (EXIT_FAILURE); // LCOV_EXCL_LINE
            }
        }
    }

    return EXIT_SUCCESS;
}

int main(int argc, char** argv) {
    return test_optprobs();
}