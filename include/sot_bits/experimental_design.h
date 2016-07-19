//
//  ExperimentalDesign.h
//  Surrogate Optimization
//
//  Created by David Eriksson on 7/30/15.
//  Copyright (c) 2015 David Eriksson. All rights reserved.
//

#ifndef __Surrogate_Optimization__ExperimentalDesign__
#define __Surrogate_Optimization__ExperimentalDesign__

#include <stdio.h>
#include <cassert>
#include "utils.h"
#include "common.h"
#include <iostream>

namespace sot {
    
    class ExpDesign {
    public:
        int dim;
        int num_points;
        virtual mat generate_points() const = 0;
    };
    
    class FixedDesign : public ExpDesign {
    protected:
        mat points;
    public:
        FixedDesign(mat& points) { this->points = points; dim = points.n_rows; num_points = points.n_cols; }
        virtual mat generate_points() const { return points; }
    };
    
    class SymmetricLatinHypercube : public ExpDesign {
    protected:
        mat create_design() const {
            std::random_device rd;
            std::default_random_engine e1(rd());
            std::uniform_real_distribution<double> uniform_dist(0, 1);

            mat points = arma::zeros<mat>(dim, num_points);
            points.row(0) = arma::linspace<vec>(1, num_points, num_points).t();

            int middleind = num_points/2;

            if (num_points % 2 == 1) {
                points.row(middleind).fill(middleind + 1);
            }

            // Fill upper
            for(int j=1; j < dim; j++) {
                for(int i=0; i < middleind;i++) {
                    if (uniform_dist(e1) < 0.5) {
                        points(j, i) = num_points -i;
                    }
                    else {
                        points(j, i) = i + 1;
                    }
                }
                // Shuffle
                points(j, arma::span(0, middleind-1)) = arma::shuffle(points(j, arma::span(0, middleind-1)));
            }

            // Fill bottom
            for(int i=middleind; i < num_points; i++) {
                points.col(i) = num_points + 1 - points.col(num_points - 1 - i);
            }

            return points/double(num_points);
        }
    public:
        SymmetricLatinHypercube(int num_points, int dim) {
            this->num_points = num_points;
            this->dim = dim;
            assert(num_points >= 2 * dim);
        }
        
        mat generate_points() const {
            arma::uword rank_pmat = 0;
            mat pmat = arma::ones<mat>(dim + 1, num_points);
            mat xsample;
            do {
                xsample = create_design();
                pmat.rows(1, dim) = xsample;
                rank_pmat = arma::rank(pmat);
            } while(rank_pmat != dim + 1);
            return xsample;
        }
    };
    
    class LatinHypercube : public ExpDesign {
    public:
        LatinHypercube(int num_points, int dim) {
            this->num_points = num_points;
            this->dim = dim;
            assert(num_points >= dim);
        }
        
        mat generate_points() const {
            mat Xbest;
            mat X;
            double bestscore = 0;

            for(int iter=0; iter < 100; iter++) {
                X = arma::zeros(dim, num_points);
                vec xvec = (arma::linspace<vec>(1, num_points, num_points) - 0.5) / num_points;

                for(int d=0; d < dim; d++) {
                    X.row(d) = xvec(arma::shuffle(arma::linspace<uvec>(0, num_points - 1, num_points))).t();
                }

                mat dists = sqrt(dim)*arma::eye(num_points, num_points) + arma::sqrt(SquaredPairwiseDistance(X, X));
                double score = arma::min((vec)arma::min(dists).t());

                if (score > bestscore) {
                    Xbest = X;
                    bestscore = score;
                }
            }

            return Xbest;
        }   
    };
    
    class TwoFactorial : public ExpDesign {
    public:
        TwoFactorial(int dim) {
            this->num_points = pow(2, dim);
            this->dim = dim;
            assert(dim <= 15);
        }
        
        mat generate_points() const {
            mat xsample = arma::zeros<mat>(dim, num_points);
            for(int i=0; i < dim; i++) {
                int elem = 0;
                int flip = pow(2,i);
                for(int j=0; j < num_points; j++) {
                    xsample(i, j) = elem;
                    if((j+1) % flip == 0) { elem = (elem + 1) % 2; }
                }
            }
            return xsample;
        }
    };
    
    class CornersMid : public ExpDesign {
    public:
        CornersMid(int dim) {
            this->num_points = 1 + pow(2, dim);
            this->dim = dim;
            assert(dim <= 15);
        }
        
        mat generate_points() const {
            mat xsample = arma::zeros<mat>(dim, num_points);

            for(int i=0; i < dim; i++) {
                int elem = 0;
                int flip = pow(2, i);
                for(int j = 0; j < num_points; j++) {
                    xsample(i, j) = elem;
                    if((j + 1) % flip == 0) { elem = (elem + 1) % 2; }
                }
            }
            xsample.col(num_points - 1).fill(0.5);

            return xsample;
        }
    };
}
#endif /* defined(__Surrogate_Optimization__ExperimentalDesign__) */
