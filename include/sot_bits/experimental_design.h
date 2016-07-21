//
//  ExperimentalDesign.h
//  Surrogate Optimization
//
//  Created by David Eriksson on 7/30/15.
//  Copyright (c) 2015 David Eriksson. All rights reserved.
//

#ifndef __Surrogate_Optimization__ExperimentalDesign__
#define __Surrogate_Optimization__ExperimentalDesign__

#include <cassert>
#include <iostream>
#include "common.h"
#include "utils.h"

namespace sot {
    
    class ExpDesign {
    protected:
        int d;
        int num_points;
    public:
        int dim() const { return this->d; }
        int npts() { return this->num_points; }
        virtual mat generate_points() const = 0;
    };
    
    class FixedDesign : public ExpDesign {
    protected:
        mat points;
    public:
        FixedDesign(mat& points) { 
            this->points = points; 
            d = points.n_rows; 
            num_points = points.n_cols; 
        }
        virtual mat generate_points() const { return points; }
    };
    
    class SymmetricLatinHypercube : public ExpDesign {
    protected:
        mat create_design() const {
            mat points = arma::zeros<mat>(d, num_points);
            points.row(0) = arma::linspace<vec>(1, num_points, num_points).t();

            int middleind = num_points/2;

            if (num_points % 2 == 1) {
                points.row(middleind).fill(middleind + 1);
            }

            // Fill upper
            for(int j=1; j < d; j++) {
                for(int i=0; i < middleind;i++) {
                    if (rand() < 0.5) {
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
        SymmetricLatinHypercube(int num_points, int d) {
            this->num_points = num_points;
            this->d = d;
            assert(num_points >= 2 * d);
        }
        
        mat generate_points() const {
            arma::uword rank_pmat = 0;
            mat pmat = arma::ones<mat>(d + 1, num_points);
            mat xsample;
            do {
                xsample = create_design();
                pmat.rows(1, d) = xsample;
                rank_pmat = arma::rank(pmat);
            } while(rank_pmat != d + 1);
            return xsample;
        }
    };
    
    class LatinHypercube : public ExpDesign {
    public:
        LatinHypercube(int num_points, int d) {
            this->num_points = num_points;
            this->d = d;
            assert(num_points >= dim);
        }
        
        mat generate_points() const {
            mat Xbest;
            mat X;
            double bestscore = 0;

            for(int iter=0; iter < 100; iter++) {
                X = arma::zeros(d, num_points);
                vec xvec = (arma::linspace<vec>(1, num_points, num_points) - 0.5) / num_points;

                for(int j=0; j < d; j++) {
                    X.row(j) = xvec(arma::shuffle(arma::linspace<uvec>(0, num_points - 1, num_points))).t();
                }

                mat dists = sqrt(d)*arma::eye(num_points, num_points) + arma::sqrt(SquaredPairwiseDistance(X, X));
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
        TwoFactorial(int d) {
            this->num_points = pow(2, d);
            this->d = d;
            assert(d <= 15);
        }
        
        mat generate_points() const {
            mat xsample = arma::zeros<mat>(d, num_points);
            for(int i=0; i < d; i++) {
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
        CornersMid(int d) {
            this->num_points = 1 + pow(2, d);
            this->d = d;
            assert(d <= 15);
        }
        
        mat generate_points() const {
            mat xsample = arma::zeros<mat>(d, num_points);

            for(int i=0; i < d; i++) {
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
