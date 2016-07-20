//
//  CandidatePoints.h
//  Surrogate Optimization
//
//  Created by David Eriksson on 7/30/15.
//  Copyright (c) 2015 David Eriksson. All rights reserved.
//

#ifndef __Surrogate_Optimization__CandidatePoints__
#define __Surrogate_Optimization__CandidatePoints__

#include <stdio.h>
#include "candidate_points.h"
#include "rbf.h"
#include "utils.h"
#include <math.h>
#include "common.h"
#include "test_problems.h"
#include <random>

namespace sot {
    
    class MeritWeightedDistance {
    private:
        vec weights = {0.3, 0.5, 0.8, 0.95};
        int next_weight = 0;
    public:
        inline mat pick_points(const mat &cand, const std::shared_ptr<Surrogate>& surf, const mat &points, int newpts, double dtol) {
            int dim = cand.n_rows; 

            // Evaluate the RBF at the candidate points
            const mat dists = arma::sqrt(SquaredPairwiseDistance<mat>(points, cand));
            vec surf_val = surf->evals(cand);
            vec val_score = UnitRescale(surf_val);
            vec min_dist = arma::min(dists).t();
            vec dist_score = 1.0 - UnitRescale(min_dist);

            mat newx = arma::zeros<mat>(dim, newpts);

            arma::uword winner;
            for(int i=0; i < newpts; i++) {
                double weight = weights[next_weight % weights.n_elem];
                next_weight++;

                // Update distances if necessary
                if (i > 0) {
                    vec new_dist = arma::sqrt(SquaredPointSetDistance<mat,vec>((vec)newx.col(i-1), cand));
                    min_dist = arma::min(min_dist, new_dist);
                    val_score(winner) = std::numeric_limits<double>::max();
                    dist_score = 1.0 - UnitRescale(min_dist);
                }

                // Pick a winner
                vec merit = weight * val_score + (1.0 -  weight) * dist_score;
                merit.elem(arma::find(min_dist < dtol)).fill(std::numeric_limits<double>::max());
                double score = merit.min(winner);
                newx.col(i) = cand.col(winner);
            }

            return newx;
        }
    };
    
    // Template for selecting the next evaluation
    class Sampling {
    protected:
        double dtol;
        int numeval;
        int budget;
    public:
        virtual void reset(int) = 0;
        virtual mat make_points(vec &xbest, const mat &points, double sigma, int newpts) = 0;
    };
    
    // DYCORS
    template<class MeritFunction = MeritWeightedDistance>
    class DYCORS : public Sampling {
    protected:
        std::shared_ptr<Problem> data;
        std::shared_ptr<Surrogate> surf;
        int ncand;
        MeritFunction merit;
    public:
        DYCORS(const std::shared_ptr<Problem>& data, const std::shared_ptr<Surrogate>& surf, int ncand, int budget) {
            this->data = std::shared_ptr<Problem>(data);
            this->surf = std::shared_ptr<Surrogate>(surf);
            this->budget = budget;
            this->ncand = ncand;
            this->dtol = 1e-3*sqrt(arma::sum(arma::square(data->rbound() - data->lbound())));
        }
        void reset(int budget) {
            this->budget = budget;
            int numeval = 0;
        }
        mat make_points(vec &xbest, const mat &points, double sigma, int newpts) {
            std::random_device rd;
            std::default_random_engine e1(rd());
            std::uniform_real_distribution<double> rand(0, 1);
            std::uniform_int_distribution<int> randi(0, data->dim() - 1);
            std::normal_distribution<double> randn(0.0, 1.0);
            
            double dds_prob = fmin(20.0/data->dim(), 1.0) * (1.0 - (log(this->numeval + 1.0) / log(this->budget)));
            mat cand = arma::repmat(xbest, 1, ncand);
            for(int i=0; i < ncand; i++) {

                int count = 0;
                for(int j=0; j < data->dim(); j++) {
                    if(rand(e1) < dds_prob) {
                        count++;
                        cand(j, i) += sigma*randn(e1);
                    }
                }
                // If no index was pertrubed we force one
                if(count == 0) {
                    int ind = randi(e1);
                    cand(ind, i) += sigma*randn(e1);
                }

                // Make sure we are still in the domain
                for(int j=0; j < data->dim(); j++) {
                    if(cand(j, i) > data->rbound()(j)) { cand(j, i) = fmax(2*data->rbound()(j) - cand(j, i), data->lbound()(j)); }
                    else if(cand(j, i) < data->lbound()(j)) { cand(j, i) = fmin(2*data->lbound()(j) - cand(j, i), data->rbound()(j)); }
                }
            }
            
            // Update counter
            numeval = numeval + newpts;
            
            return merit.pick_points(cand, surf, points, newpts, dtol);
        }
    };

    /*
    inline mat AllDim(const std::shared_ptr<Surrogate>& surf, const mat &points, const vec &xlow, const vec &xup, const vec &xbest, double sigma, int maxeval, int numeval, int initp, int dim, double dtol, vec weights, int ncand, int newpts) {
        
        assert(weights.n_elem == newpts);

        std::random_device rd;
        std::default_random_engine e1(rd());
        std::normal_distribution<double> randn(0.0, sigma);
        
        mat cand = arma::repmat(xbest, 1, ncand);
        
        // Perturbs one randomly chosen coordinate
        for(int i=0; i < ncand; i++) {
            
            for(int j=0; j < dim; j++) {
                cand(j, i) += randn(e1);
                if(cand(j, i) > xup(j)) {
                    cand(j, i) = fmax(2*xup(j) - cand(j, i), xlow(j));
                }
                else if(cand(j, i) < xlow(j)) {
                    cand(j, i) = fmin(2*xlow(j) - cand(j, i), xup(j));
                }
            }
        }

        return pick_points(cand, surf, points, dim, dtol, weights, newpts);
    };
    
    inline mat Uniform(const std::shared_ptr<Surrogate>& surf, const mat &points, const vec &xlow, const vec &xup, const vec &xbest, double sigma, int maxeval, int numeval, int initp, int dim, double dtol, vec weights, int ncand, int newpts) {
        
        assert(weights.n_elem == newpts);
        
        mat cand = arma::randu<mat>(dim, ncand);
        for(int j=0; j < dim; j++) {
            cand.row(j) = xlow(j) + (xup(j) - xlow(j)) * cand.row(j);
        }
        
        return pick_points(cand, surf, points, dim, dtol, weights, newpts);
    };
    
    inline mat Gradient(const std::shared_ptr<Surrogate>& surf, const mat &points, const vec &xlow, const vec &xup, const vec &xbest, double sigma, int maxeval, int numeval, int initp, int dim, double dtol, vec weights, int ncand, int newpts) {
        
        assert(weights.n_elem == newpts);
        
        vec gradient = arma::normalise(surf->deriv(xbest));
        mat cand = arma::repmat(xbest, 1, ncand) + sigma * sqrt(dim) * gradient * arma::randn<mat>(1, ncand);
        for(int i=0; i < ncand; i++) {
            for(int j=0; j < dim; j++) {
                if(cand(j, i) > xup(j)) {
                    cand(j, i) = fmax(2*xup(j) - cand(j, i), xlow(j));
                }
                else if(cand(j, i) < xlow(j)) {
                    cand(j, i) = fmin(2*xlow(j) - cand(j, i), xup(j));
                }
            }
        }
        
        return pick_points(cand, surf, points, dim, dtol, weights, newpts);
    };
    **/
}

#endif /* defined(__Surrogate_Optimization__CandidatePoints__) */
