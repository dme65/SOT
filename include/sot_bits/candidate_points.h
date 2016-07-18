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

namespace sot {
    
    template<class RBF>
    inline mat pick_points(const mat &cand, const RBF &rbf, const mat &centers, int dim, double dtol, vec weights, int newpts) {
        
        // Evaluate the RBF at the candidate points
        const mat dists = arma::sqrt(SquaredPairwiseDistance<mat>(centers, cand));
        //vec rbf_val = rbf.evals(cand, dists);
        vec rbf_val = rbf.evals(cand);
        vec val_score = UnitRescale(rbf_val);
        vec min_dist = arma::min(dists).t();
        vec dist_score = 1.0 - UnitRescale(min_dist);
        
        mat newx = arma::zeros<mat>(dim, newpts);
        
        arma::uword winner;
        for(int i=0; i < newpts; i++) {
            double weight = weights[i];
            
            // Update distances if necessary
            if (i > 0) {
                vec new_dist = arma::sqrt(SquaredPointSetDistance<mat,vec>((vec)newx.col(i-1), cand));
                for(int j=0; j < cand.n_cols; j++) {
                    if(new_dist(j) < min_dist(j)) {
                        min_dist(j) = new_dist(j);
                    }
                }
                val_score(winner) = std::numeric_limits<double>::max();
                dist_score = 1.0 - UnitRescale(min_dist);
            }
        merit:
            vec merit = weight * val_score + (1.0 -  weight) * dist_score;
            merit.elem(arma::find(min_dist < dtol)).fill(std::numeric_limits<double>::max());
            double score = merit.min(winner);
            if(score > 100000) {
                dtol /= 2.0;
                goto merit;
            }
            newx.col(i) = cand.col(winner);
        }
        
        return newx;
    };
    
    inline mat pick_points2(const mat &cand, const vec &rbf_val, const mat &centers, int dim, double dtol, vec weights, int newpts) {
        
        // Evaluate the RBF at the candidate points
        const mat dists = arma::sqrt(SquaredPairwiseDistance<mat>(centers, cand));
        //vec rbf_val = rbf.evals(cand, dists);
        //vec rbf_val = rbf.evals(cand);
        vec val_score = UnitRescale(rbf_val);
        vec min_dist = arma::min(dists).t();
        vec dist_score = 1.0 - UnitRescale(min_dist);
        
        mat newx = arma::zeros<mat>(dim, newpts);
        
        arma::uword winner;
        for(int i=0; i < newpts; i++) {
            double weight = weights[i];
            
            // Update distances if necessary
            if (i > 0) {
                vec new_dist = arma::sqrt(SquaredPointSetDistance<mat,vec>((vec)newx.col(i-1), cand));
                for(int j=0; j < cand.n_cols; j++) {
                    if(new_dist(j) < min_dist(j)) {
                        min_dist(j) = new_dist(j);
                    }
                }
                val_score(winner) = std::numeric_limits<double>::max();
                dist_score = 1.0 - UnitRescale(min_dist);
            }
        merit:
            vec merit = weight * val_score + (1.0 -  weight) * dist_score;
            merit.elem(arma::find(min_dist < dtol)).fill(std::numeric_limits<double>::max());
            double score = merit.min(winner);
            if(score > 100000) {
                dtol /= 2.0;
                goto merit;
            }
            newx.col(i) = cand.col(winner);
        }
        
        return newx;
    };
    
    template<class RBF>
    mat DYCORS(const RBF &rbf, const mat &centers, const vec &xlow, const vec &xup, const vec &xbest, double sigma, int maxeval, int numeval, int initp, int dim, double dtol, vec weights, int ncand, int newpts) {
        
        double dds_prob = fmin(20.0/dim, 1) * (1 - log(numeval - initp + 1)/log(maxeval - initp));
        dds_prob = fmax(dds_prob, 1.0/dim);
        vec direction_prob = dds_prob * arma::ones<vec>(dim);
        mat basis = arma::eye<mat>(dim, dim);
        return DYCORS(rbf, centers, xlow, xup, xbest, sigma, maxeval, numeval, initp, dim, dtol, weights, ncand, newpts, basis, direction_prob);
    };
    
    template<class RBF>
    mat DYCORS(const RBF &rbf, const mat &centers, const vec &xlow, const vec &xup, const vec &xbest, double sigma, int maxeval, int numeval, int initp, int dim, double dtol, vec weights, int ncand, int newpts, mat basis, vec direction_prob) {
        
        assert(weights.n_elem == newpts);
        
        std::random_device rd;
        std::default_random_engine e1(rd());
        std::uniform_real_distribution<double> rand(0, 1);
        std::uniform_int_distribution<int> randi(0, dim-1);
        std::normal_distribution<double> randn(0.0, sigma);
        
        mat cand = arma::repmat(xbest, 1, ncand);
        for(int i=0; i < ncand; i++) {
            
            int count = 0;
            for(int j=0; j < dim; j++) {
                if(rand(e1) < direction_prob(j)) {
                    count++;
                    cand.col(i) += randn(e1) * basis.col(j);
                }
            }
            // If no index was pertrubed we force one
            if(count == 0) {
                int ind = randi(e1);
                cand.col(i) += randn(e1) * basis.col(ind);
            }
            
            // Make sure we are still in the domain
            for(int j=0; j < dim; j++) {
                if(cand(j, i) > xup(j)) { cand(j, i) = fmax(2*xup(j) - cand(j, i), xlow(j)); }
                else if(cand(j, i) < xlow(j)) { cand(j, i) = fmin(2*xlow(j) - cand(j, i), xup(j)); }
            }
        }
        
        return pick_points<RBF>(cand, rbf, centers, dim, dtol, weights, newpts);
    };
    
    template<class RBF>
    mat OneDim(const RBF &rbf, const mat &centers, const vec &xlow, const vec &xup, const vec &xbest, double sigma, int maxeval, int numeval, int initp, int dim, double dtol, vec weights, int ncand, int newpts) {
        
        assert(weights.n_elem == newpts);
        
        std::random_device rd;
        std::default_random_engine e1(rd());
        std::uniform_real_distribution<double> rand(0, 1);
        std::uniform_int_distribution<int> randi(0, dim-1);
        std::normal_distribution<double> randn(0.0, sigma);
        
        mat cand = arma::repmat(xbest, 1, ncand);
        arma::ivec perturbation_coords = arma::randi<arma::ivec>(ncand, arma::distr_param(0, dim - 1));
        
        // Perturbs one randomly chosen coordinate
        for(int i=0; i < ncand; i++) {
            
            int j = perturbation_coords(i);
            cand(j, i) += randn(e1);
            if(cand(j, i) > xup(j)) {
                cand(j, i) = fmax(2*xup(j) - cand(j, i), xlow(j));
            }
            else if(cand(j, i) < xlow(j)) {
                cand(j, i) = fmin(2*xlow(j) - cand(j, i), xup(j));
            }
        }
        
        return pick_points<RBF>(cand, rbf, centers, dim, dtol, weights, newpts);
    };
    
    template<class RBF>
    mat AllDim(const RBF &rbf, const mat &centers, const vec &xlow, const vec &xup, const vec &xbest, double sigma, int maxeval, int numeval, int initp, int dim, double dtol, vec weights, int ncand, int newpts) {
        
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

        return pick_points<RBF>(cand, rbf, centers, dim, dtol, weights, newpts);
    };
    
    template<class RBF>
    mat Uniform(const RBF &rbf, const mat &centers, const vec &xlow, const vec &xup, const vec &xbest, double sigma, int maxeval, int numeval, int initp, int dim, double dtol, vec weights, int ncand, int newpts) {
        
        assert(weights.n_elem == newpts);
        
        mat cand = arma::randu<mat>(dim, ncand);
        for(int j=0; j < dim; j++) {
            cand.row(j) = xlow(j) + (xup(j) - xlow(j)) * cand.row(j);
        }
        
        return pick_points<RBF>(cand, rbf, centers, dim, dtol, weights, newpts);
    };
    
    template<class RBF>
    mat Gradient(const RBF &rbf, const mat &centers, const vec &xlow, const vec &xup, const vec &xbest, double sigma, int maxeval, int numeval, int initp, int dim, double dtol, vec weights, int ncand, int newpts) {
        
        assert(weights.n_elem == newpts);
        
        vec gradient = arma::normalise(rbf.deriv(xbest));
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
        
        return pick_points<RBF>(cand, rbf, centers, dim, dtol, weights, newpts);
    };
    
    template<class RBF>
    mat Direction(const RBF &rbf, const mat &centers, const vec &xlow, const vec &xup, const vec &xbest, const vec &direction, double sigma, int maxeval, int numeval, int initp, int dim, double dtol, vec weights, int ncand, int newpts) {
        
        assert(weights.n_elem == newpts);
        
        vec dir = arma::normalise(direction);
        mat cand = arma::repmat(xbest, 1, ncand) + sigma * sqrt(dim) * dir * arma::randn<mat>(1, ncand);
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
        
        return pick_points<RBF>(cand, rbf, centers, dim, dtol, weights, newpts);
    };
}

/*
 template<class RBF>
 mat Ellipsoid(const RBF &rbf, const mat &centers, const vec &xlow, const vec &xup, const vec &xbest, const mat &C, double sigma, int maxeval, int numeval, int initp, int dim, double dtol, int newpts) {
 
 int ncand = fmin(100 * dim, 5000);
 mat L = arma::chol(C, "lower");
 mat cand = L * arma::randn<mat>(dim, ncand);
 //double lambda = sqrt(2.0)/(sigma*sqrt(dim)) * tgamma((10.0+1.0)/2.0)/tgamma(10.0/2.0);
 //cand = arma::repmat(xbest, 1, ncand) + cand / lambda;
 cand = arma::repmat(xbest, 1, ncand) + cand * sigma;
 
 //mat cand = arma::repmat(xbest, 1, ncand) + sigma * sqrt(dim) * direction * arma::randn<mat>(1, ncand);
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
 
 return pick_points<RBF>(cand, rbf, centers, dim, dtol, newpts);
 }
 */

#endif /* defined(__Surrogate_Optimization__CandidatePoints__) */
