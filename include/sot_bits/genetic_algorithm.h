//
//  genetic_algorithm.h
//  Surrogate Optimization
//
//  Created by David Eriksson on 8/3/15.
//  Copyright (c) 2015 David Eriksson. All rights reserved.
//

#ifndef __Surrogate_Optimization__genetic_algorithm__
#define __Surrogate_Optimization__genetic_algorithm__

#include <stdio.h>
#include "utils.h"
#include "experimental_design.h"
#include "common.h"
#include "optimizer.h"
#include "test_problems.h"

namespace sot {
    
    class GeneticAlgorithm : public Optimizer  {
    protected:
        double sigma = 0.2;
        int tournament_size = 5;
        double p_cross = 0.9;
        double p_mutation;

        vec xlow;
        vec xup;
        int n_variables;
        int n_individuals;
        int n_generations;
        std::string start;
    
    public:    
        GeneticAlgorithm(Problem *data, vec &xlow, vec& xup, int n_individuals, 
                int n_generations, std::string start) {
            this->data = data;
            this->n_variables = data->dim();
            this->p_mutation = 1.0/data->dim();
            this->n_individuals = n_individuals;
            this->n_generations = n_generations;
            this->start = start;
            this->xlow = xlow;
            this->xup = xup;
            this->my_name = "Genetic algorithm";
        }
        GeneticAlgorithm(Problem *data, ExpDesign *exp_des, int maxeval) {
            this->data = data;
            this->n_variables = data->dim();
            this->p_mutation = 1.0/data->dim();
            this->n_individuals = maxeval/100;
            this->n_generations = 100;
            this->start = "SLHD";
            this->xlow = data->lbound();
            this->xup = data->rbound();
            this->my_name = "Genetic algorithm";
        }

        Result run() {
            
            int maxeval = n_individuals * n_generations;
            Result res(n_individuals * n_generations, n_individuals, data->dim());
            
            // Setup random number generators
            std::random_device rd;
            std::default_random_engine e1(rd());
            std::uniform_real_distribution<double> rand(0, 1);
            std::normal_distribution<double> randn(0.0, sigma);
            
            mat population;
            mat new_population = arma::zeros<mat>(n_variables, n_individuals);
            if (start.compare("SLHD") == 0) {
                // Use a symmetric latin hypercube to initialize the population
                SymmetricLatinHypercube my_slhd(n_individuals, n_variables);
                population =  my_slhd.generate_points();
            }
            else if (start.compare("Random") == 0) {
                // Randomly initialize the population
                population.randu(n_variables, n_individuals);
            }
            else {
                // Yeah...., dunno what you want
                throw std::invalid_argument("Unkown argument");
            }
            population = FromUnitBox(population, xlow, xup);

            //  Evaluate all individuals
            vec function_values = data->evals(population);
            
            //Save the best individual
            arma::uword ind;
            function_values.min(ind);
            vec best_individual = population.col(ind);
            double best_value = function_values(ind);

            for(int gen = 0; gen < n_generations-1; gen++) {
                
                ////////////////// Tournament selection and crossover ////////////////////
                arma::imat tournament = arma::randi<arma::imat>(tournament_size, n_individuals, arma::distr_param(0, n_individuals - 1));
                for(int i = 0; i < n_individuals/2; i++) {
                    double minval1 = std::numeric_limits<double>::max();
                    double minval2 = std::numeric_limits<double>::max();
                    int ind1, ind2;
                    for(int j=0; j < tournament_size; j++) {
                        if (function_values(tournament(j, 2*i)) < minval1) {
                            minval1 = function_values(tournament(j, 2*i));
                            ind1 = tournament(j, 2*i);
                        }
                        if (function_values(tournament(j, 2*i + 1)) < minval2) {
                            minval2 = function_values(tournament(j, 2*i + 1));
                            ind2 = tournament(j, 2*i + 1);
                        }
                    }
                    
                    double alpha = rand(e1);
                    if( rand(e1) < p_cross) {
                        new_population.col(2*i) = alpha * population.col(ind1) + (1 - alpha) * population.col(ind2);
                        new_population.col(2*i + 1) = alpha * population.col(ind2) + (1 - alpha) * population.col(ind1);
                    }
                    else {
                        new_population.col(2*i) = population.col(ind1);
                        new_population.col(2*i + 1) = population.col(ind2);
                    }
                }

                // Mutation
                for(int i=0; i < n_individuals; i++) {
                    for(int j=0; j<n_variables; j++) {
                        if(rand(e1) < p_mutation) {
                            new_population(j, i) += (xup(j) - xlow(j)) * randn(e1);
                            if(new_population(j,i) > xup(j)) {
                                new_population(j,i) = fmax(2*xup(j) - new_population(j,i), xlow(j));
                            }
                            else if(new_population(j,i) < xlow(j)) {
                                new_population(j,i) = fmin(2*xlow(j) - new_population(j,i), xup(j));
                            }
                        }
                    }
                }
                
                // Elitism
                new_population.col(n_individuals - 1) = best_individual;
                
                //  Evaluate all individuals
                function_values = data->evals(new_population);
                
                // Save the results
                res.x.cols( gen * n_individuals, gen * n_individuals + n_individuals - 1) = new_population;
                res.fx.rows(gen * n_individuals, gen * n_individuals + n_individuals - 1) = function_values;

                //Save the best individual
                arma::uword ind;
                function_values.min(ind);
                best_individual = new_population.col(ind);
                best_value = function_values(ind);
                
                // Kill the old population
                population = new_population;
            }
            
            res.xbest = best_individual;
            res.fbest = best_value;
            return res;
        }
    };
}

#endif /* defined(__Surrogate_Optimization__genetic_algorithm__) */