//
//  test_problems.h
//  Surrogate Optimization
//
//  Created by David Eriksson on 7/30/15.
//  Copyright (c) 2015 David Eriksson. All rights reserved.
//

#ifndef __Surrogate_Optimization__test_problems__
#define __Surrogate_Optimization__test_problems__

#include <cassert>
#include <cmath>
#include "common.h"
#include "problem.h"

namespace sot {    
    
    // 1
    class Sphere : public Problem {
    public:
        Sphere(int d) {
            this->my_name = "Sphere";
            this->global_min = 0.0;
            this->optimum = arma::zeros<vec>(d);
            this->d = d;
            this->xlow = -10.0 * arma::ones<vec>(d);
            this->xup = 10.0 * arma::ones<vec>(d);
        }
        double eval(vec x) const {
            assert(x.n_rows == this->d);
            return arma::sum(x % x);
        }
    };
    
    // 2
    class SumSquares : public Problem {
    public:
        SumSquares(int d) {
            this->my_name = "Sum of Squares";
            this->global_min = 0.0;
            this->optimum = arma::zeros<vec>(d);
            this->d = d;
            this->xlow = -10.0 * arma::ones<vec>(d);
            this->xup = 10.0 * arma::ones<vec>(d);
        }
        double eval(vec x) const {
            assert(x.n_rows == this->d);
            return arma::dot(arma::linspace(1, this->d, this->d), x % x);
        }
    };
    
    // 3
    class Schwefel2_22 : public Problem {
    public:
        Schwefel2_22(int d) {
            this-> my_name = "Schwefel2_22";
            this->global_min = 0.0;
            this->optimum = arma::zeros<vec>(d);
            this->d = d;
            this->xlow = -10.0 * arma::ones<vec>(d);
            this->xup = 10.0 * arma::ones<vec>(d);
        }
        double eval(vec x) const {
            assert(x.n_rows == this->d);
            return arma::sum(arma::abs(x)) + arma::prod(arma::abs(x));
        }
    };
    
    // 4
    class Exponential : public Problem {
    public:
        Exponential(int d) {
            this->my_name = "Exponential";
            this->global_min = -1.0;
            this->optimum = arma::zeros<vec>(d);
            this->d = d;
            this->xlow = -1.0 * arma::ones<vec>(d);
            this->xup = 1.0 * arma::ones<vec>(d);
        }
        double eval(vec x) const {
            assert(x.n_rows == this->d);
            return - exp(-0.5 * arma::sum(x % x));
        }
    };
    
    // 5
    class Tablet : public Problem {
    public:
        Tablet(int d) {
            this->my_name = "Tablet";
            this->global_min = 0.0;
            this->optimum = arma::zeros<vec>(d);
            this->d = d;
            this->xlow = -10.0 * arma::ones<vec>(d);
            this->xup = 10.0 * arma::ones<vec>(d);
        }
        double eval(vec x) const {
            assert(x.n_rows == this->d);
            return 1e6 * (x(0) * x(0)) + 
                    arma::sum(x.rows(1, this->d-1) %  x.rows(1, this->d-1));
        }
    };
    
    // 6
    class Step : public Problem {
    public:
        Step(int d) {
            this->my_name = "Step";
            this->global_min = 0.0;
            this->optimum = arma::zeros<vec>(d);
            this->d = d;
            this->xlow = -10.0 * arma::ones<vec>(d);
            this->xup = 10.0 * arma::ones<vec>(d);
        }
        double eval(vec x) const {
            assert(x.n_rows == this->d);
            return arma::sum((arma::floor(x + 0.5)) % (arma::floor(x + 0.5)));
        }
    };
    
    // 7
    class Zakharov : public Problem {
    public:
        Zakharov(int d) {
            this->my_name = "Zakharov";
            this->global_min = 0.0;
            this->optimum = arma::zeros<vec>(d);
            this->d = d;
            this->xlow = -5.0 * arma::ones<vec>(d);
            this->xup = 10.0 * arma::ones<vec>(d);
        }
        double eval(vec x) const {
            assert(x.n_rows == this->d);
            double term1 = arma::sum(x % x);
            double term2 = arma::sum(0.5 * arma::dot(arma::linspace(1, this->d, this->d), x));
            return term1 + (term2 * term2) + (term2 * term2 * term2 * term2);
        }
    };
    
    // 8
    class Rosenbrock : public Problem {
    public:
        Rosenbrock(int d) {
            this->my_name = "Rosenbrock";
            this->global_min = 0.0;
            this->optimum = arma::ones<vec>(d);
            this->d = d;
            this->xlow = -2.0 * arma::ones<vec>(d);
            this->xup = 2.0 * arma::ones<vec>(d);
        }
        double eval(vec x) const {
            assert(x.n_rows == this->d);
            double total = 0.0;
            for(int i=0; i < this->d-1; i++) {
                total += 50.0 * (x(i)*x(i) - x(i+1))*(x(i)*x(i) - x(i+1))  + (x(i) - 1) * (x(i) - 1);
            }
            return total;
        }
    };
    
    // 9
    class Griewank : public Problem {
    public:
        Griewank(int d) {
            this->my_name = "Griewank";
            this->global_min = 0.0;
            this->optimum = arma::zeros<vec>(d);
            this->d = d;
            this->xlow = -10.0 * arma::ones<vec>(d);
            this->xup = 10.0 * arma::ones<vec>(d);
        }
        double eval(vec x) const {
            assert(x.n_rows == this->d);
            return 1.0 + (1.0/4000) * arma::sum(x % x) - 
                    arma::prod(arma::cos(x / sqrt(arma::linspace(1, this->d, this->d))));
        }
    };
    
    // 10
    class Schaffer2 : public Problem {
    public:
        Schaffer2(int d) {
            this->my_name = "Schaffer 2";
            this->global_min = 0.0;
            this->optimum = arma::zeros<vec>(d);
            this->d = d;
            this->xlow = -10.0 * arma::ones<vec>(d);
            this->xup = 10.0 * arma::ones<vec>(d);
        }
        double eval(vec x) const {
            assert(x.n_rows == this->d);
            double total = 0.0;
            for(int i=0; i < this->d - 1; i++) {
                total += pow(x(i)*x(i) + x(i+1)*x(i+1), 0.25) * 
                        (pow(sin(50*pow(x(i)*x(i) + x(i+1)*x(i+1), 0.1)), 2.0) + 1);
            }
            return total;
        }
    };
    
    
    // 11
    class Schwefel2_26 : public Problem {
    public:
        Schwefel2_26(int d) {
            this->my_name = "Schwefel2_26";
            this->d = d;
            this->optimum = arma::zeros<vec>(d);
            this->xlow = -10.0 * arma::ones<vec>(d);
            this->xup = 10.0 * arma::ones<vec>(d);
            this->global_min =  - 418.9829 * d;
        }
        double eval(vec x) const {
            assert(x.n_rows == this->d);
            return - arma::dot(x, arma::sin(arma::sqrt(arma::abs(x))));
        }
    };
    
    // 12
    class Himmelblau : public Problem {
    public:
        Himmelblau(int d) {
            this->my_name = "Himmelblau";
            this->global_min = -78.3323;
            this->d = d;
            this->xlow = -5.0 * arma::ones<vec>(d);
            this->xup = 5.0 * arma::ones<vec>(d);
            this->optimum = -2.9035 * arma::ones<vec>(d);
        }
        double eval(vec x) const {
            assert(x.n_rows == this->d);
            return (1.0/this->d) * arma::sum( (x % x % x % x) - 16*(x % x) + 5 * x);
        }
    };
    
    // 15
    class Ackley : public Problem {
    public:
        Ackley(int d) {
            this->my_name = "Ackley";
            this->global_min = - 20 - exp(1);
            this->optimum = arma::zeros<vec>(d);
            this->d = d;
            this->xlow = -15.0 * arma::ones<vec>(d);
            this->xup = 20.0 * arma::ones<vec>(d);
        }
        double eval(vec x) const {
            assert(x.n_rows == this->d);
            return -20.0 * exp(-0.2 * sqrt(arma::sum(x % x)/double(this->d))) - 
                    exp(arma::sum(arma::cos(2.0 * __pi__ * x))/double(this->d));
        }
    };
    
    // 16
    class Rastrigin : public Problem {
    public:
        Rastrigin(int d) {
            this->my_name = "Rastrigin";
            this->d = d;
            this->optimum = arma::zeros<vec>(d);
            this->global_min = -1 * d;
            this->xlow = -4.0 * arma::ones<vec>(d);
            this->xup = 5.0 * arma::ones<vec>(d);
        }
        double eval(vec x) const {
            assert(x.n_rows == this->d);
            return arma::sum(x % x - 1 * arma::cos(2 * __pi__ * x));
        }
    };
    
    // Others
    class Michalewicz : public Problem {
    public:
        Michalewicz(int d) {
            this->my_name = "Michalewicz";
            this->d = d;
            this->xlow = arma::zeros<vec>(d);
            this->xup = __pi__ * arma::ones<vec>(d);
            this->global_min = -0.966 * d;
        }
        double eval(vec x) const {
            assert(x.n_rows == this->d);
            return - arma::sum(arma::sin(x) % arma::pow(arma::sin(
                    ((arma::linspace(1, this->d, this->d) % x % x)/__pi__)), 20));
        }
    };
    
    class Keane : public Problem {
    public:
        Keane(int d) {
            this->my_name = "Keane";
            this->d = d;
            this->xlow = arma::ones<vec>(d);
            this->xup = 10 * arma::ones<vec>(d);
            this->global_min = -0.47;
        }
        double eval(vec x) const {
            assert(x.n_rows == this->d);
            double y1 = 0.0;
            double y2 = 1.0;
            double y3 = 0.0;
            
            for(int i=0; i < this->d; i++) {
                y1 += (cos(x(i)) * cos(x(i)) * cos(x(i)) * cos(x(i)));
                y2 *= (cos(x[i]) * cos(x(i)));
                y3 += (i+1) * (x(i) * x(i));
            }
            
            return -fabs((y1 - 2.0 * y2) / sqrt(y3));
        }
    };
    
    class Levy : public Problem {
    public:
        Levy(int d) {
            this->my_name = "Levy";
            this->d = d;
            this->xlow = -5 * arma::ones<vec>(d);
            this->xup = 5 * arma::ones<vec>(d);
            this->global_min = 0;
            this->optimum = arma::ones(d);
        }
        double eval(vec x) const {
            assert(x.n_rows == this->d);
            
            vec w = arma::zeros<vec>(this->d);
            for(int i=0; i < this->d; i++) {
                w(i) = 1 + (x(i) - 1)/4;
            }
            
            double term1 = (sin(__pi__*w(0))) * (sin(__pi__*w(0)));
            double term3 = (w(this->d-1)-1) * (w(this->d-1)-1) * (1+(sin(2*__pi__*w(this->d-1))) * (sin(2*__pi__*w(this->d-1))));
            
            double term2 = 0;
            for(int i=0; i < this->d-1; i++) {
                term2 += (w(i)-1) * (w(i)-1) * (1 + 10 *(sin(__pi__*w(i)+1)) * (sin(__pi__*w(i)+1)));;
            }
            return  term1 + term2 + term3;
        }
    };
    
    class Salomon : public Problem {
    public:
        Salomon(int d) {
            this->my_name = "Salomon";
            this->d = d;
            this->optimum = arma::zeros<vec>(d);
            this->xlow = -10 * arma::ones<vec>(d);
            this->xup = 10 * arma::ones<vec>(d);
            this->global_min = 0.0;
        }
        double eval(vec x) const {
            assert(x.n_rows == this->d);
            return 1 - cos(2*__pi__*sqrt(arma::sum(x%x))) + 
                    0.1*sqrt(arma::sum(x % x));
        }
    };
    
    class Schubert3 : public Problem {
    public:
        Schubert3(int d) {
            this->my_name = "Schubert3";
            this->d = d;
            this->xlow = -10 * arma::ones<vec>(d);
            this->xup = 10 * arma::ones<vec>(d);
            this->global_min = -24.062499;
        }
        double eval(vec x) const {
            assert(x.n_rows == this->d);
            double total = 0.0;
            for(int i=0; i < this->d; i++) {
                for(int j=1; j < 6; j++) {
                    total += j*sin((j+1)*x(i)) + j;
                }
            }
            return total;
        }
    };
    
    class SineEnvelope : public Problem {
    public:
        SineEnvelope(int d) {
            this->my_name = "Sine Envelope";
            this->d = d;
            this->xlow = -20 * arma::ones<vec>(d);
            this->xup = 20 * arma::ones<vec>(d);
            this->global_min = 0.0;
        }
        double eval(vec x) const {
            assert(x.n_rows == this->d);
            double total = 0.0;
            for(int i=0; i < this->d-1; i++) {
                total += (pow(sin(sqrt(x(i)*x(i)+x(i+1)*x(i+1))-0.5),2))/
                        (pow(0.001*(x(i)*x(i)+x(i+1)*x(i+1))+1,2))+0.5;
            }
            return total;
        }
    };
    
    class Schoen : public Problem {
    protected:
        int k;
        vec fi;
        vec alpha;
        mat z;
    public:
        Schoen(int d) : Schoen(d, fmax(2^d, 500)) {}
        Schoen(int d, int k) {
            this->my_name = "Schoen, k = " + std::to_string(k);
            this->d = d;
            this->k = k;
            this->xlow = arma::zeros<vec>(d);
            this->xup = arma::ones<vec>(d);
            this->fi = 100 * arma::randu<vec>(k);
            this->fi(0) = 0;
            this->alpha = 2 + arma::randu<vec>(k);
            this->z = arma::randu<mat>(this->d, this->k);
            
            mat dists = arma::sqrt(SquaredPairwiseDistance<mat>(this->z, this->z));
            dists.diag() += arma::datum::inf;
            double min_dist = arma::min(arma::min(dists));
            
            this->global_min = arma::min(this->fi);
            arma::uvec argmin = arma::find(fi == this->global_min);
            this->optimum = this->z.col(argmin(0));
        }
        double eval(vec x) const {
            assert(x.n_rows == this->d);
            
            long double num = 0, den = 0, prodval = 0;
            for(int i=0; i < this->k; i++) {
                prodval = pow(sqrt(arma::sum(arma::square(z.col(i) - x))), this->alpha(i));
                num += this->fi(i)/prodval;
                den += 1/prodval;
            }
            return num/den;
        }
    };
    
    class SineFun : public Problem {
    protected:
        int kmax;
        double s;
        vec fk;
        vec ak;
        mat Pk;
    public:
        SineFun(int d, int kmax) {
            this->my_name = "SineFun, kmax = " + std::to_string(kmax);
            this->d = d;
            this->kmax = kmax;
            this->xlow = -4 * arma::ones<vec>(d);
            this->xup =   4 * arma::ones<vec>(d);
            this->optimum = -1 + 2 * arma::randu<vec>(d);
            this->global_min = 0;
            
            vec temp = 1*arma::randn<vec>(1);
            this->s = fabs(temp(0));
            this->ak = 5*arma::abs(arma::randn<vec>(kmax));
            this->fk = arma::randi<vec>(kmax, arma::distr_param(1, 3));
            this->Pk = arma::randn<mat>(d, kmax);
        }
        double eval(vec x) const {
            assert(x.n_rows == this->d);
            double total = s * arma::sum(arma::square(x - this->optimum));
            for(int k=0; k < kmax; k++) {
                total += ak(k) *  pow( sin(fk(k) * arma::sum(Pk.col(k) % (x - this->optimum))), 2.0);
            }
            return total;
        }
    };
    
    class CosineMixture : public Problem {
    public:
        CosineMixture(int d) {
            this->my_name = "Consine Mixture";
            this->d = d;
            this->xlow = -1 * arma::ones<vec>(d);
            this->xup = 1 * arma::ones<vec>(d);
            this->optimum = arma::zeros<vec>(d);
            this->global_min = -0.1*d;
        }
        double eval(vec x) const {
            assert(x.n_rows == this->d);
            return -0.1*arma::sum(arma::cos(5*__pi__*x)) + arma::sum(x % x);
        }
    };
    
    /*
     * A wrapper function that takes a possibly separable optimization problem
     * and turns it into a non-separable one. This is done by generating an
     * orthogonal matrix Q and a vector t, where t is in the domain of f(x). 
     * We define the new objective function
     * 
     *  g(x) = f(xopt + Q * (x - t))
     * 
     * We can see that g(t) = f(xopt) so t is the global minimum of t, and we
     * can also see that g isn't separable as long as Q isn't.
     */
    
    class RotatedProblem : public Problem {
    protected:
        vec translation;
        mat rotation;
        
        void create_translation(vec xlow, vec xup) {
            std::random_device rd;
            std::default_random_engine e1(rd());
            std::uniform_real_distribution<double> rand(0, 1);
            translation = arma::zeros<vec>(this->d);
            for(int i=0; i < this->d; i++) {
                translation(i) = xlow(i) + (xup(i) - xlow(i)) * rand(e1);
            }
        }
        void create_rotation(int dim) {
            mat Q, R;
            mat X = arma::randn<mat>(this->d, this->d);
            arma::qr(Q, R, X);
            rotation = Q;
        }
    public:
        Problem *problem;
        RotatedProblem(Problem *problem) {
            if (problem->minimum().n_elem != problem->dim()) {
                std::cout << "Optimum not specified for: " << problem->name() << std::endl;
                abort();
            }
            this->problem = problem;
            this->d = problem->dim();
            this->xlow = problem->lbound();
            this->xup = problem->rbound();
            this->global_min = problem->min();
            create_translation(xlow, xup);
            create_rotation(this->d);
            this->optimum = translation;
            this->my_name = "Rotated + Translated " + problem->name();
        }
        mat get_roatation() const { return this->rotation; }
        vec get_translation() const { return this->translation; }
        
        double eval(vec xx) const {
            assert(xx.n_rows == this->d);
            vec x = problem->minimum() + rotation * (xx - translation);
            return problem->eval(x);
        }
    };
}

#endif /* defined(__Surrogate_Optimization__test_problems__) */
