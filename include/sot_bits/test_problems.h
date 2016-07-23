//
//  test_problems.h
//  Surrogate Optimization
//
//  Created by David Eriksson on 7/30/15.
//  Copyright (c) 2015 David Eriksson. All rights reserved.
//

#ifndef Surrogate_Optimization_test_problems_h
#define Surrogate_Optimization_test_problems_h

#include <cmath>
#include "common.h"
#include "problem.h"

namespace sot {    
    
    // 1
    class Sphere : public Problem {
    protected:
        int mDim;
        vec mxLow;
        vec mxUp;
        vec mOptimum;
        double mMinimum = 0;
        std::string mName = "Sphere";
    public:
        Sphere(int dim) {
            mDim = dim;
            mOptimum = arma::zeros<vec>(dim);
            mxLow = -10.0 * arma::ones<vec>(dim);
            mxUp = 10.0 * arma::ones<vec>(dim);
        }
        vec lBounds() const { return mxLow; }
        vec uBounds() const { return mxUp; }
        int dim() const { return mDim; }
        double min() const { return mMinimum; }
        vec optimum() const { return mOptimum; }
        std::string name() const { return mName; }
        double eval(vec &x) const {
            return arma::sum(x % x);
        }
    };
    
    // 2
    class SumSquares : public Problem {
    protected:
        int mDim;
        vec mxLow;
        vec mxUp;
        vec mOptimum;
        double mMinimum = 0;
        std::string mName = "Sum of Squares";
    public:
        SumSquares(int dim) {
            mDim = dim;
            mOptimum = arma::zeros<vec>(dim);
            mxLow = -10.0 * arma::ones<vec>(dim);
            mxUp = 10.0 * arma::ones<vec>(dim);
        }
        vec lBounds() const { return mxLow; }
        vec uBounds() const { return mxUp; }
        int dim() const { return mDim; }
        double min() const { return mMinimum; }
        vec optimum() const { return mOptimum; }
        std::string name() const { return mName; }
        double eval(vec &x) const {
            return arma::dot(arma::linspace(1, mDim, mDim), x % x);
        }
    };
    
    // 3
    class Schwefel2_22 : public Problem {
    protected:
        int mDim;
        vec mxLow;
        vec mxUp;
        vec mOptimum;
        double mMinimum = 0;
        std::string mName = "Schwefel2_22";
    public:
        Schwefel2_22(int dim) {
            mDim = dim;
            mOptimum = arma::zeros<vec>(dim);
            mxLow = -10.0 * arma::ones<vec>(dim);
            mxUp = 10.0 * arma::ones<vec>(dim);
        }
        vec lBounds() const { return mxLow; }
        vec uBounds() const { return mxUp; }
        int dim() const { return mDim; }
        double min() const { return mMinimum; }
        vec optimum() const { return mOptimum; }
        std::string name() const { return mName; }
        double eval(vec &x) const {
            return arma::sum(arma::abs(x)) + arma::prod(arma::abs(x));
        }
    };
    
    // 4
    class Exponential : public Problem {
    protected:
        int mDim;
        vec mxLow;
        vec mxUp;
        vec mOptimum;
        double mMinimum = -1.0;
        std::string mName = "Exponential";
    public:
        Exponential(int dim) {
            mDim = dim;
            mOptimum = arma::zeros<vec>(dim);
            mxLow = -1.0 * arma::ones<vec>(dim);
            mxUp = 1.0 * arma::ones<vec>(dim);
        }
        vec lBounds() const { return mxLow; }
        vec uBounds() const { return mxUp; }
        int dim() const { return mDim; }
        double min() const { return mMinimum; }
        vec optimum() const { return mOptimum; }
        std::string name() const { return mName; }
        double eval(vec &x) const {
            return - exp(-0.5 * arma::sum(x % x));
        }
    };
    
    // 5
    class Tablet : public Problem {
    protected:
        int mDim;
        vec mxLow;
        vec mxUp;
        vec mOptimum;
        double mMinimum = 0.0;
        std::string mName = "Tablet";
    public:
        Tablet(int dim) {
            mDim = dim;
            mOptimum = arma::zeros<vec>(dim);
            mxLow = -10.0 * arma::ones<vec>(dim);
            mxUp = 10.0 * arma::ones<vec>(dim);
        }
        vec lBounds() const { return mxLow; }
        vec uBounds() const { return mxUp; }
        int dim() const { return mDim; }
        double min() const { return mMinimum; }
        vec optimum() const { return mOptimum; }
        std::string name() const { return mName; }
        double eval(vec &x) const {
            return 1e6 * (x(0) * x(0)) + arma::sum(x.rows(1, mDim-1) %  x.rows(1, mDim-1));
        }
    };
    
    // 6
    class Step : public Problem {
    protected:
        int mDim;
        vec mxLow;
        vec mxUp;
        vec mOptimum;
        double mMinimum = 0.0;
        std::string mName = "Step";
    public:
        Step(int dim) {
            mDim = dim;
            mOptimum = arma::zeros<vec>(dim);
            mxLow = -10.0 * arma::ones<vec>(dim);
            mxUp = 10.0 * arma::ones<vec>(dim);
        }
        vec lBounds() const { return mxLow; }
        vec uBounds() const { return mxUp; }
        int dim() const { return mDim; }
        double min() const { return mMinimum; }
        vec optimum() const { return mOptimum; }
        std::string name() const { return mName; }
        double eval(vec x) const {
            return arma::sum((arma::floor(x + 0.5)) % (arma::floor(x + 0.5)));
        }
    };
    
    // 7
    class Zakharov : public Problem {
    protected:
        int mDim;
        vec mxLow;
        vec mxUp;
        vec mOptimum;
        double mMinimum = 0.0;
        std::string mName = "Zakharov";
    public:
        Zakharov(int dim) {
            mDim = dim;
            mOptimum = arma::zeros<vec>(dim);
            mxLow = -5.0 * arma::ones<vec>(dim);
            mxUp = 10.0 * arma::ones<vec>(dim);
        }
        vec lBounds() const { return mxLow; }
        vec uBounds() const { return mxUp; }
        int dim() const { return mDim; }
        double min() const { return mMinimum; }
        vec optimum() const { return mOptimum; }
        std::string name() const { return mName; }
        double eval(vec &x) const {
            double term1 = arma::sum(x % x);
            double term2 = arma::sum(0.5 * arma::dot(arma::linspace(1, mDim, mDim), x));
            return term1 + (term2 * term2) + (term2 * term2 * term2 * term2);
        }
    };
    
    // 8
    class Rosenbrock : public Problem {
    protected:
        int mDim;
        vec mxLow;
        vec mxUp;
        vec mOptimum;
        double mMinimum = 0.0;
        std::string mName = "Rosenbrock";
    public:
        Rosenbrock(int dim) {
            mDim = dim;
            mOptimum = arma::ones<vec>(dim);
            mxLow = -2.0 * arma::ones<vec>(dim);
            mxUp = 2.0 * arma::ones<vec>(dim);
        }
        vec lBounds() const { return mxLow; }
        vec uBounds() const { return mxUp; }
        int dim() const { return mDim; }
        double min() const { return mMinimum; }
        vec optimum() const { return mOptimum; }
        std::string name() const { return mName; }
        double eval(vec &x) const {
            double total = 0.0;
            for(int i=0; i < mDim-1; i++) {
                total += 100.0 * (x(i)*x(i) - x(i+1))*(x(i)*x(i) - x(i+1))  + (x(i) - 1) * (x(i) - 1);
            }
            return total;
        }
    };
    
    // 9
    class Griewank : public Problem {
    protected:
        int mDim;
        vec mxLow;
        vec mxUp;
        vec mOptimum;
        double mMinimum = 0.0;
        std::string mName = "Griewank";
    public:
        Griewank(int dim) {
            mDim = dim;
            mOptimum = arma::zeros<vec>(dim);
            mxLow = -10.0 * arma::ones<vec>(dim);
            mxUp = 10.0 * arma::ones<vec>(dim);
        }
        vec lBounds() const { return mxLow; }
        vec uBounds() const { return mxUp; }
        int dim() const { return mDim; }
        double min() const { return mMinimum; }
        vec optimum() const { return mOptimum; }
        std::string name() const { return mName; }
        double eval(vec &x) const {
            return 1.0 + (1.0/4000) * arma::sum(x % x) - 
                    arma::prod(arma::cos(x / sqrt(arma::linspace(1, mDim, mDim))));
        }
    };
    
    // 10
    class Schaffer2 : public Problem {
    protected:
        int mDim;
        vec mxLow;
        vec mxUp;
        vec mOptimum;
        double mMinimum = 0.0;
        std::string mName = "Schaffer 2";
    public:
        Schaffer2(int dim) {
            mDim = dim;
            mOptimum = arma::zeros<vec>(dim);
            mxLow = -10.0 * arma::ones<vec>(dim);
            mxUp = 10.0 * arma::ones<vec>(dim);
        }
        vec lBounds() const { return mxLow; }
        vec uBounds() const { return mxUp; }
        int dim() const { return mDim; }
        double min() const { return mMinimum; }
        vec optimum() const { return mOptimum; }
        std::string name() const { return mName; }
        double eval(vec &x) const {
            double total = 0.0;
            for(int i=0; i < mDim - 1; i++) {
                total += pow(x(i)*x(i) + x(i+1)*x(i+1), 0.25) * 
                        (pow(sin(50*pow(x(i)*x(i) + x(i+1)*x(i+1), 0.1)), 2.0) + 1);
            }
            return total;
        }
    };
    
    
    // 11
    class Schwefel2_26 : public Problem {
    protected:
        int mDim;
        vec mxLow;
        vec mxUp;
        vec mOptimum;
        double mMinimum;
        std::string mName = "Schwefel2_26";
    public:
        Schwefel2_26(int dim) {
            mDim = dim;
            mOptimum = arma::zeros<vec>(dim);
            mxLow = -10.0 * arma::ones<vec>(dim);
            mxUp = 10.0 * arma::ones<vec>(dim);
            mMinimum = - 418.9829 * dim;
        }
        vec lBounds() const { return mxLow; }
        vec uBounds() const { return mxUp; }
        int dim() const { return mDim; }
        double min() const { return mMinimum; }
        vec optimum() const { return mOptimum; }
        std::string name() const { return mName; }
        double eval(vec &x) const {
            return - arma::dot(x, arma::sin(arma::sqrt(arma::abs(x))));
        }
    };
    
    // 12
    class Himmelblau : public Problem {
    protected:
        int mDim;
        vec mxLow;
        vec mxUp;
        vec mOptimum;
        double mMinimum = -78.3323;
        std::string mName = "Himmelblau";
    public:
        Himmelblau(int dim) {
            mDim = dim;
            mOptimum = -2.9035 * arma::ones<vec>(dim);
            mxLow = -5.0 * arma::ones<vec>(dim);
            mxUp = 5.0 * arma::ones<vec>(dim);
        }
        vec lBounds() const { return mxLow; }
        vec uBounds() const { return mxUp; }
        int dim() const { return mDim; }
        double min() const { return mMinimum; }
        vec optimum() const { return mOptimum; }
        std::string name() const { return mName; }
        double eval(vec &x) const {
            return (1.0/mDim) * arma::sum( (x % x % x % x) - 16*(x % x) + 5 * x);
        }
    };
    
    // 15
    class Ackley : public Problem {
    protected:
        int mDim;
        vec mxLow;
        vec mxUp;
        vec mOptimum;
        double mMinimum = - 20 - exp(1);
        std::string mName = "Ackley";
    public:
        Ackley(int dim) {
            mDim = dim;
            mOptimum = arma::zeros<vec>(dim);
            mxLow = -15.0 * arma::ones<vec>(dim);
            mxUp = 20.0 * arma::ones<vec>(dim);
        }
        vec lBounds() const { return mxLow; }
        vec uBounds() const { return mxUp; }
        int dim() const { return mDim; }
        double min() const { return mMinimum; }
        vec optimum() const { return mOptimum; }
        std::string name() const { return mName; }
        double eval(vec &x) const {
            return -20.0 * exp(-0.2 * sqrt(arma::sum(x % x)/double(mDim))) - 
                    exp(arma::sum(arma::cos(2.0 * __pi__ * x))/double(mDim));
        }
    };
    
    // 16
    class Rastrigin : public Problem {
    protected:
        int mDim;
        vec mxLow;
        vec mxUp;
        vec mOptimum;
        double mMinimum;
        std::string mName = "Rastrigin";
    public:
        Rastrigin(int dim) {
            mDim = dim;
            mOptimum = arma::zeros<vec>(dim);
            mxLow = -4.0 * arma::ones<vec>(dim);
            mxUp = 5.0 * arma::ones<vec>(dim);
            mMinimum = -dim;
        }
        vec lBounds() const { return mxLow; }
        vec uBounds() const { return mxUp; }
        int dim() const { return mDim; }
        double min() const { return mMinimum; }
        vec optimum() const { return mOptimum; }
        std::string name() const { return mName; }
        double eval(vec x) const {
            return arma::sum(x % x - 1 * arma::cos(2 * __pi__ * x));
        }
    };
    
    // 17
    class Michalewicz : public Problem {
    protected:
        int mDim;
        vec mxLow;
        vec mxUp;
        vec mOptimum;
        double mMinimum;
        std::string mName = "Michalewicz";
    public:
        Michalewicz(int dim) {
            mDim = dim;
            mxLow = arma::zeros<vec>(dim);
            mxUp = __pi__ * arma::ones<vec>(dim);
            mMinimum = -0.966 * dim;
            
        }
        vec lBounds() const { return mxLow; }
        vec uBounds() const { return mxUp; }
        int dim() const { return mDim; }
        double min() const { return mMinimum; }
        vec optimum() const { return mOptimum; }
        std::string name() const { return mName; }
        double eval(vec &x) const {
            return - arma::sum(arma::sin(x) % arma::pow(arma::sin(
                    ((arma::linspace(1, mDim, mDim) % x % x)/__pi__)), 20));
        }
    };
    
    // 18
    class Keane : public Problem {
    protected:
        int mDim;
        vec mxLow;
        vec mxUp;
        vec mOptimum;
        double mMinimum;
        std::string mName = "Keane";
    public:
        Keane(int dim) {
            mDim = dim;
            mxLow = arma::ones<vec>(dim);
            mxUp = 10.0 * arma::ones<vec>(dim);
            mMinimum = -0.47;
        }
        vec lBounds() const { return mxLow; }
        vec uBounds() const { return mxUp; }
        int dim() const { return mDim; }
        double min() const { return mMinimum; }
        vec optimum() const { return mOptimum; }
        std::string name() const { return mName; }
        double eval(vec &x) const {
            double y1 = 0.0;
            double y2 = 1.0;
            double y3 = 0.0;
            
            for(int i=0; i < mDim; i++) {
                y1 += (cos(x(i)) * cos(x(i)) * cos(x(i)) * cos(x(i)));
                y2 *= (cos(x[i]) * cos(x(i)));
                y3 += (i+1) * (x(i) * x(i));
            }
            
            return -fabs((y1 - 2.0 * y2) / sqrt(y3));
        }
    };
    
    // 19
    class Levy : public Problem {
    protected:
        int mDim;
        vec mxLow;
        vec mxUp;
        vec mOptimum;
        double mMinimum = 0;
        std::string mName = "Levy";
    public:
        Levy(int dim) {
            mDim = dim;
            mOptimum = arma::zeros<vec>(dim);
            mxLow = -5.0 * arma::ones<vec>(dim);
            mxUp = 5.0 * arma::ones<vec>(dim);
        }
        vec lBounds() const { return mxLow; }
        vec uBounds() const { return mxUp; }
        int dim() const { return mDim; }
        double min() const { return mMinimum; }
        vec optimum() const { return mOptimum; }
        std::string name() const { return mName; }
        double eval(vec &x) const {            
            vec w = arma::zeros<vec>(mDim);
            for(int i=0; i < mDim; i++) {
                w(i) = 1 + (x(i) - 1)/4;
            }
            
            double term1 = (sin(__pi__*w(0))) * (sin(__pi__*w(0)));
            double term3 = (w(mDim-1)-1) * (w(mDim-1)-1) * (1+(sin(2*__pi__*w(mDim-1))) * (sin(2*__pi__*w(mDim-1))));
            
            double term2 = 0;
            for(int i=0; i < mDim-1; i++) {
                term2 += (w(i)-1) * (w(i)-1) * (1 + 10 *(sin(__pi__*w(i)+1)) * (sin(__pi__*w(i)+1)));;
            }
            return  term1 + term2 + term3;
        }
    };
    
    // 20
    class Salomon : public Problem {
    protected:
        int mDim;
        vec mxLow;
        vec mxUp;
        vec mOptimum;
        double mMinimum = 0;
        std::string mName = "Salomon";
    public:
        Salomon(int dim) {
            mDim = dim;
            mOptimum = arma::zeros<vec>(dim);
            mxLow = -10.0 * arma::ones<vec>(dim);
            mxUp = 10.0 * arma::ones<vec>(dim);
        }
        vec lBounds() const { return mxLow; }
        vec uBounds() const { return mxUp; }
        int dim() const { return mDim; }
        double min() const { return mMinimum; }
        vec optimum() const { return mOptimum; }
        std::string name() const { return mName; }
        double eval(vec &x) const {
            return 1 - cos(2*__pi__*sqrt(arma::sum(x%x))) + 0.1*sqrt(arma::sum(x % x));
        }
    };
    
    // 21
    class Schubert3 : public Problem {
    protected:
        int mDim;
        vec mxLow;
        vec mxUp;
        vec mOptimum;
        double mMinimum = -24.062499;
        std::string mName = "Schubert3";
    public:
        Schubert3(int dim) {
            mDim = dim;
            mxLow = -10.0 * arma::ones<vec>(dim);
            mxUp = 10.0 * arma::ones<vec>(dim);
        }
        vec lBounds() const { return mxLow; }
        vec uBounds() const { return mxUp; }
        int dim() const { return mDim; }
        double min() const { return mMinimum; }
        vec optimum() const { return mOptimum; }
        std::string name() const { return mName; }
        double eval(vec x) const {
            double total = 0.0;
            for(int i=0; i < mDim; i++) {
                for(int j=1; j < 6; j++) {
                    total += j*sin((j+1)*x(i)) + j;
                }
            }
            return total;
        }
    };
    
    // 22
    class SineEnvelope : public Problem {
    protected:
        int mDim;
        vec mxLow;
        vec mxUp;
        vec mOptimum;
        double mMinimum = 0.0;
        std::string mName = "Sine Envelope";
    public:
        SineEnvelope(int dim) {
            mDim = dim;
            mOptimum = arma::zeros<vec>(dim);
            mxLow = -20.0 * arma::ones<vec>(dim);
            mxUp = 20.0 * arma::ones<vec>(dim);
        }
        vec lBounds() const { return mxLow; }
        vec uBounds() const { return mxUp; }
        int dim() const { return mDim; }
        double min() const { return mMinimum; }
        vec optimum() const { return mOptimum; }
        std::string name() const { return mName; }
        double eval(vec &x) const {
            double total = 0.0;
            for(int i=0; i < mDim-1; i++) {
                total += (pow(sin(sqrt(x(i)*x(i)+x(i+1)*x(i+1))-0.5),2))/
                        (pow(0.001*(x(i)*x(i)+x(i+1)*x(i+1))+1,2))+0.5;
            }
            return total;
        }
    };
    
    // 23
    class Schoen : public Problem {
    protected:
        int mDim;
        vec mxLow;
        vec mxUp;
        vec mOptimum;
        double mMinimum = 0.0;
        std::string mName;        
        int mk;
        vec mf;
        vec mAlpha;
        mat mz;
    public:
        Schoen(int dim) : Schoen(dim, fmax(2^dim, 500)) {}
        Schoen(int dim, int k) {
            mName = "Schoen, k = " + std::to_string(k);
            mDim = dim;
            mk = k;
            mxLow = arma::zeros<vec>(dim);
            mxUp = arma::ones<vec>(dim);
            mf = 100 * arma::randu<vec>(k);
            mf(0) = 0;
            mAlpha = 2 + arma::randu<vec>(k);
            mz = arma::randu<mat>(dim, k);
            mOptimum = this->mz.col(0);
        }
        vec lBounds() const { return mxLow; }
        vec uBounds() const { return mxUp; }
        int dim() const { return mDim; }
        double min() const { return mMinimum; }
        vec optimum() const { return mOptimum; }
        std::string name() const { return mName; }
        double eval(vec &x) const {            
            long double num = 0, den = 0, prodval = 0;
            for(int i=0; i < mk; i++) {
                prodval = pow(sqrt(arma::sum(arma::square(mz.col(i) - x))), mAlpha(i));
                num += mf(i) / prodval;
                den += 1 / prodval;
            }
            return num/den;
        }
    };
    
    // 24
    class CosineMixture : public Problem {
    protected:
        int mDim;
        vec mxLow;
        vec mxUp;
        vec mOptimum;
        double mMinimum;
        std::string mName = "Cosine Mixture";
    public:
        CosineMixture(int dim) {
            mDim = dim;
            mOptimum = arma::zeros<vec>(dim);
            mxLow = -1.0 * arma::ones<vec>(dim);
            mxUp = 1.0 * arma::ones<vec>(dim);
            mMinimum = -0.1 * dim;
        }
        vec lBounds() const { return mxLow; }
        vec uBounds() const { return mxUp; }
        int dim() const { return mDim; }
        double min() const { return mMinimum; }
        vec optimum() const { return mOptimum; }
        std::string name() const { return mName; }
        double eval(vec x) const {
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
        std::shared_ptr<Problem> mProblem;
        vec mTranslation;
        mat mRotation;
        int mDim;
        vec mxLow;
        vec mxUp;
        vec mOptimum;
        double mMinimum;
        std::string mName;
        
        void createTranslation() {
            mTranslation = arma::zeros<vec>(mDim);
            for(int i=0; i < mDim; i++) {
                mTranslation(i) = mxLow(i) + (mxUp(i) - mxLow(i)) * rand();
            }
        }
        void createRotation() {
            mat Q, R;
            mat X = arma::randn<mat>(mDim, mDim);
            arma::qr(Q, R, X);
            mRotation = Q;
        }
    public:
        RotatedProblem(std::shared_ptr<Problem>& problem) {
            if (problem->optimum().n_elem != problem->dim()) {
                throw std::logic_error("Optimum not specified for: " + problem->name() + " so can't create a rotated version");
            }
            mProblem = std::shared_ptr<Problem>(problem);
            mDim = problem->dim();
            mxLow = problem->lBounds();
            mxUp = problem->uBounds();
            mMinimum = problem->min();
            createTranslation();
            createRotation();
            mOptimum = mTranslation;
            mName = "Rotated + Translated " + problem->name();
        }
        mat roatation() const { return mRotation; }
        vec translation() const { return mTranslation; }
        vec lBounds() const { return mxLow; }
        vec uBounds() const { return mxUp; }
        int dim() const { return mDim; }
        double min() const { return mMinimum; }
        vec optimum() const { return mOptimum; }
        std::string name() const { return mName; }
        
        double eval(vec &x) const {
            x = mProblem->optimum() + mRotation * (x - mTranslation);
            return mProblem->eval(x);
        }
    };
}

#endif
