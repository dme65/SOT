//
//  rbf2.h
//  Surrogate Optimization
//
//  Created by David Eriksson on 9/14/15.
//  Copyright (c) 2015 David Eriksson. All rights reserved.
//
//
// Internally works on the unit hypercube
//

#ifndef Surrogate_Optimization_rbf_h
#define Surrogate_Optimization_rbf_h

#include <cassert>
#include <iostream>
#include "common.h"
#include "utils.h"
#include "surrogate.h"

namespace sot {

    class CubicKernel {
    public:
        const int phi_zero = 0;
        const int order = 2;
        inline double eval(double dist) const { return dist * dist * dist; }
        inline double deriv(double dist) const { return 3 * dist * dist; }
        inline mat eval(const mat &dists) const { return dists % dists % dists; }
        inline mat deriv(const mat &dists) const { return 3 * dists % dists; }
    };
    
    class ThinPlateKernel {
    public:
        const int phi_zero = 0;
        const int order = 2;
        inline double eval(double dist) const { return dist * dist * log(dist + 1e-10);}
        inline double deriv(double dist) const { return dist * (1.0 + 2.0 * log(dist + 1e-10)); }
        inline mat eval(const mat &dists) const { return dists % dists % arma::log(dists + 1e-10); }
        inline mat deriv(const mat &dists) const { return dists % (1 + 2.0 * arma::log(dists + 1e-10)); }
    };
    
    class LinearKernel {
    public:
        const int phi_zero = 0;
        const int order = 1;
        inline double eval(double dist) const { return dist; }
        inline double deriv(double dist) const { return 1.0; }
        inline mat eval(const mat &dists) const { return dists; }
        inline mat deriv(const mat &dists) const { return arma::ones<mat>(dists.n_rows, dists.n_cols); }
    };
    
    class LinearTail {
    public:
        const int degree = 1;
        inline mat eval(const mat &x) const { return arma::join_vert(arma::ones<mat>(1, x.n_cols), x); }
        inline vec eval(const vec &x) const {
            vec tail = arma::zeros<vec>(x.n_rows + 1);
            tail(0) = 1;
            tail.tail(x.n_rows) = x;
            return tail;
        }
        inline mat deriv(const mat &x) const { return arma::join_vert(arma::zeros<mat>(1, x.n_rows), arma::eye<mat>(x.n_rows, x.n_rows)); }
        inline int n_tail(int d) const { return 1 + d; }
    };
    
    class ConstantTail {
    public:
        const int degree = 0;
        inline mat eval(const mat &x) const { return arma::ones<mat>(x.n_rows, 1); }
        inline vec eval(const vec &x) const { return arma::ones<mat>(1, 1); }
        inline mat deriv(const mat &x) const { return arma::zeros<mat>(x.n_rows, 1); }
        inline int n_tail(int d) const { return 1; }
    };
    
    template<class Kernel, class Tail>
    class RBFInterpolant : public Surrogate {
        
    protected:
        Kernel kernel;
        Tail tail;
        mat L, U;
        uvec p;
        vec coeffs;
        vec F;
        mat centers;
        int max_points;
        int ntail;
        int num_points;
        bool dirty;
        vec xlow, xup;
        double eta = 1e-6;
        int d;
        double norm_budget = 0;
        
    public:
        
        RBFInterpolant(int max_points, int d, vec xlow, vec xup, double eta) :
            RBFInterpolant(max_points, d, xlow, xup) {
            this->eta = eta;
        }
        
        RBFInterpolant(int max_points, int d, vec xlow, vec xup) {
            this->max_points = max_points;
            this->d = d;
            this->ntail = tail.n_tail(d);
            this->num_points = 0;
            this->centers = arma::zeros<mat>(d, max_points);
            this->L = arma::zeros<mat>(max_points + ntail, max_points + ntail);
            this->U = arma::zeros<mat>(max_points + ntail, max_points + ntail);
            this->p = arma::zeros<uvec>(max_points + ntail);
            this->F = arma::zeros<vec>(max_points + ntail);
            this->coeffs = vec(max_points + ntail);
            this->dirty = false;
            this->xlow = xlow;
            this->xup = xup;
            
            if (not (kernel.order-1 <= tail.degree)) {
                throw std::logic_error("Kernel and tail mismatch");
            }
        }
        
        int dim() const {
            return this->d;
        }
        
        // Reset RBF
        void reset() {
            num_points = 0;
        }
        
        // Number of points
        int npts() {
            return this->num_points;
        }
        
        // Return points
        mat get_X() const {
            return FromUnitBox((mat)centers.cols(0, num_points - 1), xlow, xup);
        }
        
        // Return point
        vec get_X(int i) const {
            return FromUnitBox((mat)centers.col(i), xlow, xup);
        }
        
        // Return function values
        vec get_fX() const {
            return F.rows(ntail, ntail + num_points - 1);
        }
        
        // Return function value
        double get_fX(int i) const {
            return F(ntail+i);
        }
        
        vec get_coeffs() {
            if(dirty) { throw std::logic_error("RBF not updated"); }
            return coeffs;
        }
       
        // Fit the RBF
        void fit() {
            if(num_points < ntail) { throw std::logic_error("Not enough points"); }       
            if (dirty) {
                int n = num_points + ntail;
                coeffs = arma::solve(arma::trimatl(L(arma::span(0, n - 1), arma::span(0, n - 1))), F(p(arma::span(0, n - 1))));
                coeffs = arma::solve(arma::trimatu(U(arma::span(0, n - 1), arma::span(0, n - 1))), coeffs);
                dirty = false;
            }
        }
        
        // Set points
        void set_points(const mat &ppoints, const vec &fun_vals) {
            
            // Map point to be in the unit box
            mat points = ToUnitBox(ppoints, xlow, xup);
            
            num_points = (int)points.n_cols;
            int n = num_points + ntail;
            if(num_points < d + 1) { 
                throw std::logic_error("Not enough points"); 
            }
            mat px = tail.eval(points);
            mat phi = kernel.eval(arma::sqrt(SquaredPairwiseDistance(points, points)));
                    
            mat A = arma::zeros<mat>(n, n);
            A(arma::span(ntail, n - 1), arma::span(0, ntail - 1)) = px.t();
            A(arma::span(0, ntail - 1), arma::span(ntail, n - 1)) = px;
            A(arma::span(ntail, n - 1), arma::span(ntail, n - 1)) = phi;
            F.rows(ntail, n - 1) = fun_vals;
            
            // REGULARIZATION
            A += eta*arma::eye(n, n);
            
            // Compute the initial LU factorization of A
            mat LL, UU, P;
            arma::lu(LL, UU, P, A);
            
            L(arma::span(0, n - 1), arma::span(0, n - 1)) = LL;
            U(arma::span(0, n - 1), arma::span(0, n - 1)) = UU;
            
            // Convert P to a permutation vector
            for(int i = 0; i < n; i++) {
                uvec temp = find(P.row(i) > 0.5);
                p(i) = temp(0);
            }
            
            centers.cols(0, num_points - 1) = points;
            dirty = true;
        }
        
        // Add one point
        void add_point(const vec &ppoint, double fun_val) {
            if(num_points == 0) {
                std::cout << "You need to use set_points first" << std::endl;
                abort();
            }

            // Map point to be in the unit box
            vec point = ToUnitBox(ppoint, xlow, xup);
            
            int nact = ntail + num_points;
            if(num_points + 1 > max_points) { 
                throw std::logic_error("Capacity exceeded"); 
            }
      
            vec vx = arma::join_vert(tail.eval(point), kernel.eval(arma::sqrt(SquaredPointSetDistance(point, centers.cols(0, num_points - 1)))));
            vec u12 = arma::solve(arma::trimatl(L(arma::span(0, nact - 1), arma::span(0, nact - 1))), vx.rows(p.head(nact)));
            vec l21 = (arma::solve(arma::trimatl(U(arma::span(0, nact - 1), arma::span(0, nact - 1)).t()), vx));
            double u22 = kernel.phi_zero + eta - arma::dot(u12, l21);

            L(nact, arma::span(0, nact - 1)) = l21.t();
            L(nact, nact) = 1;
            U(arma::span(0, nact - 1), nact) = u12;
            U(nact, nact) = u22;
            p(nact) = nact;

            // Update F and add the centers
            F(nact) = fun_val;
            centers.col(num_points) = point;
            num_points++;
            
            dirty = true;
        }
        
        void add_points(const mat &ppoints, const vec &fun_vals) {
            if(num_points == 0) {
                return set_points(ppoints, fun_vals);
            }
            
            // Map point to be in the unit box
            mat points = ToUnitBox(ppoints, xlow, xup);
            
            int nact = ntail + num_points;
            int n_pts = (int)fun_vals.n_rows;
            if(n_pts < 2) { throw std::logic_error("Use add_point instead"); }
            if(num_points + n_pts > max_points) { throw std::logic_error("Capacity exceeded"); }
            
            auto px = tail.eval(points);
            
            mat B = arma::zeros(nact, n_pts);
            B.rows(arma::span(0, d)) = px;
            B.rows(ntail, nact - 1) = kernel.eval(arma::sqrt(SquaredPairwiseDistance<mat>(centers.cols(0, num_points - 1), points)));
            mat K = kernel.eval(arma::sqrt(SquaredPairwiseDistance<mat>(points, points)));
            
            // REGULARIZATION
            K += eta*arma::eye(n_pts, n_pts);

            // Update the LU factorization
            mat U12 = arma::solve(arma::trimatl(L(arma::span(0, nact - 1), arma::span(0, nact - 1))), B.rows(p.head(nact)));
            mat L21 = (arma::solve(arma::trimatl(U(arma::span(0, nact - 1), arma::span(0, nact - 1)).t()), B)).t();
            mat C;
            try {
                C = arma::chol(K - L21*U12, "lower");
            }
            catch (std::runtime_error) {
                std::cout << "Warning: Cholesky factorization failed, computing new LU from scratch..." << std::endl;
                // Add new points
                F.rows(nact, nact + n_pts - 1) = fun_vals;
                centers.cols(num_points, num_points + n_pts - 1) = points;
                num_points += n_pts;
                // Build LU from scratch
                set_points(get_X(), F.rows(ntail, nact + n_pts - 1));
                return;
            }
            L(arma::span(nact, nact + n_pts - 1), arma::span(0, nact - 1)) = L21;
            L(arma::span(nact, nact + n_pts - 1), arma::span(nact, nact + n_pts - 1)) = C;
            U(arma::span(0, nact - 1), arma::span(nact, nact + n_pts - 1)) = U12;
            U(arma::span(nact, nact + n_pts - 1), arma::span(nact, nact + n_pts - 1)) = C.t();
            p.rows(arma::span(nact, nact + n_pts - 1)) = arma::linspace<uvec>(nact, nact + n_pts - 1, n_pts);
            
            // Update F and add the centers
            F.rows(nact, nact + n_pts - 1) = fun_vals;
            centers.cols(num_points, num_points + n_pts - 1) = points;
            num_points += n_pts;
            
            dirty = true;
        }
        
        // Evaluate one point without provided distance
        double eval(const vec &ppoint) const {
            if(dirty) { throw std::logic_error("RBF not updated"); }       

            // Map point to be in the unit box
            vec point = ToUnitBox(ppoint, xlow, xup);
            
            assert(not dirty);
            vec px = tail.eval(point);
            vec phi = kernel.eval(arma::sqrt(SquaredPointSetDistance(point, (mat)centers.cols(0, num_points - 1))));
            vec c = coeffs.head(num_points + ntail);
            return arma::dot(c.head(ntail), px) + arma::dot(c.tail(num_points), phi);
        }
        
        // Evaluate one point with provided distance
        double eval(const vec &ppoint, const vec &dists) const {
            if(dirty) { throw std::logic_error("RBF not updated"); }       

            // Map point to be in the unit box
            vec point = ToUnitBox(ppoint, xlow, xup);
            
            assert(not dirty);
            vec px = tail.eval(point);
            vec phi = kernel.eval(dists);
            vec c = coeffs.head(num_points + ntail);
            return arma::dot(c.head(ntail), px) + arma::dot(c.tail(num_points), phi);
        }
        
        // Evaluate multiple points
        vec evals(const mat &ppoints) const {
            if(dirty) { throw std::logic_error("RBF not updated"); }       

            // Map point to be in the unit box
            mat points = ToUnitBox(ppoints, xlow, xup);
            
            assert(not dirty);
            mat px = tail.eval(points);
            mat phi = kernel.eval(arma::sqrt(SquaredPairwiseDistance((mat)centers.cols(0, num_points - 1), points)));
            vec c = coeffs.head(num_points + ntail);
            return px.t() * c.head(ntail) + phi.t() * c.tail(num_points);
        }
        
        // Evaluate multiple points with provided distances
        vec evals(const mat &ppoints, const mat &dists) const {
            if(dirty) { throw std::logic_error("RBF not updated"); }       

            // Map point to be in the unit box
            mat points = ToUnitBox(ppoints, xlow, xup);
            
            assert(not dirty);
            mat px = tail.eval(points);
            mat phi = kernel.eval(dists);
            vec c = coeffs.head(num_points + ntail);
            return px.t() * c.head(ntail) + phi.t() * c.tail(num_points);
        }
        
        // Gradient of the surface at the current point
        vec deriv(const vec &ppoint) const {
            if(dirty) { throw std::logic_error("RBF not updated"); }       

            // Map point to be in the unit box
            vec point = ToUnitBox(ppoint, xlow, xup);
            
            assert(not dirty);
            mat dpx = tail.deriv(point);
            vec c = coeffs.head(num_points + ntail);
            vec dists = arma::sqrt(SquaredPairwiseDistance<mat>(centers.cols(0, num_points - 1), point));
            dists.elem(arma::find(dists < 1e-10)).fill(1e-10);
            mat dsx = - centers.cols(0, num_points - 1);
            dsx.each_col() += point;
            dsx.each_row() %= (kernel.deriv(dists) % c.tail(num_points)/dists).t();
            return arma::sum(dsx, 1) + dpx.t() * c.head(ntail);
        }
    };

    template<class Kernel, class Tail>
    class RBFInterpolantCap : public RBFInterpolant<Kernel,Tail> {
        public:
        RBFInterpolantCap(Kernel kernel,Tail tail,int max_points, int d, vec xlow, vec xup) :
            RBFInterpolant<Kernel,Tail>(kernel, tail, max_points, d, xlow, xup) {}
        void fit() {
            assert(this->num_points > this->d);
            if (this->dirty) {
                int nact = this->num_points + this->ntail;
                vec ff = this->F.head(nact);
                double medf = arma::median(ff.rows(this->ntail, nact-1));
                for(int i=this->ntail;i<nact;i++) {
                    if (ff(i) > medf) { ff(i) = medf; }
                }
                this->coeffs = arma::solve(arma::trimatl(this->L(arma::span(0, nact - 1), arma::span(0, nact - 1))), ff(this->p));
                this->coeffs = arma::solve(arma::trimatu(this->U(arma::span(0, nact - 1), arma::span(0, nact - 1))), this->coeffs);
                this->dirty = false;
            }
        }
    };
}

#endif
