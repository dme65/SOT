//
//  Optimizer.h
//  Surrogate Optimization
//
//  Created by David Eriksson on 2/5/16.
//  Copyright © 2016 David Eriksson. All rights reserved.
//

#ifndef Optimizer_h
#define Optimizer_h

#include "test_problems.h"

namespace sot {
    
    class Optimizer {
    protected:
        Problem *data;
        ExpDesign *exp_des;
        int maxeval;
        int numeval;
        int initp;
        int dim;
        std::string my_name;
    public:
        virtual Result run() = 0;
        std::string name() { return this->my_name; }
    };
}

#endif /* Optimizer_h */
