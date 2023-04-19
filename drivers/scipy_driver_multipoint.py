#  Copyright 2019-2020, Pedro Gomes.
#
#  This file is part of FADO.
#
#  FADO is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  FADO is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with FADO.  If not, see <https://www.gnu.org/licenses/>.

import os
import time
import numpy as np
from drivers.constrained_optim_driver import ConstrainedOptimizationDriver


class ScipyDriverMultipoint(ConstrainedOptimizationDriver):
    """
    Driver to use with the SciPy optimizers, especially the constrained ones.
    """
    def __init__(self):
        ConstrainedOptimizationDriver.__init__(self)

        # list of constraintsi and variable bounds
        self._constraints = []
        self._bounds = []
        self._regularize = False
        self._regfactor = 1.0
    #end

    def preprocess(self):
        """
        Prepares the optimization problem, including preprocessing variables,
        and setting up the lists of constraints and variable bounds that SciPy
        needs. Must be called after all functions are added to the driver.
        """
        ConstrainedOptimizationDriver.preprocess(self)

        class _fun:
            def __init__(self,fun,idx,idx_range=0):
                self._f = fun
                self._i = idx
                self._idx_range = idx_range
            def __call__(self,x):
                return self._f(x,self._i,self._idx_range)
        #end

        # setup the constraint list, the callbacks are the same for all
        # constraints, an index argument (i) is used to distinguish them.
        self._constraints = []
        n_cons = 0
        i = 0
        while ( i < self._nCon ):
            if (i>= len(self._constraintsEQ) and self._constraintsGT[i-len(self._constraintsEQ)].multiCounter>0):
                save_idx = i
                multi_idx = 0
                while (i < self._nCon-1 and (self._constraintsGT[i-len(self._constraintsEQ)+1].multiCounter==self._constraintsGT[save_idx-len(self._constraintsEQ)].multiCounter)):
                    multi_idx = multi_idx+1
                    i=i+1
                self._constraints.append({'type' : ('ineq','eq')[i<len(self._constraintsEQ)],
                                      'fun' : _fun(self._eval_g,save_idx, multi_idx),
                                      'jac' : _fun(self._eval_jac_g,save_idx, multi_idx)})
                n_cons = n_cons +1
                
            else:    
                self._constraints.append({'type' : ('ineq','eq')[i<len(self._constraintsEQ)],
                                      'fun' : _fun(self._eval_g,i),
                                      'jac' : _fun(self._eval_jac_g,i)})
                n_cons = n_cons +1
            i=i+1
        #end

        # variable bounds
        self._bounds = np.array((self.getLowerBound(),self.getUpperBound()),float).transpose()

        # size the gradient and constraint jacobian
        self._grad_f = np.zeros((self._nVar,))
        self._old_grad_f = np.zeros((self._nVar,))
        self._jac_g = np.zeros((self._nVar,n_cons))
        self._old_jac_g = np.zeros((self._nVar,n_cons))
    #end

    def getConstraints(self):
        """Returns the constraint list that can be passed to SciPy."""
        return self._constraints

    def getBounds(self):
        """Return the variable bounds in a format compatible with SciPy."""
        return self._bounds

    def fun(self, x):

        """Method passed to SciPy to get the objective function value."""
        # Evaluates all functions if necessary.
        self._evaluateFunctions(x)
        sumObj = 0.0
        objCoun = 0
        for obj in self._ofval:
            if(self._objectives[objCoun].isTarget):
                sumObj+= (obj-self._objectives[objCoun].target*self._objectives[objCoun].scale)**2/self._objectives[objCoun].scale
            else:
                sumObj+= obj
            self._logObj.write(str(obj)+" ")
            objCoun = objCoun+1
        if self._regularize:
            l2norm_x = np.linalg.norm(x)
            sumObj += self._regfactor*ls2norm_x**2
        if self._logObj is not None:
            self._logObj.write(str(sumObj)+"\n")
        return sumObj
    #end

    def grad(self, x):
        """Method passed to SciPy to get the objective function gradient."""
        # Evaluates gradients and functions if necessary, otherwise it
        # simply combines and scales the results.    
        self._jacTime -= time.time()
        try:
            self._evaluateGradients(x)

            os.chdir(self._workDir)

            self._grad_f[()] = 0.0
            for obj in self._objectives:
                if(obj.isTarget):
                    self._grad_f += 2*(obj.function.getValue()-obj.target)*obj.function.getGradient(self._variableStartMask) * obj.scale
                else:
                    self._grad_f += obj.function.getGradient(self._variableStartMask) * obj.scale
            if self._regularize :
                self._grad_f += 2*self._regfactor*x
            self._grad_f /= self._varScales

            # keep copy of result to use as fallback on next iteration if needed
            self._old_grad_f[()] = self._grad_f
        except:
            if self._failureMode == "HARD": raise
            self._grad_f[()] = self._old_grad_f
        #end

        if not self._parallelEval:
            self._runAction(self._userPostProcessGrad)

        self._jacTime += time.time()
        os.chdir(self._userDir)

        return self._grad_f
    #end

    # Method passed to SciPy to expose the constraint vector.
    def _eval_g(self, x, idx, idx_range=0):
        self._evaluateFunctions(x)
        out = 0
        org_idx = idx

        while (idx <= org_idx+idx_range ):
            if idx < len(self._constraintsEQ):
                out = self._eqval[idx]
            else:
                useidx = idx-len(self._constraintsEQ)
                cons = (self._gtval[useidx])/self._constraintsGT[useidx].scale+self._constraintsGT[useidx].bound
                self._logObj.write(str(cons)+" ")
                if (self._constraintsGT[useidx].isTarget):
                    cons = (cons - self._constraintsGT[useidx].target)**2
                out = out + cons
            idx = idx+1   
        #end
        self._logObj.write(str(out)+"\n")
        out = (out - self._constraintsGT[idx-1].bound)*self._constraintsGT[idx-1].scale
        return out
    #end

    # Method passed to SciPy to expose the constraint Jacobian.
    def _eval_jac_g(self, x, idx, idx_range=0):
        self._jacTime -= time.time()
        try:
            self._evaluateGradients(x)

            os.chdir(self._workDir)

            mask = self._variableStartMask
            
            self._jac_g[:,idx] = 0.0
            f = 0.0
            org_idx = idx
          
            while(idx <= org_idx+idx_range):
                if idx < len(self._constraintsEQ):
                    con = self._constraintsEQ[idx]
                    f = -1.0 # for purposes of lazy evaluation equality is always active
                    self._jac_g[:,org_idx] += con.function.getGradient(mask)
                else:
                    useidx = idx-len(self._constraintsEQ)
                    con = self._constraintsGT[useidx]
                    consval = (self._gtval[useidx])/con.scale+con.bound
                    consgrad = 1.
                    if (con.isTarget):
                        consgrad = (consval - con.target)*2
                        consval = (consval - con.target)**2
                    f = f + consval
                    self._jac_g[:, org_idx] += con.function.getGradient(mask)*consgrad
                #end
                idx = idx + 1
            #end
            if f < 0.0 or not self._asNeeded:
                self._jac_g[:,org_idx] = self._jac_g[:,org_idx] * con.scale / self._varScales
            else:
                self._jac_g[:,org_idx] = 0.0
            #end

            # keep reference to result to use as fallback on next iteration if needed
            self._old_jac_g[:,org_idx] = self._jac_g[:,org_idx]
        except:
            if self._failureMode == "HARD": raise
            self._jac_g[:,org_idx] = self._old_jac_g[:,org_idx]
        #end

        if not self._parallelEval:
            self._runAction(self._userPostProcessGrad)

        self._jacTime += time.time()
        os.chdir(self._userDir)

        return self._jac_g[:,org_idx]
    #end

    def setRegularization(self, reg_factor):
        self._regularize = True
        self._regfactor = reg_factor
    #end
#end

