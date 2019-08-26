from abc import ABC, abstractmethod
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
import math
import numpy as np



class OptimizerResult():
    def __init__(self,func_val,x,past_hyper,past_evals):
        self.func_val   = func_val
        self.x          = x
        self.PastEvals = past_hyper
        self.Scores = past_evals


    def __str__(self):
        result = "PastEvals and Scores : \n"
        for i in range(len(self.PastEvals)):
            result += f"hyperparameter {self.PastEvals[i]} -> score: {self.Scores[i]} \n"
        return f"func_val : {self.func_val} \nx : {self.x} \n{result}"


class Optimizer(ABC):
    @abstractmethod
    def optimize(self,acquisition,gp,bounds,past_evals):
        pass


class DifferentialEvolution(Optimizer):
    def optimize(self,acquisition,gp,bounds,past_evals):
        extracted_bounds = [bound.interval for bound in bounds.bounds]
        def min_surrogate(x):
            return -acquisition.eval(x,gp,past_evals)
        return differential_evolution(min_surrogate,extracted_bounds).x



class LocalOptimizer(Optimizer):
    def optimize(self,acquisition,gp,bounds,past_evals):
        extracted_bounds = [bound.interval for bound in bounds.bounds]
        next_loc   = None
        min_surrog = math.inf
        def min_surrogate(x):
            return -acquisition.eval(x,gp,past_evals)
        return minimize(min_surrogate,
                        x0=[np.random.uniform(x,y) for x,y in extracted_bounds],
                        bounds=extracted_bounds,
                        method='L-BFGS-B').x
