from abc import ABC, abstractmethod
import math
import numpy as np

class Chooser(ABC):
    '''
    Chooser consists of one iteration of a bayesian optimization algorithm.
    *Parameters:

    ** acquisition: The surrogate model to optimize. It is an instance of the
    AcquisitionFunc class. It could either be : Upper Confidence Bound or
    Expected Improvement.

    **optimizer : The algorithm to use to optimize the acquisition. It is an
    instance of Optimizer. It could either be LocalOptimizer (L-BFGS-B), or
    Differential Evolution

    **gp : The gaussian process that fits the objective.

    **bounds: An instance of Bounds.
    '''
    @abstractmethod
    def choose(self,acquisition,
                    optimizer,
                    gp,
                    domain,
                    past_evals,
                    n_restarts):
        pass


class MaxAcquisition(Chooser):
    def choose(self,acquisition,
                    optimizer,
                    gp,
                    domain,
                    past_evals,
                    n_restarts):
        minimum = math.inf
        argmin  = None
        for i in range(n_restarts):
            curr = optimizer.optimize(acquisition,
                                      gp,
                                      domain,
                                      past_evals)
            val  = -acquisition.eval(curr,gp,past_evals)
            if val <= minimum:
                argmin  = curr
                minimum = val
        #argmin = [bounds.bounds[i].toType(argmin[i]) for i in range(len(bounds.bounds))]
        argmin = domain.correctSample(argmin)
        return argmin

class ChooseAndReduceBounds(Chooser):
    pass
