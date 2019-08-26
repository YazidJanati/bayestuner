from abc import ABC, abstractmethod
import math
import numpy as np

class Chooser(ABC):
    @abstractmethod
    def choose(self,acquisition,
                    optimizer,
                    gp,
                    bounds,
                    past_evals,
                    n_restarts):
        pass


class BasicChooser(Chooser):
    def choose(self,acquisition,
                    optimizer,
                    gp,
                    bounds,
                    past_evals,
                    n_restarts):
        minimum = math.inf
        argmin  = None
        for i in range(n_restarts):
            curr = optimizer.optimize(acquisition,
                                      gp,
                                      bounds,
                                      past_evals)
            val  = -acquisition.eval(curr,gp,past_evals)
            if val <= minimum:
                argmin  = curr
                minimum = val
        argmin = [bounds.bounds[i].toType(argmin[i]) for i in range(len(bounds.bounds))]
        return argmin

class ChooseAndReduceBounds(Chooser):
    pass
