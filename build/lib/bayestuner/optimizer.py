from abc import ABC, abstractmethod
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
import math
import numpy as np



class OptimizerResult():
    """
    A holder of the optimization result.

    Attributes
    ----------

    func_val : float
        The best objective value found throughout the optimization.

    x : numpy.ndarray
        shape = (n_features,1)
        the argument of func_val

    PastEvals : numpy.ndarray
        shape = (num_iter + init_samples,n_features)
        The visited hyperparameters throughout the optimizaiton.

    Scores: numpy.ndarray
        shape = (num_iter + init_samples,1)
        Scores associated to the visited hyperparameters.

    Methods
    -------

    __str__
        Displays information about the optimization.
    """
    def __init__(self,func_val,x,past_hyper,past_evals):
        """
        Parameters
        ----------

        func_val : float
            The best objective value found throughout the optimization.

        x : numpy.ndarray
            shape = (n_features,1)
            the argument of func_val

        PastEvals : numpy.ndarray
            shape = (num_iter + init_samples,n_features)
            The visited points throughout the optimizaiton.

        Scores: numpy.ndarray
            shape = (num_iter + init_samples,1)
            Scores associated to the visited points.
        """
        self.func_val   = func_val
        self.x          = x
        self.PastEvals  = past_hyper
        self.Scores     = past_evals


    def __str__(self):
        """Displays information about the optimization."""

        result = "PastEvals and Scores : \n"
        for i in range(len(self.PastEvals)):
            result += f"hyperparameter {self.PastEvals[i]} -> score: {self.Scores[i]} \n"
        return f"func_val : {self.func_val} \nx : {self.x} \n{result}"


class Optimizer(ABC):
    """Abstract Class. An optimizer is used to maximize the acquisition (or surrogate)."""
    @abstractmethod
    def optimize(self,acquisition,gp,domain,past_evals):
        """Parameters
        ----------

        acquisition: AcquisitionFunc object
            The surrogate model.
            Available surrogates: 'Upper Confidence Bound' or 'ExpectedImprovement'.
            Default is 'Upper Confidence Bound' with beta_t = sqrt(log(t)).

        gp : GaussianProcessRegressor object
            The gaussian process that fits the model at each iteration.

        domain : Domain object
            A description of the input space.

        past_evals: array-like
            hyperparameters visited


        Returns
        -------

        array-like
            The result of the maximization of the surrogate.
        """
        pass


class DifferentialEvolution(Optimizer):
    """
    The differential evolution algorithm. A good global optimizer but
    """
    def optimize(self,acquisition,gp,domain,past_evals):

        extracted_bounds = list(map(lambda x : [x[0],x[1]],domain.bounds))
        def min_surrogate(x):
            return -acquisition.eval(x,gp,past_evals)
        return differential_evolution(min_surrogate,extracted_bounds).x



class LBFGSB(Optimizer):
    "The L-BFGS-B algorithm"
    def optimize(self,acquisition,gp,domain,past_evals):
        extracted_bounds = list(map(lambda x : [x[0],x[1]],domain.bounds))
        def min_surrogate(x):
            return -acquisition.eval(x,gp,past_evals)
        return minimize(min_surrogate,
                        x0=[np.random.uniform(x,y) for x,y in extracted_bounds],
                        bounds=extracted_bounds,
                        method='L-BFGS-B').x
