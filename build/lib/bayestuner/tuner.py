import numpy as np
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from .acquisitionfunc import UCB, EI
from .optimizer import DifferentialEvolution,LBFGSB, OptimizerResult
from .chooser import MaxAcquisition
import seaborn as sns
import matplotlib.pyplot as plt
from .bounds import Domain
from .initialization import Normal,Uniform
from scipy.optimize import differential_evolution
import math
import copy


class BayesTuner :
    '''
    BayesTuner is the main component of ....

    Attributes
    ----------

    objective : function
        Real valued function to maximize.

    bounds : list
        size : n_features.
        List of tuples. Each tuple specifies a dimension of the input space.
        A dimension in characterized by : lower bound, upper bound, type.
        Type is either 'continuous' if the restriction of the input space to the
        dimension is a continuous domain, or 'discrete'. discrete means a set of
        integers spanning [lower bound, upper bound].
        e.g. : [(-10,12,'continuous'),(2,10,'discrete')] if the objective has both
        continuous and discrete hyperparameters.
        Note that if the hyperparameters are discrete but not integers, you can
        always transform them to integers.

    n_iter : int
        Number of iterations.

    init_samples : int
        Onitial number of samples to use for the fitting of the gaussian process.

    optimizer : Optimizer, optional
        Optimizer to use for the maximization of the surrogate model.
        Available optimizers: 'L-BFGS-B' or 'DifferentialEvolution'

    acquisition : AcquisitionFunc, optional
        The surrogate model.
        Available surrogates: 'Upper Confidence Bound' or 'ExpectedImprovement'.
        Default is 'Upper Confidence Bound' with beta_t = sqrt(log(t)).

    chooser : Chooser, optional
        The way you choose the next point where you evaluate the objective.
        The default chooser is the one that chooses the maximum of the surrogate.

    initialization : Initialization, optional
        The distribution to sample from.
        Default is using the gaussian distribution.

    kernel : Kernel, optional
        The kernel to use for the gaussian process regression.
        Default is ConstantKernel * Matern(nu = 2.5)

    alpha : float, optional
        Value added to the diagonal of the kernel matrix during fitting.
        Larger values correspond to increased noise level in the observations.
        This can also prevent a potential numerical issue during fitting, by
        ensuring that the calculated values form a positive definite matrix.
        Default : 1e-2.

    n_restarts : int, optional
        Number of restarts of the surrogate optimizer. Default : 5.


    Methods
    -------

    tune(verbose = True)
        Optimizes the objective using Bayesian Optimization.

    '''
    def __init__(self,
                 objective,
                 bounds,
                 n_iter,
                 init_samples,
                 optimizer = LBFGSB(),
                 acquisition = lambda i : UCB(i, lambda x : np.log(x)),
                 chooser = MaxAcquisition(),
                 initialization = Uniform(),
                 kernel = ConstantKernel(1.0)*Matern(nu = 2.5),
                 alpha = 1e-2,
                 n_restarts = 5):
        """
        Attributes
        ----------

        domain : Domain
            A description of the input space.

        objective : function
            Real valued function to maximize.

        bounds : list
            List of tuples. Each tuple specifies a dimension of the input space.
            A dimension in characterized by : lower bound, upper bound, type.
            Type is either 'continuous' if the restriction of the input space to the
            dimension is a continuous domain, or 'discrete'. discrete means a set of
            integers spanning [lower bound, upper bound].
            e.g. : [(-10,12,'continuous'),(2,10,'discrete')] if the objective has both
            continuous and discrete hyperparameters.
            Note that if the hyperparameters are discrete but not integers, you can
            always transform them to integers.

        n_iter : int
            Number of iterations.

        init_samples : int
            Onitial number of samples to use for the fitting of the gaussian process.

        past_hyper : array-like
            initial shape : (init_samples,n_features)
            Contains all the hyperparemeters visited throughout the bayesian optimization of
            the objective. It initially contains the first sampled hyperparemeters.

        past_evals : array-like
            initial shape : (init_samples,1)
            Contains the scores of the hyperparemeters visited throughout the bayesian
            optimization of the objective. Initially contains the scores of the first
            sampled hyperparemeters.

        optimizer : Optimizer object, optional (default: LBFGSB)
            Optimizer to use for the maximization of the surrogate model.
            Available optimizers: 'L-BFGS-B' or 'DifferentialEvolution'

        acquisition : AcquisitionFunc object, optional (default: UCB)
            The surrogate model.
            Available surrogates: 'Upper Confidence Bound' or 'ExpectedImprovement'.
            Default is 'Upper Confidence Bound' with beta_t = sqrt(log(t)).

        chooser : Chooser, optional (default: MaxAcquisition )
            The way you choose the next point where you evaluate the objective.

        initialization : Initialization, optional (default: Normal)
            The distribution to sample from.

        kernel : Kernel, optional (default : ConstantKernel * Matern(nu = 2.5) )
            The kernel to use for the gaussian process regression.

        alpha : float, optional (default: 1e-2)
            Value added to the diagonal of the kernel matrix during fitting.
            Larger values correspond to increased noise level in the observations.
            This can also prevent a potential numerical issue during fitting, by
            ensuring that the calculated values form a positive definite matrix.

        n_restarts : int, optional (default: 5)
            Number of restarts of the surrogate optimizer.
        """

        self.domain         =   Domain(bounds)
        self.objective      =   objective
        self.n_iter         =   n_iter
        self.init_samples   =   init_samples
        self.initialization =   initialization
        self.past_hyper     =   initialization.generate(self.domain,init_samples)
        self.past_evals     =   np.array([objective(x) for x in self.past_hyper]).reshape(-1,1)
        self.optimizer      =   optimizer
        self.acquisition    =   acquisition
        self.chooser        =   chooser
        self.kernel         =   kernel
        self.alpha          =   alpha
        self.n_restarts     =   n_restarts
        self.gp             =   GaussianProcessRegressor(kernel = self.kernel,
                                                         alpha = self.alpha,
                                                         n_restarts_optimizer = 3,
                                                         normalize_y = True).fit(self.past_hyper,self.past_evals)
        self.gps            =   [copy.copy(self.gp)]

    def tune(self,verbose = False):
        """
        Performs a bayesian optimization of the objective function.

        Parameters
        ----------

        verbose : bool, optional (default: False)
            whether to print the current iteration, the chosen point, its image
            and the best point / image found yet.

        Returns
        -------

        OptimizerResult
            Object that contains relevant information about the optimization.
            * OptimizerResult.x to get the argmax
            * OptimizerResult.func_val to get the value of the maximum found.
            * OptimizerResult.PastEvals to get the visited hyperparemeters.
            * OptimizerResult.Scores to get the scores of the visited hyperparemeters.
        """
        '''def optimizer(obj_func, initial_theta, bounds):
            obj = lambda theta : obj_func(theta,eval_gradient = False)
            bounds = np.array([[0,7],[0,10]])
            res = differential_evolution(obj,bounds)
            theta_opt = res.x
            func_min  = res.fun
            return theta_opt, func_min'''
        idx_best_yet = np.argmax(self.past_evals)
        best_yet     = self.past_evals[idx_best_yet]
        for i in range(1,self.n_iter):
            next_eval = self.chooser.choose(self.acquisition(i),
                                self.optimizer,
                                self.gp,
                                self.domain,
                                self.past_evals,
                                self.n_restarts)
            score_next_eval = self.objective(next_eval)
            if score_next_eval >= best_yet:
                best_yet = score_next_eval
                idx_best_yet = i
            if verbose == True:
                print(f"{i} / {self.n_iter} | current eval : {next_eval} / score : {score_next_eval} |\n \
-> best score yet: {best_yet} \n")
            self.past_hyper = np.vstack((self.past_hyper,next_eval))
            self.past_evals = np.vstack((self.past_evals,score_next_eval))
            self.gp.fit(self.past_hyper,self.past_evals)
            self.gps.append(copy.copy(self.gp))
        idx_argmax = np.argmax(self.past_evals)
        argopt  = self.past_hyper[idx_argmax]
        optimum = self.past_evals[idx_argmax]
        result  = OptimizerResult(optimum,argopt,self.past_hyper,self.past_evals)
        return result

    def supertuner(self,runs, verbose = False):
        self.n_iter = runs[0]
        self.tune(verbose)
        for run in runs[1:-1]:
            print(f'***New run: number of calls : {run}')
            grid = [(bound[0],bound[1],bound[2]/2) for bound in self.domain.bounds]
            self.domain = Domain(grid)
            self.n_iter = run
            self.tune(verbose)
        idx_argmax = np.argmax(self.past_evals)
        argopt  = self.past_hyper[idx_argmax]
        last_domain = [(argopt_ - (bound[1]-bound[0])/5,argopt_ + (bound[1]-bound[0])/5,0) \
                        for (argopt_,bound) in zip(argopt,self.domain.bounds)]
        self.domain = Domain(last_domain)
        self.n_iter = runs[-1]
        self.tune(verbose)
        idx_argmax = np.argmax(self.past_evals)
        argopt  = self.past_hyper[idx_argmax]
        optimum = self.past_evals[idx_argmax]
        result  = OptimizerResult(optimum,argopt,self.past_hyper,self.past_evals)
        return result





    '''def plot_progress(self):
        sns.set_style("darkgrid")
        if len(self.bounds.bounds) > 1:
            raise ValueError("Can't plot for dimensions > 1")
        gp = GaussianProcessRegressor(kernel = self.kernel,
                                      alpha = self.alpha,
                                      n_restarts_optimizer = self.n_restarts)
        extract_bound = np.array(self.bounds.bounds[0].interval)
        space = np.linspace(extract_bound[0],extract_bound[1],1000)
        list_axes = []
        for i in range(self.n_iter):
            if i%5 == 0:
                figure, ax = plt.subplots(nrows = 5,ncols = 2,figsize = (10,20))
                list_axes.append(ax)
            plt.subplots_adjust(hspace = 0.5)
            ax[i%5][0].set_title(f"Iteration {i}")
            ax[i%5][0].plot(space,[self.objective(x) for x in space])
            ax[i%5][1].set_title("UCB")
            ax[i%5][1].plot(space,[self.acquisition(i).eval(x,
                                                         gp,
                                                         self.past_evals) for x in space],'r')
            ax[i%5][0].plot(self.past_hyper,self.past_evals,'gD',markersize = 6)
            next_eval = self.chooser.choose(self.acquisition(i),
                                self.optimizer,
                                gp,
                                self.bounds,
                                self.past_evals,
                                self.n_restarts)
            obj_val   = self.objective(next_eval)
            ax[i%5][0].plot(next_eval,obj_val,'ro',markersize = 6)
            self.past_hyper = np.vstack((self.past_hyper,next_eval))
            self.past_evals = np.vstack((self.past_evals,obj_val))
            gp.fit(self.past_hyper,self.past_evals)
        plt.show()'''
