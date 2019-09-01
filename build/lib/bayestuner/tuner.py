import numpy as np
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from .acquisitionfunc import UCB, EI
from .optimizer import DifferentialEvolution,LocalOptimizer, OptimizerResult
from .chooser import BasicChooser
import seaborn as sns
import matplotlib.pyplot as plt
from .bounds import Domain
import math

class BayesTuner :
    '''
    BayesTuner is the main component of the Bayesian Optimization algorithm.

    * Parameters:

    ** objective: ndarray -> double

    ** bounds: List[Bound]:
    The domain on which the objective is optimized. It must be a list of Bound
    objects, with each Bound




    '''
    def __init__(self,
                 objective,
                 bounds,
                 n_iter,
                 init_samples,
                 optimizer = LocalOptimizer(),
                 acquisition = lambda i : UCB(i, lambda x : np.log(x))  ,
                 chooser = BasicChooser(),
                 kernel = ConstantKernel(1.0)*Matern(nu = 2.5),
                 alpha = 1e-2,
                 n_restarts = 5):

        self.domain       =   Domain(bounds)
        self.objective    =   objective
        self.n_iter       =   n_iter
        self.init_samples =   init_samples
        self.past_hyper   =   self.domain.genSamples(self.init_samples)
        self.past_evals   =   np.array([objective(x) for x in self.past_hyper]).reshape(-1,1)
        self.optimizer    =   optimizer
        self.acquisition  =   acquisition
        self.chooser      =   chooser
        self.kernel       =   kernel
        self.alpha        =   alpha
        self.n_restarts   =   n_restarts

    def tune(self,print_score = False):
        gp = GaussianProcessRegressor(kernel = self.kernel,
                                      alpha = self.alpha,
                                      n_restarts_optimizer = self.n_restarts,
                                      normalize_y = True)
        idx_best_yet = np.argmax(self.past_evals)
        best_yet     = self.past_evals[idx_best_yet]
        for i in range(1,self.n_iter):
            next_eval = self.chooser.choose(self.acquisition(i),
                                self.optimizer,
                                gp,
                                self.domain,
                                self.past_evals,
                                self.n_restarts)
            score_next_eval = self.objective(next_eval)
            if score_next_eval >= best_yet:
                best_yet = score_next_eval
                idx_best_yet = i
            if print_score == True:
                print(f"{i} / {self.n_iter} | current eval : {next_eval} / score : {score_next_eval} |\n \
-> best score yet: {best_yet} \n")
            self.past_hyper = np.vstack((self.past_hyper,next_eval))
            self.past_evals = np.vstack((self.past_evals,score_next_eval))
            gp.fit(self.past_hyper,self.past_evals)
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
