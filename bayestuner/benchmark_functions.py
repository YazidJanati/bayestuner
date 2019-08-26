#import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from bounds import Bounds, Bound
from abc import ABC, abstractmethod
from mayavi import mlab
import math


#Functions are supposed to be maximized.
class Benchmark(ABC):
    @abstractmethod
    def __init__(self,dimension):
        self.bounds = Bounds([Bound([0,0],"continuous")]*dimension)
        self.global_optimum = [0]*dimension

    @abstractmethod
    def evaluate(self,argument):
        pass
    @abstractmethod
    def plot_func(self,dimension):
        pass


class Ackley(Benchmark):
    def __init__(self,dimension):
        self.dimension = dimension
        self.bounds = [Bound([-32,32],"continuous")]*dimension
        self.global_optimum = [0]*dimension

    def evaluate(self,X):
        if not hasattr(X,'__len__'):
            X = [X]
        first_term = -20*np.exp(-0.2*np.sqrt(sum(X[i]**2 for i in range(self.dimension))/self.dimension))
        second_term = - np.exp(sum(np.cos(2.*np.pi*X[i]) for i in range(self.dimension))/self.dimension)
        return first_term + second_term + 20 + np.exp(1)

    def plot_func(self):
        if self.dimension > 2 or self.dimension == 0:
            raise ValueError("dimension must be lower than 2")
        if self.dimension == 1 :
            X = np.linspace(self.bounds[0].interval[0],
                            self.bounds[0].interval[1],1000)
            plt.plot(X,[- self.evaluate(x) for x in X],'r')
        else:
            #x,y = np.mgrid[-32:32:100j,-32:32:100j]
            X = Y = np.linspace(-32,32,1000)
            x,y = np.meshgrid(X,Y)
            to_plot = lambda x,y : - self.evaluate(np.array([x,y]))
            z = to_plot(x,y)
            print(z)
            plot = mlab.surf(z,warp_scale = 'auto',colormap = 'magma')
            return plot
