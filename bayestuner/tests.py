from tuner import BayesTuner
import numpy as np
import math
from acquisitionfunc import EI
from skopt import gp_minimize
from optimizer import DifferentialEvolution, LBFGSB
from acquisitionfunc import AcquisitionFunc
from initialization import Normal

rastringin_ = lambda X : -(10*len(X) + sum([x**2 - 10*np.cos(2*math.pi*x) for x in X]))
square = lambda X : sum([x**2 for x in X])

'''tuner = BayesTuner(objective = rastringin_ ,
                   bounds = [(-5.12,5.12,'discrete')]*2,
                   optimizer = LBFGSB(),
                   n_iter = 50,
                   init_samples = 20)'''

tuner =  BayesTuner(objective = rastringin_ ,
                    bounds = [(-5.12,5.12,'continuous')]*2,
                    optimizer = DifferentialEvolution(),
                    n_iter = 70,
                    init_samples = 20)


result = tuner.tune(verbose = True)


#class A(AcquisitionFunc):
#    pass

#print([cls.__name__ for cls in AcquisitionFunc.__subclasses__()])
#res = gp_minimize(rastringin_,[(-5.12,5.12),(-5.12,5.12)],verbose = True)
