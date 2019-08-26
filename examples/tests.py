from bayestuner.bayestuner import BayesTuner
from bayestuner.bounds import Bound
import numpy as np
import math

rastringin_ = lambda X : (10*len(X) + sum([x**2 - 10*np.cos(2*math.pi*x) for x in X]))


tuner1 = BayesTuner(objective = rastringin_,
                    bounds = [Bound([-5.12,5.12],'continuous')],
                    num_iter = 30,
                    num_samples = 20)
print(tuner1.tune())
