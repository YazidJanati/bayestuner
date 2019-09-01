import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import norm



class AcquisitionFunc(ABC) :

    def __init__(self,i):
        pass

    @abstractmethod
    def eval(self,curr_loc,gp,past_evals):
        pass


class UCB(AcquisitionFunc):
    def __init__(self,i,temperature):
        self.i = i
        self.temperature = temperature

    def eval(self,curr_loc,gp,past_evals):
        #pay attention to the shape of curr_loc
        m,s = gp.predict(curr_loc.reshape(1,-1),return_std = True)
        return m[0] + self.temperature(self.i) * s[0]

class EI(AcquisitionFunc):
    def __init__(self,i):
        self.i = i

    def eval(self,curr_loc,gp,past_evals):
        #put the right shape to curr_loc beforehand
        m,s = gp.predict(curr_loc.reshape(1,-1),return_std= True)
        if s[0] == 0:
            ei = 0
        else:
            y_max  = np.max(past_evals)
            delta  = m[0] - y_max
            Z      = delta / s[0]
            ei     = s[0]*norm.pdf(Z) + delta*norm.cdf(Z)
        return ei
