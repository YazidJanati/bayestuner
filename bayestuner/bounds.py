import numpy as np

class Bound:
    def __init__(self,interval,type):
        if not isinstance(interval,list):
            raise TypeError("Interval must be a list")
        if type not in ["continuous","discrete"]:
            raise ValueError("type must be either continous or discrete")
        self.interval = interval
        self.type = type

    def toType(self,number):
        if self.type == "continuous":
            return number
        else:
            if number - int(number) <= 0.5:
                return int(number)
            else:
                return int(number) + 1


class Bounds:
    def __init__(self,bounds):
        self.bounds = bounds

    def generate_samples(self,num_samples):
        samples = [[bound.toType(np.random.uniform(bound.interval[0],bound.interval[1])) \
                    for bound in self.bounds] \
                    for i in range(num_samples)]
        return np.array(samples)
