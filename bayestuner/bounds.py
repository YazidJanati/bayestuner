import numpy as np

'''class Bound:
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
'''

class Domain:

    def __init__(self,bounds):
        '''
        Domain is an object that has a list of tuples as attribute.
        e.g. : [(-1,1,'continous'),(-2,1,'discrete')]
        '''
        for bound in bounds:
            if len(bound) != 3 :
                raise ValueError("You forgot to specify a parameter")
            if bound[0] >= bound[1]:
                raise ValueError("Lower bound can't be bigger than upper bound")
            if bound[2] not in ['continuous','discrete']:
                raise ValueError('Type of domain must be either continuous \
                or discrete')
        self.bounds = bounds


    def correctSample(self,sample):
        '''
        to document
        '''
        corrected_sample = []
        for (x,type) in zip(sample,list(map(lambda x : x[2],self.bounds))):
            if type == 'continuous':
                corrected_sample.append(x)
            elif type == 'discrete':
                corrected_sample.append(round(x))
        return corrected_sample

    def genSamples(self,num_samples):
        samples = [[np.random.uniform(bound[0],bound[1])\
                    for bound in self.bounds] \
                    for i in range(num_samples)]
        return [self.correctSample(sample) for sample in samples]
