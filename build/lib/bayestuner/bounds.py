import numpy as np

class Domain:
    """
    A class that describes the input space of the objectives.

    Attributes
    ----------

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

    Methods
    -------

    correctSample(sample)
        Makes sure that the components of every point sampled respect the type
        of the corresponding bound.

    """
    def __init__(self,bounds):
        """
        Parameters
        ----------

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

        Raises
        ______

        ValueError
            If one of the bounds has a missing element.
            If the lower bound of a bound is larger than the upper bound.
            If the type is not 'continuous' or 'discrete'.
        """
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
        Makes sure that the components of every point sampled respect the type
        of the corresponding bound.

        Parameters
        ----------
        sample : numpy.ndarray
            the sample to correct

        Returns
        -------
        numpy.ndarray
            the same sample with corrected components.
        
        '''
        corrected_sample = []
        for (x,type) in zip(sample,list(map(lambda x : x[2],self.bounds))):
            if type == 'continuous':
                corrected_sample.append(x)
            elif type == 'discrete':
                corrected_sample.append(round(x))
        return corrected_sample
