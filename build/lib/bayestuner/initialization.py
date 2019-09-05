import numpy as np
from .bounds import Domain

class Initialization:
    """
    Probability distribution to use when initializing samples.

    Methods
    -------

    generate(domain,num_samples)
        Generates samples from the probability distribution.

    check_within_bounds(sample,domain)
        Checks if the samples generated are within the bounds.

    """
    def generate(self,domain,num_samples):
        """
        Generates an array of samples from a specified probability distribution.

        Parameters
        ----------

        domain : Domain
            Domain object has 'bounds' attribute which contains a list.
        num_samples : Int
            Number of samples to generate.

        Returns
        -------
        numpy.ndarray
            a (n_samples,n_features) array.
        """
        pass
    def check_within_bounds(self,sample,domain):
        """
        Checks if every sample generated is within the specified bounds.

        Parameters
        ----------

        sample : numpy.ndarray
            Generated sample from a probability distribution.
        domain : Domain


        Returns
        -------

        bool
            True if within bounds, False if not.
        """
        for val,bound in zip(sample,domain.bounds):
            if val < bound[0] or val > bound[1]:
                return False
        return True


class Uniform(Initialization):
    def generate(self,domain,num_samples):
        """
        Generates an array of samples from a specified probability distribution.

        Parameters
        ----------

        domain : Domain
            Domain object has 'bounds' attribute which contains a list.
        num_samples : Int
            Number of samples to generate.

        Returns
        -------
        numpy.ndarray
            a (n_samples,n_features) array.
        """
        samples = [[np.random.uniform(bound[0],bound[1])\
                    for bound in domain.bounds] \
                    for i in range(num_samples)]
        return [domain.correctSample(sample) for sample in samples]


class Normal(Initialization):
    def generate(self,domain,num_samples):
        """
        Generates an array of samples from a specified probability distribution.

        Parameters
        ----------

        domain : Domain
            Domain object has 'bounds' attribute which contains a list.
        num_samples : Int
            Number of samples to generate.

        Returns
        -------
        numpy.ndarray
            a (n_samples,n_features) array.
        """
        samples = []
        mean = np.array([(bound[0]+bound[1])/2 for bound in domain.bounds])
        print(mean)
        cov  = np.diag([np.abs((bound[0]-bound[1])/2) for bound in domain.bounds])
        while  len(samples) < num_samples:
            sample = np.random.multivariate_normal(mean,cov)
            if self.check_within_bounds(sample,domain):
                samples.append(domain.correctSample(sample))
        return np.array(samples).reshape(-1,len(domain.bounds))
