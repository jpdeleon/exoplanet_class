import numpy as np
from scipy import stats

class Prior(dict):

    def __init__(self, kind='', name=''):
        self['kind'] = kind
        self['name'] = name

    @property
    def kind(self):
        return self['kind']

    @property
    def name(self):
        return self['name']

    def __str__(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError


class GaussianPrior(Prior):

    def __init__(self, name='', mu=0, sigma=1):
        super(GaussianPrior, self).__init__('Gaussian', name)
        self['mu'] = mu
        self['sigma'] = sigma

    @property
    def mu(self):
        return self['mu']

    @property
    def sigma(self):
        return self['sigma']

    @property
    def bounds(self):
        return self.mu-5*self.sigma, self.mu+5*self.sigma

    def __str__(self):
        return "{} ~ N({}, {})".format(self.name, self.mu, self.sigma)

    def test(self, val):
        return np.log(stats.norm.pdf(val, self.mu, self.sigma))


class UniformPrior(Prior):

    def __init__(self, name='', a=0, b=1):
        super(UniformPrior, self).__init__('Uniform', name)
        self['name'] = name
        self['a'] = a
        self['b'] = b

    @property
    def a(self):
        return self['a']

    @property
    def b(self):
        return self['b']

    @property
    def bounds(self):
        return self.a, self.b

    def __str__(self):
        return "{} ~ U({}, {})".format(self.name, self.a, self.b)

    def test(self, val):
        if (self.a <= val and val <= self.b):
            return 0.0
        return -np.inf


class PriorSet(dict):

    def __init__(self, name, *priors):
        self.name = name
        if len(priors) > 0:
            for p in priors:
                self[p.name] = p

    def __str__(self):
        return '\n'.join([str(i) for i in self.values()])

    @property
    def param_names(self):
        return self.keys()

    def lnprior(self, d):
        vals = []
        for key, val in d.items():
            if key in self:
                pdfval = self[key].test(val)
                if np.isfinite(pdfval):
                    vals.append(pdfval)
                else:
                    return -np.inf
            else:
                raise ValueError("Unexpected parameter name")
        return sum(vals)
