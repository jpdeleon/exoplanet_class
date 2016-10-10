import io
import like
import prior
import plot
import util
import numpy as np


class Fit(dict):

    def __init__(self):
        self['datasets'] = {}
        self['priorsets'] = {}
        self['lnlikes'] = {}

    @property
    def datasets(self):
        return self['datasets']

    @property
    def priorsets(self):
        return self['priorsets']

    @property
    def lnlikes(self):
        return self['lnlikes']

    @property
    def param_names(self):
        names = []
        for ps in self.priorsets.values():
            for p in ps:
                if p not in names:
                    names.append(p)
        return names

    @property
    def bounds(self):
        bounds = []
        for key in self.param_names:
            for ps in self.priorsets.values():
                if key in ps.keys():
                    bounds.append(ps[key].bounds)
        return bounds

    @property
    def ndim(self):
        return len(self.param_names)

    def add_data(self, *datasets):
        for ds in datasets:
            self.datasets[ds.name] = ds

    def add_priorsets(self, *priorsets):
        for ps in priorsets:
            self.priorsets[ps.name] = ps

    def add_lnlikes(self, *lnlikes):
        for lnlike in lnlikes:
            self.lnlikes[lnlike.name] = lnlike

    def _d(self, theta, key=None):
        d = dict(zip(self.param_names, theta))
        if key:
            return {k:d[k] for k in self.priorsets[key].param_names}
        return d

def lnprior(theta, fit):
    lps = []
    for key in fit.priorsets:
        d = fit._d(theta, key)
        lp = fit.priorsets[key].lnprior(d)
        lps.append(lp)
    lp = sum(lps)
    if not np.isfinite(lp):
        return -np.inf
    return lp

def lnlike(theta, fit):
    lls = []
    for key in fit.lnlikes:
        d = fit._d(theta, key)
        ll = fit.lnlikes[key](d, *fit.datasets[key].unpack())
        lls.append(ll)
    ll = sum(lls)
    return ll

def lnprob(theta, fit):
    lp = lnprior(theta, fit)
    if not np.isfinite(lp):
        return -np.inf
    ll = lnlike(theta, fit)
    return lp + ll
