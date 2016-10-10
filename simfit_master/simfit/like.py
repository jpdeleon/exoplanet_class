import numpy as np
from copy import copy
from pytransit import MandelAgol

MA_K2 = MandelAgol(supersampling=8, exptime=0.02)
MA_SP = MandelAgol()


class LnLikeK2(object):

    def __init__(self, name, fixed=None, bl=True, tc=None):
        self.name = name
        self.keys = ['k', 'tc', 'a', 'i', 'p', 'sig', 'u1', 'u2']
        self.fixed = fixed
        self.with_bl = bl
        self.tc = tc
        if self.with_bl:
            assert self.tc is not None

    def _baseline(self, d, t):
        cm, cb = d['cm'], d['cb']
        return cm * t + cb

    def get_corrected(self, d, t, f):
        bl = self._baseline(self._strip(d), t - self.tc)
        return f * bl

    def _strip(self, d):
        # hacky way to reuse this class for multiple K2 datasets
        # (and/or simultaneously with Spitzer datasets)
        dc = copy(d)
        for k,v in dc.items():
            if '_' in k:
                dc.pop(k)
                kn = k.split('_')[0]
                dc[kn] = v
        return dc

    def __call__(self, d, t, f, get_model=False):
        dc = self._strip(d)
        if self.fixed:
            for k,v in self.fixed.items():
                dc[k] = v
        k, tc, a, i, p, sig, u1, u2 = [dc.get(k) for k in self.keys]
        model = MA_K2.evaluate(t, k, (u1, u2), tc, p, a, i)
        if self.with_bl:
            model *= self._baseline(dc, t - self.tc)
        if get_model:
            return model
        inv_sig2 = 1./sig**2
        return -0.5 * np.sum( (f - model) ** 2 * inv_sig2 - np.log(inv_sig2) )


class LnLikeSpz(object):

    def __init__(self, name, fixed=None, unc=False, bl=True, tc=None, alpha=False, geom=3, pld_order=1):
        self.name = name
        self.keys = ['k', 'tc', 'a', 'i', 'p', 'sig', 'u1', 'u2']
        self.fixed = fixed
        self.with_unc = unc
        self.with_bl = bl
        self.tc = tc
        if self.with_bl:
            assert self.tc is not None
        self.with_alpha = alpha
        self.geom = geom
        self.pld_order = pld_order
        if self.with_unc:
            self.keys.pop(self.keys.index('sig'))
        if self.with_alpha:
            assert self.with_unc
            self.keys.append('alpha')
        if self.pld_order == 1 and self.geom == 3:
            self.pld_keys = ['c{}'.format(j) for j in range(1,10)]
        elif self.pld_order == 1 and self.geom == 5:
            self.pld_keys = ['c{}'.format(j) for j in range(1,26)]
        elif self.pld_order == 2 and self.geom == 3:
            self.pld_keys = ['c{}'.format(j) for j in range(1,19)]
        elif self.pld_order == 2 and self.geom == 5:
            self.pld_keys = ['c{}'.format(j) for j in range(1,51)]
        else:
            raise ValueError("PLD geometry and/or order not supported.")

    def _baseline(self, d, t):
        cm, cb = d['cm'], d['cb']
        return cm * t + cb

    def pld_model(self, d, t, f, pix, sys=False, full=False):
        if self.with_unc and not self.with_alpha:
            k, tc, a, i, p, u1, u2 = [d.get(key) for key in self.keys]
        elif self.with_alpha:
            k, tc, a, i, p, u1, u2, alpha = [d.get(key) for key in self.keys]
        else:
            k, tc, a, i, p, sig, u1, u2 = [d.get(key) for key in self.keys]
        u = (u1, u2)
        coeff = np.array([d.get(key) for key in self.pld_keys])
        syste = (coeff * pix).sum(axis=1)
        if self.with_bl:
            syste *= self._baseline(d, t - self.tc)
        if sys:
            return syste
        elif full:
            return MA_SP.evaluate(t, k, u, tc, p, a, i) + syste
        else:
            return MA_SP.evaluate(t, k, u, tc, p, a, i)

    def __call__(self, d, t, f, pix, s=None, get_model=False, **kwargs):
        # hacky way to reuse this class for multiple Spitzer datasets
        dc = copy(d)
        for k,v in dc.items():
            if '_' in k:
                dc.pop(k)
                kn = k.split('_')[0]
                dc[kn] = v
        if self.fixed:
            for k,v in self.fixed.items():
                dc[k] = v
        if get_model:
            return self.pld_model(dc, t, f, pix, **kwargs)
        model = self.pld_model(dc, t, f, pix, full=True)
        if s is not None:
            sig2 = s ** 2
            if self.with_alpha:
                inv_sig2 = 1. / (sig2 * dc['alpha'])
            else:
                inv_sig2 = 1. / sig2
        else:
            inv_sig2 = 1. / dc['sig']**2
        ll = -0.5 * np.sum( (f - model) ** 2 * inv_sig2 - np.log(inv_sig2) )
        if np.isnan(ll):
            return -np.inf
        return ll
