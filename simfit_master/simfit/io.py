import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl



class DataSet(dict):

    def __init__(self, name, t, f):
        self.name = name
        self['t'] = t
        self['f'] = f

    def __str__(self):
        return self.name

    def df(self):
        df = pd.DataFrame(self)
        return df[['t', 'f']]

    def arr(self):
        return np.c_[self['t'], self['f']]

    def unpack(self):
        return self['t'], self['f']

    def plot(self, **kwargs):
        pl.plot(self['t'], self['f'], 'ko', **kwargs)


class SpzDataSet(DataSet):

    def __init__(self, name, t, f, pix, s=None, pld_order=1):
        self.name = name
        self.pld_order = pld_order
        self['t'] = t
        self['f'] = f
        self['pix'] = pix
        if self.pld_order == 1:
            pass
        elif self.pld_order == 2:
            self['pix'] = np.append(pix, pix ** 2, axis=1)
        else:
            raise ValueError('PLD order not supported.')
        if s is not None:
            self['s'] = s
            self.has_unc = True
        else:
            self.has_unc = False

    def unpack(self):
        if self.has_unc:
            return self['t'], self['f'], self['pix'], self['s']
        else:
            return self['t'], self['f'], self['pix']
