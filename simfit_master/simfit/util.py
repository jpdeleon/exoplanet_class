import os
import sys
import yaml
import shutil
import numpy as np
import emcee
import pickle
from scipy import stats
import scipy.optimize as op
import statsmodels.api as sm
from astropy import constants as c
from astropy import units as u
from astropy.stats import sigma_clip
from photutils.morphology import centroid_com


def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if shutil.fnmatch.fnmatch(basename, pattern):
                filepath = os.path.join(root, basename)
                yield filepath


def ignore_oserror(f):
    """
    rather silly decorator for ignoring OSError exceptions
    """
    def wrapper(*args):
        try:
            f(*args)
        except OSError as e:
            print(e)
    return wrapper


@ignore_oserror
def mkdir(d):
    os.makedirs(d)


def mkdirs(*args):
    for d in args:
        mkdir(d)


def rms(x):
    return np.sqrt((x**2).sum()/x.size)


def geom_mean(x):
    x = np.abs(x)
    gm = np.sqrt(np.product(x)) if x.size > 1 else x
    return gm


def binned(a, binsize, fun=np.mean):
    return np.array([fun(a[i:i+binsize], axis=0) for i in range(0, a.shape[0], binsize)])



def center_transit(t0, time, period):
    n = 0
    t = t0
    while t < time.min():
        n += 1
        t += period
    return t, n


def print_summary(t0, time, period, p):
    t, n = center_transit(t0, time, period)
    print("center of transit [BJD]: {}".format(t))
    print("number of orbits since t0: {}".format(n))
    print("depth of transit [ppm]: {}".format(p**2*1e6))


def impact(a, i):
    return np.abs(a * np.cos(i))


def scaled_a(p, t14, k, i=np.pi/2.):
    numer = np.sqrt( (k + 1) ** 2 )
    denom = np.sin(i) * np.sin(t14 * np.pi / p)
    return float(numer / denom)


def tdur_circ(p, a, k, i=np.pi/2.):
    b = impact(a, i)
    alpha = np.sqrt( (k + 1) ** 2 - b ** 2 )
    return (p / np.pi) * np.arcsin( alpha / np.sin(i) / a )


def t14(a, p, rstar, k):
    return p * np.arcsin( (k + 1) * rstar / a ) / np.pi


def parse_setup(fp):
    setup = yaml.load(open(fp))
    transit = setup['transit']
    if not transit['i']:
        transit['i'] = np.pi/2
    if not transit['a']:
        try:
            p = transit['p']
            t14 = transit['t14']
            k = transit['k']
            i = transit['i']
            transit['a'] = scaled_a(p, t14, k, i)
        except KeyError as e:
            msg = "{} is missing! unable to compute scaled semi-major axis"
            print(msg.format(e))
    setup['transit'] = transit
    return setup


def mode(x, bins=100):
    counts, edges = np.histogram(x, bins=bins)
    idx = np.argmax(counts)
    return edges[idx:idx+2].mean()


def simple_clip(x, width=100, sigma=3, norm=False, return_idx=False):
    x = np.array(x, copy=True)
    n = x.size
    idx = np.repeat(False, n)
    for i in range(0, n-width):
        sub = x[i:i+width]
        mu = np.nanmean(sub)
        thresh = sigma * np.nanstd(sub)
        idx[i:i+width][np.abs(sub-mu) > thresh] = True
    if return_idx:
        return idx
    x[idx] = np.nan
    if norm:
        return x/np.median(x[~np.isnan(x)])
    return x


def iter_clip(x, width=100, sigma=3, norm=False, return_idx=False):
    x = np.array(x, copy=True)
    n = x.size
    msk = np.repeat(False, n)
    for i in range(0, n-width):
        sub = x[i:i+width]
        c,l,u = stats.sigmaclip(sub, sigma, sigma)
        idx = (sub < l) | (sub > u)
        msk[i:i+width][idx] = True
    if return_idx:
        return msk
    x[msk] = np.nan
    if norm:
        return x/np.median(x[~np.isnan(x)])
    return x


def rhostar(p, a):
    """
    Eq.4 of http://arxiv.org/pdf/1311.1170v3.pdf. Assumes circular orbit.
    """
    p = p * u.d
    gpcc = u.g / u.cm ** 3
    rho_mks = 3 * np.pi / c.G / p ** 2 * a ** 3
    return rho_mks.to(gpcc)


def logg(rho, r):
    r = (r * u.R_sun).cgs
    gpcc = u.g / u.cm ** 3
    rho = rho * gpcc
    g = 4 * np.pi / 3 * c.G.cgs * rho * r
    return np.log10(g.value)


def rho(logg, r):
    r = (r * u.R_sun).cgs
    g = 10 ** logg * u.cm / u.s ** 2
    rho = 3 * g / (r * c.G.cgs * 4 * np.pi)
    return rho


def beta(residuals, timestep, start_min=5, stop_min=20, return_betas=False):

    """
    residuals : data - model
    timestep : time interval between datapoints in seconds
    """

    ndata = len(residuals)

    sigma1 = np.std(residuals)

    min_bs = int(start_min * 60 / timestep)
    max_bs = int(stop_min * 60 / timestep)

    betas = []
    for bs in range(min_bs, max_bs + 1):
        nbins = ndata / bs
        sigmaN_theory = sigma1 / np.sqrt(bs) * np.sqrt( nbins / (nbins - 1) )
        sigmaN_actual = np.std(binned(residuals, bs))
        beta = sigmaN_actual / sigmaN_theory
        betas.append(beta)

    if return_betas:
        return betas

    return np.median(betas)


def get_ephem(vals, orb):
    # TODO: simplify inputs, maybe just individual lists of lo, mu, hi, orb_num
    """
    vals : dictionary of 'lo', 'mu', 'hi' values for individual Tc params
    orb : array-like, orbit numbers (epochs) corresponding to each Tc in vals
    """
    keys = ['tc_{}'.format(o) for o in orb]
    tt = [vals[k]['mu'] for k in keys]
    ett = [geom_mean([vals[k]['hi']-vals[k]['mu'], vals[k]['mu']-vals[k]['lo']]) for k in keys]
    return orb, tt, ett


def fit_linear_ephem(orb, tt, ett):
    # TODO: could just be a generic OLS function
    """
    Ordinary Least-Squares (OLS).
    orb : array-like, orbit numbers (x)
    tt : array-like, Tc values / transit times (y)
    ett : array-like, Tc uncertainties (y_unc)
    """
    x, y, yerr = map(np.array, [orb, tt, ett])
    A = np.vstack((np.ones_like(x), x)).T
    C = np.diag(yerr * yerr)
    cov = np.linalg.inv(np.dot(A.T, np.linalg.solve(C, A)))
    b_ls, m_ls = np.dot(cov, np.dot(A.T, np.linalg.solve(C, y)))
    b_sig, m_sig = np.sqrt(np.diag(cov))
    return m_ls, m_sig, b_ls, b_sig


def simple_ols(x, y, intercept=True):
    """
    Simple OLS with no y uncertainties.
    x : array-like, abscissa
    y : array-like, ordinate
    """
    if intercept:
        X = np.c_[np.ones_like(x), x]
    else:
        X = x
    return np.dot( np.dot( np.linalg.inv( np.dot(X.T, X) ), X.T), y )


def bic(lnlike, ndata, nparam):
    return -2 * lnlike + nparam * np.log(ndata)


def save_bic_summary(fit, fp, verbose=True):

    with open(fp, 'w') as w:

        bicvals = []

        for dsname in sorted(fit.datasets.keys()):
            ds = fit.datasets[dsname]
            d = fit._d(fit.best, dsname)
            lnlike = fit.lnlikes[dsname](d, *ds.unpack())
            ndata = len(ds['t'])
            nparam = len(d)
            bicval = bic(lnlike, ndata, nparam)
            bicvals.append(bicval)
            line = "{}: {} datapoints, {} params, lnlike: {}"
            line = line.format(dsname, ndata, nparam, lnlike)
            if verbose: print line
            w.write(line+'\n')
            line = "BIC: {}\n".format(bicval)
            if verbose: print line
            w.write(line+'\n')

        line = "Total BIC: {}".format(sum(bicvals))
        if verbose: print line
        w.write(line)



def save_samples(fit, sampler, fp):
    acor = sampler.acor.mean().astype(int)
    accept_frac = sampler.acceptance_fraction.mean()
    ndim = sampler.dim
    try:
        nwalkers = sampler.k
    except:
        nwalkers = sampler.nwalkers
    try:
        nsteps = sampler.iterations
    except:
        nsteps = -1
    param_names = fit.param_names
    lnprob = sampler.lnprobability
    np.savez_compressed(fp, chain=sampler.chain, lnprob=lnprob,
                        accept_frac=accept_frac, acor=acor, nsteps=nsteps,
                        ndim=ndim, nwalkers=nwalkers, param_names=param_names)


def sample_rhostar(fit, samples, epochs):
    rho = []
    n = int(1e4) if len(samples) > 1e4 else len(samples)
    for theta in samples[np.random.randint(len(samples), size=n)]:
        d = dict(zip(fit.param_names, theta))
        tt = sorted([v for k,v in d.items() if 'tc' in k])
        res = simple_ols(epochs, tt)
        p = res[1]
        a = d['a']
        rho.append(rhostar(p, a).value)
    return np.array(rho)


def sample_logg(rho_samples, rstar, urstar):
    """
    Given samples of the stellar density and the stellar radius
    (and its uncertainty), compute a sample of logg
    """
    rs = rstar + urstar * np.random.randn(len(rho_samples))
    idx = rs > 0
    return logg(rho_samples[idx], rs[idx])


def get_tns(t, p, t0):

    idx = t != 0
    t = t[idx]

    while t0-p > t.min():
        t0 -= p
    if t0 < t.min():
        t0 += p

    tns = [t0+p*i for i in range(int((t.max()-t0)/p+1))]

    while tns[-1] > t.max():
        tns.pop()

    while tns[0] < t.min():
        tns = tns[1:]

    return tns


def extract_individual(star, p, t0, width=1):

    t, f = star.time, star.flux
    idx = np.isfinite(t) & np.isfinite(f)
    t, f = t[idx], f[idx]

    tns =  get_tns(t, p, t0)

    transits = []
    for i,tn in enumerate(tns):
        idx = (t > tn - width/2.) & (t < tn + width/2.)
        if idx.sum() == 0:
            continue
        ti = t[idx].tolist()
        fi = f[idx].tolist()
        transits.append((ti,fi))

    return transits


def fold(t, f, p, t0, width=0.4, clip=False, bl=False, t14=0.2):
    tns = get_tns(t, p, t0)
    tf, ff = np.empty(0), np.empty(0)
    for i,tn in enumerate(tns):
        idx = (t > tn - width/2.) & (t < tn + width/2.)
        ti = t[idx]-tn
        fi = f[idx]
        fi /= np.nanmedian(fi)
        if bl:
            idx = (ti < -t14/2.) | (ti > t14/2.)
            assert np.isfinite(ti[idx]).all() & np.isfinite(fi[idx]).all()
            assert idx.sum() > 0
            try:
                res = sm.RLM(fi[idx], sm.add_constant(ti[idx])).fit()
                if np.abs(res.params[1]) > 1e-2:
                    print "bad data probably causing poor fit"
                    print "transit {} baseline params: {}".format(i, res.params)
                    continue
                model = res.params[0] + res.params[1] * ti
                fi = fi - model + 1
            except:
                print "error computing baseline for transit {}".format(i)
                print "num. points: {}".format(idx.sum())
                print ti
        tf = np.append(tf, ti)
        ff = np.append(ff, fi / np.nanmedian(fi))
    idx = np.argsort(tf)
    tf = tf[idx]
    ff = ff[idx]
    if clip:
        fc = sigma_clip(ff, sigma_lower=10, sigma_upper=2)
        tf, ff = tf[~fc.mask], ff[~fc.mask]
    return tf, ff


def spz_chi2(theta, fit, dsname, reduced=True, scalefac=1, norm_test=False, verbose=False):
    d = fit._d(theta, dsname)
    ds = fit.datasets[dsname]
    t, f, pix, s = [i.copy() for i in ds.unpack()]
    s *= scalefac
    mf = fit.lnlikes[dsname](d, t, f, pix, s, get_model=True, full=True)
    N, n = len(f), len(d)
    dof = N - n
    resid = mf - f
    if verbose:
        print "N = {}, n = {} --> dof = {}".format(N, n, dof)
        print "expected chi-square: {} +/- {}".format(dof, np.sqrt(2*dof))
    if norm_test:
        std_resid = (resid - resid.mean()) / resid.std()
        print stats.mstats.normaltest( std_resid )
        print stats.kstest( std_resid, 'norm')
    if reduced:
        return sum( (resid / s) ** 2 ) / (dof)
    else:
        return sum( (resid / s) ** 2 )
