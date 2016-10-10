import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
from matplotlib.colors import LogNorm
from photutils.morphology import centroid_com, centroid_1dg, centroid_2dg
from scipy import stats
import scipy.optimize as op
import statsmodels.api as sm
import functools
import corner
import util
import seaborn as sb

ncolors = 5
cp = [sb.desaturate(pl.cm.spectral((j+1)/float(ncolors+1)), 0.75) for j in range(ncolors)]


def k2(fit, samples, save=None):
    """
    fit : Fit object containing the data, priors, likelihoods, etc.
    samples : array-like, samples to use (assumes burnin already removed)
    """

    with sb.axes_style('white'):

        dsnames = sorted([key for key in fit.datasets.keys() if 'k2' in key])
        nds = len(dsnames)
        if nds == 1:
            # this should never happen, but anyway...
            fig, axs = pl.subplots(1, nds, figsize=(5,5),
                sharex=True, sharey=True)
            fig = [fig] # hacky
        else:
            ncol = 2
            nrow = nds/2
            nbot = 2
            if nds % 2 > 0:
                nrow += 1
                nbot -= 1
            fig, axs = pl.subplots(nrow, ncol, figsize=(12,nds),
                sharex=True, sharey=True)

        for j,dsname in enumerate(dsnames):
            ds = fit.datasets[dsname]
            t, f = ds['t'], ds['f']
            lnlike = fit.lnlikes[ds.name]
            models = []
            tdurs = []
            tcs = []
            tm = np.linspace(t.min(), t.max(), 100)
            for theta in samples[np.random.randint(len(samples), size=96)]:
                d = fit._d(theta, dsname)
                p = lnlike.fixed['p']
                a,k,i = [d.get(z) for z in 'a,k,i'.split(',')]
                key = filter(lambda x: 'tc' in x, d.keys())[0]
                tcs.append(d[key])
                tdurs.append(util.tdur_circ(p,a,k,i))
                models.append(lnlike(d, tm, f, get_model=True))
            tdur = np.median(tdurs) * 24
            print "T14 [hours]: {}".format(tdur)

            tc = np.median(tcs)

            axs.flat[j].plot((t-tc)*24, f, 'ko', ms=10, alpha=1)
            for model in models:
                axs.flat[j].plot((tm-tc)*24, model, color=cp[4], alpha=0.1, lw=1.5)
            axs.flat[j].set_xlabel('Time from mid-transit [hours]', fontsize=16)
            axs.flat[j].set_ylabel('Normalized flux', fontsize=16)
            axs.flat[j].set_xlim(-2*tdur, 2*tdur)
            axs.flat[j].set_title(dsname)
            if j < nds - nbot:
                pl.setp(axs.flat[j], xlabel='')

        fig.tight_layout()
        if save:
            fig.savefig(save, dpi=75)
            pl.close()


def ephem(vals, orb, title=None, save=None):
    # TODO: simplify inputs, maybe just individual lists of lo, mu, hi, orb_num
    """
    Ephemeris (linear) plot
    vals : dictionary of 'lo', 'mu', 'hi' values for individual Tc params
    orb : array-like, orbit numbers (epochs) corresponding to each Tc in vals
    """
    x, y, yerr = util.get_ephem(vals, orb)
    m_ls, m_sig, b_ls, b_sig = util.fit_linear_ephem(x, y, yerr)

    with sb.axes_style('white'):

        pl.figure(figsize=(8,6))
        for i in zip(x, y, yerr):
            pl.errorbar(*i, fmt='ko')
            xl = [x[0]-1, x[-1]+1]

        pl.plot(xl, [b_ls+m_ls*xl[0], b_ls+m_ls*xl[-1]], 'r-', lw=1.5)
        pl.xlim(*xl)
        xlim, ylim = pl.xlim(), pl.ylim()
        text1 = "$slope = {0:0.8f}\pm{1:0.8f}$".format(m_ls, m_sig)
        text2 = "$intercept = {0:0.8f}\pm{1:0.8f}$".format(b_ls, b_sig)
        pl.text(xlim[0]+0.05*np.diff(xlim), ylim[0]+0.9*np.diff(ylim), text1, fontsize=18)
        pl.text(xlim[0]+0.05*np.diff(xlim), ylim[0]+0.84*np.diff(ylim), text2, fontsize=18)
        pl.xlabel("Epoch")
        pl.ylabel("Mid-transit Time [BJD]")
        if title:
            pl.title(title)
        if save:
            pl.savefig(save)
            pl.close()

def oc(vals, orb, fig=None, title=None, save=None):
    # TODO: simplify inputs, maybe just individual lists of lo, mu, hi, orb_num
    """
    Observed minus Predicted (O-C) plot.
    vals : dictionary of 'lo', 'mu', 'hi' values for individual Tc params
    orb : array-like, orbit numbers (epochs) corresponding to each Tc in vals
    """
    x, y, yerr = util.get_ephem(vals, orb)
    m_ls, m_sig, b_ls, b_sig = util.fit_linear_ephem(x, y, yerr)

    tns = [b_ls+m_ls*i for i in x]
    y = (y - np.array(tns)) * 24 * 60
    keys = ['tc_{}'.format(o) for o in orb]
    yerr = [[vals[k]['mu']-vals[k]['lo'], vals[k]['hi']-vals[k]['mu']] for k in keys]
    yerr = np.array(yerr).T
    yerr *= 24 * 60

    with sb.axes_style('white'):
        if fig is None:
            fig, ax = pl.subplots(1, 1, figsize=(8,6))
        ax = fig.get_axes()[0]
        ax.errorbar(x, y, yerr, fmt='ko', capthick=0)
        xl = [x[0]-1, x[-1]+1]
        ax.set_xlim(*xl)
        ax.hlines([m_sig*24*60, -m_sig*24*60], *pl.xlim(), linestyles='dashed')
        ax.set_xlabel('Orbit Number')
        ax.set_ylabel('Observed minus predicted [minutes]')
        if title:
            ax.set_title(title)
        if save:
            fig.savefig(save)
            pl.close()


def t14(fit, samples, save=None, return_samples=False):
    # TODO: strip out the sampling part maybe
    tdurs = []
    p = fit.lnlikes.values()[0].fixed['p']
    for theta in samples[np.random.randint(len(samples), size=int(1e5))]:
        d = dict(zip(fit.param_names, theta))
        a,k,i = [d.get(z) for z in 'a,k,i'.split(',')]
        tdurs.append(util.tdur_circ(p,a,k,i) * 24)
    if return_samples:
        return [tdurs]
    with sb.axes_style('white'):
        fig, ax = pl.subplots(1, 1, figsize=(3,3))
        # corner.corner(tdurs, fig=fig, labels=['$T_{14}$'],
        #               quantiles=[0.16, 0.5, 0.84],
        #               show_titles=True, title_fmt=".4f")
        ax.hist(tdurs, bins=20, histtype='step', lw=2, alpha=0.8, color='k')
        percs = np.percentile(tdurs, [16,50,84])
        ax.vlines(percs, *ax.get_ylim(), linestyles='--')
        pl.setp(ax, xlabel='$T_{14}$ [hours]', yticks=[])
        fig.tight_layout()
        if save:
            fig.savefig(save)
            pl.close()
    return percs[1] / 24.


def ephem_posteriors(fit, samples, orb, return_res=False, save=None):
    # TODO: refactor so samples are produced by a util func
    """
    Given samples of the individual Tc values, produce corner plot for a
    linear ephemeris, and optionally return the T0 and P samples.
    fit : Fit object containing the data, priors, likelihoods, etc.
    samples : array-like, samples to use (assumes burnin already removed)
    orb : array-like, orbit numbers (epochs) corresponding to each Tc in vals
    """

    t0s, pers = [], []

    for theta in samples[np.random.randint(len(samples), size=int(1e5))]:

        d = dict(zip(fit.param_names, theta))
        tt = sorted([v for k,v in d.items() if 'tc' in k])
        res = util.simple_ols(orb, tt)
        t0s.append(res[0])
        pers.append(res[1])

    with sb.axes_style('white'):

        fig, axs = pl.subplots(2, 2, figsize=(5,5))

        corner.corner(np.c_[t0s, pers], fig=fig,
                      labels=['$T_0$', '$P$'],
                      quantiles=[0.16, 0.5, 0.84],
                      hist_kwargs=dict(lw=2, alpha=0.5),
                      title_kwargs=dict(fontdict=dict(fontsize=12)),
                      show_titles=True, title_fmt=".4f")

        if save:
            fig.tight_layout()
            fig.savefig(save)
            pl.close()

        if return_res:
            return np.c_[t0s, pers]


def multi_gauss_fit(samples, p0, save=None, return_popt=False):

    def multi_gauss(x, *args):
        n = len(args)
        assert n % 3 == 0
        g = np.zeros(len(x))
        for i in range(0,n,3):
            a, m, s = args[i:i+3]
            g += a * stats.norm.pdf(x, m, s)
        return g

    hist, edges = np.histogram(samples, bins=100, normed=True)
    bin_width = np.diff(edges).mean()
    x, y = edges[:-1] + bin_width/2., hist
    try:
        popt, pcov = op.curve_fit(multi_gauss, x, y, p0=p0)
    except RuntimeError as e:
        print e
        with sb.axes_style('white'):
            fig,ax = pl.subplots(1,1, figsize=(7,3))
            ax.hist(samples, bins=30, normed=True,
                    histtype='stepfilled', color='gray', alpha=0.6)
        return

    ncomp = len(p0)/3
    names = 'amp mu sigma'.split() * ncomp
    comp = []
    for i in range(ncomp):
        comp += (np.zeros(3) + i).astype(int).tolist()
    for i,(p,u) in enumerate(zip(popt, np.sqrt(np.diag(pcov)))):
        print "{0}{1}: {2:.6f} +/- {3:.6f}".format(names[i], comp[i], p, u)

    a_,mu_,sig_ =[],[],[]
    for i in range(len(p0)/3):
        a_.append(popt[i*3])
        mu_.append(popt[i*3+1])
        sig_.append(popt[i*3+2])

    with sb.axes_style('white'):
        fig,ax = pl.subplots(1,1, figsize=(7,3))
        ax.hist(samples, bins=edges, normed=True,
                histtype='stepfilled', color=cp[1], alpha=0.6)
        for a,m,s in zip(a_,mu_,sig_):
            ax.plot(x, a * stats.norm.pdf(x, m, s), linestyle='-', color=cp[0])
        ax.plot(x, multi_gauss(x, *popt), linestyle='--', color=cp[4], lw=3)
        pl.setp(ax, xlim=[x.min(), x.max()], yticks=[])
        fig.tight_layout()
        if save:
            fig.savefig(save)
            pl.close()

    if return_popt:
        return popt


def spz(fit, epoch, samples, best, plot_best=True,
              save=None, dpi=120, bs=128, scale=1):

    dsname = 'spz-e{}'.format(epoch)
    ds = fit.datasets[dsname]
    lnlike = fit.lnlikes[dsname]

    t, f, s = ds['t'], ds['f'], ds['s']
    s *= scale
    d = fit._d(best, dsname)
    fcor = f - lnlike(d, *ds.unpack(), get_model=True, sys=True)
    mbest = lnlike(d, *ds.unpack(), get_model=True)

    fc = samples
    mp = np.median(fc, 0)

    flux_pr = []
    for pv in fc[np.random.permutation(fc.shape[0])[:1000]]:
        d = fit._d(pv, dsname)
        m = lnlike(d, *ds.unpack(), get_model=True)
        flux_pr.append(m)

    flux_pr = np.array(flux_pr)
    flux_pc = np.array(np.percentile(flux_pr, [50, 0.15,99.85, 2.5,97.5, 16,84], 0))

    sig = util.binned(s, bs) / np.sqrt(bs)
    tc = mp[fit.param_names.index('tc_{}'.format(epoch))]
    tn = (t - tc) * 24
    tb = util.binned(tn, bs)
    fb = util.binned(fcor, bs)

    with sb.axes_style('white'):
        zx1,zx2,zy1,zy2 = -0.08,0.08, 0.9992, 0.9994
        fig, ax = pl.subplots(1,1, figsize=(13,4))
        ax.errorbar(tb, fb, sig, fmt='.', c=cp[1], alpha=0.75)
        [ax.fill_between(tn,*flux_pc[i:i+2,:],alpha=0.2,facecolor=cp[4]) for i in range(1,6,2)]
        ax.plot(tn, flux_pc[0], c=cp[4], label='median')
        if plot_best:
            ax.plot(tn, mbest, c='k', alpha=0.9, label='best fit')
            ax.legend(loc=4)
        pl.setp(ax, xlim=tn[[0,-1]], xlabel='Time from mid-transit [hours]', ylabel='Normalized flux')
        fig.tight_layout()

    if save:
        fig.savefig(save, dpi=dpi)
        pl.close()


def spz_new(fit, theta, dsname, binned=False, save=None):
    """
    fit : Fit object containing the data, priors, likelihoods, etc.
    theta : array-like, parameter vector to use for the plot
    dsname : string, name of dataset belonging to fit object
    """
    d = fit._d(theta, dsname)
    t, f, pix, s = [i.copy() for i in fit.datasets[dsname].unpack()]
    mt = fit.lnlikes[dsname](d, t, f, pix, s, get_model=True)
    mf = fit.lnlikes[dsname](d, t, f, pix, s, get_model=True, full=True)
    ms = fit.lnlikes[dsname](d, t, f, pix, s, get_model=True, sys=True)
    fcor = f - ms
    resid = mf - f
    with sb.axes_style('white'):
        pl.plot(t, f, 'k.', t, mf, 'r-')
        pl.plot(t, fcor - 0.025, 'k.', ms=5, alpha=0.5)
        if binned:
            bs = 256
            tb = util.binned(t, bs)
            fb = util.binned(fcor - 0.025, bs)
            pl.plot(tb, fb, 'bo', ms=10, alpha=1, mew=0)
        pl.plot(t, mt - 0.025, 'r-', lw=3)
        pl.plot(t, resid + 0.95, 'k.', ms=3, alpha=0.3)
        pl.xlim(t.min(), t.max())
        pl.xlabel('Time [BJD]')
        pl.ylabel('Normalized Flux')
        if save:
            pl.savefig(save)
            pl.close()


def spz_old(fit, theta, dsname, save=None):

    d = fit._d(theta, dsname)
    ds = fit.datasets[dsname]
    lnlike = fit.lnlikes[dsname]

    full_model = lnlike(d, *ds.unpack(), get_model=True, full=True)
    sys_only = lnlike(d, *ds.unpack(), get_model=True, sys=True)
    tra_only = lnlike(d, *ds.unpack(), get_model=True)
    f = ds['f']
    fcor = f - sys_only
    t = ds['t']

    with sb.axes_style('white'):

        fig, axs = pl.subplots(3, 1, figsize=(12,6))
        axs[0].plot(t, f, 'ko', alpha=0.2)
        axs[0].plot(t, full_model, 'r-')
        axs[0].set_xlim(t.min(), t.max())
        axs[1].plot(t, fcor, 'ko', alpha=0.2)
        axs[1].plot(t, tra_only, 'r-', lw=3)
        axs[1].set_xlim(t.min(), t.max())

        resid = f - full_model
        axs[2].hist(resid, bins=50, normed=True,
                histtype='stepfilled', color='gray', label='residuals')
        xl = pl.xlim()
        xi = np.linspace(xl[0], xl[1], 100)
        axs[2].plot(xi, stats.norm.pdf(xi, *stats.norm.fit(resid)), 'r-', lw=3)
        fig.tight_layout()
        if save:
            fig.savefig(save)
            pl.close()
