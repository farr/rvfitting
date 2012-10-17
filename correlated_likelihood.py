import numpy as np
import numpy.linalg as nl
import numpy.random as nr
import parameters as params
import rv_model as rv
import scipy.linalg as sl

def correlated_gaussian_loglikelihood(xs, means, cov):
    """Returns the likelihood for data xs, assumed to be multivariate
    Gaussian with the given means and covariance."""
    lu,piv=sl.lu_factor(cov)

    lambdas=np.diag(lu)

    ndim=xs.shape[0]
    
    ds=(xs-means)*sl.lu_solve((lu,piv), xs-means)/2.0

    return -np.log(2.0*np.pi)*(ndim/2.0)-0.5*np.sum(np.log(lambdas))-np.sum(ds)

def generate_covariance(ts, sigma, tau):
    """Generates a covariance matrix according to an exponential
    autocovariance: cov(t_i, t_j) =
    sigma*sigma*exp(-|ti-tj|/tau)."""

    ndim = ts.shape[0]

    tis = np.tile(np.reshape(ts, (-1, 1)), (1, ndim))
    tjs = np.tile(ts, (ndim, 1))

    return sigma*sigma*np.exp(-np.abs(tis-tjs)/tau)

class LogPrior(object):
    """Log of the prior function."""

    def __init__(self, pmin=None, pmax=None, npl=1, nobs=1):
        """Initialize with the given bounds on the priors."""

        self._pmin = pmin
        self._pmax = pmax
        self._npl = npl
        self._nobs = nobs

    def __call__(self, p):
        p = params.Parameters(p, npl=self._npl, nobs=self._nobs)

        if self._pmin is None:
            self._pmin = 0.0*p
            self._pmin.V = float('-inf')

        if self._pmax is None:
            self._pmax = p + float('inf')
            if npl > 0:
                self._pmax.chi = 1.0
                self._pmax.e = 1.0
                self._pmax.omega = 2.0*np.pi

        # Check bounds
        if np.any(p <= self._pmin) or np.any(p >= self._pmax):
            return float('-inf')

        # Ensure unique labeling of planets: in increasing order of
        # period
        if p.npl > 1 and np.any(p.P[1:] < p.P[:-1]):
            return float('-inf')

        if np.any(p.K < np.min(p.sigma)/10.0):
            return float('-inf')

        pr=0.0

        # Uniform prior on velocity offset

        # Jeffreys scale prior on sigma
        for s in p.sigma:
            pr -= np.sum(np.log(s))

        # Jeffreys scale prior on tau
        for t in p.tau:
            pr -= np.sum(np.log(t))

        # Jeffreys scale prior on K
        for k in p.K:
            pr -= np.sum(np.log(k))

        # Jeffreys scale prior on n
        for n in p.n:
            pr -= np.sum(np.log(n))

        # Uniform prior on chi

        # Thermal prior on e
        for e in p.e:
            pr += np.sum(np.log(e))

        # Uniform prior on omega

        return pr

class LogLikelihood(object):
    """Log likelihood."""
    def __init__(self, ts, rvs):
        self._ts = ts
        self._rvs = rvs

    @property
    def ts(self):
        return self._ts

    @property
    def rvs(self):
        return self._rvs

    def __call__(self, p):
        nobs=len(self.rvs)
        npl=(p.shape[-1]-3*nobs)/5

        p = params.Parameters(p, nobs=nobs, npl=npl)

        ll=0.0

        for t, rvobs, V, sigma, tau in zip(self.ts, self.rvs, p.V, p.sigma, p.tau):
            if npl == 0:
                rvmodel=np.zeros_like(t)
            else:
                rvmodel=np.sum(rv.rv_model(t, p), axis=0)

            residual=rvobs-rvmodel-V
            cov=generate_covariance(t, sigma, tau)

            ll += correlated_gaussian_loglikelihood(residual, np.zeros_like(residual), cov)

        return ll
            
def min_rv_mean_error(rvs):
    """Returns the minimum error across all observations on the mean
    of the radial velocities."""

    muerr=float('inf')
    for rv in rvs:
        err=np.std(rv)/np.sqrt(len(rv))
        muerr=min(err,muerr)

    return muerr

def prior_bounds_from_data(npl, ts, rvs):
    """Returns conservative prior bounds (pmin, pmax) given sampling
    times for each observatory."""

    nobs=len(ts)

    dts=[np.diff(t) for t in ts]
    min_dt=reduce(min, [np.min(dt) for dt in dts])

    tobss=[t[-1]-t[0] for t in ts]
    max_obst=reduce(max, tobss)

    pmin=params.Parameters(nobs=nobs,npl=npl)
    pmax=params.Parameters(nobs=nobs,npl=npl)

    pmin[:]=0.0
    pmax[:]=float('inf')

    pmin.V = float('-inf')

    pmin.tau = min_dt/10.0
    pmax.tau = 10.0*max_obst

    if npl >= 1:
        pmin.n = 2.0*np.pi/(max_obst)
        pmax.n = 2.0*np.pi/(min_dt)

        pmax.chi = 1.0
        
        pmax.e = 1.0
    
        pmax.omega = 2.0*np.pi

    return pmin, pmax

def generate_initial_sample(ts, rvs, ntemps, nwalkers, nobs=1, npl=1):
    """Generates an initial sample of parameters for
    single-observatory, single-planet setups."""

    ts=np.concatenate(ts)
    rvs=np.concatenate(rvs)

    mu=np.mean(rvs)
    sig=np.std(rvs)
    sqrtN=np.sqrt(rvs.shape[0])

    T=np.amax(ts)-np.amin(ts)
    dtmin=np.min(np.diff(np.sort(ts)))

    nmin=2.0*np.pi/T
    nmax=2.0*np.pi/dtmin

    samps=params.Parameters(arr=np.zeros((ntemps, nwalkers, 3*nobs+5*npl)), nobs=nobs, npl=npl)

    samps.V = nr.normal(loc=mu, scale=sig/sqrtN, size=(ntemps,nwalkers,nobs))
    samps.sigma = nr.lognormal(mean=np.log(sig), sigma=1.0/sqrtN, size=(ntemps,nwalkers,nobs))
    samps.tau = nr.uniform(low=dtmin, high=T, size=(ntemps, nwalkers,nobs))
    if npl >= 1:
        samps.K = np.reshape(np.min(samps.sigma, axis=2)/10.0, (ntemps, nwalkers, 1)) + nr.lognormal(mean=np.log(sig), sigma=1.0/sqrtN, size=(ntemps,nwalkers,npl))

        # Make sure that periods are increasing
        samps.n = np.sort(nr.uniform(low=nmin, high=nmax, size=(ntemps, nwalkers,npl)))[:,:,::-1]

        samps.e = nr.uniform(low=0.0, high=1.0, size=(ntemps, nwalkers,npl))
        samps.chi = nr.uniform(low=0.0, high=1.0, size=(ntemps, nwalkers,npl))
        samps.omega = nr.uniform(low=0.0, high=2.0*np.pi, size=(ntemps, nwalkers,npl))

    return samps

def recenter_samples(ts, chains, logls, sigmafactor=0.1):
    """Generates a suite of samples around the maximum likelihood
    point in chains, with a reasonable error distribution."""

    sf=sigmafactor

    T=ts[-1]-ts[0]
    
    ibest=np.argmax(logls)
    p0=params.Parameters(np.reshape(chains, (-1, chains.shape[-1]))[ibest, :])

    ncycle=T/p0.P
    ncorr=T/p0.tau
    nobs=len(ts)

    samples=params.Parameters(np.copy(chains))

    assert samples.npl == 1, 'require exactly one planet'
    assert samples.nobs == 1, 'require exactly one observatory'

    samples.V = np.random.normal(loc=p0.V, scale=sf*p0.sigma/np.sqrt(nobs), size=samples.V.shape)
    samples.sigma = np.random.lognormal(mean=np.log(p0.sigma), sigma=sf/np.sqrt(ncorr), size=samples.sigma.shape)
    samples.tau = np.random.lognormal(mean=np.log(p0.tau), sigma=sf/np.sqrt(ncorr), size=samples.tau.shape)
    samples.K = np.random.normal(loc=p0.K, scale=sf*p0.K/np.sqrt(nobs), size=samples.K.shape)
    samples.n = np.random.lognormal(mean=np.log(p0.n), sigma=sf/np.sqrt(ncycle), size=samples.n.shape)
    samples.chi = np.random.lognormal(mean=np.log(p0.chi), sigma=sf/np.sqrt(ncycle), size=samples.chi.shape)
    samples.e = np.random.lognormal(mean=np.log(p0.e), sigma=sf/np.sqrt(ncycle), size=samples.e.shape)
    samples.omega = np.random.lognormal(mean=np.log(p0.omega), sigma=sf/np.sqrt(ncycle), size=samples.omega.shape)

    return samples
    
