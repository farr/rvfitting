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

def generate_covariance(ts, sigma0, tau):
    """Generates a covariance matrix according to an exponential
    autocovariance: cov(t_i, t_j) =
    sigma0*sigma0*exp(-|ti-tj|/tau)."""

    ndim = ts.shape[0]

    tis = np.tile(np.reshape(ts, (-1, 1)), (1, ndim))
    tjs = np.tile(ts, (ndim, 1))

    return sigma0*sigma0*np.exp(-np.abs(tis-tjs)/tau)

class LogPrior(object):
    """Log of the prior function."""

    def __init__(self, pmin=None, pmax=None):
        """Initialize with the given bounds on the priors."""

        self._pmin = pmin
        self._pmax = pmax

    def __call__(self, p):
        p = params.Parameters(p)

        if self._pmin is None:
            self._pmin = 0.0*p
            self._pmin.V = float('-inf')

        if self._pmax is None:
            self._pmax = p + float('inf')
            self._pmax.chi = 1.0
            self._pmax.e = 1.0
            self._pmax.omega = 2.0*np.pi

        # Check bounds
        if np.any(p <= self._pmin) or np.any(p >= self._pmax):
            return float('-inf')

        pr=0.0

        # Uniform prior on velocity offset

        # Jeffreys scale prior on sigma0
        for s in p.sigma0:
            pr -= np.log(s)

        # Jeffreys scale prior on tau
        for t in p.tau:
            pr -= np.log(t)

        # Jeffreys scale prior on K
        for k in p.K:
            pr -= np.log(k)

        # Jeffreys scale prior on n
        for n in p.n:
            pr -= np.log(n)

        # Uniform prior on chi

        # Thermal prior on e
        for e in p.e:
            pr += np.log(e)

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
        p = params.Parameters(p)

        ll=0.0

        for t, rvobs, V, sigma0, tau in zip(self.ts, self.rvs, p.V, p.sigma0, p.tau):
            rvmodel=np.sum(rv.rv_model(t, p), axis=0)

            residual=rvobs-rvmodel-V
            cov=generate_covariance(t, sigma0, tau)

            ll += correlated_gaussian_loglikelihood(residual, np.zeros_like(residual), cov)

        return ll
            
def prior_bounds_from_times(npl, ts):
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

    pmin.n = 2.0*np.pi/(10.0*max_obst)
    pmax.n = 2.0*np.pi/(min_dt/10.0)

    pmax.chi = 1.0

    pmax.e = 1.0
    
    pmax.omega = 2.0*np.pi

    return pmin, pmax

def generate_initial_sample(p, n):
    """Generates an initial sample of n parameters around the values
    in p."""

    assert p.nobs == 1, 'Cannot generate initial data for more than one observatory yet.'
    assert p.npl == 1, 'Cannot generate initial data for more than one planet yet.'

    if p.V == 0:
        V_scale = 1.0
    else:
        V_scale = np.abs(p.V)

    Vs=nr.normal(loc=p.V, scale=V_scale, size=n)
    sigma0s=nr.lognormal(mean=np.log(p.sigma0), size=n)
    taus=nr.lognormal(mean=np.log(p.tau), size=n)
    Ks=nr.lognormal(mean=np.log(p.K), size=n)
    nns=nr.lognormal(mean=np.log(p.n), size=n)
    chis=nr.rand(n)
    es=nr.rand(n)
    omegas=2.0*np.pi*nr.rand(n)

    samps=[]
    for V,sigma0,tau,K,nn,chi,e,omega in zip(Vs, sigma0s, taus, Ks, nns, chis, es, omegas):
        samps.append(params.Parameters(V=V,sigma0=sigma0,tau=tau,K=K,n=nn,chi=chi,e=e,omega=omega))

    return np.array(samps)

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

    samples.V = np.random.normal(loc=p0.V, scale=sf*p0.sigma0/np.sqrt(nobs), size=samples.V.shape)
    samples.sigma0 = np.random.lognormal(mean=np.log(p0.sigma0), sigma=sf/np.sqrt(ncorr), size=samples.sigma0.shape)
    samples.tau = np.random.lognormal(mean=np.log(p0.tau), sigma=sf/np.sqrt(ncorr), size=samples.tau.shape)
    samples.K = np.random.normal(loc=p0.K, scale=sf*p0.K/np.sqrt(nobs), size=samples.K.shape)
    samples.n = np.random.lognormal(mean=np.log(p0.n), sigma=sf/np.sqrt(ncycle), size=samples.n.shape)
    samples.chi = np.random.lognormal(mean=np.log(p0.chi), sigma=sf/np.sqrt(ncycle), size=samples.chi.shape)
    samples.e = np.random.lognormal(mean=np.log(p0.e), sigma=sf/np.sqrt(ncycle), size=samples.e.shape)
    samples.omega = np.random.lognormal(mean=np.log(p0.omega), sigma=sf/np.sqrt(ncycle), size=samples.omega.shape)

    return samples
    
