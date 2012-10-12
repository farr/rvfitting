import acor
import emcee as em
import multiprocessing as multi
import numpy as np
import numpy.random as nr

class PTPost(object):
    """Wrapper for posterior used in emcee."""
    
    def __init__(self, logl, logp, beta):
        """Initialize with given log-likelihood, log-prior, and beta = 1/T."""

        self._logl = logl
        self._logp = logp
        self._beta = beta

    def __call__(self, x):
        """Returns lnpost(x), lnlike(x) (the second value will be
        treated as a blob by emcee), where lnpost(x) = beta*lnlike(x)
        + lnprior(x)."""

        lp = self._logp(x)

        # If outside prior bounds, return 0.
        if lp == float('-inf'):
            return lp, lp

        ll = self._logl(x)

        return self._beta*ll+lp, ll

class PTSampler(object):
    """A parallel-tempered ensemble sampler."""

    def __init__(self, ntemps, nwalkers, dim, logl, logp, threads=1, pool=None):
        """Initialize a sampler with the given log-likelihood and
        log-prior functions.  If nthreads > 1, the sampler will use a
        multiprocessing pool to perform the ensemble moves."""

        self.ntemps = ntemps
        self.nwalkers = nwalkers
        self.dim = dim

        self.betas = exponential_beta_ladder(ntemps)

        self.nswap = np.zeros(ntemps, dtype=np.float)
        self.nswap_accepted = np.zeros(ntemps, dtype=np.float)

        self.pool = pool
        if threads > 1 and pool is None:
            self.pool = multi.Pool(threads)

        self.samplers = [em.EnsembleSampler(nwalkers, dim, PTPost(logl, logp, b), pool=self.pool) for b in self.betas]

    def reset(self):
        """Clear the stored samplers."""
    
        for s in self.samplers:
            s.reset()

    def sample(self, p0, lnprob0=None, logl0=None, iterations=1, storechain=True):
        """Advance the chains iterations steps as a generator.  p0
        should have shape (ntemps, nwalkers, dim)."""

        p = p0

        # If we have no lnprob or blobs, then run at least one
        # iteration to compute them.
        if lnprob0 is None or logl0 is None:
            iterations -= 1
            
            lnprob = []
            logl = []
            for i,s in enumerate(self.samplers):
                for psamp, lnprobsamp, rstatesamp, loglsamp in s.sample(p[i,...], storechain=storechain):
                    p[i,...] = psamp
                    lnprob.append(lnprobsamp)
                    logl.append(loglsamp)

            lnprob = np.array(lnprob) # Dimensions (ntemps, nwalkers)
            logl = np.array(logl)

            p,lnprob,logl = self.temperature_swaps(p, lnprob, logl)
        else:
            lnprob = lnprob0
            logl = logl0

        for i in range(iterations):
            for i,s in enumerate(self.samplers):
                for psamp, lnprobsamp, rstatesamp, loglsamp in s.sample(p[i,...], lnprob0=lnprob[i,...], blobs0=logl[i,...], storechain=storechain):
                    p[i,...] = psamp
                    lnprob[i,...] = lnprobsamp
                    logl[i,...] = np.array(loglsamp)

            p,lnprob,logl = self.temperature_swaps(p, lnprob, logl)

            yield p, lnprob, logl

    def temperature_swaps(self, p, lnprob, logl):
        """Perform parallel-tempering temperature swaps on the state
        in p with associated lnprob and logl."""

        ntemps=self.ntemps

        for i in range(ntemps-1, 0, -1):
            bi=self.betas[i]
            bi1=self.betas[i-1]

            dbeta = bi1-bi

            for j in range(self.nwalkers):
                self.nswap[i] += 1
                self.nswap[i-1] += 1

                ii=nr.randint(self.nwalkers)
                jj=nr.randint(self.nwalkers)

                paccept = dbeta*(logl[i, ii] - logl[i-1, jj])

                if paccept > 0 or np.log(nr.rand()) < paccept:
                    self.nswap_accepted[i] += 1
                    self.nswap_accepted[i-1] += 1

                    ptemp=np.copy(p[i, ii, :])
                    logltemp=logl[i, ii]
                    lnprobtemp=lnprob[i, ii]

                    p[i,ii,:]=p[i-1,jj,:]
                    logl[i,ii]=logl[i-1, jj]
                    lnprob[i,ii] = lnprob[i-1,jj] - dbeta*logl[i-1,jj]

                    p[i-1,jj,:]=ptemp
                    logl[i-1,jj]=logltemp
                    lnprob[i-1,jj]=lnprobtemp + dbeta*logltemp

        return p, lnprob, logl

    @property 
    def chain(self):
        """Returns the chain of samples.  Will have shape that is
        (Ntemps, Nwalkers, Nsteps, Ndim)."""

        return np.array([s.chain for s in self.samplers])

    @property
    def lnprobability(self):
        """Matrix of lnprobability values, of shape (Ntemps,
        Nwalkers, Nsteps)"""
        return np.array([s.lnprobability for s in self.samplers])

    @property
    def lnlikelihood(self):
        """Matrix of ln-likelihood values of shape (Ntemps, Nwalkers,
        Nsteps)."""
        return np.array([np.transpose(np.array(s.blobs)) for s in self.samplers])

    @property
    def tswap_acceptance_fraction(self):
        """Returns an array of accepted temperature swap fractions for
        each temperature."""
        return self.nswap_accepted / self.nswap

    @property
    def acceptance_fraction(self):
        """Matrix of shape (Ntemps, Nwalkers) detailing the acceptance
        fraction for each walker."""
        return np.array([s.acceptance_fraction for s in self.samplers])

    @property
    def acor(self):
        """Returns a matrix of autocorrelation lengths of shape
        (Ntemps, Ndim)."""
        return np.array([s.acor for s in self.samplers])

def exponential_beta_ladder(ntemps):
    """Exponential ladder in T, increasing by sqrt(2) each step, with
    ntemps in total."""
    return np.exp(np.linspace(0, -(ntemps-1)*0.5*np.log(2), ntemps))

def thermodynamic_log_evidence(logls):
    """Computes the evidence integral from the (Nsamples,
    Ntemperatures, Nwalkers) set of log(L) samples."""

    nsamp,ntemp,nwalk=logls.shape

    mean_logls=np.mean(np.mean(logls, axis=2), axis=0)

    betas=exponential_beta_ladder(ntemp)
    betas=np.concatenate((betas, np.array([0.0])))

    return -np.sum(mean_logls*np.diff(betas))

def burned_in_samples(pts, fburnin=0.1):
    """Returns the samples from pts after an initial burnin
    fraction.  pts should have shape (Nsamples, ...)."""

    iburnin=int(fburnin*pts.shape[0]+1)

    post_burnin_shape=(-1,) + pts.shape[1:]

    return np.reshape(np.reshape(pts, (pts.shape[0], -1))[iburnin:, :], post_burnin_shape)

def decorrelated_samples(pts):
    """Returns a subset of pts that is downsampled by the longest
    correlation length in pts.  pts should have shape (Nsamples,
    Nwalkers, Ndim)."""

    means=np.mean(pts, axis=1)

    taumax=float('-inf')
    for j in range(means.shape[1]):
        tau,mu,sigma=acor.acor(means[:,j])
        taumax=max(tau,taumax)

    taumax=int(taumax+1)

    return pts[::taumax, :, :]

    
