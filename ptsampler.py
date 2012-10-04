import emcee as em
import multiprocessing as multi
import numpy as np
import numpy.random as nr

class DEStep(object):
    """Callable class that implements one step of differential
    evolution for multiprocessing pool."""

    def __init__(self, pts, logls, logps, beta, logl, logp):
        """Init with given points (for de proposal), log-likelihoods,
        log-priors, and beta (inverse temperature)."""
        self.pts=pts
        self.logls=logls
        self.logps=logps
        self.beta=beta
        self.logl=logl
        self.logp=logp

    def __call__(self, i):
        """Return a differentially-evolved new point for walker i."""
        pts=self.pts
        logls=self.logls
        logps=self.logps
        beta=self.beta

        nwalkers,ndim=pts.shape

        x=np.copy(pts[i,:])
        lx=logls[i]
        px=logps[i]

        ii=nr.randint(nwalkers)
        jj=nr.randint(nwalkers)
        while jj == ii:
            jj=nr.randint(nwalkers)

        dx=2.38/np.sqrt(2.0*ndim)*nr.randn()*(pts[jj,:]-pts[ii,:])

        xnew=x+dx
        lxnew=self.logl(xnew)
        pxnew=self.logp(xnew)

        if beta*lxnew + pxnew > beta*lx + px or np.log(nr.rand()) < beta*lxnew+pxnew-beta*lx-px:
            # Accept step:
            return xnew, lxnew, pxnew
        else:
            # Reject step:
            return x, lx, px

class LogLLogP(object):
    def __init__(self, pts, fn):
        self.pts=pts
        self.fn=fn

    def __call__(self, j):
        return self.fn(self.pts[j,:])

class ChunkedMap(object):
    def __init__(self, pool, chunksize):
        self.pool = pool
        self.chunksize = chunksize

    def __call__(self, fn, iterable):
        if self.pool is not None:
            return self.pool.map(fn, iterable, chunksize=self.chunksize)
        else:
            return map(fn, iterable)

class PTSampler(object):
    """A parallel-tempered ensemble sampler."""

    def __init__(self, logl, logp, nthreads=1):
        """Initialize a sampler with the given log-likelihood and
        log-prior functions.  If nthreads > 1, the sampler will use a
        multiprocessing pool to perform the ensemble moves."""
        self.logl=logl
        self.logp=logp
        self.nthreads=nthreads

        if nthreads > 1:
            self.pool = multi.Pool(processes=nthreads)
        else:
            self.pool = None

    def samples(self, pts, logls=None, logps=None, niters=None):
        """Given a (num_temps, n_walkers, ndim) array of initial
        points, returns a sequence of (newpts, logls, logps).  logls
        and logps should have shape (n_temps, n_walkers), if they are
        given.  The iteration will terminate after niters steps
        (unless niters is None)."""

        logl=self.logl
        logp=self.logp

        ntemps,nwalkers,ndim = pts.shape

        betas=np.linspace(1,0,ntemps+1)[:-1]

        if logls is None:
            logls=np.zeros((ntemps,nwalkers))
            for i in range(ntemps):
                ll=LogLLogP(pts[i,:,:], logl)
                logls[i,:]=np.array(ChunkedMap(self.pool, nwalkers/self.nthreads+1)(ll, range(nwalkers)))

        if logps is None:
            logps=np.zeros((ntemps,nwalkers))
            for i in range(ntemps):
                lp=LogLLogP(pts[i,:,:], logp)
                logps[i,:]=np.array(ChunkedMap(self.pool, nwalkers/self.nthreads+1)(lp, range(nwalkers)))

        iiter=0
        while True:
            # Run one step of de sampling:
            for i,beta in enumerate(betas):
                evolvefn=DEStep(pts[i,:,:], logls[i,:], logps[i,:], beta, logl, logp)
                new_states=ChunkedMap(self.pool, nwalkers/self.nthreads+1)(evolvefn, range(nwalkers))

                for j,(x,lx,px) in enumerate(new_states):
                    pts[i,j,:]=x
                    logls[i,j]=lx
                    logps[i,j]=px

            # Now do temperature swaps
            for i,beta1 in enumerate(betas[:-1]):
                beta2=betas[i+1]
                
                for j in range(nwalkers):
                    ii=nr.randint(nwalkers)
                    jj=nr.randint(nwalkers)

                    l1=logls[i,ii]
                    l2=logls[i+1, jj]

                    ll=(beta2-beta1)*l1 + (beta1-beta2)*l2

                    if ll > 0.0 or np.log(nr.rand()) < ll:
                        # Accept swap
                        temp=np.copy(pts[i,ii,:])
                        templ=logls[i,ii]
                        tempp=logps[i,ii]

                        pts[i,ii,:]=pts[i+1,jj,:]
                        logls[i,ii]=logls[i+1,jj]
                        logps[i,ii]=logps[i+1,jj]

                        pts[i+1,jj,:]=temp
                        logls[i+1,jj]=templ
                        logps[i+1,jj]=tempp
                        
            iiter+=1
            yield np.copy(pts), np.copy(logls), np.copy(logps)

            if niters is not None and iiter >= niters:
                break
