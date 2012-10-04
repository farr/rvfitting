#!/usr/bin/env python

import acor
import matplotlib.pyplot as pp
import numpy as np
import numpy.random as nr
import ptsampler as pt
import scipy.stats as ss

mu0=0.5
mu1=-0.5

sigma=0.1

class Logp(object):
    def __call__(self, x):
        return 0.0

class Logl(object):
    def __call__(self, x):
        x0=np.array([mu0, mu0])
        x1=np.array([mu1, mu1])

        d0=np.sum(-(x-x0)*(x-x0)/(2*sigma*sigma))
        d1=np.sum(-(x-x1)*(x-x1)/(2*sigma*sigma))

        return np.logaddexp(d0,d1)

def truedist(x,T):
    return 0.5*(ss.norm(loc=mu0, scale=sigma*np.sqrt(T)).pdf(x) + ss.norm(loc=mu1, scale=sigma*np.sqrt(T)).pdf(x))

if __name__ == '__main__':
    nthreads=3
    nsamps=10000
    ntemp=12
    nwalkers=100

    logl=Logl()
    logp=Logp()

    ts=1/np.linspace(1,0,ntemp+1)[:-1]

    sampler=pt.PTSampler(logl, logp, nthreads=nthreads)

    savepts=[]
    for i, (pts, logls, logps) in enumerate(sampler.samples(nr.rand(ntemp, nwalkers,2), niters=nsamps)):
        savepts.append(pts)

    savepts=np.array(savepts)

    print savepts.shape

    xs=np.linspace(-2,2,10000)

    for i in range(ntemp):
        samps=savepts[:,i,:,0]
        samps=samps[samps.shape[0]/2:, :]
        means=np.mean(samps, axis=1)
        tau,dummy1,dummy2=acor.acor(means)
        print 'Temperature %d (%g) has acl = %g, mean=%g, sigma = %g'%(i,ts[i],tau,dummy1,dummy2)
        samps=samps[-1::-int(tau), :].flatten()
        pp.plot(xs, ss.gaussian_kde(samps)(xs), label='MCMC')
        pp.plot(xs, truedist(xs, ts[i]), label='True')
        pp.legend()
        pp.show()
