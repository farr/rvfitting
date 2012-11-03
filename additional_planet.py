#!/usr/bin/env python

from argparse import ArgumentParser
import correlated_likelihood as cl
from gzip import GzipFile
import numpy as np
import numpy.random as nr
import parameters as pr

def draw_logarithmic(min, max, size=1):
    """Returns an array of samples drawn from a flat-in-log distribution.

    :param min: Minimum value of a sample.

    :param max: Maximum value of a sample.

    :param size (optional): Size of the result array."""

    lmin = np.log(min)
    lmax = np.log(max)

    return np.exp(nr.uniform(low=lmin, high=lmax, size=size))

if __name__ == '__main__':
    parser=ArgumentParser()
    
    parser.add_argument('--input', metavar='FILE', required=True, help='input chain file')
    parser.add_argument('--prefix', metavar='FILE', default='chain', help='output file prefix')
    parser.add_argument('--rvs', metavar='FILE', default=[], action='append', help='radial velocity file')

    parser.add_argument('--nwalkers', metavar='N', default=1000, type=int, help='number of ensemble walkers')
    parser.add_argument('--npl', metavar='N', default=1, type=int, help='number of planets in chain')
    parser.add_argument('--ntemps', metavar='N', default=20, type=int, help='number of temperatures')

    args=parser.parse_args()

    ts=[]
    rvs=[]
    for f in args.rvs:
        data=np.loadtxt(f)
        ts.append(data[:,0])
        rvs.append(data[:,1])

    nobs = len(ts)
    npl = args.npl
    newnpl = npl + 1

    pmin,pmax = cl.prior_bounds_from_data(newnpl, ts, rvs)

    chain = pr.Parameters(arr=np.loadtxt(args.input)[-args.nwalkers:, 2:], npl=npl, nobs=nobs)
    newchain = pr.Parameters(arr=np.zeros((args.nwalkers, chain.shape[1]+5)), npl=newnpl, nobs=nobs)

    newchain[:, :-5] = chain
    
    newks = newchain.K
    newks[:, -1] = draw_logarithmic(pmin.K[0], pmax.K[0], size=args.nwalkers)
    newchain.K = newks

    newes = newchain.e
    newes[:,-1] = nr.uniform(low=0.0, high=1.0, size=args.nwalkers)
    newchain.e = newes

    newchis = newchain.chi
    newchis[:,-1] = nr.uniform(low=0.0, high=1.0, size=args.nwalkers)
    newchain.chi = newchis

    newomegas = newchain.omega
    newomegas[:,-1] = nr.uniform(low=0.0, high=2.0*np.pi, size=args.nwalkers)
    newchain.omega = newomegas

    newns = newchain.n
    nmin = pmin.n[0]
    for i in range(args.nwalkers):
        newns[i,-1] = draw_logarithmic(nmin, newns[i,-2])
    newchain.n = newns

    logl = cl.LogLikelihood(ts, rvs)
    logp = cl.LogPrior(pmin, pmax, newnpl, nobs)

    oldpmin,oldpmax = cl.prior_bounds_from_data(npl, ts, rvs)

    oldlogp = cl.LogPrior(oldpmin, oldpmax, npl, nobs)

    for i in range(args.nwalkers):
        if oldlogp(chain[i,:]) == float('-inf'):
            print 'Found one'

    logls=np.zeros(args.nwalkers)
    logps=np.zeros(args.nwalkers)
    for i in range(args.nwalkers):
        logls[i] = logl(newchain[i,:])
        logps[i] = logp(newchain[i,:])

    for i in range(args.ntemps):
        header=newchain.header[0] + ' logl logp' + newchain.header[1:]
        with GzipFile('%s.%02d.txt.gz'%(args.prefix, i), 'w') as out:
            out.write(header)
            np.savetxt(out, np.column_stack((logls, logps, newchain)))
