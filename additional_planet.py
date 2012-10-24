#!/usr/bin/env python

from argparse import ArgumentParser
import correlated_likelihood as cl
from gzip import GzipFile
import multiprocessing
import numpy as np
import numpy.random as nr
from parameters import Parameters

if __name__ == '__main__':
    parser=ArgumentParser()

    parser.add_argument('--input', metavar='FILE', required=True, help='input file')
    parser.add_argument('--rvs', metavar='FILE', required=True, action='append', default=[], help='rv file')
    
    parser.add_argument('--prefix', metavar='FILE', default='chain', help='filename prefix')

    parser.add_argument('--nwalkers', metavar='N', default=100, type=int, help='number of ensemble walkers')

    parser.add_argument('--nthreads', metavar='N', default=2, type=int, help='number of parallel threads')

    parser.add_argument('--npl', metavar='N', default=1, type=int, help='number of planets')
    parser.add_argument('--nobs', metavar='N', default=1, type=int, help='number of observatories')

    parser.add_argument('--ntemps', metavar='N', default=20, type=int, help='number of temperatures')

    args=parser.parse_args()

    state=np.loadtxt(args.input)[..., 2:]
    state=Parameters(state[-args.nwalkers:,:], npl=args.npl, nobs=args.nobs)

    ts=[]
    rvs=[]
    for rv in args.rvs:
        data=np.loadtxt(rv)
        ts.append(data[:,0])
        rvs.append(data[:,1])

    pmin,pmax=cl.prior_bounds_from_data(args.npl+1, ts, rvs)

    logl=cl.LogLikelihood(ts, rvs)
    logp=cl.LogPrior(pmin=pmin, pmax=pmax, npl=args.npl+1, nobs=args.nobs)

    newstate=Parameters(np.zeros((args.nwalkers, 3*args.nobs+5*(args.npl + 1))), npl=args.npl+1, nobs=args.nobs)

    newstate[:, :-5] = state

    sigmas=np.min(newstate.sigma, axis=1)

    temp = newstate.K
    temp[:,-1] = sigmas/10.0 + nr.lognormal(mean=np.log(np.mean(sigmas)), sigma=0.1, size=args.nwalkers)
    newstate.K = temp

    temp = newstate.n
    temp[:,-1] = np.exp(nr.uniform(size=args.nwalkers)*(np.log(newstate.n[:,-2]) - np.log(pmin.n[0])) + np.log(pmin.n[0]))
    newstate.n = temp

    temp = newstate.e
    temp[:, -1] = nr.uniform(size=args.nwalkers)
    newstate.e = temp

    temp=newstate.omega
    temp[:,-1] = 2.0*np.pi*nr.uniform(size=args.nwalkers)
    newstate.omega=temp

    temp=newstate.chi
    temp[:,-1] = nr.uniform(size=args.nwalkers)
    newstate.chi=temp

    if args.nthreads > 1:
        pool=multiprocessing.Pool(args.nthreads)
        mm=pool.map
    else:
        mm=map
    logls=np.array(mm(logl, newstate))
    logps=np.array(mm(logp, newstate))

    for i in range(args.ntemps):
        with GzipFile('%s.%02d.txt.gz'%(args.prefix, i), 'w') as out:
            out.write(newstate.header[:1] + ' logl logp' + newstate.header[1:])
            np.savetxt(out, np.column_stack((logls, logps, newstate)))
