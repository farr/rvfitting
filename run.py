#!/usr/bin/env python

import acor
from argparse import ArgumentParser
import correlated_likelihood as cl
import numpy as np
import os
from parameters import Parameters
from ptsampler import PTSampler
import tempfile
import sys

def load_data(files):
    ts=[]
    rvs=[]

    for file in files:
        data=np.loadtxt(file)
        ts.append(data[:,0])
        rvs.append(data[:,1])

    return ts,rvs

def overwrite(file, data):
    with tempfile.NamedTemporaryFile(dir=os.path.dirname(file), delete=False) as out:
        name=out.name
        np.savetxt(out, np.reshape(data, (-1, data.shape[-1])))

    os.rename(name, file)

if __name__ == '__main__':
    parser=ArgumentParser()

    parser.add_argument('--output', metavar='FILE', required=True, help='output file')
    parser.add_argument('--likelihood', metavar='FILE', required=True, default=None, help='likelihood file')
    parser.add_argument('--prior', metavar='FILE', required=True, default=None, help='prior file')

    parser.add_argument('--nthreads', metavar='N', type=int, default=1, help='number of parallel threads')
    parser.add_argument('--nplanets', metavar='N', type=int, default=1, help='number of planets')
    parser.add_argument('--nthin', metavar='N', type=int, default=4, help='iterations between output')

    parser.add_argument('--ntemps', metavar='N', type=int, default=10, help='number of temperatures')
    parser.add_argument('--nwalkers', metavar='N', type=int, default=100, help='number of walkers')

    parser.add_argument('--reset', default=False, action='store_const', const=True, help='overwrite contents of output with new samples')

    parser.add_argument('--rvs', metavar='FILE', required=True, default=[], action='append', help='file of times and RV\'s')

    args=parser.parse_args()

    pts=np.loadtxt(args.output)
    logls=np.loadtxt(args.likelihood)
    logps=np.loadtxt(args.prior)

    # Reshape
    pts=np.reshape(pts, (-1, args.ntemps, args.nwalkers, pts.shape[1]))[-1, :,:,:]
    logls=np.reshape(logls, (-1, args.ntemps, args.nwalkers))[-1,:,:]
    logps=np.reshape(logps, (-1, args.ntemps, args.nwalkers))[-1,:,:]

    if args.reset:
        overwrite(args.output, pts)
        overwrite(args.likelihood, logls)
        overwrite(args.prior, logps)
    
    ts, rvs=load_data(args.rvs)

    pmin,pmax=cl.prior_bounds_from_times(args.nplanets, ts)

    log_likelihood=cl.LogLikelihood(ts, rvs)
    log_prior=cl.LogPrior(pmin=pmin, pmax=pmax)

    sampler=PTSampler(log_likelihood, log_prior, nthreads=args.nthreads)

    start_logl=np.amax(logls)

    print 'max(log(L)) med(log(L)) min(log(L)) <afrac> acorr(<log(L)>)'
    sys.stdout.flush()

    mean_logls=[np.mean(logls[0,:])]
    while True:
        for pts, logls, logps, afrac in sampler.samples(pts, logls=logls, logps=logps, niters=args.nthin):
            pass

        mean_logls.append(np.mean(logls[0,:]))

        n=len(mean_logls)/10

        try:
            tau,mu,sigma=acor.acor(np.array(mean_logls[n:]))
        except:
            tau=float('inf')

        print '%11.1f %11.1f %11.1f %7.2f %15.1f'%(np.amax(logls[0,:]), np.median(logls[0,:]), np.min(logls[0,:]), np.mean(afrac[0,:]), tau)
        sys.stdout.flush()
            
        with open(args.output, 'a') as out:
            np.savetxt(out, np.reshape(pts, (-1, pts.shape[-1])))
        with open(args.likelihood, 'a') as out:
            np.savetxt(out, logls)
        with open(args.prior, 'a') as out:
            np.savetxt(out, logps)
