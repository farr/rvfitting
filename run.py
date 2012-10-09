#!/usr/bin/env python

import acor
from argparse import ArgumentParser
import correlated_likelihood as cl
from gzip import GzipFile
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

if __name__ == '__main__':
    parser=ArgumentParser()

    parser.add_argument('--prefix', metavar='PRE', default='chain', help='output prefix (files will be <prefix>.NN.txt.gz)')

    parser.add_argument('--nthreads', metavar='N', type=int, default=1, help='number of parallel threads')
    parser.add_argument('--nplanets', metavar='N', type=int, default=1, help='number of planets')
    parser.add_argument('--nthin', metavar='N', type=int, default=4, help='iterations between output')

    parser.add_argument('--ntemps', metavar='N', type=int, default=10, help='number of temperatures')
    parser.add_argument('--nwalkers', metavar='N', type=int, default=100, help='number of walkers')

    parser.add_argument('--rvs', metavar='FILE', required=True, default=[], action='append', help='file of times and RV\'s')

    args=parser.parse_args()

    ts, rvs=load_data(args.rvs)

    pmin,pmax=cl.prior_bounds_from_data(args.nplanets, ts, rvs)

    try:
        pts=[]
        logls=[]
        logps=[]

        for i in range(args.ntemps):
            data=np.loadtxt('%s.%02d.txt.gz'%(args.prefix, i))
            pts.append(data[-args.nwalkers:, 2:])
            logls.append(data[-args.nwalkers:,0])
            logps.append(data[-args.nwalkers:,1])

        pts=np.array(pts)
        logls=np.array(logls)
        logps=np.array(logps)
    except:
        pts=cl.generate_initial_sample(ts, rvs, args.ntemps, args.nwalkers, nobs=len(args.rvs), npl=args.nplanets)
        logls=None
        logps=None

    log_likelihood=cl.LogLikelihood(ts, rvs)
    log_prior=cl.LogPrior(pmin=pmin, pmax=pmax, npl=args.nplanets, nobs=len(args.rvs))

    sampler=PTSampler(log_likelihood, log_prior, nthreads=args.nthreads)

    print 'max(log(P)) med(log(P)) min(log(P)) <afrac> <tfrac> acorr(<log(P)>)'
    sys.stdout.flush()

    mean_logps=[]
    while True:
        for pts, logls, logps, afrac, tfrac in sampler.samples(pts, logls=logls, logps=logps, niters=args.nthin):
            pass

        mean_logps.append(np.mean(logls[0,:]+logps[0,:]))

        n=len(mean_logps)/10

        try:
            tau,mu,sigma=acor.acor(np.array(mean_logps[n:]))
        except:
            tau=float('inf')

        for i in range(args.ntemps):
            with GzipFile('%s.%02d.txt.gz'%(args.prefix, i), 'a') as out:
                np.savetxt(out, np.column_stack((logls[i,...], logps[i,...], pts[i,...])))

        with GzipFile('%s.accept.txt.gz'%args.prefix, 'a') as out:
            np.savetxt(out, np.reshape(np.mean(afrac, axis=1), (1, -1)))
        with GzipFile('%s.aswaps.txt.gz'%args.prefix, 'a') as out:
            np.savetxt(out, np.reshape(tfrac, (1, -1)))

        print '%11.1f %11.1f %11.1f %7.2f %7.2f %15.1f'%(np.amax(logls[0,:]+logps[0,:]), np.median(logls[0,:]+logps[0,:]), np.min(logls[0,:]+logps[0,:]), np.mean(afrac[0,:]), tfrac[0], tau)
        sys.stdout.flush()
            
