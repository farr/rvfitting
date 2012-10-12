#!/usr/bin/env python

import acor
from argparse import ArgumentParser
import correlated_likelihood as cl
from gzip import GzipFile
import numpy as np
import os
from parameters import Parameters
from emcee.ptsampler import PTSampler
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
    parser.add_argument('--niter', metavar='N', type=int, default=10000, help='total number of iterations')

    parser.add_argument('--ntemps', metavar='N', type=int, default=10, help='number of temperatures')
    parser.add_argument('--nwalkers', metavar='N', type=int, default=100, help='number of walkers')

    parser.add_argument('--rvs', metavar='FILE', required=True, default=[], action='append', help='file of times and RV\'s')

    args=parser.parse_args()

    ts, rvs=load_data(args.rvs)

    pmin,pmax=cl.prior_bounds_from_data(args.nplanets, ts, rvs)

    try:
        pts=[]
        logls=[]
        lnprobs=[]

        for i in range(args.ntemps):
            data=np.loadtxt('%s.%02d.txt.gz'%(args.prefix, i))
            pts.append(data[-args.nwalkers:, 2:])
            logls.append(data[-args.nwalkers:,0])
            lnprobs.append(data[-args.nwalkers:,1]+logls[-1])

        pts=np.array(pts)
        logls=np.array(logls)
        lnprobs=np.array(lnprobs)
    except:
        pts=cl.generate_initial_sample(ts, rvs, args.ntemps, args.nwalkers, nobs=len(args.rvs), npl=args.nplanets)
        logls=None
        lnprobs=None
        p=Parameters(npl=args.nplanets, nobs=len(args.rvs))
        header = p.header[0] + ' logl logp' + p.header[1:]
        for i in range(args.ntemps):
            with GzipFile('%s.%02d.txt.gz'%(args.prefix, i), 'w') as out:
                out.write(header)

    log_likelihood=cl.LogLikelihood(ts, rvs)
    log_prior=cl.LogPrior(pmin=pmin, pmax=pmax, npl=args.nplanets, nobs=len(args.rvs))

    sampler=PTSampler(args.ntemps, args.nwalkers, pts.shape[-1], log_likelihood, log_prior, threads=args.nthreads)

    print 'max(log(P)) med(log(P)) min(log(P)) <afrac> <tswap> acorr'
    sys.stdout.flush()

    mean_lnprobs=[]
    for i, (pts, lnprobs, logls) in enumerate(sampler.sample(pts, lnprob0=lnprobs, logl0=logls, iterations=args.niter, storechain=False)):
        if (i+1) % args.nthin == 0:
            mean_lnprobs.append(np.mean(lnprobs[0,:]))
            try:
                ac=acor.acor(np.array(mean_lnprobs)[len(mean_lnprobs)/10:])[0]
            except:
                ac=float('inf')

            for i in range(args.ntemps):
                with GzipFile('%s.%02d.txt.gz'%(args.prefix, i), 'a') as out:
                    np.savetxt(out, np.column_stack((logls[i,...], lnprobs[i,...]-logls[i,...], pts[i,...])))

            with GzipFile('%s.accept.txt.gz'%args.prefix, 'a') as out:
                np.savetxt(out, np.reshape(np.mean(sampler.acceptance_fraction, axis=1), (1, -1)))

            with GzipFile('%s.aswaps.txt.gz'%args.prefix, 'a') as out:
                np.savetxt(out, np.reshape(sampler.tswap_acceptance_fraction, (1, -1)))
                
            print '%11.1f %11.1f %11.1f %7.2f %7.2f %5.1f'%(np.amax(lnprobs[0,:]), np.median(lnprobs[0,:]), np.min(lnprobs[0,:]), np.mean(sampler.acceptance_fraction[0, :]), sampler.tswap_acceptance_fraction[0], ac)
            sys.stdout.flush()
        else:
            pass
            
