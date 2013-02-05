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
    parser.add_argument('--nthin', metavar='N', type=int, default=10, help='iterations between output')
    parser.add_argument('--nensembles', metavar='N', type=int, default=100, help='number of ensembles to output')

    parser.add_argument('--nburnin', metavar='N', type=int, default=100, help='number of initial ensembles to discard as burnin')

    parser.add_argument('--ntemps', metavar='N', type=int, default=20, help='number of temperatures')
    parser.add_argument('--nwalkers', metavar='N', type=int, default=100, help='number of walkers')

    parser.add_argument('--rvs', metavar='FILE', required=True, default=[], action='append', help='file of times and RV\'s')

    parser.add_argument('--restart', action='store_true', help='restart an old run')
    parser.add_argument('--init', metavar='FILE', help='file storing initial point')
    parser.add_argument('--delta', metavar='DPARAM', type=float, default=1e-3, help='fractional width about initial point')

    args=parser.parse_args()

    ts, rvs=load_data(args.rvs)

    pmin,pmax=cl.prior_bounds_from_data(args.nplanets, ts, rvs)
    
    ndim = 5*args.nplanets + 4*len(ts)

    # If re-starting a run, burnin = nthin, so that output continues
    # to be evenly-spaced
    if args.restart:
        args.nburnin = args.nthin - 1

    if args.restart:
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
    elif args.init is not None:
        p0=Parameters(np.loadtxt(args.init))
        if len(ts) > 1 or args.nplanets > 1:
            raise NotImplementedError('cannot init from more than one observatory and one planet')
        pts=Parameters(np.zeros((args.ntemps, args.nwalkers, ndim)))
        pts.V = np.random.normal(p0.V, p0.sigma0*args.delta, size=pts.V.shape[0:2])
        pts.sigma0 = np.random.lognormal(np.log(p0.sigma0), args.delta, size=pts.sigma0.shape[0:2])
        pts.sigma = np.random.lognormal(np.log(p0.sigma), args.delta, size=pts.sigma.shape[0:2])
        pts.tau = np.random.lognormal(np.log(p0.tau), args.delta, size=pts.tau.shape[0:2])
        pts.K = np.random.normal(p0.K, p0.sigma0*args.delta, size=pts.K.shape[0:2])
        pts.n = np.random.lognormal(np.log(p0.n), args.delta, size=pts.n.shape[0:2])
        pts.chi = np.random.normal(p0.chi, args.delta, size=pts.chi.shape[0:2])
        pts.e = np.random.normal(p0.e, args.delta, size=pts.e.shape[0:2])
        pts.omega = np.random.normal(p0.omega, args.delta, size=pts.omega.shape[0:2])
        logls=None
        lnprobs=None

        header = pts.header[0] + ' logl logp' + pts.header[1:]
        for i in range(args.ntemps):
            with GzipFile('%s.%02d.txt.gz'%(args.prefix, i), 'w') as out:
                out.write(header)
    else:
        pts=cl.generate_initial_sample(pmin, pmax, args.ntemps, args.nwalkers)
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

    print 'max(log(P)) med(log(P)) min(log(P)) <afrac> <tswap>'
    sys.stdout.flush()

    np.savetxt('%s.betas.txt'%args.prefix, np.reshape(sampler.betas, (1, -1)))

    for pts, lnprobs, logls in sampler.sample(pts, iterations=args.nburnin):
        pass

    sampler.reset()

    for i, (pts, lnprobs, logls) in enumerate(sampler.sample(pts, iterations=args.nthin*args.nensembles, thin=args.nthin)):
        if i % args.nthin == 0:
            for j in range(args.ntemps):
                with GzipFile('%s.%02d.txt.gz'%(args.prefix, j), 'a') as out:
                    np.savetxt(out, np.column_stack((logls[j,...], lnprobs[j,...]-logls[j,...], pts[j,...])))

            with GzipFile('%s.accept.txt.gz'%args.prefix, 'a') as out:
                np.savetxt(out, np.reshape(np.mean(sampler.acceptance_fraction, axis=1), (1, -1)))

            with GzipFile('%s.aswaps.txt.gz'%args.prefix, 'a') as out:
                np.savetxt(out, np.reshape(sampler.tswap_acceptance_fraction, (1, -1)))
                
            print '%11.1f %11.1f %11.1f %7.2f %7.2f'%(np.amax(lnprobs[0,:]), np.median(lnprobs[0,:]), np.min(lnprobs[0,:]), np.mean(sampler.acceptance_fraction[0, :]), sampler.tswap_acceptance_fraction[0])
            sys.stdout.flush()

    print 'Run completed.'
    
    try:
        ac = sampler.acor
        print 'Autocorrelation matrix is ', ac
        print 'Max is ', np.max(ac)
    except:
        print 'Autocorrelation too long to compute.'
