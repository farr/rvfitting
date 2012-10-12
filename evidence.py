#!/usr/bin/env python

from argparse import ArgumentParser
import numpy as np
import ptutils as pt

if __name__ == '__main__':
    parser=ArgumentParser()

    parser.add_argument('--prefix', metavar='PRE', default='chain', help='prefix for chain files')

    parser.add_argument('--ntemps', metavar='N', default=20, type=int, help='number of temperatures')
    parser.add_argument('--nwalkers', metavar='N', default=100, type=int, help='number of ensemble walkers')

    parser.add_argument('--fburnin', metavar='F', default=0.5, type=float, help='fraction of samples to discard as burnin')

    args=parser.parse_args()

    inlogls=[]
    for i in range(args.ntemps):
        inlogls.append(np.loadtxt('%s.%02d.txt.gz'%(args.prefix, i))[:,0])

    betas=np.loadtxt('%s.betas.txt'%args.prefix)

    # What if we have a few more outputs in some files than in others?
    minlen=reduce(min, [logl.shape[0] for logl in inlogls])
    inlogls=[logl[:minlen] for logl in inlogls]

    # Now ordered as (Nsamples*Nwalkers, Nchains)
    inlogls=np.transpose(np.array(inlogls))

    # Resahpe to (Nsamples, Nchains, Nwalkers)
    logls=np.zeros((inlogls.shape[0]/args.nwalkers, args.ntemps, args.nwalkers))
    for i in range(args.ntemps):
        logls[:, i, :] = np.reshape(inlogls[:,i], (-1, args.nwalkers))

    # burn in...
    logls=logls[int(args.fburnin*logls.shape[0])+1:, ...]

    print pt.thermodynamic_log_evidence(logls, betas)
