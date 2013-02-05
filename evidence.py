#!/usr/bin/env python

from argparse import ArgumentParser
import numpy as np
import ptutils as pt

if __name__ == '__main__':
    parser=ArgumentParser()

    parser.add_argument('--prefix', metavar='PRE', default='chain', help='prefix for chain files')

    parser.add_argument('--ntemps', metavar='N', default=20, type=int, help='number of temperatures')
    parser.add_argument('--nwalkers', metavar='N', default=100, type=int, help='number of ensemble walkers')

    parser.add_argument('--fburnin', metavar='F', default=0.1, type=float, help='fraction of samples to discard as burnin')

    args=parser.parse_args()

    meanlogls=[]
    for i in range(args.ntemps):
        data=np.loadtxt('%s.%02d.txt.gz'%(args.prefix, i))
        logls=np.reshape(data[:,0], (-1, args.nwalkers))
        chain=np.reshape(data[:, 2:], (-1, args.nwalkers, data.shape[1]-2))

        istart = int(args.fburnin*chain.shape[0] + 0.5)

        logls = logls[istart:, :]
        chain = chain[istart:,:,:]

        chain,logls = pt.burned_in_samples(chain,logls)
        
        meanlogls.append(np.mean(logls.flatten()))

    meanlogls = np.array(meanlogls)

    inbetas=np.loadtxt('%s.betas.txt'%args.prefix)
    inbetas2 = inbetas[::2]
    betas = np.zeros(inbetas.shape[0] + 1)
    betas2 = np.zeros(inbetas2.shape[0] + 1)
    betas[:-1] = inbetas
    betas2[:-1] = inbetas2

    dbetas = np.diff(betas)
    dbetas2 = np.diff(betas2)

    lnZ = -np.sum(dbetas*meanlogls)
    lnZ2 = -np.sum(dbetas2*meanlogls[::2])

    dlnZ = np.abs(lnZ2 - lnZ)

    print '# ln(Z) dln(Z)'
    print np.sum(dbetas*np.array(meanlogls)), dlnZ
