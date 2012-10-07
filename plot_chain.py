#!/usr/bin/env python

from argparse import ArgumentParser
import matplotlib.pyplot as pp
import numpy as np
import os
import parameters as pa
import ptsampler as pt
import scipy.stats as ss

if __name__ == '__main__':
    parser=ArgumentParser()

    parser.add_argument('--input', metavar='FILE', required=True, help='input chain')
    parser.add_argument('--outdir', metavar='DIR', default=None, help='output directory')
    parser.add_argument('--trueparams', metavar='FILE', default=None, help='true parameters')

    parser.add_argument('--fburnin', metavar='F', default=0.1, type=float, help='fraction to discard as burned in')

    parser.add_argument('--nwalkers', metavar='N', default=100, type=int, help='number of ensemble walkers')
    parser.add_argument('--ntemps', metavar='N', default=10, type=int, help='number of temperatures')

    args=parser.parse_args()

    pts=np.loadtxt(args.input)

    # (Nsamples, Ntemps, Nwalkers, Ndim)
    pts=np.reshape(pts, (-1, args.ntemps, args.nwalkers, pts.shape[-1]))
    
    # Only want the zero-temp chain
    pts=pts[:, 0, :, :]

    # Discard burnin
    pts=pt.burned_in_samples(pts, fburnin=args.fburnin)

    # Decorrelate
    pts=pt.decorrelated_samples(pts)

    names=pa.Parameters(arr=pts[0,0,:]).header.split()[1:]

    if args.trueparams is not None:
        true=np.loadtxt(args.trueparams)
    else:
        true=None

    try:
        if args.outdir is not None:
            os.makedirs(args.outdir)
    except:
        # Ignore errors
        pass

    for i in range(pts.shape[-1]):
        ipts=pts[:,:,i]
        xs=np.linspace(np.amin(ipts), np.amax(ipts), 1000)
        
        pp.subplot(2,1,1)
        pp.plot(xs, ss.gaussian_kde(ipts.flatten())(xs))

        if true is not None:
            pp.axvline(true[i], color='k')

        pp.xlabel(names[i])
        pp.ylabel(r'$p\left( \mathrm{' + names[i] + r'} \right)$')
        pp.title(names[i])

        pp.subplot(2,1,2)
        pp.plot(ipts.flatten(), ',')

        pp.ylabel(names[i])
        pp.title('Chain evolution')

        if args.outdir is not None:
            pp.savefig(os.path.join(args.outdir, names[i] + '.pdf'))

        pp.show()
