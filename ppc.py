#!/usr/bin/env python

import acor
from argparse import ArgumentParser
import correlated_likelihood as cl
import numpy as np
import scipy.stats as ss

if __name__ == '__main__':
    parser=ArgumentParser()

    parser.add_argument('--input', required=True, metavar='FILE', help='input chain file')
    parser.add_argument('--rvs', required=True, metavar='FILE', default=[], action='append',
                        help='radial velocity file(s)')

    parser.add_argument('--output', required=True, metavar='FILE', help='output quantile file')

    parser.add_argument('--npl', default=1, type=int, metavar='N', help='number of planets')
    parser.add_argument('--nwalkers', default=100, type=int, metavar='N', help='number of walkers')
    parser.add_argument('--fburnin', default=0.1, type=float, metavar='N', help='fraction of samples to discard as burnin')

    args=parser.parse_args()

    ts=[]
    rvs=[]
    for rvfile in args.rvs:
        data=np.loadtxt(rvfile)
        ts.append(data[:,0])
        rvs.append(data[:,1])

    params=np.loadtxt(args.input)
    params=params[:,2:]
    params=np.reshape(params, (-1, args.nwalkers, params.shape[-1]))

    params=params[int(args.fburnin*params.shape[0]+0.5):, ...]

    taumax=float('-inf')
    for k in range(params.shape[-1]):
        tau=acor.acor(params[:,:,k].T)[0]
        taumax=max(taumax, tau)

    taumax=int(taumax+0.5)
    params=params[::taumax, ...]

    print 'Averaging ', params.shape[0]*params.shape[1], ' posterior quantiles'

    params=np.reshape(params, (-1, params.shape[-1]))

    qs=cl.posterior_data_mean_quantiles(ts, rvs, params)

    D,p = ss.kstest(qs, lambda x: x)

    print 'KS p-value = ', p

    np.savetxt(args.output, qs)
