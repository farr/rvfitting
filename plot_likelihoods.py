#!/usr/bin/env python

from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as pp

if __name__ == '__main__':
    parser=ArgumentParser()

    parser.add_argument('--input', metavar='FILE', required=True, help='logl file')

    parser.add_argument('--ntemps', metavar='N', type=int, default=10, help='number of temperatures')
    parser.add_argument('--nwalkers', metavar='N', type=int, default=100, help='number of walkers')

    args=parser.parse_args()

    data=np.loadtxt(args.input)

    data=np.reshape(data, (-1, args.ntemps, args.nwalkers))
    data=data[:,0,:]

    pp.plot(np.mean(data, axis=1))
    pp.show()
