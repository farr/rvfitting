import numpy as np
import numpy.linalg as nl

def correlated_gaussian_loglikelihood(xs, means, cov):
    """Returns the likelihood for data xs, assumed to be multivariate
    Gaussian with the given means and covariance."""
    lambdas=nl.eigvalsh(cov)

    ndim=xs.shape[0]
    
    ds=(xs-means)*nl.solve(cov, xs-means)/2.0

    return -np.log(2.0*np.pi)*(ndim/2.0)-np.sum(np.log(lambdas))-np.sum(ds)

def generate_covariance(ts, sigma0, tau):
    """Generates a covariance matrix according to an exponential
    autocovariance: cov(t_i, t_j) = sigma0*exp(-|ti-tj|/tau)."""

    ndim = ts.shape[0]

    tis = np.tile(np.reshape(ts, (-1, 1)), (1, ndim))
    tjs = np.tile(ts, (ndim, 1))

    return sigma0*np.exp(-np.abs(tis-tjs)/tau)
