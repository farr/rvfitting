import acor
import numpy as np

def exponential_beta_ladder(ntemps):
    return np.exp(np.linspace(0, -(ntemps-1)*0.5*np.log(2), ntemps))

def thermodynamic_log_evidence(logls, betas):
    """Computes the evidence integral.

    :param logls: The ln(likelihood) samples, of shape ``(Nsamples,
    Ntemperatures, Nwalkers)``."""

    nsamp,ntemp,nwalk=logls.shape

    mean_logls=np.mean(np.mean(logls, axis=2), axis=0)

    betas=np.concatenate((betas, np.array([0.0])))

    return -np.sum(mean_logls*np.diff(betas))

def burned_in_samples(pts, fburnin=0.1):
    """Returns the samples from ``pts`` after an initial burnin
    fraction.  pts should have shape ``(Nsamples, ...)``."""

    iburnin=int(fburnin*pts.shape[0]+1)

    post_burnin_shape=(-1,) + pts.shape[1:]

    return np.reshape(np.reshape(pts, (pts.shape[0], -1))[iburnin:, :], post_burnin_shape)

def decorrelated_samples(pts):
    """Returns a subset of ``pts`` that is downsampled by the longest
    correlation length in ``pts``.  pts should have shape ``(Nsamples,
    Nwalkers, Ndim)``."""

    if acor is None:
        raise ImportError('acor')
    
    means=np.mean(pts, axis=1)

    taumax=float('-inf')
    for j in range(means.shape[1]):
        tau,mu,sigma=acor.acor(means[:,j])
        taumax=max(tau,taumax)

    taumax=int(taumax+1)

    return pts[::taumax, :, :]

    
