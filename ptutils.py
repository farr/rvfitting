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

def burned_in_samples(pts, logls):
    """Automatic burn-in criterion.  

    The run is considered burned-in whenever the mean log(L) over reaches a
    point within one standard deviation of the maximum value

    :param pts: The points in the chain, of shape ``(Nsamples,
    Nwalkers, Ndim)``.

    :param logls: The log(L) values of the chain, shape ``(Nsamples,
    Nwalkers)``.

    :return pts, logls: The burned-in points and logls."""

    mean_logls = np.mean(logls, axis=1)
    max_mean = np.max(mean_logls)
    
    # An estimate of the uncertainty in the mean log(L) under the
    # distribution at the final sampling point.
    sigma_mean = np.std(logls[-1, :])/np.sqrt(logls.shape[1])

    istart = np.nonzero(mean_logls > max_mean - sigma_mean)[0][0]

    return pts[istart:,...], logls[istart:, :]

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

    taumax=int(round(taumax))

    return pts[::taumax, :, :]

    
