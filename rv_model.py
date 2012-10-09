import numpy as np
import parameters as params
import scipy.optimize as so

def kepler_f(M, E, e):
    """Returns the residual of Kepler's equation with mean anomaly M,
    eccentric anomaly E and eccentricity e."""
    return E - e*np.sin(E) - M

def kepler_solve_ea(n, e, t):
    """Solve for the eccentric anomaly for an orbit with mean motion
    n, eccentricity e, and time since pericenter passage t."""

    t=np.atleast_1d(t)

    M = np.fmod(n*t, 2.0*np.pi)

    while np.any(M < 0.0):
        M[M<0.0] += 2.0*np.pi

    Emin=np.zeros_like(t)
    Emax=np.zeros_like(t)+2.0*np.pi
    fEmin=kepler_f(M, Emin, e)
    fEmax=kepler_f(M, Emax, e)

    while np.any(Emax-Emin > 1e-8):
        Emid = 0.5*(Emin+Emax)
        fEmid=kepler_f(M, Emid, e)

        low_sel=(fEmid*fEmin < 0)
        high_sel= (fEmid*fEmax < 0)

        fEmax[low_sel] = fEmid[low_sel]
        Emax[low_sel] = Emid[low_sel]

        fEmin[high_sel] = fEmid[high_sel]
        Emin[high_sel] = Emid[high_sel]

    return (Emin*fEmax - Emax*fEmin)/(fEmax-fEmin)        

def kepler_solve_ta(n, e, t):
    """Solve for the true anomaly of a Keplerian orbit with mean
    motion n, eccentricity e at time t since pericenter passage."""

    E=kepler_solve_ea(n,e,t)

    f = 2.0*np.arctan(np.sqrt((1.0+e)/(1.0-e))*np.tan(E/2.0))

    # Get positive f, either [0, pi/2] or [3pi/2, 2pi]
    if f < 0.0:
        f += 2.0*np.pi

    return f

def rv_model(ts, ps):
    """Returns the radial velocity measurements associated with the
    planets in parameters ps at times ts.  The returned array has
    shape (Npl, Nts)."""

    rvs=np.zeros((ps.npl, ts.shape[0]))

    for i,(K,e,omega,chi,n) in enumerate(zip(ps.K, ps.e, ps.omega, ps.chi, ps.n)):

        ecw=e*np.cos(omega)
        esw=e*np.sin(omega)

        t0 = -chi*2.0*np.pi/n

        fs=np.array([kepler_solve_ta(n, e, (t-t0)) for t in ts])

        rvs[i,:]=K*(np.sin(fs + omega) + ecw)

    return rvs

    
