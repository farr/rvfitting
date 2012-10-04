import numpy as np
import scipy.optimize as so

def kepler_f(M, E, e):
    """Returns the residual of Kepler's equation with mean anomaly M,
    eccentric anomaly E and eccentricity e."""
    return E - e*np.sin(E) - M

def kepler_solve_ea(n, e, t):
    """Solve for the eccentric anomaly for an orbit with mean motion
    n, eccentricity e, and time since pericenter passage t."""

    M = np.fmod(n*t, 2.0*np.pi)

    while M < 0.0:
        M += 2.0*np.pi

    return so.brentq(lambda E: kepler_f(M, E, e), 0.0, 2.0*np.pi, xtol=1e-8)

def kepler_solve_ta(n, e, t):
    """Solve for the true anomaly of a Keplerian orbit with mean
    motion n, eccentricity e at time t since pericenter passage."""

    E=kepler_solve_ea(n,e,t)

    f = 2.0*np.arctan(np.sqrt((1.0+e)/(1.0-e))*np.tan(E/2.0))

    # Get positive f, either [0, pi/2] or [3pi/2, 2pi]
    if f < 0.0:
        f += 2.0*np.pi

    return f

def rv_model(ts, params):
    """Returns the radial velocity measured at the given times for an
    orbit with the given parameters.  The parameters are
    [K,t0,e,omega,n], where

    * K is the radial velocity semi-amplitude.
    
    * e is the eccentriticy of the orbit.

    * omega is the longitude of periastron.

    * chi is the fraction of the orbit that has passed at t = 0.

    * n is the mean motion of the planet (n = 2*Pi/P, with P the
      period)."""

    K,e,omega,chi,n=params

    ecw=e*np.cos(omega)
    esw=e*np.sin(omega)

    t0 = -chi*2.0*np.pi/n

    fs=np.array([kepler_solve_ta(n, e, (t-t0)) for t in ts])

    return K*(np.sin(fs + omega) + ecw)

    
