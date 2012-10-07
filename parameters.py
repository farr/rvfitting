import numpy as np

class Parameters(np.ndarray):
    """Parameters for radial velocity fitting for a single telescope
    observing a single planet."""

    def __new__(subclass, arr=None, nobs=1, npl=1, 
                V=None, sigma0=None, tau=None, K=None, n=None, chi=None, e=None, omega=None):
        """Create a parameter object out of the given array (or a
        fresh array, if none given), with nobs observatories and npl
        planets."""

        assert nobs >= 1, 'must have at least one observatory'
        assert npl >= 1, 'must have at least one planet'
        assert arr is None or arr.shape[-1] == 3*nobs+5*npl, 'final array dimensions must match 3*nobs + 5*npl'

        if arr is None:
            arr = np.zeros(nobs*3+npl*5)
        
        obj = np.asarray(arr).view(subclass)

        obj._nobs = nobs
        obj._npl = npl

        if V is not None:
            obj.V = V

        if sigma0 is not None:
            obj.sigma0 = sigma0

        if tau is not None:
            obj.tau = tau

        if K is not None:
            obj.K = K

        if n is not None:
            obj.n = n

        if chi is not None:
            obj.chi = chi

        if e is not None:
            obj.e = e
    
        if omega is not None:
            obj.omega = omega

        return obj

    def __array_finalize__(self, other):
        if other is None:
            pass
        else:
            self._nobs = getattr(other, 'nobs', 1)
            self._npl = getattr(other, 'npl', 1)

    @property
    def header(self):
        """A suitable header (including newline) to describe parameter data."""
        if self.nobs == 1 and self.npl == 1:
            return '# V sigma0 tau K n chi e omega\n'
        else:
            header = '# '

            for i in range(self.nobs):
                header += 'V%d sigma0%d tau%d '%(i,i,i)

            for i in range(self.npl):
                header += 'K n chi e omega '%(i,i,i,i,i)

            header[-1]='\n'

            return header

    @property
    def V(self):
        """The velocity offset of the observatory or observatories."""
        return np.array(self[...,0:3*self.nobs:3])

    @V.setter
    def V(self, vs):
        self[...,0:3*self.nobs:3] = vs
       
    @property
    def sigma0(self):
        """The variance at zero lag of the telescope errors."""
        return np.array(self[...,1:3*self.nobs:3])

    @sigma0.setter
    def sigma0(self, s0):
        self[...,1:3*self.nobs:3] = s0
        
    @property
    def tau(self):
        """The exponential decay timescale for correlations in
        telescope errors."""
        return np.array(self[...,2:3*self.nobs:3])

    @tau.setter
    def tau(self, t):
        self[...,2:3*self.nobs:3] = t
        
    @property
    def K(self):
        """The amplitude of the radial velocity."""
        return np.array(self[...,3*self.nobs::5])
        
    @K.setter
    def K(self, k):
        self[...,3*self.nobs::5] = k

    @property
    def n(self):
        """Mean motion (2*pi/P)."""
        return np.array(self[...,3*self.nobs+1::5])

    @n.setter
    def n(self, nn):
        self[...,3*self.nobs+1::5] = nn
        
    @property
    def chi(self):
        """The fraction of an orbit completed at t = 0."""
        return np.array(self[...,3*self.nobs+2::5])

    @chi.setter
    def chi(self, c):
        self[...,3*self.nobs+2::5] = c
        
    @property
    def e(self):
        """The orbital eccentricity."""
        return np.array(self[...,3*self.nobs+3::5])

    @e.setter
    def e(self, ee):
        self[...,3*self.nobs+3::5]=ee
        
    @property
    def omega(self):
        """The longitude of perastron."""
        return np.array(self[...,3*self.nobs+4::5])
        
    @omega.setter
    def omega(self, o):
        self[...,3*self.nobs+4::5]=o

    @property
    def obs(self):
        """Returns an (N,3) array of observatory parameters."""
        return np.array(np.reshape(self[:3*self.nobs], (-1, 3)))

    @obs.setter
    def obs(self, o):
        self[:3*self.nobs] = o

    @property
    def planets(self):
        """Returns an (N,5) array of planet parameters."""
        return np.array(np.reshape(self[3*self.nobs:], (-1, 5)))

    @planets.setter
    def planets(self, p):
        self[3*self.nobs:] = p

    @property
    def nobs(self):
        return self._nobs

    @property
    def npl(self):
        return self._npl

    
        
