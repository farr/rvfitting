import matplotlib.pyplot as pp
import numpy as np
import scipy.stats as ss

def plot_kde(samples, *args, **kwargs):
    """Plots the KDE estimate of samples."""
    xs=np.linspace(np.min(samples), np.max(samples), 1000)
    pp.plot(xs, ss.gaussian_kde(samples)(xs), *args, **kwargs)
