
import numpy as np
from functools import lru_cache


@lru_cache()
def factorial(n):

    if n == 0:
        return 1
    else:
        return n * factorial(n-1)


def hermite(n, x):

    if n == 0:
        return x*0 + 1
    elif n == 1:
        return 2*x
    else:
        return 2*x*hermite(n-1, x) - 2*n*hermite(n-2, x)


def calc_harmonic_energies(k, m, max_n=10, max_x=1E-12):
    """
    Calculate energy of harmonic oscillator as both classical and quantum
    entity

    Parameters
    ----------
    k : float
        Force constant (N/m)
    m : float
        Reduced mass (g mol-1)
    max_n : int
        maximum value of n used for Harmonic states - used to find maximum
        displacement
    Returns
    -------
    np.ndarray:
        Harmonic state energies for quantum oscillator in Joules
    np.ndarray:
        Harmonic energies for classical oscillator in Joules
    np.ndarray:
        Displacements used for classical oscillator in metres
    float:
        Zero point displacement in metres
    """

    # Convert mass to kg
    m *= 1.6605E-27  # kg (g mol^-1)

    # Angular frequency
    omega = np.sqrt(k/m)  # s^-1

    hbar = 1.0545718E-34  # m2 kg / s
    state_E = np.array([hbar*omega*(n + 0.5) for n in range(0, max_n+1)])  # J

    # Harmonic potential
    # E = 1/2 kx^2
    max_x = np.sqrt((max_n + 0.5) * 2*hbar*omega/k)

    displacement = np.linspace(-max_x, max_x, 100)  # m
    harmonic_E = 0.5*k*displacement**2  # J

    # Find zero point displacement
    zpd = np.sqrt(hbar * omega / k)  # m

    return state_E, harmonic_E, displacement, zpd


def calculate_mu(w, k):
    """
    Calculates "reduced mass" from angular frequency and force constant

    Parameters
    ----------
    w : float
        Angular frequency Omega (s^-1)
    k : float
        Force constant k (N m^-1)

    Returns
    -------
    float:
        Reduced mass mu (g mol^-1)
    """

    mu = k/w**2
    mu /= 1.6605E-27

    return mu


def calculate_k(w, mu):
    """
    Calculates force constant from angular frequency and reduced mass

    Parameters
    ----------
    w : float
        Angular frequency Omega (s^-1)
    mu : float
        Reduced mass (g mol^-1)

    Returns
    -------
    float:
        Force constant k (N m^-1)
    """

    mu *= 1.6605E-27
    k = mu * w**2

    return k


def harmonic_wf(n, x):
    """
    Calculates normalised harmonic wavefunction for nth state over x

    Parameters
    ----------
    n : int
        Harmonic oscillator quantum number
    x : np.ndarray
        Displacement

    Returns
    -------
    np.ndarray:
        Harmonic wavefunction of nth levels evaluated at each point in x
    """

    h = hermite(n, x)

    N = 1./(2**n * factorial(n)*np.sqrt(np.pi)**0.5)

    wf = h * N * np.exp(-x**2 * 0.5)

    return wf
