import numpy as np


def find_b(h, j, k, alpha):
    """Calculates constant b for simulation.

    Args:
        h (float): Time step.
        j (int): Time index.
        k (int): Time index.
        alpha (float): Diffusion exponent.
    """
    b = h**alpha / alpha * ((k-j+1)**alpha - (k-j)**alpha)
    return b


def find_a(h, j, k, alpha):
    """Calculates constant a for simulation.

    Args:
        h (float): Time step.
        j (int): Time index.
        k (int): Time index.
        alpha (float): Diffusion exponent.
    """
    a = h**alpha/(alpha * (alpha+1))
    if j == 0:
        a *= k**(alpha+1) - (k-alpha)*(k+1)**alpha
    elif j == k+1:
        a *= 1
    else:
        a *= (k-j+2)**(alpha+1) + (k-j)**(alpha+1) - 2*(k-j+1)**(alpha+1)
    return a


def find_RHS_tw(x, lr, L, f_coeff, V0, x0, F, H, fgn):
    """Find force function on RHS.

    Args:
        x (float): Position.
        f_coeff (float): Friction coefficient.
        V0 (float): Washboard amplitude.
        x0 (float): Washboard period.
        F (float): Washboard bias.
        H (float): Holder exponent.
        sigma (float): Noise strength.
        fgn (float): Fractional Gaussian noise.
    """
    if H < 0.5:
        H = 0.5
    sigma = np.sqrt(lr*f_coeff/(H*(2*H-1)))
    rhs = -2*np.pi*V0/x0 * np.sin(2*np.pi*x/x0) + F + sigma*fgn
    if x > L:
        rhs = -rhs
    return rhs


def find_RHS_harmonic(x, lr, f_coeff, H, sharpness, fgn):
    """Find force function on RHS.

    Args:
        x (float): Position.
        f_coeff (float): Friction coefficient.
        V0 (float): Washboard amplitude.
        x0 (float): Washboard period.
        F (float): Washboard bias.
        H (float): Holder exponent.
        sigma (float): Noise strength.
        fgn (float): Fractional Gaussian noise.
    """
    if H < 0.5:
        H = 0.5
    sigma = np.sqrt(lr*f_coeff/(H*(2*H-1)))
    rhs = - sharpness * x + sigma*fgn
    return rhs


def find_RHS_combined(x, lr, sharpness, L, f_coeff, V0, x0, F, H, fgn):
    """Find force function on RHS.

    Args:
        x (float): Position.
        f_coeff (float): Friction coefficient.
        V0 (float): Washboard amplitude.
        x0 (float): Washboard period.
        F (float): Washboard bias.
        H (float): Holder exponent.
        sigma (float): Noise strength.
        fgn (float): Fractional Gaussian noise.
    """
    if np.abs(x) > L:
        rhs = find_RHS_tw(x, lr, L, f_coeff, V0, x0, F, H, fgn)
    else:
        rhs = find_RHS_harmonic(x, lr, f_coeff, H, sharpness, fgn)
    return rhs
