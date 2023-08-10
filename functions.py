import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

def integral_Ib(p_0, exponent_b):
    b = exponent_b
    if b != 4:
        return 4.0 * (pow(p_0, b) - p_0 ** 4) / (4.0 - b)
    else: # b == 4:
        return -4.0 * p_0 ** 4 * np.log(p_0)
    
def integral_Ib_plus(p_0, p_1, exponent_b):
    b = exponent_b
    if b != 4:
        return 4.0 * p_0 ** 4 * (1 - pow(p_1, b-4)) / (4.0 - b)
    else: # b == 4:
        return 4.0 * p_0 ** 4 * np.log(p_1)
    
def integral_I2bb(p_0, exponent_b1, exponent_b2):
    return integral_Ib(p_0, exponent_b1 + exponent_b2) - \
            integral_Ib(p_0, exponent_b1) - \
            integral_Ib(p_0, exponent_b2) + \
            integral_Ib(p_0, 0.0)

# Used for debugging
def Phi_b_C0(p_0, p_1, exponent_b):
    one_over_denom = 1.0 / (1 - (p_0/p_1)**4)
    return one_over_denom * \
           integral_I2bb(p_0, exponent_b, exponent_b)

# Used for debugging
def Phi_b_C1(p_0, p_1, exponent_b):
    one_over_denom = 1.0 / (1 - (p_0/p_1)**4)
    return one_over_denom * \
           (integral_I2bb(p_0, exponent_b, exponent_b) - \
           2 * exponent_b / 3.0 * integral_I2bb(p_0, exponent_b, 3) + \
           exponent_b ** 2 / 9.0 * integral_I2bb(p_0, 3, 3))

# Used for debugging
def match_floor_approximant_C0(p_0, p_1, exponent_b, beta, u_IM):
    return 1.0 - 0.5 * beta ** 2 * pow(u_IM, 2.0 * b) * Phi_b_C0(p_0, p_1, exponent_b)

# Used for debugging
def match_floor_approximant_C1(p_0, p_1, exponent_b, beta, u_IM):
    return 1.0 - 0.5 * beta ** 2 * pow(u_IM, 2.0 * b) * Phi_b_C1(p_0, p_1, exponent_b)


def matrix_P(p_0, p_1):
    two_over_denom = 2.0 / (1 - (p_0/p_1)**4)
    P11 = integral_Ib(p_0, 6) + integral_Ib_plus(p_0, p_1, 6)
    P12 = -integral_Ib(p_0, 3) - integral_Ib_plus(p_0, p_1, 3)
    P21 = P12
    P22 = integral_Ib(p_0, 0) + integral_Ib_plus(p_0, p_1, 0)
    return two_over_denom * np.array([[P11,P12],[P21,P22]])

def vector_p_b(p_0, p_1, exponent_b):
    two_over_denom = 2.0 / (1 - (p_0/p_1)**4)
    p1 = -integral_Ib(p_0, exponent_b + 3) + integral_Ib(p_0, 3) + \
          exponent_b / 3.0 * (integral_Ib(p_0, 6) - integral_Ib(p_0, 3))
    p2 = integral_Ib(p_0, exponent_b) - integral_Ib(p_0, 0) - \
          exponent_b / 3.0 * (integral_Ib(p_0, 3) - integral_Ib(p_0, 0))
    return two_over_denom * np.array([p1, p2])
    
def term_under_sqrt(p_0, p_1, exponent_b):
    vec_p_b = vector_p_b(p_0, p_1, exponent_b)
    vec_theta_0 = -np.dot(inv(matrix_P(p_0, p_1)), vec_p_b)
    return Phi_b_C1(p_0, p_1, exponent_b) + 0.5 * np.dot(vec_theta_0, vec_p_b)


def beta_from_beta_tilde_paper(beta_tilde, p_0, p_1, exponent_b, u_IM):
    """Produces ppE beta from the given parameters. This formulation is in
    terms of the parameters used in the paper.

    Parameters
    ----------
    beta_tilde : float
        The rescaled beta term (which can be computed from a desired mismatch).
    p_0 : float
        The lower frequency cutoff of the waveform, in units of fractional
        inspiral velocity: p_0 = (f_0 / f_IM)^(1/3)
    p_1 : float
        The upper frequency cutoff of the waveform, in units of fractional
        inspiral velocity: p_1 = (f_1 / f_IM)^(1/3)
    exponent_b : float
        The ppE exponent b.
    u_IM: float
        The Keplerian velocity at which the inspiral regime ends.

    Returns
    -------
    beta : float
         The ppE parameter beta.
    """
    denominator = pow(u_IM, exponent_b) * np.sqrt(term_under_sqrt(p_0, p_1, exponent_b))
    return beta_tilde / denominator

def beta_from_beta_tilde_wrapped(beta_tilde, f_min_Hz, Mf_max, exponent_b, Mf_IM, total_mass):
    """Produces ppE beta from the given parameters. This formulation differs
    from that given in the paper.

    Parameters
    ----------
    beta_tilde : float
        The rescaled beta term (which can be computed from a desired mismatch).
    f_min_Hz : float
        The lower frequency cutoff of the waveform, in units of Hz.
    Mf_max : float
        The upper frequency cutoff of the waveform, in dimensionless units.
    exponent_b : float
        The ppE exponent b.
    Mf_IM: float
        The dimensionless frequency at which the inspiral regime ends.
    total_mass: float
        The total mass of the binary in units of solar masses.

    Returns
    -------
    beta : float
         The ppE parameter beta.
    """
    # Units of solar mass can be converting to units of time using the conversion
    # factor: 1 M_solar = 4.925491025543576e-06s.
    # Then 1 M_solar^-1 = 203025.44351700248Hz
    # To convert Hz to inverse solar masses we divide by 203025.44351700248.
    Mf_min = total_mass * f_min_Hz * 4.925491025543576e-06
    p_0 = pow(Mf_min / Mf_IM, 1.0/3.0)
    p_1 = pow(Mf_max / Mf_IM, 1.0/3.0)
    u_IM = pow(np.pi * Mf_IM, 1.0/3.0)
    return beta_from_beta_tilde_paper(beta_tilde, p_0, p_1, exponent_b, u_IM)

## Only used for debugging
def beta_from_beta_tilde_floor_C0(beta_tilde, p_0, p_1, exponent_b, u_IM):
    denominator = pow(u_IM, exponent_b) * np.sqrt(Phi_b_C0(p_0, p_1, exponent_b))
    return beta_tilde / denominator

## Only used for debugging
def beta_from_beta_tilde_floor_C1(beta_tilde, p_0, p_1, exponent_b, u_IM):
    denominator = pow(u_IM, exponent_b) * np.sqrt(Phi_b_C1(p_0, p_1, exponent_b))
    return beta_tilde / denominator