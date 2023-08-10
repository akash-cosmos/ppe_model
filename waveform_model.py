"""
File to store the PPE formalism model.
"""

import numpy as np
from bilby.gw.source import *


import numpy as np
import pycbc.waveform as pycbc_wf
import bilby
import bilby.gw.conversion as b_cnv
import pycbc.conversions as p_cnv
from astropy import constants as const
import math
import warnings
from functions import *
warnings.filterwarnings("ignore", category=RuntimeWarning) 

M_F_NTRL_UNIT_CNV = const.G.value/(const.c.value**3)

print('Version 1.1')


def ppe_c1_eps_model(frequency_array, mass_1, mass_2, a_1, a_2, tilt_1, tilt_2, luminosity_distance, theta_jn, phi_12,          phi_jl, phase, b, beta, delta_eps, a=0.0, alpha=0.0,  **kwargs):
    
    reference_frequency = kwargs.get('reference_frequency', 50.0)
    kwargs['waveform_approximant'] = 'IMRPhenomPv2'
    ht_ppe = lal_binary_black_hole(frequency_array, mass_1, mass_2, luminosity_distance,a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, **kwargs)
    
    hptilde, hctilde = ht_ppe["plus"], ht_ppe["cross"]
    
    #this reassignment ensures the beta_eps correction as extension of beta_c1 correction
    eps = beta + delta_eps       
    
    f_im = 0.018 / (M_F_NTRL_UNIT_CNV * (mass_1+mass_2)*const.M_sun.value)
    
    f_rd = ringdown_frequency(frequency_array, hptilde, mass_1 + mass_2)  
    f_int_end = 0.75*f_rd      
        
    f_im_index = np.argmax(frequency_array > f_im)
    f_int_end_index = np.argmax(frequency_array > f_int_end)    

    u = u_fn(mass_1, mass_2, frequency_array)
    u_im = u_fn(mass_1, mass_2, f_im)
    
    #amplitude term
    amp_term = alpha * (u**a)
    
    phase_term = np.zeros(len(frequency_array))

    #inspiral phase
    phase_term[1:f_im_index] = inspiral_phase(beta, b, u[1:f_im_index]) -post_inspiral_phase(beta, b, eps, u[1:f_im_index], u_im) 
  
    #intermediate phase
    u_int_end = u_fn(mass_1, mass_2, f_int_end)
    freq_arr_int = frequency_array[f_im_index:f_int_end_index]
    phase_term[f_im_index:f_int_end_index] = intermediate_phase(beta, b, eps, f_im, u_im, f_int_end, u_int_end,freq_arr_int) -post_inspiral_phase(beta, b, eps, u[f_im_index:f_int_end_index], u_im)
    
    #print(phase_term)
    
    #complete ppe term
    ppe_term = (1 + amp_term) * np.exp(1j * phase_term)
    #print(ppe_term)    
    hptilde = hptilde * ppe_term
    hctilde = hctilde * ppe_term
    
    return dict(plus=hptilde, cross=hctilde)



def ppe_c1_eps_model_rescaled(frequency_array, mass_1, mass_2, a_1, a_2, tilt_1, tilt_2, luminosity_distance, theta_jn, phi_12,          phi_jl, phase, b, rescaled_beta, rescaled_eps, a=0.0, alpha=0.0,  **kwargs):
    
    reference_frequency = kwargs.get('reference_frequency', 50.0)
    kwargs['waveform_approximant'] = 'IMRPhenomPv2'
    f_min_Hz = 20
    f_max_Hz = 1024
    Mf_max = f_max_Hz
    
    Mf_IM = 0.018
    total_mass = (mass_1 + mass_2)
    Mf_min = total_mass * f_min_Hz * 4.925491025543576e-06
    Mf_max = total_mass * f_max_Hz * 4.925491025543576e-06
    
    beta = beta_from_beta_tilde_wrapped(rescaled_beta, f_min_Hz, Mf_max, b, Mf_IM, total_mass)
    eps = beta*(1+rescaled_eps)
    
    
    ht_ppe = lal_binary_black_hole(frequency_array, mass_1, mass_2, luminosity_distance,a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, **kwargs)
    
    hptilde, hctilde = ht_ppe["plus"], ht_ppe["cross"]
    
    #this reassignment ensures the beta_eps correction as extension of beta_c1 correction
    #eps = beta + delta_eps       
    
    f_im = 0.018 / (M_F_NTRL_UNIT_CNV * (mass_1+mass_2)*const.M_sun.value)
    
    f_rd = ringdown_frequency(frequency_array, hptilde, mass_1 + mass_2)  
    f_int_end = 0.75*f_rd      
        
    f_im_index = np.argmax(frequency_array > f_im)
    f_int_end_index = np.argmax(frequency_array > f_int_end)    

    u = u_fn(mass_1, mass_2, frequency_array)
    u_im = u_fn(mass_1, mass_2, f_im)
    
    #amplitude term
    amp_term = alpha * (u**a)
    
    phase_term = np.zeros(len(frequency_array))

    #inspiral phase
    phase_term[1:f_im_index] = inspiral_phase(beta, b, u[1:f_im_index]) -post_inspiral_phase(beta, b, eps, u[1:f_im_index], u_im) 
  
    #intermediate phase
    u_int_end = u_fn(mass_1, mass_2, f_int_end)
    freq_arr_int = frequency_array[f_im_index:f_int_end_index]
    phase_term[f_im_index:f_int_end_index] = intermediate_phase(beta, b, eps, f_im, u_im, f_int_end, u_int_end,freq_arr_int) -post_inspiral_phase(beta, b, eps, u[f_im_index:f_int_end_index], u_im)
    
    #print(phase_term)
    
    #complete ppe term
    ppe_term = (1 + amp_term) * np.exp(1j * phase_term)
    #print(ppe_term)    
    hptilde = hptilde * ppe_term
    hctilde = hctilde * ppe_term
    
    return dict(plus=hptilde, cross=hctilde)



def ppe_c1_model(frequency_array, mass_1, mass_2, luminosity_distance, theta_jn, phase, b, beta, a=0.0, alpha=0.0,
                 a_1=0.0, a_2=0.0, tilt_1=0.0, tilt_2=0.0, phi_12=0.0, phi_jl=0.0, **kwargs):
    
    reference_frequency = kwargs.get('reference_frequency', 50.0)
    kwargs['waveform_approximant'] = 'IMRPhenomD'
    ht_ppe = lal_binary_black_hole(frequency_array, mass_1, mass_2, luminosity_distance,
                                    a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, **kwargs)
    
    hptilde, hctilde = ht_ppe["plus"], ht_ppe["cross"]
  
    f_im = 0.018 / (M_F_NTRL_UNIT_CNV * (mass_1+mass_2)*const.M_sun.value)    
    f_im_index = np.argmax(frequency_array > f_im)
    
#     print(f_im, f_im_index)
    u = u_fn(mass_1, mass_2, frequency_array)
    u_im = u_fn(mass_1, mass_2, f_im)
    
    #amplitude term
    amp_term = alpha * (u**a)
    
    phase_term = np.zeros(len(frequency_array))

    #inspiral phase
    phase_term[1:f_im_index] = inspiral_phase(beta, b, u[1:f_im_index]) -post_inspiral_phase(beta, b, beta, u[1:f_im_index], u_im) 
  
    #print(phase_term)
    
    #complete ppe term
    ppe_term = (1 + amp_term) * np.exp(1j * phase_term)
    #print(ppe_term)    
    hptilde = hptilde * ppe_term
    hctilde = hctilde * ppe_term
    
    return dict(plus=hptilde, cross=hctilde)







def inspiral_phase(beta, b, u):
    return beta * u**b



def intermediate_phase(beta, b, eps, f_start, u_im, f_end, u_end, f):
    phase_start_diff = beta * b * u_im**b / (3 * f_start)
    phase_end_diff = eps * b * u_im**(b-3) * u_end**3 / (3 * f_end)
    
    phase_arr = np.array([inspiral_phase(beta, b, u_im),
                          phase_start_diff,
                          post_inspiral_phase(beta, b, eps, u_end, u_im),
                          phase_end_diff])
    
    f_mtx = np.array([[1., f_start,    math.log(f_start),    -1/3 * (f_start ** -3)     ], 
                      [0,  1,          1./f_start,           f_start ** -4              ],
                      [1., f_end,      math.log(f_end),      -1/3 * ((f_end) ** -3)     ],
                      [0,  1,          1./f_end,             (f_end) ** -4              ]])
    beta_array = np.linalg.solve(f_mtx, phase_arr)
    #print('betas -> ', beta_array)
    return beta_array[0] + beta_array[1]*f + beta_array[2]*np.log(f) - beta_array[3]/(3.*(f**3))


def post_inspiral_phase(beta, b, eps, u, u_im):
    return beta*(u_im**b) + (b/3)*eps*(u_im**b)*(((u/u_im)**3)-1)


def u_fn(mass_1, mass_2, f):
    return (np.pi * (mass_1 + mass_2) * const.M_sun.value * f *  M_F_NTRL_UNIT_CNV) ** (1/3)

def ringdown_frequency(freq_arr, fd_waveform, total_mass):
    """Numerically finds the ringdown frequency, f_RD, given a frequency
    domain waveform. The ringdown frequency is as defined in:
    https://arxiv.org/pdf/1508.07253.pdf
    Parameters
    ----------
    freq_arr : array_like
        The frequencies at which the strain is calculated
    fd_waveform : numpy array
        The frequency domain waveform of the coalescence.
    total_mass : float
        The total mass of the binary, in solar masses.
    Returns
    -------
    f_RD : float
         The ringdown frequency, in Hz.
    """
    
    light_ring_frequency = 1 / (6 * np.pi * total_mass *const.M_sun.value * M_F_NTRL_UNIT_CNV)
   
    light_ring_index = np.argmax(freq_arr > light_ring_frequency)
    two_times_light_ring_index = np.argmax(freq_arr > 2 * light_ring_frequency)
    
    mindex = np.argmin(np.gradient(np.unwrap(np.angle(fd_waveform[light_ring_index:two_times_light_ring_index]))))
    
    f_RD = freq_arr[mindex + light_ring_index]
    return f_RD