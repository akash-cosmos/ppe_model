import sys
sys.path.append('/mnt/pfs/akash.mishra/ppe_model/')
import numpy as np
import matplotlib.pyplot as plt
import waveform_model
# import pycbc.ppe.ppe_tools as ppe_tools

import pycbc
import bilby
import numpy as np
import matplotlib.pylab as plt
import pycbc.psd
from waveform_model import ppe_c1_eps_model_rescaled
np.set_printoptions(threshold=sys.maxsize)
import pycbc.waveform as wf
from prior import DiscreteUniform
import bilby
from gwpy.timeseries import TimeSeries

logger = bilby.core.utils.logger
outdir = "results"
label = "ppe_inj"
bilby.core.utils.setup_logger(outdir=outdir, label=label)

# Set up a random seed for result reproducibility.  This is optional!
np.random.seed(88170235)

duration = 4.0
sampling_frequency = 1024


injection_parameters = dict(
    mass_1=36.0,
    mass_2=29.0,
    a_1=0.4,
    a_2=0.3,
    tilt_1=0.5,
    tilt_2=1.0,
    phi_12=1.7,
    phi_jl=0.3,
    luminosity_distance=450.0,
    theta_jn=0.4,
    psi=2.659,
    phase=1.3,
    geocent_time=1126259642.413,
    ra=1.375,
    dec=-1.2108,
    a = 0.0,
    alpha = 0.0,
    b = -5,
    rescaled_beta = -0.7,
    rescaled_eps = 2,
)

waveform_arguments = dict(
    waveform_approximant="IMRPhenomXPHM",
    reference_frequency=50.0,
    minimum_frequency=20.0,
)

# Create the waveform_generator using a LAL BinaryBlackHole source function
# the generator will convert all the parameters
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=ppe_c1_eps_model_rescaled,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments,
)

ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency,
    duration=duration,
    start_time=injection_parameters["geocent_time"] - 2,
)

ifos.inject_signal(
    waveform_generator=waveform_generator, parameters=injection_parameters
)

priors = bilby.gw.prior.BBHPriorDict()
priors["geocent_time"] = bilby.core.prior.Uniform(
    minimum=injection_parameters["geocent_time"] - 0.1,
    maximum=injection_parameters["geocent_time"] + 0.1,
    name="geocent_time",
    latex_label="$t_c$",
    unit="$s$",
)

priors['mass_1'].minimum = 5
priors['mass_2'].minimum = 5
priors['mass_1'].maximum = 70
priors['mass_2'].maximum = 70
priors['luminosity_distance'].maximum = 1000
priors['luminosity_distance'].minimum = 100

priors['alpha'] = 0.0
priors['a'] = 0.0
priors['b']=DiscreteUniform(-6,2,name='b', latex_label='$b$')
### ppe recovery
#priors['b']=DiscreteUniform(-7,1,name='b', latex_label='$b$')
priors['rescaled_beta']=bilby.prior.Uniform(-3.0,3.0,name='beta', latex_label='$\beta$')
priors['rescaled_eps']=bilby.prior.Uniform(-3.0,3.0,name='delta_eps', latex_label='$\delta$')

likelihood = bilby.gw.GravitationalWaveTransient(
    interferometers=ifos,
    waveform_generator=waveform_generator,
    priors=priors,
    distance_marginalization=False,
    phase_marginalization=False,
    time_marginalization=False,

)

result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="dynesty",
    nlive=2000,
    injection_parameters=injection_parameters,
    outdir=outdir,
    label=label,
    check_point_delta_t=300,
    check_point_plot=True,
    
)




