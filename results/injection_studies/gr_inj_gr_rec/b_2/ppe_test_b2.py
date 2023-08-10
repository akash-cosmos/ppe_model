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
from waveform_model import ppe_c1_eps_model
np.set_printoptions(threshold=sys.maxsize)
import pycbc.waveform as wf

import bilby
from gwpy.timeseries import TimeSeries

logger = bilby.core.utils.logger
outdir = "results_b2_updated"
label = "GW150914"


trigger_time = 1126259462.4
detectors = ["H1", "L1"]
maximum_frequency = 512
minimum_frequency = 20
roll_off = 0.4  # Roll off duration of tukey window in seconds, default is 0.4s
duration = 4  # Analysis segment duration
post_trigger_duration = 2  # Time between trigger time and end of segment
end_time = trigger_time + post_trigger_duration
start_time = end_time - duration

psd_duration = 32 * duration
psd_start_time = start_time - psd_duration
psd_end_time = start_time

ifo_list = bilby.gw.detector.InterferometerList([])
for det in detectors:
    logger.info("Downloading analysis data for ifo {}".format(det))
    ifo = bilby.gw.detector.get_empty_interferometer(det)
    data = TimeSeries.fetch_open_data(det, start_time, end_time)
    ifo.strain_data.set_from_gwpy_timeseries(data)

    logger.info("Downloading psd data for ifo {}".format(det))
    psd_data = TimeSeries.fetch_open_data(det, psd_start_time, psd_end_time)
    psd_alpha = 2 * roll_off / duration
    psd = psd_data.psd(
        fftlength=duration, overlap=0, window=("tukey", psd_alpha), method="median"
    )
    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
        frequency_array=psd.frequencies.value, psd_array=psd.value
    )
    ifo.maximum_frequency = maximum_frequency
    ifo.minimum_frequency = minimum_frequency
    ifo_list.append(ifo)

    
priors = bilby.gw.prior.BBHPriorDict(filename="GW150914.prior")

# Add the geocent time prior
priors["geocent_time"] = bilby.core.prior.Uniform(
    trigger_time - 0.1, trigger_time + 0.1, name="geocent_time"
)
priors["beta"] = bilby.core.prior.Uniform(
    -5, 5, name= "beta"
)
priors["delta_eps"] = bilby.core.prior.Uniform(
    -5, 5, name= "delta_eps"
)
priors["a"] = 0.0
priors["alpha"] = 0.0
priors['b'] = -2

waveform_generator = bilby.gw.WaveformGenerator(
    frequency_domain_source_model=ppe_c1_eps_model,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments={
        "waveform_approximant": "IMRPhenomPv2",
        "reference_frequency": 50,
    },)


likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    ifo_list,
    waveform_generator,
    priors=priors,
    time_marginalization=False,
    phase_marginalization=False,
    distance_marginalization=False,
)

result = bilby.run_sampler(
    likelihood,
    priors,
    sampler="dynesty",
    outdir=outdir,
    label=label,
    nlive=2000,
    check_point_delta_t=300,
    check_point_plot=True,
    npool=1,
    conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
)



