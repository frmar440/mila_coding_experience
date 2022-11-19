"""Interface an oscilloscope to probe electrical signals

"""

# install the PicoSDK C librairies: https://www.picotech.com/downloads/linux
# install the Python bindings: https://github.com/picotech/picosdk-python-wrappers
# Programmer's Guide: https://www.picotech.com/download/manuals/picoscope-2000-series-a-api-programmers-guide.pdf
import ctypes
from picosdk.ps2000a import ps2000a as ps
from picosdk.functions import assert_pico_ok, adc2mV

import time

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import TransferFunction, bode, hilbert, find_peaks
from scipy.optimize import curve_fit

import pandas as pd
from datetime import datetime
import os


def getFilter(Q:int, f_0:int) -> TransferFunction:
    """Get transfert function of band-pass filter.

    Args:
        Q (int): Quality factor.
        omega_0 (int): Central frequency.

    Returns:
        TransferFunction: Band-pass filter transfert function.
    """
    omega_0 = 2*np.pi*f_0
    
    num = [1, 0]
    den = [Q/omega_0, 1, Q*omega_0]

    return TransferFunction(num, den)


def prior_knowledge_curve_fit(freq_z:np.ndarray, H_mag:np.ndarray) -> tuple:
    """Curve fit algorithm with prior knowledge.

    Args:
        freq_z (np.ndarray): Array of frequencies (from experiment).
        H_mag (np.ndarray): Array of magnitudes (from experiment).

    Returns:
        tuple: Tuple of parameter estimators.
    """

    def magnitude(freq_z:np.ndarray, Q:int, f_0:int) -> np.ndarray:
        """Generate magnitude (dB) of RLC filter.

        Args:
            freq_z (np.ndarray): Array of frequencies.
            Q (int): Quality factor.
            f_0 (int): Central frequency.

        Returns:
            np.ndarray: Array of magnitudes.
        """
        omega = 2*np.pi*freq_z
        omega_0 = 2*np.pi*f_0
        H_s = 1 / (1 + 1.j*Q*(omega/omega_0 - omega_0/omega)) # transfert function

        return 20*np.log10( np.absolute( H_s ) )

    # prior knowledge
    max_index = np.argmax(H_mag)
    f_0_prior = freq_z[max_index]
    cutoff_indices = np.where(-3 < H_mag)[0] # cutoff at -3 dB

    if cutoff_indices.size < 2:
        cutoff_indices = [max_index-1, max_index+1]
        
    Q_prior = f_0_prior / (freq_z[ cutoff_indices[-1] ] - freq_z[ cutoff_indices[0] ])

    # curve fit with prior knowledge
    (Q_pred, f_0_pred), _ = curve_fit(magnitude, freq_z, H_mag, p0=[Q_prior, f_0_prior])
        
    return abs(Q_pred), abs(f_0_pred)

# number of measurements
N_ITER = 3

start = time.time() # timer

# -------------------------------------------------------------------------- #
#                                  HARDWARE                                  #
# -------------------------------------------------------------------------- #

# Gives the device a handle
status = {}
chandle = ctypes.c_int16()

# Open the scope
status["openunit"] = ps.ps2000aOpenUnit(ctypes.byref(chandle), None)

try:
    assert_pico_ok(status["openunit"])
except:
    # powerstate becomes the status number of openunit
    powerstate = status["openunit"]

    # If powerstate is the same as 282 then it will run this if statement
    if powerstate == 282:
        # Changes the power input to "PICO_POWER_SUPPLY_NOT_CONNECTED"
        status["ChangePowerSource"] = ps.ps2000aChangePowerSource(chandle, 282)
    # If the powerstate is the same as 286 then it will run this if statement
    elif powerstate == 286:
        # Changes the power input to "PICO_USB3_0_DEVICE_NON_USB3_0_PORT"
        status["ChangePowerSource"] = ps.ps2000aChangePowerSource(chandle, 286)
    else:
        raise

    assert_pico_ok(status["ChangePowerSource"])


# Select channel A ranges and AC/DC coupling
chARange = 6
status["setChA"] = ps.ps2000aSetChannel(chandle, # handle = chandle
                                        0, # channel = PS2000A_CHANNEL_A = 0
                                        1, # enabled = 1
                                        1, # coupling type = PS2000A_DC = 1
                                        chARange, # range = PS2000A_1V = 6
                                        0) # analogue offset = 0 V
assert_pico_ok(status["setChA"])

# Select channel B ranges and AC/DC coupling
chBRange = 6
status["setChB"] = ps.ps2000aSetChannel(chandle, # handle = chandle
                                        1, # channel = PS2000A_CHANNEL_B = 1
                                        1, # enabled = 1
                                        1, # coupling type = PS2000A_DC = 1
                                        chBRange, # range = PS2000A_1V = 6
                                        0) # analogue offset = 0 V
assert_pico_ok(status["setChB"])

preTriggerSamples = 8000000
postTriggerSamples = 8000000
noSamples = preTriggerSamples + postTriggerSamples # loads buffer memory to 32 MS

# Select timebase
timebase = 1
timeIntervalns = ctypes.c_float()
returnedMaxSamples = ctypes.c_int32()
oversample = ctypes.c_int16(0)
status["getTimebase2"] = ps.ps2000aGetTimebase2(chandle, # handle = chandle
                                                timebase, # timebase = 1 (2**1 / 500,000,000 = 4 ns = 250 MHz)
                                                noSamples, # noSamples = noSamples
                                                ctypes.byref(timeIntervalns), # pointer to timeIntervalNanoseconds = ctypes.byref(timeIntervalNs)
                                                oversample, # not used
                                                ctypes.byref(returnedMaxSamples), # pointer to maxSamples = ctypes.byref(returnedMaxSamples)
                                                0) # segment index = 0
assert_pico_ok(status["getTimebase2"])

# Set up the trigger
status["trigger"] = ps.ps2000aSetSimpleTrigger(chandle, # handle = chandle
                                               1, # enabled = 1
                                               0, # source = PS2000A_CHANNEL_A = 0
                                               0, # threshold = 0 ADC counts
                                               2, # direction = PS2000A_RISING = 2
                                               0, # delay = 0 s
                                               0) # auto Trigger = 0 ms (wait indefinitely for a trigger)
assert_pico_ok(status["trigger"])

bufferAMax = (ctypes.c_int16 * noSamples)()
bufferAMin = (ctypes.c_int16 * noSamples)()
bufferBMax = (ctypes.c_int16 * noSamples)()
bufferBMin = (ctypes.c_int16 * noSamples)()

# Memory buffer A
status["setDataBuffersA"] = ps.ps2000aSetDataBuffers(chandle, # handle = chandle
                                                     0, # source = PS2000A_CHANNEL_A = 0
                                                     ctypes.byref(bufferAMax), # pointer to buffer max = ctypes.byref(bufferDPort0Max)
                                                     ctypes.byref(bufferAMin), # pointer to buffer min = ctypes.byref(bufferDPort0Min)
                                                     noSamples, # buffer length = noSamples
                                                     0, # segment index = 0
                                                     0) # ratio mode = PS2000A_RATIO_MODE_NONE = 0
assert_pico_ok(status["setDataBuffersA"])

# Memory buffer B
status["setDataBuffersB"] = ps.ps2000aSetDataBuffers(chandle, # handle = chandle
                                                     1, # source = PS2000A_CHANNEL_B = 1
                                                     ctypes.byref(bufferBMax), # pointer to buffer max = ctypes.byref(bufferBMax)
                                                     ctypes.byref(bufferBMin), # pointer to buffer min = ctypes.byref(bufferBMin)
                                                     noSamples, # buffer length = noSamples
                                                     0, # segment index = 0
                                                     0) # ratio mode = PS2000A_RATIO_MODE_NONE = 0
assert_pico_ok(status["setDataBuffersB"])

# Set up signal generator
# Output a square wave peak-to-peak 1 V and frequency 1 Hz
wavetype = ctypes.c_int16(1)
sweepType = ctypes.c_int32(0)
triggertype = ctypes.c_int32(0)
triggerSource = ctypes.c_int32(0)
status["SetSigGenBuiltIn"] = ps.ps2000aSetSigGenBuiltIn(chandle, # handle = chandle
                                                        0, # offsetVoltage = 0
                                                        1000000, # pkToPk = 1,000,000
                                                        wavetype, # waveType = ctypes.c_int16(1) = PS2000A_SQUARE
                                                        1, # startFrequency = 1 Hz
                                                        1, # stopFrequency = 1 Hz
                                                        0, # increment = 0
                                                        1, # dwellTime = 1
                                                        sweepType, # sweepType = ctypes.c_int16(1) = PS2000A_UP
                                                        0, # operation = 0
                                                        0, # shots = 0
                                                        0, # sweeps = 0
                                                        triggertype, # triggerType = ctypes.c_int16(0) = PS2000A_SIGGEN_RISING
                                                        triggerSource, # triggerSource = ctypes.c_int16(0) = PS2000A_SIGGEN_NONE
                                                        1) # extInThreshold = 1
assert_pico_ok(status["SetSigGenBuiltIn"])

# Start the scope
status["runBlock"] = ps.ps2000aRunBlock(chandle, # handle = chandle
                                        preTriggerSamples, # number of pre-trigger samples = preTriggerSamples
                                        postTriggerSamples, # number of post-trigger samples = PostTriggerSamples
                                        timebase, # timebase = 1
                                        oversample, # not used
                                        None, # value is not required
                                        0, # segment index = 0
                                        None, # lpReady = None (using ps2000aIsReady)
                                        None) # value is not required
assert_pico_ok(status["runBlock"])

# Wait until the scope is ready
ready = ctypes.c_int16(0)
check = ctypes.c_int16(0)
while ready.value == check.value:
    status["isReady"] = ps.ps2000aIsReady(chandle, # handle = chandle
                                          ctypes.byref(ready)) # pointer to ready = ctypes.byref(ready)

# Transfer the block of data from the scope
cTotalSamples = ctypes.c_int32(noSamples)
overflow = ctypes.c_int16()
status["getValues"] = ps.ps2000aGetValues(chandle, # handle = chandle
                                          0, # start index = 0
                                          ctypes.byref(cTotalSamples), # pointer to noOfSamples = ctypes.byref(cTotalSamples)
                                          0, # downsample ratio = 0
                                          0, # downsample ratio mode = PS2000A_RATIO_MODE_NONE
                                          0, # segment index = 0
                                          ctypes.byref(overflow)) # pointer to overflow = ctypes.byref(overflow))
assert_pico_ok(status["getValues"])


# Get maximum ADC count in GetValues call
maxADC = ctypes.c_int16()
status["maximumValue"] = ps.ps2000aMaximumValue(chandle, # handle = chandle
                                                ctypes.byref(maxADC)) # pointer to value = ctypes.byref(maxADC)
assert_pico_ok(status["maximumValue"])

# Raw ADC count values to mV 
ChA1 =  np.array(adc2mV(bufferAMax, chARange, maxADC))*1e-3
ChB1 =  np.array(adc2mV(bufferBMax, chBRange, maxADC))*1e-3
# Time data
t = np.linspace(0, ((cTotalSamples.value)-1) * timeIntervalns.value, cTotalSamples.value)*1e-9

max_voltage = np.amax(np.abs(ChB1))

if max_voltage < 0.020:
    chBRange = 1
elif max_voltage < 0.050:
    chBRange = 2
elif max_voltage < 0.100:
    chBRange = 3
elif max_voltage < 0.200:
    chBRange = 4
elif max_voltage < 0.500:
    chBRange = 5
elif max_voltage < 1:
    chBRange = 6
else:
    chBRange = 7

status["setChB"] = ps.ps2000aSetChannel(chandle, # handle = chandle
                                        1, # channel = PS2000A_CHANNEL_B = 1
                                        1, # enabled = 1
                                        1, # coupling type = PS2000A_DC = 1
                                        chBRange, # chBRange
                                        0) # analogue offset = 0 V
assert_pico_ok(status["setChB"])

f_0_preds = np.zeros(N_ITER)
Q_preds = np.zeros(N_ITER)
beta_preds = np.zeros(N_ITER)

# loop on number of measurements
for i in range(N_ITER):

    # Start the scope
    status["runBlock"] = ps.ps2000aRunBlock(chandle, # handle = chandle
                                            preTriggerSamples, # number of pre-trigger samples = preTriggerSamples
                                            postTriggerSamples, # number of post-trigger samples = PostTriggerSamples
                                            timebase, # timebase = 1
                                            oversample, # not used
                                            None, # value is not required
                                            0, # segment index = 0
                                            None, # lpReady = None (using ps2000aIsReady)
                                            None) # value is not required
    assert_pico_ok(status["runBlock"])

    # Wait until the scope is ready
    ready = ctypes.c_int16(0)
    check = ctypes.c_int16(0)
    while ready.value == check.value:
        status["isReady"] = ps.ps2000aIsReady(chandle, # handle = chandle
                                            ctypes.byref(ready)) # pointer to ready = ctypes.byref(ready)

    # Transfer the block of data from the scope
    status["getValues"] = ps.ps2000aGetValues(chandle, # handle = chandle
                                            0, # start index = 0
                                            ctypes.byref(cTotalSamples), # pointer to noOfSamples = ctypes.byref(cTotalSamples)
                                            0, # downsample ratio = 0
                                            0, # downsample ratio mode = PS2000A_RATIO_MODE_NONE
                                            0, # segment index = 0
                                            ctypes.byref(overflow)) # pointer to overflow = ctypes.byref(overflow))
    assert_pico_ok(status["getValues"])


    # Get maximum ADC count in GetValues call
    status["maximumValue"] = ps.ps2000aMaximumValue(chandle, # handle = chandle
                                                    ctypes.byref(maxADC)) # pointer to value = ctypes.byref(maxADC)
    assert_pico_ok(status["maximumValue"])

    # Raw ADC count values to mV 
    ChA2 =  np.array(adc2mV(bufferAMax, chARange, maxADC))*1e-3
    ChB2 =  np.array(adc2mV(bufferBMax, chBRange, maxADC))*1e-3
    # Time data
    t = np.linspace(0, ((cTotalSamples.value)-1) * timeIntervalns.value, cTotalSamples.value)*1e-9

    # -------------------------------------------------------------------------- #
    #                                  SOFTWARE                                  #
    # -------------------------------------------------------------------------- #

    SAMPLING_RATE = 1 / (timeIntervalns.value*1e-9) # Hz


    # step response time window
    trigger_index = ChB2.size // 2
    analytic_signal = hilbert( ChB2[ trigger_index: ] ) # compute analytic signal of response
    instantaneous_phase = np.unwrap( np.angle( analytic_signal ) )
    instantaneous_frequency = (np.diff( instantaneous_phase ) / (2*np.pi) * SAMPLING_RATE) # compute instantaneous frequency of analytic signal
    noise_frequency_threshold = 0.95*np.amax( instantaneous_frequency ) # max instant frequency is reached in noise region, set threshold at 95% of max

    window_index = np.where( instantaneous_frequency > noise_frequency_threshold )[0][0]
    extended_window_index = min( instantaneous_frequency.size, window_index*10 )

    # signal processing
    x_t_process = np.copy( ChA2[ trigger_index-window_index : trigger_index+extended_window_index ] ) # cut time signals to minimize noise integration
    y_t_process = np.copy( ChB2[ trigger_index-window_index : trigger_index+extended_window_index ] )
    t_process = np.copy( t[ trigger_index-window_index : trigger_index+extended_window_index ] )

    x_dot_t_process = np.diff( x_t_process ) # time derivative of signals (emulates delta dirac from a step signal)
    y_dot_t_process = np.diff( y_t_process )

    X_dot_z_process = np.fft.fft( x_dot_t_process ) # Fourier transform to switch in frequency domain
    Y_dot_z_process = np.fft.fft( y_dot_t_process )

    freq_z = np.fft.fftfreq( y_dot_t_process.size, d=1/SAMPLING_RATE )

    # frequency response
    f_lower = 1e2
    f_upper = 1e6

    i_lower = int( f_lower*y_dot_t_process.size/SAMPLING_RATE )+1
    i_upper = int( f_upper*y_dot_t_process.size/SAMPLING_RATE )

    freq_z = freq_z[ i_lower : i_upper ] # cut frequency signals between f_lower and f_upper (fitting bounds)
    X_dot_z_process = X_dot_z_process[ i_lower : i_upper ]
    Y_dot_z_process = Y_dot_z_process[ i_lower : i_upper ]

    # H estimator
    H_mag_norm = 20*np.log10( np.absolute( Y_dot_z_process ) / np.absolute( X_dot_z_process ) )
    H_phase_norm = np.angle( Y_dot_z_process, deg=True ) - np.angle( X_dot_z_process, deg=True )

    H_phase_norm[H_phase_norm < -180] %= (180) # normalize phase within [-pi, pi]
    H_phase_norm[H_phase_norm > 180] %= (-180)

    # curve fit
    peaks_norm, _ = find_peaks( H_mag_norm, threshold=8 ) # cut noise region before fitting in frequency domain, irregular peaks are found in noise, detection threshold is set to 8 dB
    peak_norm_index = peaks_norm[1] if peaks_norm.size > 1 else H_mag_norm.size-1

    Q_pred_norm, f_0_pred_norm = prior_knowledge_curve_fit( freq_z[ :peak_norm_index ], H_mag_norm[ :peak_norm_index ] ) # curve fit with prior knowledge (see function definition)
    beta_pred_norm = f_0_pred_norm / Q_pred_norm

    f_0_preds[i] = f_0_pred_norm
    Q_preds[i] = Q_pred_norm
    beta_preds[i] = beta_pred_norm


# Stop the scope
status["stop"] = ps.ps2000aStop(chandle) # handle = chandle
assert_pico_ok(status["stop"])

# Close the unit
status["close"] = ps.ps2000aCloseUnit(chandle) # handle = chandle
assert_pico_ok(status["close"])

# Display status
print(status)


f_0_pred = f_0_preds.mean() # average N_ITER measurements
Q_pred = Q_preds.mean()
beta_pred = beta_preds.mean()


f = np.logspace(2, 6, 1000000) # generate data for Bode diagram
rlc_filter_pred_norm = getFilter(Q=Q_pred_norm, f_0=f_0_pred_norm)
_, mag_norm, phase_norm = bode(rlc_filter_pred_norm, w=2*np.pi*f)


d = datetime.now()
date_label = d.strftime("%Y-%m-%d-%H:%M:%S")
date_text = d.strftime("%Y-%m-%d, %H:%M:%S")

figname = f'reports1/bodes/bode_{date_label}.png'
reportname_txt = f'reports1/txt/report_{date_label}.txt'
reportname_pdf = f'reports1/report_{date_label}.pdf'


# generate bode diagram
with plt.style.context(['mplstyle/ieee.mplstyle']): # matplotlib style for IEEE journal figures

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)

    # magnitude
    ax1.semilogx(freq_z, H_mag_norm, label='Mesure')
    ax1.semilogx(f, mag_norm, label='Fit')
    ax1.set_ylabel('Amplitude (dB)')
    ax1.legend()

    # phase
    ax2.semilogx(freq_z, H_phase_norm)
    ax2.semilogx(f, phase_norm)
    ax2.set_xlabel('Fréquence (Hz)')
    ax2.set_ylabel('Phase ($^\circ$)')

    plt.savefig(figname, dpi=600, bbox_inches='tight')


stop = time.time()
delta = stop - start


# generate report
report = '---\n'\
        'title: Rapport de caractérisation\n'\
        'author: LundiM - Équipe 1\n'\
        f'date: {date_text}\n'\
        '---\n'\
        '\n'\
        f'![Diagramme de Bode du filtre passe-bande]({figname})\n'\
        '\n'\
        '| Paramètre | Valeur | Erreur relative |\n'\
        '| :----: | :----: | :----: |\n'\
        f'| Fréquence centrale $f_0$ | {f_0_pred:.3f} (Hz) | $\pm 7\%$ |\n'\
        f'| Largeur de bande | {beta_pred:.3f} (Hz) | $\pm 12\%$ |\n'\
        f'| Facteur de qualité $Q$ | {Q_pred:.3f} (-) | $\pm 5\%$ |\n'\
        '\n'\
        f'Temps requis pour {N_ITER} mesures: {delta:.3f} (s)'

with open(reportname_txt, 'w') as f:
    f.write(report)


# compile report
os.system(f'pandoc {reportname_txt} --pdf-engine=xelatex -o {reportname_pdf}')
os.system('play -nq -t alsa synth .1 sine 700')
