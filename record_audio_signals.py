"""Interface a soundcard to record live audio signals from a piezoelectric sensor

"""
import pickle
import queue
import sys

from scipy.signal import correlate2d
import numpy as np
import sounddevice as sd


DEVICE = None
WINDOW = 1000 # ms
DOWNSAMPLE = 1
SAMPLERATE = sd.query_devices(DEVICE, 'input')['default_samplerate']
THRESHOLD = 0.005
N_SAMPLES = 1000

MIN_RESPONSE_TIME = 4410 # ~100 ms (avoid double threshold from strong resonance)

RESPONSE_TIME = max(N_SAMPLES, MIN_RESPONSE_TIME)

# initialize queue
q = queue.Queue()


def audio_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(indata[::DOWNSAMPLE, [0]])


def update_data():
    global data
    while True:
        try:
            block = q.get_nowait()
        except queue.Empty:
            break
        shift = len(block)
        data = np.roll(data, -shift, axis=0)
        data[-shift:] = block[:,0]


# initialize data array
length = int(WINDOW * SAMPLERATE / (1000 * DOWNSAMPLE))
data = np.zeros(length)

# load storage (signals, labels, notes) in memory
ref_signals = np.load('storage/signals.npy')#[:-2] # array [n_samples, n_features]
labels = ['c3', 'd3', 'e3', 'f3', 'g3', 'a4', 'b4', 'c-3', 'd-3', 'f-3', 'g-3', 'a-4']
with open('storage/notes.pkl', 'rb') as f:
    notes = pickle.load(f) # dictionnary of tuples (note, fs)

stream = sd.InputStream(
    device=DEVICE, channels=1,
    samplerate=SAMPLERATE, callback=audio_callback)

with stream: # audio_callback stream on a different thread
    # main loop
    while True:
        update_data() # update data to keep the last %WINDOW ms in memory
        index_max = np.argmax( np.abs( data[:-RESPONSE_TIME] ) )

        if np.abs( data[index_max] ) > THRESHOLD:
            
            # signal read out and normalization
            signal = data[ index_max : index_max+N_SAMPLES ]
            signal /= np.linalg.norm(signal)
            signal = np.reshape(signal, (1,-1)) # array [1, n_features]

            # correlation
            probabilities = np.sort(
                np.amax( 
                    correlate2d(ref_signals, signal, mode='same'), axis=1 
                    ).reshape(12, 10), axis=1
                )[:,1:-1].mean(axis=1)

            label = labels[ np.argmax(probabilities) ]
            note, fs = notes[label]
            sd.play(note, fs, device=1)
            
            data = np.zeros(length) # reset data (avoid double threshold)
