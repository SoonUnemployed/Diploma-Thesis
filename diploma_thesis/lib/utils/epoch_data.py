import mne 
from pathlib import Path
import numpy as np

def epoch_data(load_path: Path) -> mne.epochs.Epochs:
    
    #Load data 
    processed  = mne.io.read_raw_fif(load_path, preload = True)

    #Get when the stimuli happened
    events = mne.find_events(processed, stim_channel = "Status", shortest_event = 1)

    #Drop the Status channel
    processed = processed.pick(picks = "eeg")

    #Calculate decim to resample signal
    current_sfreq = processed.info["sfreq"]
    desired_sfreq = 256  # Hz
    decim = np.round(current_sfreq / desired_sfreq).astype(int)
    obtained_sfreq = current_sfreq / decim
    lowpass_freq = obtained_sfreq / 3.0 

    #Low-pass filter the signal
    processed_filt = processed.copy().filter(l_freq = None, h_freq = lowpass_freq)
    
    #Epoch the data based on the events
    epochs = mne.Epochs(processed_filt, events = events, tmin = -0.2, tmax = 2.3, baseline = (None, 0), preload = True, decim = decim)

    return epochs