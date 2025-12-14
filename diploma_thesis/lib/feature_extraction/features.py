import numpy as np
from scipy.signal import welch
import mne

def bandpower(signal: mne.io.Raw, fmin: float, fmax: float, fs: int = 256) -> np.float64:
    freqs, psd = welch(signal, fs)
    idx = np.logical_and(freqs >= fmin, freqs <= fmax)
    return psd[idx].mean()

#Hjorth parameters
def hjorth(signal: mne.io.Raw) -> tuple[float, float, float]:
    first_deriv = np.diff(signal)
    second_deriv = np.diff(signal, n=2)

    var0 = np.var(signal)
    var1 = np.var(first_deriv)
    var2 = np.var(second_deriv)

    activity = var0
    mobility = np.sqrt(var1 / var0) if var0 > 0 else 0
    complexity = np.sqrt(var2 / var1) / mobility if var1 > 0 and mobility > 0 else 0

    return activity, mobility, complexity

def spectral_entropy(signal: mne.io.Raw , fs: int = 256):
    _, psd = welch(signal, fs)
    psd_norm = psd / psd.sum()
    return -(psd_norm * np.log2(psd_norm + 1e-12)).sum()


def features_per_epoch(epoch: mne.epochs.Epochs) -> np.array:
    #Initialize feature list
    features = []
    #Frequency bands for band power
    bands = [(1,4), (4,8), (8,13), (13,30), (30,45)]

    #Calculate features 
    for ch in range(epoch.shape[0]):
        signal = epoch[ch]
        #Band power
        bp = []
        for fmin, fmax in bands:
            bp.append(bandpower(signal, fmin, fmax))
        #Hjorth parameters
        activity, mobility, complexity = hjorth(signal)
        #RMS
        rms = np.sqrt(np.mean(signal**2))
        #Peak-to-peak
        p2p = signal.max() - signal.min()
        #Spectral entropy
        entr = spectral_entropy(signal)
        #Combine all features in one vector
        channel_features = bp + [activity, mobility, complexity, entr, rms, p2p]
        features.extend(channel_features)

    features = np.array(features)

    return features