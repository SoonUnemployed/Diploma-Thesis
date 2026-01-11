import mne 

def ica_filters(raw: mne.io.edf.edf.RawEDF) -> mne.io.edf.edf.RawEDF:

    filt_raw = raw.copy().filter(l_freq = 1.0, h_freq = None)

    filt_raw.notch_filter(freqs = (50, 100, 150), method = "spectrum_fit")

    return filt_raw


def filters(raw: mne.io.edf.edf.RawEDF) -> mne.io.edf.edf.RawEDF:

    clean_signal = raw.copy().filter(l_freq = 0.1, h_freq = None)

    clean_signal.notch_filter(freqs = (50, 100, 150), method = "spectrum_fit")

    return clean_signal