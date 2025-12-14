import mne
from mne.preprocessing import ICA
import mne_icalabel
from lib.preprocessing.filtering import ica_filters, filters

def choose_components(filtered_signal: mne.io.edf.edf.RawEDF, ica: mne.preprocessing.ica.ICA) -> list:
    
    #Label ICA components using ICALABEL
    ica_labels = mne_icalabel.label_components(filtered_signal, ica, method = "iclabel")

    #Find eye-blink artifacts using Fp1 and Fp2 channels
    
    eog_inds, _ = ica.find_bads_eog(filtered_signal, ch_name = ["Fp1", "Fp2"])

    #Compare EOG and ICALABEL results for eye-blink artifacts
    iclabel_eye_components = [i for i, label in enumerate(ica_labels["labels"]) if label == "eye blink"]
    overlap = set(iclabel_eye_components) & set(eog_inds)

    #Mark components as EOG correctly
    if bool(overlap) == 0:
        if eog_inds:
            eog_components = eog_inds
        else:
            eog_components = iclabel_eye_components
    else:
        eog_components = list(overlap)

    #All components that need to be excluded (We keep "other")
    exclude_comp =  [i for i, label in enumerate(ica_labels["labels"]) if label not in ["brain", "other", "eye blink"]] + eog_components
    exclude_comp.sort()

    return exclude_comp


def compute_ica(raw: mne.io.edf.edf.RawEDF) -> mne.preprocessing.ICA:
    
    #Filter to compute ica
    filtered_signal  = ica_filters(raw)

    #Set Common average reference for ICA
    filtered_signal = filtered_signal.set_eeg_reference("average", projection = False)

    #Initialize ICA
    ica = ICA(n_components = 0.99, max_iter = "auto", method = "fastica", random_state = 19) #, fit_params = dict(extended = True)) for infomax
    
    #Fit ICA to signal
    ica.fit(filtered_signal)

    #Find which components to exclude from the signal
    exclude_comp = choose_components(filtered_signal, ica)
    
    #Exclude them from the ica
    ica.exclude = exclude_comp 

    return ica


def artifact_removal(raw: mne.io.edf.edf.RawEDF, ica: mne.preprocessing.ICA) -> mne.io.edf.edf.RawEDF:

    #Copy raw signal
    clean_signal = raw.copy()

    #Apply ICA
    ica.apply(clean_signal)

    #Filter signal
    clean_signal = filters(raw)

    return clean_signal