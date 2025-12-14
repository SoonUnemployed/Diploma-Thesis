import pathlib
import mne
from lib.utils.utils import channels, cut_sig
from lib.preprocessing.artifacts import artifact_removal, compute_ica

def pipeline(input_path: pathlib.PosixPath) -> mne.io.edf.edf.RawEDF:
    
    #Get file
    raw = mne.io.read_raw_bdf(input_path, preload = True)

    #Fix channels
    raw = channels(raw)

    #Clip signal
    raw = cut_sig(raw)

    #Compute ICA
    ica = compute_ica(raw)
    
    #Remove artifacts
    clean_signal = artifact_removal(raw, ica)
    
    return clean_signal