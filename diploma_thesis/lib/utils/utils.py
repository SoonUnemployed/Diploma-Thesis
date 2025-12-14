import mne

def channels(raw: mne.io.edf.edf.RawEDF) -> mne.io.edf.edf.RawEDF:

    #Delete unused Biosemi channels
    keep_ch = raw.ch_names[:32] + ["Status"]
    raw = raw.pick(keep_ch)

    #Change names to Biosemi32 standard
    montage = mne.channels.make_standard_montage(kind = "biosemi32")

    rename_dict = dict(zip(raw.ch_names, montage.ch_names)) 
    raw.rename_channels(rename_dict) 

    #Set montage
    raw.set_montage(montage)
    
    return raw

def cut_sig(raw: mne.io.edf.edf.RawEDF) -> mne.io.edf.edf.RawEDF:
    
    #Get the events
    events = mne.find_events(raw, stim_channel = "Status", shortest_event = 1)
    
    #Clip signal
    first_stim = events[events[:, 2] == 2][0, 0] #find the first code 2 event in frames
    last_stim = events[events[:, 2] == 2][-1, 0]

    first_stim_time = raw.times[first_stim]
    last_stim_time = raw.times[last_stim]

    start_time = max(2.0, first_stim_time - 2.0)
    end_time = min(raw.times[-1], last_stim_time  + 1.5  # stim duration
                                                  + 3.0)  # extra time until resting state screen closes
                                                  #+ 0.3) # extra time to be sure    

    raw.crop(tmin = start_time, tmax = end_time)

    return raw