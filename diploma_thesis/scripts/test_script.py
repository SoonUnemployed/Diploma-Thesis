from pathlib import Path 
import numpy as np
from braindecode.models import EEGNet


model = EEGNet(32, 15, n_times=718, sfreq=256.0)

print(model)