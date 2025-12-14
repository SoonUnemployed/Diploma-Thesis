import argparse
from pathlib import Path
import os 
import numpy as np

from lib.utils.epoch_data import epoch_data
from lib.feature_extraction.features import features_per_epoch
from lib.utils.skip_processed import skip_processed

def arg_parcer() -> argparse.ArgumentParser:
    
    parser = argparse.ArgumentParser(description = "Feature extraction")
    
    parser.add_argument(
        "--input",
        type = Path,
        required = True,
        help = "Path to feature files"
    )

    parser.add_argument(
        "--output",
        type = Path,
        required = True,
        help = "Path to results folder"
    )

    args = parser.parse_args()

    return args

def main(args: argparse.ArgumentParser):
    
    input_path = args.input
    output_path = args.output

    files = [f for f in os.listdir(input_path) if f.endswith(".fif")]
    
    for file in files:
        #Check if already preprocessed
        if skip_processed(file, ".npy", output_path):
            continue
        features = []
        #Get path to load file 
        load_path = input_path / file 
        #Epoch the data based on the acoustic stimuli 
        epochs = epoch_data(load_path)
        #Get features
        for ep in epochs:
            f = features_per_epoch(ep)
            features.append(f)
        
        feat_array = np.vstack(features)
        #Save features
        file = file[:-4] #Drop ".fif"
        save_path = output_path / file
        np.save(save_path, feat_array)
    
    return 


if __name__ == "__main__":
    args = arg_parcer()
    main(args)