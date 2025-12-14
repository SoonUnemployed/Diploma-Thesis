import argparse
from pathlib import Path
import os 

from lib.utils.stimuli_labels import get_labels
from lib.utils.load_features import load_features

def arg_parcer() -> argparse.ArgumentParser:
    
    parser = argparse.ArgumentParser(description = "Cosine similarity")
    
    parser.add_argument(
        "--database",
        type = Path,
        required = True,
        help = "Path to database feature arrays"
    )

    parser.add_argument(
        "--output",
        type = Path,
        required = True,
        help = "Path to results folder"
    )

    parser.add_argument(
        "--labels",
        type = Path,
        required = True,
        help = "Path to stimuli labels"
    )
    
    args = parser.parse_args()

    return args

def main(args: argparse.ArgumentParser):
    
    input_path = args.input
    output_path = args.output
    stim_labels_path = args.labels 

    files = [f for f in os.listdir(input_path) if f.endswith(".npy")]

    for file in files:
        #Get path to load file 
        load_path = input_path / file
        #Load feature file
        features = load_features(load_path)
        #Get stimuli sequence (0 -> 500Hz, 1 -> 2000Hz, 2 -> 4000Hz)
        labels = get_labels(file, stim_labels_path)
        #Calculate cosine similarity of features



if __name__ == "__main__":
    args = arg_parcer()
    main(args)