import argparse
from pathlib import Path
import os 
from lib.preprocessing.pipeline import pipeline
from lib.preprocessing.split import split_data
from lib.utils.skip_processed import skip_processed

def arg_parcer() -> argparse.ArgumentParser:
    
    parser = argparse.ArgumentParser(description = "EEG data preprocessing")
    
    parser.add_argument(
        "--input",
        type = Path,
        required = True,
        help = "Path to the raw EEG files"
    )

    parser.add_argument(
        "--output",
        type = Path,
        required = True,
        help = "Output folder path"
    )

    args = parser.parse_args()

    return args

def main(args: argparse.ArgumentParser):
    
    input_path = args.input
    output_path = args.output

    files = [f for f in os.listdir(input_path) if f.endswith(".bdf")]
    
    for file in files:
        #Check if already preprocessed
        if skip_processed(file, ".fif", output_path):
            continue

        #Get path to load file 
        load_path = input_path / file 

        #Clean signal
        clean_signal = pipeline(load_path) 

        #Save signal
        file = file[:-4] #Drop ".bdf"
        save_path = output_path / (file + ".fif")
        clean_signal.save(save_path, overwrite = True)

    #Split the files to train, validation and testing Datasets
    split_data(output_path)

if __name__ == "__main__":
    args = arg_parcer()
    main(args)