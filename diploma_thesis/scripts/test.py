from pathlib import Path 
import argparse
import torch
import joblib
from itertools import groupby
import numpy as np

from lib.utils.set_seed import set_seed
from lib.train.get_csv import get_csv
from lib.train.dataset import load_dataset, reshape_dataset, apply_scalar
from lib.models.EEGNet import eegnet, test_eegnet, test_eegnet_per_freq
from lib.models.XGBoost import XGBoost_Classifier, test_xgboost, test_xgboost_per_freq
from lib.train.remap import remap
from lib.test.save_metrics import save_metrics, save_metrics_per_frequency
from lib.utils.stimuli_labels import load_stim, select_seq, arrange_stim

def arg_parcer() -> argparse.ArgumentParser:
    
    parser = argparse.ArgumentParser(description = "Train/Evaluation of models")
    
    parser.add_argument(
        "--input_pre",
        type = Path,
        required = True,
        help = "Path to preprocessed data"
    )
    
    parser.add_argument(
        "--input_features",
        type = Path,
        required = True,
        help = "Path to manually extracted features"
    )

    parser.add_argument(
        "--output",
        type = Path,
        required = True,
        help = "Path to results folder"
    )

    parser.add_argument(
        "--model",
        type = str,
        required = True,
        help = "eegnet or xgboost"
    )

    parser.add_argument(
        "--model_path",
        type = Path,
        required = True,
        help = "Path to model weights"
    )

    parser.add_argument(
        "--stim",
        type = Path,
        required = True,
        help = "Path to stimuli sequences"
    )

    args = parser.parse_args()

    return args

def main(args: argparse.ArgumentParser):
    
    preprocessed_path = args.input_pre
    features_path = args.input_features 
    output_path = args.output
    model_path = args.model_path
    model_type = args.model
    stim_path = args.stim
    model_type = model_type.lower()

    #Set seed 
    set_seed()
    #Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #Get train-eval-test split csv 
    split_df = get_csv(preprocessed_path)
    
    #Load train dataset for XGBoost
    X, y, group = load_dataset(split_df, preprocessed_path, features_path, "test", model_type)
    #Apply scaler
    X = apply_scalar(X, model_type, "test", model_path)
    #Reshape data based on the model
    X = reshape_dataset(X, model_type)
    #Remap the labels
    y = remap(y, model_path)

    #Get stimuli sequence
    stim_seq = load_stim(stim_path)
    #Get the training stim seq
    stim_seq = select_seq(stim_seq, split_df, "test")
    #Arrange stim sequences in a array
    freq = arrange_stim(stim_seq, group)

    #Get and test model
    if model_type == "eegnet":
        model = eegnet(X, y, device)
        #results, y_pred = test_eegnet(model, X, y, model_path, device)
        results = test_eegnet_per_freq(model, X, y, freq, model_path, device)
    elif model_type == "xgboost":
        model = XGBoost_Classifier(y)
        #results, y_pred = test_xgboost(model, X, y, model_path)
        results = test_xgboost_per_freq(model, X, y, freq, model_path)
    
    #Save resulting metrics
    #save_metrics(results, y_pred, y, model_type, output_path)
    save_metrics_per_frequency(results, y, model_type, output_path)

    return 


if __name__ == "__main__":
    args = arg_parcer()
    main(args)