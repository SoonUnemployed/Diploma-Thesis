import argparse
from pathlib import Path
import os 
import numpy as np
import torch

from lib.utils.set_seed import set_seed
from lib.train.get_csv import get_csv
from lib.train.dataset import load_dataset, reshape_dataset, apply_scalar
from lib.models.EEGNet import eegnet, eegnet_feature_extractor
from lib.utils.epoch_data import epoch_data
from lib.feature_extraction.features import features_per_epoch
from lib.utils.skip_processed import skip_processed
from lib.feature_extraction.save_embeddings import save_embeddings

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

    parser.add_argument(
        "--model",
        type = str,
        required = True,
        help = "EEGNet or XGBoost"
    )

    parser.add_argument(
        "--model_path",
        type = Path,
        required = True,
        help = "Path to model weights"
    )

    args = parser.parse_args()

    return args

def main(args: argparse.ArgumentParser):
    
    input_path = args.input
    output_path = args.output
    model_type = args.model
    model_type = model_type.lower()
    model_path = args.model_path

    if model_type == "xgboost":

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

    elif model_type == "eegnet":

        #Set seed 
        set_seed()
        #Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        #Get train-eval-test split csv 
        split_df = get_csv(input_path)   
  
        _ = "Not needed"
        #Load train dataset
        X, y, group_train = load_dataset(split_df, input_path, _, "train", model_type)
        #Load evaluation dataset
        X_val, y_val, group_val = load_dataset(split_df, input_path, _, "val", model_type)
        #Load test dataset
        X_test, y_test, group_test = load_dataset(split_df, input_path, _, "test", model_type)
        #Load impostor dataset
        X_imp, y_imp, group_imp = load_dataset(split_df, input_path, _, "impostor", model_type)


        #Apply scalar to dataset
        X = apply_scalar(X, model_type, "Not", model_path)
        X_val = apply_scalar(X_val, model_type, "Not", model_path)
        X_test = apply_scalar(X_test, model_type, "Not", model_path)
        X_imp = apply_scalar(X_imp, model_type, "Not", model_path)
        
        #Reshape data based on the model
        X = reshape_dataset(X, model_type)
        X_val = reshape_dataset(X_val, model_type)
        X_test = reshape_dataset(X_test, model_type)
        X_imp = reshape_dataset(X_imp, model_type)

        #Get model and extract embeddings
        model = eegnet(X, y, device)
        model_path = model_path / "eegnet.pth"
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        model.eval()

        embeddings_train = eegnet_feature_extractor(model, X, y, device)
        embeddings_val = eegnet_feature_extractor(model, X_val, y_val, device)
        embeddings_test = eegnet_feature_extractor(model, X_test, y_test, device)
        embeddings_imp = eegnet_feature_extractor(model, X_imp, y_imp, device)
        
        #Save embeddings per session
        save_embeddings(embeddings_train, group_train, y, output_path, "train")
        save_embeddings(embeddings_val, group_val, y_val, output_path, "val")
        save_embeddings(embeddings_test, group_test, y_test, output_path, "test")
        save_embeddings(embeddings_imp, group_imp, y_imp, output_path, "impostor")
    
    return 


if __name__ == "__main__":
    args = arg_parcer()
    main(args)