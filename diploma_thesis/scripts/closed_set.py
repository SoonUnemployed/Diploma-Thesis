from pathlib import Path
import argparse
import torch
import numpy as np

from lib.utils.set_seed import set_seed
from lib.train.get_csv import get_csv
from lib.train.dataset import load_dataset, reshape_dataset, apply_scalar
from lib.models.EEGNet import eegnet, train_eegnet
from lib.models.XGBoost import XGBoost_Classifier, train_xgboost
from lib.train.save_model import save_model
from lib.train.label_mapping import label_mapping
from lib.train.remap import remap
from lib.train.cv import cross_validation

def arg_parcer() -> argparse.ArgumentParser:
    
    parser = argparse.ArgumentParser(description = "Closed-set model")
    
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
        help = "Path save model"
    )

    parser.add_argument(
        "--model",
        type = str,
        required = True,
        help = "eegnet or xgboost"
    )


    args = parser.parse_args()

    return args

def main(args: argparse.ArgumentParser):
    
    preprocessed_path = args.input_pre
    features_path = args.input_features
    output_path = args.output
    model_type = args.model
    model_type = model_type.lower()

    #Set seed 
    set_seed()
    #Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #Get train-eval-test split csv 
    split_df = get_csv(preprocessed_path)
    
    #Map labels
    label_mapping(split_df, output_path)   
    
   #NO CV CODE (train + evaluation)    
    #Load train dataset
    X, y, _ = load_dataset(split_df, preprocessed_path, features_path, "train", model_type)
    #Load evaluation dataset
    X_val, y_val, _ = load_dataset(split_df, preprocessed_path, features_path, "val", model_type)
    
    #Apply scalar to dataset
    X = apply_scalar(X, model_type, "train", output_path)
    X_val = apply_scalar(X_val, model_type, "val", output_path)
    
    #Reshape data based on the model
    X = reshape_dataset(X, model_type)
    X_val = reshape_dataset(X_val, model_type)

    #Remap to new labels 
    y = remap(y, output_path)
    y_val = remap(y_val, output_path)

    #Get and train model
    if model_type == "eegnet":
        model = eegnet(X, y, device)
        model, _ = train_eegnet(model, X, y, X_val, y_val, device)
    elif model_type == "xgboost":
        model = XGBoost_Classifier(y)
        model, _ = train_xgboost(model, X, y, X_val, y_val)
    
    #Save trained model
    save_model(model, model_type, output_path)

    '''
    #CV CODE 
    #Load train dataset for XGBoost
    X, y, groups = load_dataset(split_df, preprocessed_path, features_path, "none", model_type)

    #Split into sets(ONLY WHEN DOING CROSS VALIDATION)
    cv = cross_validation()

    #Remap to new labels 
    y = remap(y, output_path)

    scores = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups), 1):
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        print(f"\nFold {fold}")
        print("  Train epochs:", len(X_train))
        print("  Val epochs:", len(X_val))

        #Apply scalar to dataset
        X_train = apply_scalar(X_train, model_type, "train", output_path)
        X_val = apply_scalar(X_val, model_type, "val", output_path)
        
        #Reshape data based on the model
        X_train = reshape_dataset(X_train, model_type)
        X_val = reshape_dataset(X_val, model_type)


        #Get and train model
        if model_type == "eegnet":
            model = eegnet(X_train, y_train, device)
            model, acc = train_eegnet(model, X_train, y_train, X_val, y_val, device)
        elif model_type == "xgboost":
            model = XGBoost_Classifier(y_train)
            model, acc = train_xgboost(model, X_train, y_train, X_val, y_val)
        
        scores.append(acc)

        print(f"Fold {fold}: val acc = {acc:.3f}")

    print("Model CV: ", np.mean(scores), "±", np.std(scores))
    '''

    return 


if __name__ == "__main__":
    args = arg_parcer()
    main(args)