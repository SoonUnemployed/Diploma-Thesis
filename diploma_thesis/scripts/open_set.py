import argparse
from pathlib import Path
import pandas as pd
import torch

from lib.utils.stimuli_labels import get_labels
from lib.train.dataset import load_dataset
from lib.open_set.load_emb import load_embeddings
from lib.open_set.aggregate import aggregate_by_session
from lib.open_set.build_templates import build_templates, calibrate_threshold
from lib.open_set.separate import separate
from lib.open_set.evaluate import eval_known, eval_unknown
from lib.open_set.save_results import save_results

def arg_parcer() -> argparse.ArgumentParser:
    
    parser = argparse.ArgumentParser(description = "Open-set model")
    
    parser.add_argument(
        "--embeddings",
        type = Path,
        required = True,
        help = "Path to embeddings"
    )
    
    parser.add_argument(
        "--split",
        type = Path,
        required = True,
        help = "Path to split csv"
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

    parser.add_argument(
        "--model_type",
        type = str,
        required = True,
        help = "Model (EEGNet or XGBoost) to open the correct type of array file"
    )
    
    args = parser.parse_args()

    return args

def main(args: argparse.ArgumentParser):
    
    split_path = args.split
    feature_path = args.embeddings
    output_path = args.output
    stim_labels_path = args.labels 
    model_type = args.model_type
    model_type = model_type.lower()

    if model_type == "xgboost":

        path = output_path / ("xgboost_open_set.npz")
    
        split_path = split_path / ("split.csv")
        split_df = pd.read_csv(split_path)
        
        _ = "Not_used"
        train_emb, train_y, train_grp = load_dataset(split_df, _, feature_path, "train", model_type)
        val_emb, val_y, val_grp = load_dataset(split_df, _, feature_path, "val", model_type)
        test_emb, test_y, test_grp = load_dataset(split_df, _, feature_path, "test", model_type)
        imp_emb, imp_y, imp_grp = load_dataset(split_df, _, feature_path, "impostor", model_type)

        #Apply z-score
        train_emb = torch.from_numpy(train_emb).float()
        val_emb = torch.from_numpy(val_emb).float()
        test_emb = torch.from_numpy(test_emb).float()
        imp_emb = torch.from_numpy(imp_emb).float()

        target_far = 1.0
            

    elif model_type == "eegnet":

        path = output_path / ("eeg_open_set.npz")
        
        #Load all sets
        load_path = feature_path / ("train_embeddings.pt")
        train_emb, train_y, train_grp = load_embeddings(load_path)

        load_path = feature_path / ("val_embeddings.pt")
        val_emb, val_y, val_grp = load_embeddings(load_path)

        load_path = feature_path / ("test_embeddings.pt")
        test_emb, test_y, test_grp = load_embeddings(load_path)

        load_path = feature_path / ("impostor_embeddings.pt")
        imp_emb, imp_y, imp_grp = load_embeddings(load_path)
        
        target_far = 0.7

    #Aggregate embeddings
    emb_train_ses, y_train_ses, train_grp = aggregate_by_session(train_emb, train_y, train_grp)    
    emb_val_ses, y_val_ses, val_grp = aggregate_by_session(val_emb, val_y, val_grp)
    emb_test_ses, y_test_ses, test_grp = aggregate_by_session(test_emb, test_y, test_grp)
    emb_imp_ses, y_imp_ses, imp_grp = aggregate_by_session(imp_emb, imp_y, imp_grp)

    #Create enrollment set 
    templates = build_templates(emb_train_ses, y_train_ses)

    #Calibrate threshold (Closed_set val + Closed_set test)
    emb_thr = torch.cat([emb_val_ses, emb_test_ses], dim = 0)
    y_thr = torch.cat([y_val_ses, y_test_ses], dim = 0)

    thresholds = calibrate_threshold(emb_thr, y_thr, templates, target_far)

    #Split know - unknown from impostor list
    emb_known, y_known, grp_known, emb_unkn, y_unkn, grp_unkn = separate(emb_imp_ses, y_imp_ses, imp_grp, templates)

    #Evaluate the impostor sessions
    FRR, TP, FN = eval_known(emb_known, y_known, grp_known, templates, thresholds)
    FAR, FP, TN, imp_matches = eval_unknown(emb_unkn, y_unkn, grp_unkn, templates, thresholds)

    #Save results
    save_results(path, TP, FN, FP, TN, FAR, FRR, thresholds, imp_matches)

    print("===== AUTHENTICATION RESULTS =====")

    print(f"TP (True Accepts): {TP}")
    print(f"FN (False Rejects): {FN}")
    print(f"FP (False Accepts): {FP}")
    print(f"TN (True Rejects): {TN}")

    print("\n----- Rates -----")
    print(f"FRR (False Reject Rate): {FRR:.4f}")
    print(f"FAR (False Accept Rate): {FAR:.4f}")
    print(f"TAR (True Accept Rate): {TP / max(1, TP + FN):.4f}")

    print("\n----- Dataset sizes -----")
    print(f"Genuine attempts: {TP + FN}")
    print(f"Impostor attempts: {FP + TN}")

    print("\n----- Impostor attribution -----")
    if len(imp_matches) == 0:
        print("No impostors were accepted.")
    else:
        for imp_sess, users in imp_matches.items():
            print(f"{imp_sess} → accepted as users {users}")

    print("=================================")
    
    return 

if __name__ == "__main__":
    args = arg_parcer()
    main(args)