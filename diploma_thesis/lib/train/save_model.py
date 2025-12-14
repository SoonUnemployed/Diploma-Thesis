from pathlib import Path
import torch 

def save_model(model, model_type: str, output_path: Path):

    if model_type == "eegnet":
        output_path = output_path / (model_type + ".pth")
        torch.save(model.state_dict(), output_path)

    elif model_type == "xgboost":
        output_path = output_path / (model_type + ".json")
        model.save_model(output_path)
    
    else:
        raise TypeError("Unknown model type")