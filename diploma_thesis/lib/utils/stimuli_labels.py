from pathlib import Path 

def get_labels(file: str, stim_labels_path: Path) -> list:
    
    #Fix label path
    temp = file[:-4]
    label_path = stim_labels_path / (temp + ".txt")
    
    #Get label sequence from txt file
    with open(label_path, "r") as f:
        content = f.read()
    labels = [int(x) for x in content.split(" ")]

    return labels 