from pathlib import Path
import pandas as pd
import os

def get_csv(path: Path) -> pd.DataFrame:

    csv_files = [f for f in os.listdir(path) if f.lower().endswith(".csv")]

    if len(csv_files) != 1:
        print(csv_files)
        raise SystemExit("Wrong number of csv files")
    
    csv_path = path / csv_files[0]

    df = pd.read_csv(csv_path)

    return df