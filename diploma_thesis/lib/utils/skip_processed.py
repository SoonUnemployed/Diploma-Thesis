from pathlib import Path
import os

def skip_processed(file: str, file_type: str, output_path: Path) -> bool:

    temp = file.split(".")[0]

    files = [f for f in os.listdir(output_path) if f.endswith(file_type)]

    if not files:
        return False
    
    files = [f.split(".")[0] for f in files ]

    if temp in files: 
        return True
    
    else:
        return False   