import sys
import os
import shutil
import re
from pathlib import Path

# 1. Define the path to the submodule
# distinct from the current script location
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory of your project
COMM_DIR = ROOT / 'communication_submodule'

# 2. Add the submodule to sys.path so Python can "see" it
if str(COMM_DIR) not in sys.path:
    sys.path.append(str(COMM_DIR))

from async_communication import AsyncSender



IMAGE_BATCH_SIZE = 4


import run_yolo
import sender 

def get_max_increment_folder_name(path: Path, prefix: str):
    """
    Finds the folder in YOLO_IN with the highest number appended (e.g., 'run12').
    If no folders exist, returns a path for 'run1' (but does not create it).
    """
    # Regex to capture folders named 'run' followed by numbers
    pattern = re.compile(rf'^{prefix}(\d+)$')
    
    max_num = 0
    
    if path.exists():
        for item in path.iterdir():
            if item.is_dir():
                match = pattern.match(item.name)
                if match:
                    # Extract number and compare
                    num = int(match.group(1))
                    if num > max_num:
                        max_num = num
    
    # If max_num is 0, it means no folders exist, so we start at run1
    # If folders exist (e.g. run5), we return that. 
    
    name = f'run{max_num}' if max_num > 0 else ''
    return name



def create_new_increment_folder(path: Path, prefix: str):
    """Creates the next incremental run folder."""
    current =  get_max_increment_folder_name(path, prefix)
    
    
    match = re.search(rf'{prefix}(\d+)$', current)
    next_num = int(match.group(1)) + 1
    next_run = f'run{next_num}'
    
    new_in_folder = path / next_run
    new_in_folder.mkdir(parents=True, exist_ok=True)
    new_out_folder = path / next_run
    new_out_folder.mkdir(parents=True, exist_ok=True)
    
    return next_run


def get_image_count(directory):
    """Counts image files in a directory."""
    directory = Path(directory)
    if not directory.exists():
        return 0
        
    # Set of valid extensions to avoid counting system files
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    # Generator expression is memory efficient
    return sum(1 for f in directory.iterdir() if f.is_file() and f.suffix.lower() in valid_exts)




def copy_image(image_path, dest_dir):
    """Copies a single image to the destination directory."""
    image_path = Path(image_path)
    dest_dir = Path(dest_dir)
    
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # copy2 preserves metadata (timestamps)
        shutil.copy2(image_path, dest_dir / image_path.name)
        return False
    except Exception as e:
        print(f"Error copying {image_path.name}: {e}")
        return True


def copy_images(in_dir, dest_dir):
    """Copies all images from source to destination."""
    in_dir = Path(in_dir)
    dest_dir = Path(dest_dir)
    
    if not in_dir.exists():
        return
        
    for item in in_dir.iterdir():
        if item.is_file():
            copy_image(item, dest_dir)
            
            




    
