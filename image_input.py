import sys
import os
import shutil
import re
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory of your project


YOLO_IN = ROOT / 'yolo_in'
YOLO_OUT = ROOT / 'yolo_out'

IMAGE_BATCH_SIZE = 4


import run_yolo
import sender 

def get_current_run_folder_name():
    """
    Finds the folder in YOLO_IN with the highest number appended (e.g., 'run12').
    If no folders exist, returns a path for 'run1' (but does not create it).
    """
    # Regex to capture folders named 'run' followed by numbers
    pattern = re.compile(r'^run(\d+)$')
    
    max_num = 0
    
    if YOLO_IN.exists():
        for item in YOLO_IN.iterdir():
            if item.is_dir():
                match = pattern.match(item.name)
                if match:
                    # Extract number and compare
                    num = int(match.group(1))
                    if num > max_num:
                        max_num = num
    
    # If max_num is 0, it means no folders exist, so we start at run1
    # If folders exist (e.g. run5), we return that. 
    if max_num == 0:
        max_num = 1 
        path = YOLO_IN / f'run{max_num}'
        path.mkdir(parents=True, exist_ok=True)
    
    return 'run{max_num}'



def create_new_run_folder():
    """Creates the next incremental run folder."""
    current = YOLO_IN / get_current_run_folder_name()
    
    
    match = re.search(r'run(\d+)$', current.name)
    next_num = int(match.group(1)) + 1
    next_run = f'run{next_num}'
    new_folder = YOLO_IN / next_run

    new_folder.mkdir(parents=True, exist_ok=True)
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


def is_run_folder_ready(run_dir):
    return get_image_count(run_dir) >= IMAGE_BATCH_SIZE


def copy_image(image_path, dest_dir):
    """Copies a single image to the destination directory."""
    image_path = Path(image_path)
    dest_dir = Path(dest_dir)
    
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # copy2 preserves metadata (timestamps)
        shutil.copy2(image_path, dest_dir / image_path.name)
        return True
    except Exception as e:
        print(f"Error copying {image_path.name}: {e}")
        return False


def copy_images(in_dir, dest_dir):
    """Copies all images from source to destination."""
    in_dir = Path(in_dir)
    dest_dir = Path(dest_dir)
    
    if not in_dir.exists():
        return
        
    for item in in_dir.iterdir():
        if item.is_file():
            copy_image(item, dest_dir)
            
            
def resend_pending_transfer(transfer_queue):
    """
    Retries transfers in the queue.
    transfer_queue should be a set to avoid duplicates.
    """
    # Create a copy of the list/set to iterate over, 
    # so we can safely remove items from the original `transfer_queue`
    folders_to_process = list(transfer_queue)
    
    for folder in folders_to_process:
        try:
            print(f"Retrying transfer for: {folder}")
            sender.transfer_folder_recursive(folder)
            # If successful, remove from the main queue
            transfer_queue.remove(folder) 
        except Exception as e:
            print(f"Retry failed for {folder}: {e}")
            # It stays in the queue for next time


def start_run(run_name, transfer_queue, batch=2, workers=2):
    in_folder = YOLO_IN / run_name
    out_folder = YOLO_OUT / run_name
    
    # Ensure output folder exists before running YOLO
    out_folder.mkdir(parents=True, exist_ok=True)
    
    run_yolo.run_yolo(
        weights=ROOT / 'yolo_model/weights/best.pt',    # Path to your trained model
        source=in_folder,   # Path to images
        project=out_folder,
        name='',
        batch_size=batch,
        workers=workers
    )
    
    try:
        sender.transfer_folder_recursive(in_folder)
        sender.transfer_folder_recursive(out_folder)
    except Exception:
        print("COMMUNICATION ERROR: RECONNECT SERVERS")
        transfer_queue.add(in_folder)
        transfer_queue.add(out_folder)
        return True
    return False

    
    
