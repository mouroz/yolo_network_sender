import sys
import os
from pathlib import Path

# 1. Define the path to the submodule
# distinct from the current script location
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory of your project
YOLO_DIR = ROOT / 'yolov5_submodule'

# 2. Add the submodule to sys.path so Python can "see" it
if str(YOLO_DIR) not in sys.path:
    sys.path.append(str(YOLO_DIR))

# 3. Import the specific function
# Now python treats 'yolov5_submodule' as a source folder
try:
    from detectBatchv3 import run as run_yolo
    print("YOLOv5 module imported successfully.")
except ImportError as e:
    print(f"Error importing YOLO: {e}")
    # Common fix: ensure __init__.py exists or check submodule path

# 4. Running the function
def process_images(source, project, name, batch_size=2, workers=2):
    # Usually, YOLO 'run' functions accept specific kwargs.
    # Check the detectBatchv3.py definition to match arguments exactly.
    
    return run_yolo(
        weights=ROOT / 'yolo_model/weights/best.pt',    # Path to your trained model
        source=source,   # Path to images
        project=project,
        name=name,
        save_txt=True, # Save results to *.txt files on project/labels
        #exist_ok=True, # Append results instead of creating new folders
        imgsz=(2048, 2048),
        batch_size=batch_size,
        workers=workers,
        prepoc_queue_size=4,
        # Add other arguments defined in detectBatchv3 here
    )

if __name__ == "__main__":
    save_dir = process_images()
    print(save_dir)