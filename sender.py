
import asyncio
import os
from pathlib import Path
import sys
import argparse

# 1. Define the path to the submodule
# distinct from the current script location
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory of your project
COMM_DIR = ROOT / 'communication_submodule'

# 2. Add the submodule to sys.path so Python can "see" it
if str(COMM_DIR) not in sys.path:
    sys.path.append(str(COMM_DIR))




import run_yolo
import async_communication as comm
import image_input as img



CURRENT_RUN_NAME = ''
CURRENT_RUN_FOLDER = ''
TRANSFER_QUEUE = []
WORKERS = 2
BATCH = 2

def init_global_variables():
    CURRENT_RUN_NAME = img.get_current_run_folder_name()
    CURRENT_RUN_FOLDER = img.YOLO_IN / CURRENT_RUN_NAME

async def start_connection(HOST='127.0.0.1', PORT=5555):
    # 2. Start Server in background
    sender = comm.AsyncSender(HOST, PORT)
    await sender.connect()


def run_and_update():
    comm_status = img.start_run(run_name=CURRENT_RUN_FOLDER, transfer_queue=TRANSFER_QUEUE, batch=BATCH, workers=WORKERS)
    CURRENT_RUN_NAME = img.create_new_run_folder()
    CURRENT_RUN_FOLDER =  img.YOLO_IN / CURRENT_RUN_NAME
    return comm_status
    
def add_file(path):
    img.copy_image(path, CURRENT_RUN_FOLDER)
    if img.is_run_folder_ready(CURRENT_RUN_FOLDER):
        print(f'Got a sizable batch of {img.get_image_count(CURRENT_RUN_FOLDER)}, starting run')
        return run_and_update()
    return False

def add_folder(path):
    img.copy_images(path, CURRENT_RUN_FOLDER)
    if img.is_run_folder_ready(CURRENT_RUN_FOLDER):
        print(f'Got a sizable batch of {img.get_image_count(CURRENT_RUN_FOLDER)}, starting run')
        return run_and_update()
    return False        

def force_run():
    return run_and_update()




def is_image(path):
    """Checks if path exists and has an image extension."""
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    return os.path.exists(path) and os.path.isfile(path) and Path(path).suffix.lower() in valid_exts


async def operation_terminal(sender_instance):
    """
    Interactive terminal for manual control.
    Args:
        sender_instance: The active AsyncSender object.
    """
    global current_sender
    current_sender = sender_instance

    print("\n" + "="*40)
    print("      YOLO & TRANSFER COMMAND CENTER      ")
    print("="*40)

    while True:
        print("\n--- OPTIONS ---")
        print("1. Restart Connection")
        print("2. Copy Image to Buffer")
        print("3. Copy Folder of Images to Buffer")
        print("4. Force Run Detection")
        print("5. Retry Pending Transfer Queue")
        print("6. List Cached Runs")
        print("7. Resend Run Folder to Server")
        print("0. Exit Terminal")
        
        choice = input("\nSelect option: ").strip()

        if choice == '1':
            # Restart Connection
            new_host = input("Enter HOST IP: ").strip()
            new_port = input("Enter PORT: ").strip()
            
            if new_port.isdigit():
                print(f"Reconnecting to {new_host}:{new_port}...")
                try:
                    # Assuming sender has disconnect/connect methods
                    await current_sender.disconnect()
                    current_sender.host = new_host
                    current_sender.port = int(new_port)
                    await current_sender.connect()
                    # Optional: Re-authenticate if using handshake
                    # await current_sender.authenticate("my_id") 
                    print("✅ Connection restarted successfully.")
                except Exception as e:
                    print(f"❌ Connection failed: {e}")
            else:
                print("❌ Invalid Port.")

        elif choice == '2':
            # Copy Image
            img_path = input("Enter image path: ").strip().strip("'").strip('"') # Clean quotes
            
            if is_image(img_path):
                dest = img.get_current_run_folder()
                print(f"Copying to current batch: {dest.name}")
                if img.copy_image(img_path, dest):
                    print("✅ Image copied.")
                else:
                    print("❌ Copy failed.")
            else:
                print("❌ Invalid path or not an image file.")

        elif choice == '3':
            # Copy Folder
            folder_path = input("Enter folder path: ").strip().strip("'").strip('"')
            
            if os.path.exists(folder_path) and os.path.isdir(folder_path):
                dest = img.get_current_run_folder()
                print(f"Copying images to current batch: {dest.name}")
                img.copy_images(folder_path, dest)
                print("✅ Operation complete.")
            else:
                print("❌ Invalid folder path.")

        elif choice == '4':
            # Force Run Detection
            print("🚀 Forcing detection run...")
            
            # Using the start_run from utility_script
            # Note: start_run logic likely needs the run name, e.g., 'run5'
            try:
                force_run()
  
            except Exception as e:
                print(f"❌ Error during run: {e}")

        elif choice == '5':
            # Retry Pending Queue
            print("Checking transfer queue...")
            if not TRANSFER_QUEUE:
                print("ℹ️  Queue is empty. Nothing to send.")
            else:
                print(f"Found {len(TRANSFER_QUEUE)} items pending.")
                img.resend_pending_transfer(TRANSFER_QUEUE)
                print("✅ Retry attempt finished.")

        elif choice == '6':
            # List Cached Runs
            if not img.YOLO_IN.exists():
                print("ℹ️  YOLO_IN directory does not exist yet.")
            else:
                runs = [d for d in img.YOLO_IN.iterdir() if d.is_dir() and d.name.startswith('run')]
                print(f"\n📂 Cached Runs in {img.YOLO_IN}:")
                for run in sorted(runs, key=lambda x: x.name):
                    # Count images inside
                    count = img.get_image_count(run)
                    print(f" - {run.name}: {count} images")
                print(f"Total runs: {len(runs)}")

        elif choice == '7':
            # Resend Specific Run
            run_name = input("Enter run name (e.g., 'run1'): ").strip()
            folder_to_send = img.YOLO_OUT / run_name
            
            if not folder_to_send.exists():
                print(f"❌ Folder not found: {folder_to_send}")
            else:
                print(f"Sending {folder_to_send}...")
                try:
                    # Calling the async sender method
                    await current_sender.send_folder_recursive(str(folder_to_send))
                    print("✅ Sent successfully.")
                except Exception as e:
                    print(f"❌ Failed to send: {e}")
                    print("Adding to pending queue.")
                    TRANSFER_QUEUE.add(folder_to_send)

        elif choice == '0':
            print("Exiting terminal...")
            break
        
        else:
            print("❌ Invalid option.")
            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Transfer Client Terminal")
    parser.add_argument("host", type=str, help="Server IP address", default="127.0.0.1")
    parser.add_argument("port", type=int, help="Server Port", default=8888)

    args = parser.parse_args()
    init_global_variables()
    
    sender = None
    async def main():
        sender = comm.AsyncSender(args.host, args.port)
        
        print(f"Attempting connection to {args.host}:{args.port}...")
        try:
            await sender.connect()
            print("✅ Connection established.")
            
            # Start the terminal loop
            await operation_terminal(sender)
            
        except (ConnectionRefusedError, OSError) as e:
            print(f"❌ CRITICAL: Could not connect to server.")
            print(f"Details: {e}")
            sys.exit(1)
        except KeyboardInterrupt:
            print("\nUser interrupted.")
        finally:

            if sender is not None:
                try:
                    await sender.disconnect()
                except:
                    pass

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass