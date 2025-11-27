
import asyncio
import os
from pathlib import Path
import sys
import argparse
import traceback 
# 1. Define the path to the submodule
# distinct from the current script location
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory of your project
COMM_DIR = ROOT / 'communication_submodule'

# 2. Add the submodule to sys.path so Python can "see" it
if str(COMM_DIR) not in sys.path:
    sys.path.append(str(COMM_DIR))




import run_yolo
from async_communication import AsyncSender
import image_input as img



YOLO_IN = ROOT / 'yolo_in'
YOLO_OUT = ROOT / 'yolo_out'
PREFIX = 'run'

class BatchManager:
    def __init__(self, sender: AsyncSender, batch_size=2, workers=2, transfer_batch_size=4):
        # Configuration (Defaults that can be overridden)
        self.batch_size = batch_size
        self.workers = workers
        self.transfer_batch_size=transfer_batch_size # Amount of images added to current run before its sent to yolo automatically
        self.transfer_queue = []
        self.sender = sender
        
        
        # State: Initialize the first run folder immediately upon creation
        self.current_run_name = img.get_max_increment_folder_name(YOLO_IN, PREFIX)
        if self.current_run_name == "":
            self.current_run_name = PREFIX + "1"
            
        self.current_folder = YOLO_IN / self.current_run_name
        self.result_folder = YOLO_OUT / self.current_run_name
        
        print(f"Initialized BatchManager. Output: {self.current_folder}")
        

    

    async def _run_and_rotate_folder(self):        
        # Ensure output folder exists before running YOLO
        self.result_folder.mkdir(parents=True, exist_ok=True)
        
        run_yolo.run_yolo(
            weights=ROOT / 'yolo_model/weights/best.pt',    # Path to your trained model
            source=self.current_folder,   # Path to images
            project=self.result_folder,
            name='',
            batch_size=self.batch_size,
            workers=self.workers,
            exist_ok=True
        )
        
        self.current_run_name = img.create_new_increment_folder(YOLO_IN, PREFIX)
        
        error=False
        try:
            # Pass ROOT as base_path to preserve yolo_in/runX and yolo_out/runX structure
            await self.sender.send_folder_recursive(str(self.current_folder), str(ROOT))
            await self.sender.send_folder_recursive(str(self.result_folder), str(ROOT))
            
        except Exception as e:
            print(f"COMMUNICATION ERROR: {e}")
            traceback.print_exc()
            self.transfer_queue.append(self.current_folder)
            self.transfer_queue.append(self.result_folder)
            error = True    
        
        finally:
            self.current_folder =  YOLO_IN / self.current_run_name
            self.result_folder = YOLO_OUT / self.current_run_name
    
            
        return error
    
   
        
    def is_run_folder_ready(self):
        print(img.get_image_count(self.current_folder))
        return img.get_image_count(self.current_folder) >= self.transfer_batch_size
    
    async def async_add_file(self, path):
        img.copy_image(path, self.current_folder)
        if self.is_run_folder_ready():
            print(f'Got a sizable batch of {img.get_image_count(self.current_folder)}, starting run')
            return await self._run_and_rotate_folder()
        return False

    async def async_add_folder(self, path):
        img.copy_images(path, self.current_folder)
        if self.is_run_folder_ready():
            print(f'Got a sizable batch of {img.get_image_count(self.current_folder)}, starting run')
            return await self._run_and_rotate_folder()
        return False        

    async def async_force_run(self):
        if img.get_image_count(self.current_folder) > 0:
            return await self._run_and_rotate_folder()
        else:
            print("Current folder is empty. Nothing to force run.")
            return False
        
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





def is_image(path):
    """Checks if path exists and has an image extension."""
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    return os.path.exists(path) and os.path.isfile(path) and Path(path).suffix.lower() in valid_exts


async def operation_terminal(sender_instance:AsyncSender, manager:BatchManager):
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
            if not img_path.startswith("/"):
                img_path = ROOT / img_path
                
            if is_image(img_path):
                
                print(f"Copying to current batch: {manager.current_folder}")
                if await manager.async_add_file(img_path):
                    print("❌ Copy failed.")
                else:
                    print("✅ Image copied.")
            else:
                print(f"❌ Invalid path or not an image file: {img_path}.")

        elif choice == '3':
            # Copy Folder
            folder_path = input("Enter folder path: ").strip().strip("'").strip('"')
            if not folder_path.startswith("/"):
                folder_path = ROOT / folder_path
            
            if os.path.exists(folder_path) and os.path.isdir(folder_path):
                print(f"Copying images to current batch: {manager.current_folder}")
                if await manager.async_add_folder(folder_path):
                    print("❌ Communication error!")
                else:
                    print("✅ Operation complete.")
            else:
                print(f"❌ Invalid folder path {folder_path}")

        elif choice == '4':
            # Force Run Detection
            print("🚀 Forcing detection run...")
            
            # Using the start_run from utility_script
            # Note: start_run logic likely needs the run name, e.g., 'run5'
            try:
                await manager.async_force_run()
  
            except Exception as e:
                print(f"❌ Error during run: {e}")

        elif choice == '5':
            # Retry Pending Queue
            print("Checking transfer queue...")
            if not manager.transfer_queue:
                print("ℹ️  Queue is empty. Nothing to send.")
            else:
                print(f"Found {len(manager.transfer_queue)} items pending.")
                img.resend_pending_transfer(manager.transfer_queue)
                print("✅ Retry attempt finished.")

        elif choice == '6':
            # List Cached Runs
            if not YOLO_IN.exists():
                print("ℹ️  YOLO_IN directory does not exist yet.")
            else:
                runs = [d for d in YOLO_IN.iterdir() if d.is_dir() and d.name.startswith('run')]
                print(f"\n📂 Cached Runs in {YOLO_IN}:")
                for run in sorted(runs, key=lambda x: x.name):
                    # Count images inside
                    count = img.get_image_count(run)
                    print(f" - {run.name}: {count} images")
                print(f"Total runs: {len(runs)}")

        elif choice == '7':
            # Resend Specific Run
            run_name = input("Enter run name (e.g., 'run1'): ").strip()
            folder_to_send = YOLO_OUT / run_name
            
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
                    manager.transfer_queue.append(folder_to_send)

        elif choice == '0':
            print("Exiting terminal...")
            break
        
        else:
            print("❌ Invalid option.")
            




        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Transfer Client Terminal")
    parser.add_argument("host", type=str, help="Server IP address", default="127.0.0.1")
    parser.add_argument("port", type=int, help="Server Port", default=8888)
    parser.add_argument("--folder-batch-size", type=int, help="Minimum number of Images before batching to send to model", default=4)
    parser.add_argument("--yolo-batch-size", type=int, help="Size of batches used per detection", default=4)
    parser.add_argument("--yolo-workers-count", type=int, help="Number of workers (thread) to run concurrently", default=2)
    

    args = parser.parse_args()

    
    sender = None
    async def main():
        sender = AsyncSender(args.host, args.port)
        
        print(f"Attempting connection to {args.host}:{args.port}...")
        try:
            await sender.connect()
            print("✅ Connection established.")
            
            manager = BatchManager(
                sender=sender,
                batch_size=args.yolo_batch_size, 
                workers=args.yolo_workers_count, 
                transfer_batch_size=args.folder_batch_size
            )
                
            # Start the terminal loop
            await operation_terminal(sender, manager)
            
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