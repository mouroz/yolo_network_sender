import asyncio
from pathlib import Path
import random
import string
import shutil

from async_communication import AsyncCommunication, AsyncReceiver, AsyncSender

def create_dummy_data(root_dir, folder_name, file_count=3):
    """Creates a folder with random text files."""
    path = Path(root_dir) / folder_name
    path.mkdir(parents=True, exist_ok=True)
    
    for i in range(file_count):
        fname = f"file_{i}.txt"
        content = ''.join(random.choices(string.ascii_letters, k=500))
        with open(path / fname, 'w') as f:
            f.write(content)
    
    # Create a subfolder to test recursion
    sub = path / "sub"
    sub.mkdir(exist_ok=True)
    with open(sub / "deep_file.log", 'w') as f:
        f.write("Deep log content")
        
    return path

# ==========================================
# 3. MAIN TEST EXECUTION
# ==========================================

# async def dummy_test():
#     TEST_ROOT = Path("test_env")
#     SRC_DIR = TEST_ROOT / "source"
#     src1 = create_dummy_data(SRC_DIR, "data_s1")
#     src2 = create_dummy_data(SRC_DIR, "data_s2")
#     test(src1, src2, "a")
    
async def test(src1, src2, dest):
    HOST = "127.0.0.1"
    PORT = 8888
    
    # --- Setup Directories ---
    
    DEST_DIR = dest
    print("--- 1. Generating Dummy Data ---")

    # --- Start Receiver ---
    print("\n--- 2. Starting Receiver ---")
    receiver = AsyncReceiver(host=HOST, port=PORT, save_dir=str(DEST_DIR))
    # Run server in background task
    server_task = asyncio.create_task(receiver.start())
    await asyncio.sleep(0.5) # Give it a moment to bind

    # --- Run Senders Concurrently ---
    print("\n--- 3. Starting Senders ---")
    
    async def run_sender_1():
        sender = AsyncSender(HOST, PORT)
        await sender.connect()
        # This sends "data_s1/..." but prefixes it with "sender1"
        # Result on server: ./received_test/sender1/data_s1/file_0.txt
        await sender.send_folder_recursive(str(src1))
        await sender.disconnect()
        print(">> Sender 1 Finished")

    async def run_sender_2():
        sender = AsyncSender(HOST, PORT)
        await sender.connect()
        # Result on server: ./received_test/sender2/data_s2/file_0.txt
        await sender.send_folder_recursive(str(src2))
        await sender.disconnect()
        print(">> Sender 2 Finished")

    # Run both senders at the same time
    await asyncio.gather(run_sender_1(), run_sender_2())

    # --- Verification ---
    print("\n--- 4. Verifying Results ---")
    await asyncio.sleep(1) # Ensure file buffers flush
    

    # Cleanup server
    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        pass

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory of your project

if __name__ == "__main__":
    try:
        asyncio.run(test(ROOT / 'yolo_out/yolo_results', ROOT / 'yolo_out/yolo_results2', ROOT / 'test'))
    except KeyboardInterrupt:
        pass