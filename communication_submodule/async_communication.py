import asyncio
from enum import Enum
import json
import os
import struct
from pathlib import Path
import uuid


class AsyncCommunication:
    BUFFER_SIZE = 65536
    MSG_TYPE_TEXT = "TEXT"
    MSG_TYPE_CSV = "CSV"
    MSG_TYPE_IMAGE = "IMAGE"
    MSG_TYPE_UNKNOWN = "UNKNOWN"
    
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        # In asyncio, we don't hold a single 'socket' object the same way
        # We deal with (reader, writer) pairs per interaction.
    
        
    def get_msg_type(self, filepath: str) -> str:
        """Helper to determine message type from extension."""
        _, ext = os.path.splitext(filepath)
        ext = ext.lower()

        if ext in ['.txt', '.md', '.log', '.json', '.xml']:
            return self.MSG_TYPE_TEXT
        elif ext == '.csv':
            return self.MSG_TYPE_CSV
        elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']:
            return self.MSG_TYPE_IMAGE
        else:
            return self.MSG_TYPE_UNKNOWN
        
    async def _send_header(self, writer: asyncio.StreamWriter, msg_type: str, 
                           data_size: int, filename: str = ""):
        
        header = {
            "type": msg_type,
            "size": data_size,
            "filename": filename
        }
        header_json = json.dumps(header).encode('utf-8')
        header_size = struct.pack("!I", len(header_json))
        
        # Write to the buffer
        writer.write(header_size)
        writer.write(header_json)
        
        # 'drain' yields control to the event loop until the OS buffer is free.
        # This prevents the program from running too far ahead of the network.
        await writer.drain()

    async def _receive_header(self, reader: asyncio.StreamReader) -> tuple[str, int, str]:
        try:
            header_size_data = await reader.readexactly(4)
            header_size = struct.unpack("!I", header_size_data)[0]
            
            header_json = await reader.readexactly(header_size)
            header = json.loads(header_json.decode('utf-8'))
            
            return header["type"], header["size"], header["filename"]
        
        except asyncio.IncompleteReadError:
            raise ConnectionError("Connection closed during header receive")
        
        
class AsyncReceiver(AsyncCommunication):
    def __init__(self, host="0.0.0.0", port=5555, save_dir="./received"):
        super().__init__(host, port)
        self.server = None
        self.save_dir = save_dir

    async def start(self):
        # Starts the server and registers the callback
        self.server = await asyncio.start_server(
            self.handle_client, self.host, self.port
        )
        print(f"Async Server listening on {self.host}:{self.port}")
        
        async with self.server:
            await self.server.serve_forever()

    async def _stream_messages(self, reader):
        """Yields headers as they arrive. Handles the loop logic here."""
        while True:
            try:
                # Wait for next header
                msg_type, data_size, filename = await self._receive_header(reader)
                yield msg_type, data_size, filename
            except (asyncio.IncompleteReadError, ConnectionError):
                # Client disconnected naturally
                break
            
    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """
        This runs CONCURRENTLY for every connected sender.
        To avoid overwrites, each filename must be unique. It is up to the sender to ensure that,
        as they may have more meaningfull ways of identifying themselves instead of dynamic IP
        """
        addr = writer.get_extra_info('peername')
        session_id = str(uuid.uuid4())[:8]
        client_tag = f"[ID:{session_id} | {addr[0]}:{addr[1]}]"
        
        
        try:
            async for msg_type, size, filename in self._stream_messages(reader):
            
                save_path = os.path.join(self.save_dir, filename)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                # Add identifier to filename to avoid overwrites
                #if os.path.exists(filepath):
                #    base, ext = os.path.splitext(filename)
                #    filepath = os.path.join(self.save_dir, f"{base}_{session_id}{ext}")
                
                if msg_type == self.MSG_TYPE_CSV:
                    await self._receive_and_save_stream(reader, save_path, size, 'ab') #Append csv instead of overwrite
                    print(f"{client_tag} Success. Appended to: {save_path}")
                else:
                    await self._receive_and_save_stream(reader, save_path, size)
                    print(f"{client_tag} Success. Saved to: {save_path}")
                
                
            
        except asyncio.IncompleteReadError:
            # Triggered if the client closes the connection while we were expecting data
            print(f"{client_tag} WARNING: Client disconnected during transmission (Data truncated).")
        except ConnectionResetError:
            # Triggered if the client crashed or their OS forcibly closed the socket
            print(f"{client_tag} ALERT: Connection reset by peer (Hard disconnect).")     
        except Exception as e:
            print(f"Error with {addr}: {e}")
        finally:
            print(f"Closing connection {addr}")
            writer.close()
            await writer.wait_closed()


    async def _receive_and_save_stream(self, reader, filepath, total_size, op_type = 'wb'):
        """
        Reads from network and writes to disk incrementally.
        Uses a thread for the disk write to avoid blocking the event loop.
        """
        remaining = total_size
        
        with open(filepath, op_type) as f:
            while remaining > 0:
                chunk_size = min(self.BUFFER_SIZE, remaining)
                
                # Non-blocking network read
                data = await reader.readexactly(chunk_size)
                
                # Blocking disk write (offloaded to thread)
                await asyncio.to_thread(f.write, data)
                
                remaining -= len(data)
                
                

class AsyncSender(AsyncCommunication):
    def __init__(self, host, port=5555):
        super().__init__(host, port)
        self.reader = None
        self.writer = None

    async def connect(self):
        self.reader, self.writer = await asyncio.open_connection(self.host, self.port)

    async def send_text(self, text: str):
        data = text.encode('utf-8')
        await self._send_header(self.writer, self.MSG_TYPE_TEXT, len(data))
        
        self.writer.write(data)
        await self.writer.drain()

    async def send_file(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Not found: {filepath}")

        filename = os.path.basename(filepath)
        file_size = os.path.getsize(filepath)

        msg_type = self.get_msg_type(filepath)
        
        try:
            await self._send_header(self.writer, msg_type, file_size, filename)

            # Send file in chunks
            with open(filepath, 'rb') as f:
                bytes_sent = 0
                while bytes_sent < file_size:
                    # Read from disk in thread (to not freeze UI/Loop)
                    chunk = await asyncio.to_thread(f.read, self.BUFFER_SIZE)
                    if not chunk:
                        break
                    
                    self.writer.write(chunk)
                    # Important: Wait for OS to send before reading more
                    await self.writer.drain() 
                    
                    bytes_sent += len(chunk)
                    print(f"\rSending: {bytes_sent/file_size*100:.1f}%", end='')
                    
        except (BrokenPipeError, ConnectionResetError):
            print(f"\nCRITICAL: Lost connection to server {self.host}:{self.port}")    
            
            try:
                self.writer.close()
                await self.writer.wait_closed()
            except Exception:
                pass # Ignore errors during forced close

            raise ConnectionResetError(f"Failed to send {filename}")
    
    
    async def send_folder_recursive(self, folder_path: str):
        """
        Recursively sends a folder and its contents.
        Preserves structure by sending 'folder/subfolder/file.ext' as the filename.
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        # specific logic to capture the folder name itself in the path
        # If folder_path is "./my_run", we want the files to be named "my_run/file.txt"
        # So we calculate path relative to the *parent* of the target folder.
        base_path = os.path.dirname(os.path.normpath(folder_path))

        for root, _, files in os.walk(folder_path):
            for file in files:
                full_path = os.path.join(root, file)
                
                # Calculate relative path: e.g. "my_run/subdir/image.png"
                rel_path = os.path.relpath(full_path, start=base_path)
                
                # IMPORTANT: Normalize slashes to forward slashes '/' 
                # This ensures Windows clients can send to Linux servers correctly.
                remote_filename = rel_path.replace(os.sep, '/')

                await self.send_file(full_path)
    
    async def disconnect(self):
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()