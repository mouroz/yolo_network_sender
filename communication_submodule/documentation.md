# Async TCP Communication System Documentation

This module provides a robust, asynchronous TCP server and client implementation using Python's asyncio. It supports sending text messages and streaming large files without blocking the event loop or freezing the UI.

## Core Features

Asynchronous I/O: Uses asyncio streams (StreamReader, StreamWriter) for high concurrency.

Non-Blocking File Transfer: Offloads disk I/O to threads to keep the network heartbeat alive during large transfers.

Header-Based Protocol: Every message is preceded by a JSON header containing metadata (type, size, filename).

## Protocol Structure

The communication follows a strict sequence for every message:

Header Size (4 bytes): Unsigned integer (!I) indicating the length of the JSON header.

JSON Header (Variable): Contains {"type": "...", "size": 12345, "filename": "data.bin"}.

Payload (Variable): The actual raw data (text bytes or file binary).

## Usage Guide

### 1. Starting the Server (AsyncReceiver)

The server listens for incoming connections and processes each one in a separate concurrent task.

```
import asyncio
from async_communication_test import AsyncReceiver

async def run_server():
    # Initialize server on all interfaces (0.0.0.0) port 5555
    server = AsyncReceiver(host="0.0.0.0", port=5555)
    
    print("Starting server...")
    # This starts the server in the background
    await server.start()

# Run the async loop
if __name__ == "__main__":
    asyncio.run(run_server())
```

### 2. Sending Text (AsyncSender)

The client connects, sends a message, and should disconnect gracefully.
```
import asyncio
from async_communication_test import AsyncSender

async def send_text_message():
    client = AsyncSender(host="127.0.0.1", port=5555)
    
    await client.connect()
    
    # Send a UTF-8 string
    await client.send_text("Hello from Async Client!")
    
    await client.disconnect()

asyncio.run(send_text_message())
```

### 3. Sending Files (AsyncSender)

Large files are read in chunks (default 64KB) to minimize memory usage.
```
import asyncio
from async_communication_test import AsyncSender, AsyncCommunication

async def upload_file():
    client = AsyncSender(host="127.0.0.1", port=5555)
    
    await client.connect()
    
    # Send a file (ensure file exists)
    await client.send_file(
        filepath="./my_large_image.png",
        msg_type=AsyncCommunication.MSG_TYPE_IMAGE
    )
    
    await client.disconnect()

asyncio.run(upload_file())
```

## Key Concepts for Developers

### The drain() Method

In the AsyncSender class, you will see await self.writer.drain() called after writing data. This is critical. It pauses the writing loop until the OS network buffer has enough space. Without this, a fast script could fill the RAM with 1GB of data instantly before it is actually sent over the network.

### Mixing Threads and Async

The system uses a "Sandwich Pattern" for file I/O to ensure the server remains responsive:

Async Loop: Reads raw bytes from the network.

Thread Pool: Writes those bytes to the hard drive (await asyncio.to_thread(f.write, data)).

Async Loop: Repeats.

This prevents the server from freezing (blocking) while waiting for slow hard drives.