#!/usr/bin/env python3
"""
Prefect Embedded Server Configuration

This module provides configuration for running Prefect server embedded within the ApeRAG API process.
This reduces the number of processes but couples the services together.
"""

import asyncio
import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from prefect.server.api.server import create_app
from prefect.settings import PREFECT_API_URL, PREFECT_SERVER_API_HOST, PREFECT_SERVER_API_PORT

logger = logging.getLogger(__name__)


class EmbeddedPrefectServer:
    """Embedded Prefect server that runs in a separate thread"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 4200):
        self.host = host
        self.port = port
        self.server: Optional[uvicorn.Server] = None
        self.thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
    
    def start(self):
        """Start the Prefect server in a separate thread"""
        if self.thread and self.thread.is_alive():
            logger.warning("Prefect server is already running")
            return
        
        logger.info(f"Starting embedded Prefect server on {self.host}:{self.port}")
        
        # Set Prefect server configuration
        os.environ[PREFECT_SERVER_API_HOST.name] = self.host
        os.environ[PREFECT_SERVER_API_PORT.name] = str(self.port)
        os.environ[PREFECT_API_URL.name] = f"http://{self.host}:{self.port}/api"
        
        # Create and start server in thread
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.thread.start()
        
        # Wait a moment for server to start
        import time
        time.sleep(2)
        
    def _run_server(self):
        """Run the Prefect server in the current thread"""
        try:
            # Create Prefect app
            app = create_app()
            
            # Configure uvicorn server
            config = uvicorn.Config(
                app=app,
                host=self.host,
                port=self.port,
                log_level="info",
                access_log=False  # Reduce log noise
            )
            
            self.server = uvicorn.Server(config)
            
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run server
            loop.run_until_complete(self.server.serve())
            
        except Exception as e:
            logger.error(f"Prefect server failed: {e}")
        finally:
            self._shutdown_event.set()
    
    def stop(self):
        """Stop the Prefect server"""
        if self.server:
            logger.info("Stopping embedded Prefect server...")
            if hasattr(self.server, 'should_exit'):
                self.server.should_exit = True
        
        if self.thread:
            self._shutdown_event.wait(timeout=10)
            if self.thread.is_alive():
                logger.warning("Prefect server thread did not stop gracefully")
    
    def is_running(self) -> bool:
        """Check if the server is running"""
        return self.thread is not None and self.thread.is_alive() and not self._shutdown_event.is_set()


# Global embedded server instance
_embedded_server: Optional[EmbeddedPrefectServer] = None


def start_embedded_prefect_server(host: str = "127.0.0.1", port: int = 4200):
    """Start embedded Prefect server"""
    global _embedded_server
    
    if _embedded_server and _embedded_server.is_running():
        logger.info("Embedded Prefect server is already running")
        return _embedded_server
    
    _embedded_server = EmbeddedPrefectServer(host, port)
    _embedded_server.start()
    return _embedded_server


def stop_embedded_prefect_server():
    """Stop embedded Prefect server"""
    global _embedded_server
    
    if _embedded_server:
        _embedded_server.stop()
        _embedded_server = None


@asynccontextmanager
async def prefect_lifespan(app):
    """FastAPI lifespan context manager for embedded Prefect server"""
    # Startup
    logger.info("Starting embedded Prefect server with API...")
    server = start_embedded_prefect_server()
    
    yield
    
    # Shutdown
    logger.info("Stopping embedded Prefect server...")
    stop_embedded_prefect_server()


def configure_embedded_mode():
    """Configure environment for embedded mode"""
    os.environ["TASK_SCHEDULER_TYPE"] = "prefect"
    os.environ["PREFECT_API_URL"] = "http://127.0.0.1:4200/api"
    os.environ["PREFECT_LOGGING_LEVEL"] = "WARNING"  # Reduce log noise
    
    logger.info("Configured for embedded Prefect mode")


if __name__ == "__main__":
    # Test the embedded server
    import time
    
    logging.basicConfig(level=logging.INFO)
    
    server = start_embedded_prefect_server()
    
    try:
        print("Embedded Prefect server started. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_embedded_prefect_server()
        print("Server stopped.") 