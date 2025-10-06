import queue
import sys
import threading
import h5py
from pathlib import Path
import numpy as np
from datetime import datetime

import logging, logging.handlers

# Import AppConfig from daq_config.py
from lib_ipmu_daq_config import AppConfig


class DAQApp:
    """
    Orchestrator for the DAQ application.
    Initializes and holds shared resources like queues and stop events.
    """
    def __init__(self, config: AppConfig, DEBUG: bool = False):
        """Initializes the DAQApp with configuration."""
        self.cfg = config
        self.buf_q = None
        self.quad_q = None
        self.stop_event = None
        self.h5f = None
        self.dset = None
        self.runs_dir = None
        self.DEBUG = DEBUG
        self.logger = None
        self.log_q = None

    def setup(self):
        if self.DEBUG:
            logger = logging.getLogger("debug")
            if logger.hasHandlers():
                logger.handlers.clear()
            logger.setLevel(logging.INFO)


        """Sets up shared resources for the application components."""
        self.buf_q = queue.Queue(maxsize=self.cfg.io.queue_depth)
        self.quad_q = queue.Queue(maxsize=self.cfg.io.quad_depth)
        self.stop_event = threading.Event()
        print("Shared resources (Queues, Stop Event) are set up.")

        if self.DEBUG:
            print("DEBUG mode is ON.")
            self.log_q = queue.Queue(maxsize=0)
            queue_h = logging.handlers.QueueHandler(self.log_q)
            logger = logging.getLogger("debug")
            logger.setLevel(logging.INFO)
            logger.addHandler(queue_h)
            self.logger = logger

    def signalStop(self):
        """Signals all components to stop their loops."""
        if self.stop_event:
            print("Signaling all components to stop...")
            self.stop_event.set()

    def initLogger(self):
        if not self.DEBUG:
            print("Logger setup skipped: DEBUG mode is OFF.")
            return

        handler = logging.StreamHandler(sys.stdout)
        self.log_listener = logging.handlers.QueueListener(self.log_q, handler)
        self.log_listener.start()

    def initStorer(self, runs_dir: Path):
        """Initializes settings for saving data to an HDF5 file."""
        self.runs_dir = runs_dir
        self.runs_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%y%m%d%H%M%S")
        filepath = self.runs_dir / f"{timestamp}.h5"

        self.h5f = h5py.File(filepath, "w")
        self.dset = self.h5f.create_dataset(
            "log",
            shape=(0, self.cfg.logging.log_data_num),
            maxshape=(None, self.cfg.logging.log_data_num),
            dtype=np.float32,
            chunks=(self.cfg.logging.log_chunk, self.cfg.logging.log_data_num),
            compression="gzip"
        )
        print(f"HDF5 dataset created at: {filepath}")

    def shutdown(self):
        print("Shutting down the application...")
        self.signalStop()

        if self.log_listener:
            self.log_listener.stop()
            print("Logging listener stopped.")

