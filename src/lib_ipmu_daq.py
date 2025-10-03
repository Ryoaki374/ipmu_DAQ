import queue
import threading

# Import AppConfig from daq_config.py
from lib_ipmu_daq_config import AppConfig


class DAQApp:
    """
    Orchestrator for the DAQ application.
    Initializes and holds shared resources like queues and stop events.
    """
    def __init__(self, config: AppConfig):
        """Initializes the DAQApp with configuration."""
        self.cfg = config
        self.buf_q = None
        self.quad_q = None
        self.stop_event = None

    def setup(self):
        """Sets up shared resources for the application components."""
        self.buf_q = queue.Queue(maxsize=self.cfg.io.queue_depth)
        self.quad_q = queue.Queue(maxsize=self.cfg.io.quad_depth)
        self.stop_event = threading.Event()
        print("Shared resources (Queues, Stop Event) are set up.")
    
    def signalStop(self):
        """Signals all components to stop their loops."""
        if self.stop_event:
            print("Signaling all components to stop...")
            self.stop_event.set()