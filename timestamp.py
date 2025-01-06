import time
from threading import Lock

class Timestamp:
    def __init__(self):
        self.timestamp = time.time()  # Initialize with the current time
        self.lock = Lock()  # Lock for synchronization

    def get_timestamp(self):
        with self.lock:
            return self.timestamp

    def update_timestamp(self):
        with self.lock:
            self.timestamp = time.time()

# Global instance for shared timestamp
shared_timestamp = Timestamp()
