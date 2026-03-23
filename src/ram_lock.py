"""Cross-process RAM preload lock using file-based locking."""
import fcntl
import time
import os

LOCK_PATH = "/tmp/patchcore_ram_preload.lock"

class RAMPreloadLock:
    """Ensures only one process does RAM preload at a time."""
    def __init__(self):
        self._fd = None

    def __enter__(self):
        self._fd = open(LOCK_PATH, "w")
        pid = os.getpid()
        waited = 0
        while True:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._fd.write(str(pid))
                self._fd.flush()
                if waited > 0:
                    print(f"  [Lock] Acquired after {waited}s wait (PID {pid})")
                return self
            except (IOError, OSError):
                if waited == 0:
                    print(f"  [Lock] Another process is preloading RAM, waiting... (PID {pid})")
                time.sleep(5)
                waited += 5

    def __exit__(self, *args):
        if self._fd:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            self._fd.close()
            self._fd = None
