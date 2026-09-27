"""JSON-lines event logging; no images recorded unless explicitly requested."""
import json
import time
from pathlib import Path

class EventLog:
    def __init__(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open('w', encoding='utf-8')

    def write(self, kind: str, **data):
        self.file.write(json.dumps({'kind':kind, 'logged_at':time.monotonic(), **data}, allow_nan=False)+'\n')
        self.file.flush()

    def close(self):
        self.file.close()
