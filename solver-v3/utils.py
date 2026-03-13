"""Run-level logging utilities."""

import sys


class _Tee:
    """Write stdout to multiple streams simultaneously."""
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()


def log_step(number, title):
    print(f"\n{'='*60}")
    print(f"  STEP {number}: {title}")
    print(f"{'='*60}")

def log(msg):
    print(f"  {msg}")

def log_ok(msg):
    print(f"  [OK] {msg}")

def log_err(msg):
    print(f"  [ERROR] {msg}")

def log_out(path):
    print(f"  [OUT] {path}")
