"""Bounded rpc-server / llama-server logs for the peer agent.

The agent starts its processes with GGML_SCHED_DEBUG=3 / GGML_RPC_DEBUG=3
(the Flow Visualizer reads those lines), which logs every graph compute.
Under steady traffic one rpc-server log grew to 1.18 GB in about 50
minutes, and logs_tail read the whole file for every request, so the
agent stopped answering in time and the farm showed the peer "offline".

- open_append(): the child's log handle is opened in OS-level append mode
  (O_APPEND on POSIX, FILE_APPEND_DATA on Windows). Every write lands at
  the current end of the file, so the file can be truncated while the
  process keeps writing.
- rotate(): copies the log to "<log>.1" and truncates it.
- start_rotation(): background thread rotating every half of
  PEER_AGENT_LOG_KEEP_SEC (default 3600 = 60 min, the longest window the
  Performance tab shows; 0 disables), so at most the last KEEP_SEC
  seconds stay on disk (current log + "<log>.1"). Only logs
  this agent opened in append mode are rotated (a marker file next to the
  log records that), since truncating a log a process writes at a fixed
  offset would leave a run of NUL bytes instead of freeing space.
- tail_lines(): reads only the end of the file.
"""
import os
import shutil
import threading
import time
from pathlib import Path

KEEP_SEC = int(os.environ.get("PEER_AGENT_LOG_KEEP_SEC", "3600") or 0)
MARKER_PREFIX = "."
MARKER_SUFFIX = ".append"

_started = False


def _marker(path: Path) -> Path:
    return path.with_name(f"{MARKER_PREFIX}{path.name}{MARKER_SUFFIX}")


def open_append(path, fresh=True):
    """Open `path` for the child process in OS-level append mode.

    fresh=True truncates it first (a new run starts a new log). Returns a
    binary file object; pass it as Popen's stdout. A marker file records
    that the log is safe to rotate."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        f = _open_append_windows(str(path), fresh)
    else:
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        if fresh:
            flags |= os.O_TRUNC
        f = os.fdopen(os.open(str(path), flags, 0o644), "ab", buffering=0)
    try:
        _marker(path).write_text("append\n", encoding="utf-8")
    except OSError:
        pass
    return f


def _open_append_windows(path, fresh):
    import ctypes
    import msvcrt
    from ctypes import wintypes
    FILE_APPEND_DATA = 0x0004
    SYNCHRONIZE = 0x00100000
    FILE_SHARE_ALL = 0x1 | 0x2 | 0x4          # read | write | delete
    CREATE_ALWAYS, OPEN_ALWAYS = 2, 4
    FILE_ATTRIBUTE_NORMAL = 0x80
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.restype = wintypes.HANDLE
    k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD,
                                wintypes.HANDLE]
    h = k32.CreateFileW(path, FILE_APPEND_DATA | SYNCHRONIZE, FILE_SHARE_ALL,
                        None, CREATE_ALWAYS if fresh else OPEN_ALWAYS,
                        FILE_ATTRIBUTE_NORMAL, None)
    if h in (None, wintypes.HANDLE(-1).value):
        raise ctypes.WinError(ctypes.get_last_error())
    fd = msvcrt.open_osfhandle(h, os.O_WRONLY | os.O_APPEND)
    return os.fdopen(fd, "ab", buffering=0)


def is_rotatable(path) -> bool:
    return _marker(Path(path)).exists()


def rotate(path) -> bool:
    """Keep the current log as "<log>.1" and empty the log. Returns
    whether it rotated (only append-mode logs, see module doc)."""
    path = Path(path)
    if not is_rotatable(path) or not path.exists():
        return False
    try:
        if path.stat().st_size == 0:
            return False
        shutil.copyfile(path, path.with_name(path.name + ".1"))
        with open(path, "r+b") as f:
            f.truncate(0)
        return True
    except OSError:
        return False


def tail_lines(path, n, block=64 * 1024):
    """Last `n` lines of a text file, reading only the end of it."""
    path = Path(path)
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        end = f.tell()
        data = b""
        pos = end
        while pos > 0 and data.count(b"\n") <= n:
            step = min(block, pos)
            pos -= step
            f.seek(pos)
            data = f.read(step) + data
    lines = data.decode("utf-8", errors="replace").splitlines()
    return lines[-n:] if n else []


def rotate_dir(log_dir, pattern_re):
    """Rotate every append-mode log in `log_dir` whose name matches."""
    rotated = []
    try:
        entries = list(Path(log_dir).iterdir())
    except OSError:
        return rotated
    for p in entries:
        if p.is_file() and pattern_re.match(p.name) and rotate(p):
            rotated.append(p.name)
    return rotated


def start_rotation(log_dir, pattern_re, keep_sec=None):
    """Keep at most keep_sec seconds of each append-mode log: rotate every
    keep_sec / 2 (current + "<log>.1" = keep_sec). 0 disables."""
    global _started
    keep = KEEP_SEC if keep_sec is None else int(keep_sec)
    if _started or keep <= 0:
        return False
    _started = True
    every = max(60, keep // 2)

    def loop():
        while True:
            time.sleep(every)
            rotate_dir(log_dir, pattern_re)
    threading.Thread(target=loop, daemon=True, name="log-rotation").start()
    return True
