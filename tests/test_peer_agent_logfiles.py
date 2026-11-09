"""Peer agent log bounding: append-mode logs, rotation, tail from the end.

Under steady traffic an rpc-server log (GGML_RPC_DEBUG=3) reached 1.18 GB
in ~50 minutes and logs_tail read the whole file per request, so the
agent stopped answering in time.
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "peer_agent"))

from llama_peer_agent import logfiles, processes  # noqa: E402
from llama_peer_agent.config import AgentConfig   # noqa: E402

LOG_RE = re.compile(r"^rpc-server(?:-\d+)?\.log$")


def test_tail_lines_reads_only_the_end(tmp_path):
    p = tmp_path / "rpc-server-51000.log"
    p.write_bytes(b"".join(b"line %d\n" % i for i in range(50000)))
    assert logfiles.tail_lines(p, 3) == ["line 49997", "line 49998", "line 49999"]
    assert logfiles.tail_lines(p, 3, block=7) == ["line 49997", "line 49998", "line 49999"]
    q = tmp_path / "short.log"
    q.write_bytes(b"a\nb")                       # no trailing newline, fewer lines
    assert logfiles.tail_lines(q, 10) == ["a", "b"]


def test_append_mode_survives_truncation(tmp_path):
    """The point of open_append: after another handle truncates the file,
    the writer continues at the new end (0), not at its old offset."""
    p = tmp_path / "rpc-server-51001.log"
    f = logfiles.open_append(p)
    f.write(b"x" * 10000)
    with open(p, "r+b") as g:
        g.truncate(0)
    f.write(b"after\n")
    f.close()
    data = p.read_bytes()
    assert data == b"after\n"                    # no 10000 NUL bytes in front
    assert logfiles.is_rotatable(p)


def test_rotate_keeps_previous_and_empties_current(tmp_path):
    p = tmp_path / "rpc-server-51002.log"
    f = logfiles.open_append(p)
    f.write(b"old\n")
    assert logfiles.rotate(p)
    f.write(b"new\n")
    f.close()
    assert (tmp_path / "rpc-server-51002.log.1").read_bytes() == b"old\n"
    assert p.read_bytes() == b"new\n"


def test_rotate_skips_logs_not_opened_in_append_mode(tmp_path):
    p = tmp_path / "rpc-server-51003.log"
    p.write_bytes(b"written by an older agent at a fixed offset\n")
    assert not logfiles.rotate(p)
    assert p.read_bytes().startswith(b"written")
    assert not (tmp_path / "rpc-server-51003.log.1").exists()


def test_rotate_dir_and_disabled_rotation(tmp_path, monkeypatch):
    a = tmp_path / "rpc-server-51004.log"
    f = logfiles.open_append(a)
    f.write(b"data\n")
    f.close()
    (tmp_path / "rpc-server-51005.log").write_bytes(b"legacy\n")
    (tmp_path / "notes.txt").write_bytes(b"x")
    assert logfiles.rotate_dir(tmp_path, LOG_RE) == ["rpc-server-51004.log"]
    monkeypatch.setattr(logfiles, "_started", False)
    assert logfiles.start_rotation(tmp_path, LOG_RE, keep_sec=0) is False


def test_agent_logs_tail_uses_the_end(tmp_path):
    cfg = AgentConfig(token="t", log_dir=str(tmp_path))
    p = tmp_path / "rpc-server-51006.log"
    p.write_bytes(b"".join(b"[graph_compute] %d\n" % i for i in range(20000)))
    out = processes.logs_tail(cfg, "rpc-server-51006.log", 2)
    assert out["lines"] == ["[graph_compute] 19998", "[graph_compute] 19999"]
    assert out["text"] == "[graph_compute] 19998\n[graph_compute] 19999"
