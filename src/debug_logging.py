"""Verbose llama.cpp debug logging for the Flow Visualizer — off by default.

The Visualizer animates per-tensor device assignment from llama.cpp's own
debug output: GGML_SCHED_DEBUG / GGML_RPC_DEBUG on llama-server and every
rpc-server, plus llama-server's -v "streamed chunk" lines. That output is
one or more lines per graph compute: under steady embedding traffic one
peer's rpc-server log reached 1.18 GB in about 50 minutes, and the
host-side llama-server output overwrote llama-swap's log buffer within
minutes (a crash's last lines were already gone when looked at).

It is now a setting (settings.yaml `debug_logging`, Settings > Defaults),
off unless turned on. Measured with ggml-rpc-server: GGML_RPC_DEBUG=0
behaves exactly like the variable being unset (no per-command lines), so
the farm always sets the variables explicitly — "0" also overrides the
peer agent's own "3" default.
"""
import os

from runtime_paths import data_dir

SETTING = "debug_logging"


def enabled():
    """True when settings.yaml turns the Visualizer debug output on."""
    try:
        import yaml
        p = os.path.join(data_dir(), "config", "settings.yaml")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return bool(data.get(SETTING, False))
    except Exception:
        pass
    return False


def level():
    return "3" if enabled() else "0"


def env():
    """GGML debug variables for llama-server / rpc-server environments."""
    lv = level()
    return {"GGML_SCHED_DEBUG": lv, "GGML_RPC_DEBUG": lv}


def exports():
    """The same as a space-separated VAR=value prefix (trailing space)."""
    return "".join(f"{k}={v} " for k, v in env().items())
