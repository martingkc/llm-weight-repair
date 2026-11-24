"""A persistent, sandboxed Python kernel for the programmer.

The kernel is a separate interpreter (.venv-kernel: numpy, scipy, safetensors,
tokenizers; no torch) started under macOS sandbox-exec. It cannot use the network,
cannot read the user's home directory except the exported weights, its own work
folder and its interpreter, and cannot write outside its work folder. So it cannot
see the question splits, the reference answers, the results, other runs or the
API key.

Variables survive between code runs. A run that exceeds the time limit kills the
kernel; the next run starts a fresh one and the caller is told.
"""
from __future__ import annotations

import json
import os
import select
import subprocess
import time
from pathlib import Path

from steer.model import ROOT

KERNEL_PYTHON = ROOT / ".venv-kernel" / "bin" / "python"
WEIGHTS = ROOT / "data" / "kernel_weights"
MAX_OUTPUT = 10_000

PRELUDE = r'''
import json, os, math
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=160)
WORK = os.getcwd()
_W = os.environ["KERNEL_WEIGHTS"]
META = json.load(open(os.path.join(_W, "meta.json")))
CONFIG = META["config"]
_ALIASES = META.get("aliases", {})
NAMES = sorted(list(META["base"]) + list(_ALIASES))   # tensor names you can load
ADAPTED = sorted(META["adapter"])            # module names of a separate adapter, if one is provided

def load(name):
    """A weight tensor as a float32 array."""
    return np.load(os.path.join(_W, "base", _ALIASES.get(name, name) + ".npy"))

def lora(module):
    """(A, B, scaling) of the adapter on `module`. Misaligned weight = load(module + ".weight") + scaling * B @ A."""
    if not META["adapter"]:
        raise RuntimeError("there is no separate adapter in this task: you only have the one merged model")
    a = np.load(os.path.join(_W, "adapter", module + ".A.npy"))
    b = np.load(os.path.join(_W, "adapter", module + ".B.npy"))
    return a, b, META["adapter"][module]["scaling"]

def delta(module):
    """The adapter's dense update of `module`: scaling * B @ A."""
    a, b, s = lora(module)
    return s * (b @ a)

try:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(_W, "tokenizer.json"))
except Exception as _e:
    tok = None

os.makedirs(os.path.join(WORK, "vectors"), exist_ok=True)

def save_vector(name, array):
    """Save a candidate steering vector (shape (hidden_size,), finite) under a name."""
    a = np.asarray(array, dtype=np.float32)
    if a.shape != (CONFIG["hidden_size"],):
        raise ValueError(f"a steering vector must have shape ({CONFIG['hidden_size']},), got {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError("the vector has non-finite values")
    if not str(name).replace("_", "").replace("-", "").isalnum():
        raise ValueError("use letters, digits, - and _ in vector names")
    np.save(os.path.join(WORK, "vectors", f"{name}.npy"), a)
    return f"saved {name}: norm {float(np.linalg.norm(a)):.4f}"

def vectors():
    """Names of the saved vectors."""
    return sorted(f[:-4] for f in os.listdir(os.path.join(WORK, "vectors")) if f.endswith(".npy"))

def note(text):
    """Append to your notebook notes.md. It is handed back to you if the conversation restarts."""
    with open(os.path.join(WORK, "notes.md"), "a") as f:
        f.write(str(text).rstrip() + "\n\n")
    return "noted"
'''

SERVER = r'''
import ast, contextlib, io, json, sys, traceback
_ns = {"__name__": "__main__"}
exec(compile(open(sys.argv[1]).read(), "<prelude>", "exec"), _ns)
sys.__stdout__.write(json.dumps({"ready": True}) + "\n"); sys.__stdout__.flush()
for _line in sys.stdin:
    _req = json.loads(_line)
    _buf = io.StringIO()
    _err = None
    try:
        _tree = ast.parse(_req["code"], mode="exec")
        _last = None
        if _tree.body and isinstance(_tree.body[-1], ast.Expr):
            _last = ast.Expression(_tree.body.pop().value)
        with contextlib.redirect_stdout(_buf), contextlib.redirect_stderr(_buf):
            exec(compile(_tree, "<cell>", "exec"), _ns)
            if _last is not None:
                _val = eval(compile(_last, "<cell>", "eval"), _ns)
                if _val is not None:
                    print(repr(_val))
    except BaseException:
        _err = traceback.format_exc(limit=6)
    sys.__stdout__.write(json.dumps({"out": _buf.getvalue(), "err": _err}) + "\n"); sys.__stdout__.flush()
'''


def sandbox_profile(work: Path, weights: Path, python: Path) -> str:
    home = Path.home().resolve()
    venv = Path(os.path.abspath(python)).parent.parent          # the kernel's virtual environment
    interpreter = Path(os.path.realpath(python)).parent.parent  # the real interpreter behind its symlink
    allow_read = [work.resolve(), weights.resolve(), venv, interpreter]
    reads = " ".join(f'(subpath "{p}")' for p in allow_read)
    return f'''(version 1)
(allow default)
(deny network*)
(deny file-read* (subpath "{home}"))
(allow file-read-metadata (subpath "{home}"))
(allow file-read* {reads})
(deny file-write* (subpath "/"))
(allow file-write* (subpath "{work.resolve()}") (literal "/dev/null") (literal "/dev/dtracehelper"))
'''


class Kernel:
    def __init__(self, work: Path, timeout_s: int = 120, weights: Path = WEIGHTS,
                 python: Path = KERNEL_PYTHON, sandbox: bool = True):
        self.work = Path(work)
        self.work.mkdir(parents=True, exist_ok=True)
        self.timeout_s, self.weights, self.python, self.sandbox = timeout_s, Path(weights), Path(python), sandbox
        self.proc: subprocess.Popen | None = None
        self.restarts = 0
        self._lost = False  # a kernel was killed, so the next start is a restart
        (self.work / "_prelude.py").write_text(PRELUDE)
        (self.work / "_server.py").write_text(SERVER)

    def _start(self) -> None:
        cmd = [str(self.python), "-u", str(self.work / "_server.py"), str(self.work / "_prelude.py")]
        if self.sandbox:
            cmd = ["sandbox-exec", "-p", sandbox_profile(self.work, self.weights, self.python)] + cmd
        env = {"KERNEL_WEIGHTS": str(self.weights.resolve()), "PATH": "/usr/bin:/bin", "HOME": str(self.work.resolve()),
               "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "4"}
        self.proc = subprocess.Popen(cmd, cwd=str(self.work.resolve()), env=env, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        ready = self._read_line(60)
        if not ready or not json.loads(ready).get("ready"):
            err = self.proc.stderr.read().decode(errors="replace") if self.proc.poll() is not None else ""
            self.stop()
            raise RuntimeError(f"kernel failed to start: {err[-2000:]}")

    def _read_line(self, timeout: float) -> bytes | None:
        assert self.proc and self.proc.stdout
        fd = self.proc.stdout.fileno()
        buf = bytearray()
        deadline = time.time() + timeout
        while True:
            left = deadline - time.time()
            if left <= 0:
                return None
            r, _, _ = select.select([fd], [], [], left)
            if not r:
                return None
            chunk = os.read(fd, 65536)
            if not chunk:
                return None
            buf += chunk
            if buf.endswith(b"\n"):
                return bytes(buf)

    def run(self, code: str) -> dict:
        """Returns {'output': str, 'error': str|None, 'restarted': bool, 'timed_out': bool}."""
        restarted = False
        if self.proc is None or self.proc.poll() is not None:
            restarted = self._lost
            self._lost = False
            self._start()
            if restarted:
                self.restarts += 1
        assert self.proc and self.proc.stdin
        self.proc.stdin.write((json.dumps({"code": code}) + "\n").encode())
        self.proc.stdin.flush()
        line = self._read_line(self.timeout_s)
        if line is None:
            died = self.proc.poll() is not None
            self.stop()
            self._lost = True
            msg = ("the kernel died (memory?)" if died else f"time limit of {self.timeout_s} s exceeded")
            return {"output": "", "error": msg + "; the kernel was restarted and its variables are lost",
                    "restarted": restarted, "timed_out": not died}
        res = json.loads(line)
        out = res["out"]
        if len(out) > MAX_OUTPUT:
            out = out[:MAX_OUTPUT] + f"\n[output cut at {MAX_OUTPUT} characters]"
        return {"output": out, "error": res["err"], "restarted": restarted, "timed_out": False}

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass
