#!/usr/bin/env python3
"""Compose a `docker run` argv for a throwaway copy of a running SGLang container WITHOUT speculative decoding.

Everything is carried over from `docker inspect <container>` — image, env, host config (network / ipc / gpus / shm /
ulimits / cpuset / cap-add / security-opt), bind mounts and the launch_server arguments — except: every `--speculative-*`
flag (and its value) is dropped, `--port` is replaced, the container gets a new name and no restart policy. This is how
the no-speculation baseline in README §1 was measured: the same target, same flags, same image, speculation off.

Usage:
    docker inspect qwen3.8-27b-sglang > inspect.json
    python bench/compose_nospec.py --inspect inspect.json --port 8891 --name kearuga-nospec --out argv.json
    # then run the argv in argv.json (a JSON list) with `docker run`, e.g.
    # python -c "import json,subprocess; subprocess.run(json.load(open('argv.json')), check=True)"
"""
from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path


def strip_speculative(cmd: list[str]) -> list[str]:
    """Drop every `--speculative-*` flag with its value (pure). A value is the next token when it does not
    start with `--`; boolean-style speculative flags (no value) are dropped alone."""
    out, i = [], 0
    while i < len(cmd):
        tok = cmd[i]
        if tok.startswith("--speculative-"):
            i += 2 if (i + 1 < len(cmd) and not cmd[i + 1].startswith("--")) else 1
            continue
        out.append(tok)
        i += 1
    return out


def set_flag(cmd: list[str], flag: str, value: str) -> list[str]:
    out = list(cmd)
    if flag in out:
        out[out.index(flag) + 1] = value
    else:
        out += [flag, value]
    return out


def compose_nospec_argv(inspect_doc, name: str, port: int) -> list[str]:
    """Pure: inspect JSON (list with one container, or the container dict) -> docker run argv."""
    insp = inspect_doc[0] if isinstance(inspect_doc, list) else inspect_doc
    cfg, hc = insp["Config"], insp["HostConfig"]
    cmd = set_flag(strip_speculative(list(cfg["Cmd"])), "--port", str(port))
    if any(t.startswith("--speculative-") for t in cmd):
        raise ValueError("speculative flag survived stripping")
    argv = ["docker", "run", "-d", "--name", name]
    if hc.get("NetworkMode") == "host":
        argv += ["--network", "host"]
    if hc.get("IpcMode") == "host":
        argv += ["--ipc", "host"]
    if hc.get("DeviceRequests"):
        argv += ["--gpus", "all"]
    if hc.get("ShmSize"):
        argv += ["--shm-size", str(hc["ShmSize"])]
    for u in hc.get("Ulimits") or []:
        argv += ["--ulimit", f"{u['Name']}={u['Soft']}:{u['Hard']}"]
    for cap in hc.get("CapAdd") or []:
        argv += ["--cap-add", cap]
    if hc.get("CpusetCpus"):
        argv += ["--cpuset-cpus", hc["CpusetCpus"]]
    for s in hc.get("SecurityOpt") or []:
        argv += ["--security-opt", s]
    for b in hc.get("Binds") or []:
        argv += ["-v", b]
    for e in cfg.get("Env") or []:      # same composition as start-dflash2.sh
        argv += ["-e", e]
    argv.append(cfg["Image"])
    argv += cmd
    return argv


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inspect", required=True, help="path to `docker inspect <resident>` JSON")
    p.add_argument("--name", default="kearuga-nospec")
    p.add_argument("--port", type=int, default=8891)
    p.add_argument("--out", required=True, help="argv written as a JSON list (consumed by the window script)")
    a = p.parse_args(argv)
    run_argv = compose_nospec_argv(json.loads(Path(a.inspect).read_text(encoding="utf-8")), a.name, a.port)
    Path(a.out).write_text(json.dumps(run_argv, indent=1) + "\n", encoding="utf-8")
    print(" ".join(shlex.quote(x) for x in run_argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
