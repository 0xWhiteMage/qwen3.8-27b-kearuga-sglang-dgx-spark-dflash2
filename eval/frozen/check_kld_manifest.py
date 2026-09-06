#!/usr/bin/env python3
"""Verify that the frozen KLD evaluation files are byte-identical to kld-manifest.json (exit 1 on any mismatch)."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "kld-manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(manifest_path: Path = DEFAULT_MANIFEST, root: Path | None = None) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = root or manifest_path.parent
    rows, ok = [], True
    for name, meta in manifest["files"].items():
        path = root / name
        row = {"file": name, "expected_sha256": meta["sha256"], "expected_size": meta["size"], "source": meta.get("source")}
        if not path.is_file():
            row["status"] = "missing"
        else:
            row.update(size=path.stat().st_size, sha256=sha256_file(path))
            row["status"] = "ok" if (row["size"], row["sha256"]) == (meta["size"], meta["sha256"]) else "mismatch"
        ok = ok and row["status"] == "ok"
        rows.append(row)
    listed = set(manifest["files"]) | {manifest_path.name, Path(__file__).name}
    unlisted = sorted(p.name for p in root.iterdir() if p.is_file() and p.name not in listed)
    return {"ok": ok, "manifest": str(manifest_path), "root": str(root), "rows": rows, "unlisted": unlisted}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--root", type=Path, default=None, help="directory holding the files (default: manifest dir)")
    parser.add_argument("--strict", action="store_true", help="also fail when the directory holds unlisted files")
    args = parser.parse_args(argv)
    result = check(args.manifest, args.root)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] and not (args.strict and result["unlisted"]) else 1


if __name__ == "__main__":
    sys.exit(main())
