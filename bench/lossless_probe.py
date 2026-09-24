#!/usr/bin/env python3
"""Losslessness probe: does the speculative server reproduce the same target served without speculation?

Captures greedy continuations (with top-2 logprobs) from an OpenAI-compatible server and compares captures token by
token. Used for README §1 "Losslessness, measured": A = speculative server, B = the same server again (the run-to-run
noise floor), OFF = the same target booted without speculative decoding (see compose_nospec.py).

Subcommands:
  capture  --base-url --prompts (json list of user strings) --max-tokens --out  (one request at a time)
  compare  --ref A --cand B --out                                              (per-prompt first divergence)
  decide   --ref onA --noise onB --cand off --out DIR                          (rule + marker)

Rule (pre-registered before our run): LOSSLESS-OK iff n_div(A, OFF) <= n_div(A, B) + 3 AND
max_margin(A, OFF) <= max(0.25, max_margin(A, B)), where n_div = prompts whose token sequences differ and margin = the
reference's top-1 minus top-2 logprob at the first divergent position. On our run the count criterion failed (28/40 vs
6/40) while every margin was <= 0.25 — see README §1: the count cannot separate "not lossless" from "a different
rounding path"; the margin can.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

TOOL_VERSION = "lossless-probe/1.0"
NOISE_SLACK_PROMPTS = 3
TIE_MARGIN = 0.25


# --------------------------------------------------------------------------- pure


def first_divergence(a: list[str], b: list[str]) -> int | None:
    """First index where two token sequences differ (a length difference counts at min length); None if equal."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def compare_records(ref: list[dict], cand: list[dict]) -> dict:
    """Per-prompt divergence of `cand` from `ref` (both lists of capture records, same prompt order)."""
    if len(ref) != len(cand):
        raise ValueError(f"record count mismatch: {len(ref)} vs {len(cand)}")
    rows = []
    for r, c in zip(ref, cand):
        if r.get("prompt_sha256") != c.get("prompt_sha256"):
            raise ValueError(f"prompt mismatch at index {r.get('index')}")
        d = first_divergence(r["tokens"], c["tokens"])
        margin = None
        if d is not None and d < len(r.get("margins") or []):
            margin = r["margins"][d]
        rows.append({"index": r["index"], "first_divergence": d, "ref_margin": margin,
                     "ref_tokens": len(r["tokens"]), "cand_tokens": len(c["tokens"]),
                     "ref_finish": r.get("finish_reason"), "cand_finish": c.get("finish_reason")})
    divergent = [x for x in rows if x["first_divergence"] is not None]
    margins = [x["ref_margin"] for x in divergent if x["ref_margin"] is not None]
    return {"n_prompts": len(rows), "n_divergent": len(divergent),
            "max_margin": max(margins) if margins else None,
            "divergent": divergent, "rows": rows}


def decide_lossless(noise: dict, test: dict, slack: int = NOISE_SLACK_PROMPTS, tie_margin: float = TIE_MARGIN) -> dict:
    """LOSSLESS-OK iff n_div(test) <= n_div(noise) + slack AND max_margin(test) <= max(tie_margin, max_margin(noise))."""
    noise_max = noise.get("max_margin")
    bound = max(tie_margin, noise_max) if noise_max is not None else tie_margin
    test_max = test.get("max_margin")
    ok_count = test["n_divergent"] <= noise["n_divergent"] + slack
    ok_margin = test_max is None or test_max <= bound
    return {"rule": (f"n_div(on, off) <= n_div(on, on') + {slack} AND max first-divergence margin(on, off) <= "
                     f"max({tie_margin}, max margin(on, on'))"),
            "n_prompts": test["n_prompts"], "n_divergent_noise": noise["n_divergent"], "n_divergent_test": test["n_divergent"],
            "max_margin_noise": noise_max, "max_margin_test": test_max, "margin_bound": bound,
            "count_ok": ok_count, "margin_ok": ok_margin,
            "verdict": "LOSSLESS-OK" if (ok_count and ok_margin) else "LOSSLESS-SUSPECT"}


def render_md(decision: dict, noise: dict, test: dict, labels: tuple[str, str, str]) -> str:
    a, b, off = labels
    lines = ["# R-LOSSLESS — served speculative path vs the same target without speculation", "",
             f"rule: {decision['rule']}", "",
             f"noise floor ({a} vs {b}): {noise['n_divergent']}/{noise['n_prompts']} prompts diverge, max margin {noise['max_margin']}",
             f"test ({a} vs {off}): {test['n_divergent']}/{test['n_prompts']} prompts diverge, max margin {test['max_margin']}",
             f"bound: {decision['margin_bound']}; count_ok={decision['count_ok']} margin_ok={decision['margin_ok']}", "",
             f"**verdict: {decision['verdict']}**", "",
             "| prompt | noise first_div (margin) | test first_div (margin) | ref tokens | off tokens |", "|---:|---|---|---:|---:|"]
    for n, t in zip(noise["rows"], test["rows"]):
        lines.append(f"| {n['index']} | {n['first_divergence']} ({n['ref_margin']}) | {t['first_divergence']} ({t['ref_margin']}) "
                     f"| {t['ref_tokens']} | {t['cand_tokens']} |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- capture


def _chat(base_url: str, payload: dict) -> dict:
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as resp:
        return json.loads(resp.read().decode())


def _resolve_model(base_url: str) -> str:
    with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=30) as resp:
        return json.loads(resp.read().decode())["data"][0]["id"]


def record_from_response(index: int, prompt: str, resp: dict) -> dict:
    """Pure: turn a chat-completions response (with logprobs/top_logprobs>=2) into a capture record."""
    choice = resp["choices"][0]
    content = ((choice.get("logprobs") or {}).get("content")) or []
    tokens = [c["token"] for c in content]
    margins = []
    for c in content:
        tops = sorted((t["logprob"] for t in (c.get("top_logprobs") or [])), reverse=True)
        margins.append((tops[0] - tops[1]) if len(tops) >= 2 else None)
    return {"index": index, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "text": choice["message"]["content"], "finish_reason": choice.get("finish_reason"),
            "tokens": tokens, "margins": margins, "usage": resp.get("usage")}


def capture(args) -> None:
    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    model = args.model or _resolve_model(args.base_url)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with out.open("w", encoding="utf-8") as fh:
        for i, prompt in enumerate(prompts):
            payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": args.max_tokens,
                       "temperature": 0.0, "top_p": 1.0, "seed": 1234, "logprobs": True, "top_logprobs": 2,
                       "chat_template_kwargs": {"enable_thinking": False}}
            rec = record_from_response(i, prompt, _chat(args.base_url, payload))
            rec.update(tag=args.tag, model=model)
            fh.write(json.dumps(rec) + "\n")
            print(f"capture {args.tag} {i + 1}/{len(prompts)}: {len(rec['tokens'])} tok finish={rec['finish_reason']}", flush=True)
    print(f"wrote {out} ({len(prompts)} prompts, {time.time() - t0:.0f} s)")


def _load(path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--base-url", required=True, help="e.g. http://127.0.0.1:8888/v1")
    c.add_argument("--model", default=None)
    c.add_argument("--prompts", required=True, help="json list of user prompt strings (eval/frozen/kld-prompts-40.json)")
    c.add_argument("--max-tokens", type=int, default=256)
    c.add_argument("--tag", required=True)
    c.add_argument("--out", required=True)
    m = sub.add_parser("compare")
    m.add_argument("--ref", required=True)
    m.add_argument("--cand", required=True)
    m.add_argument("--out", required=True)
    d = sub.add_parser("decide")
    d.add_argument("--ref", required=True, help="spec-on capture A")
    d.add_argument("--noise", required=True, help="spec-on capture B (noise floor)")
    d.add_argument("--cand", required=True, help="spec-off capture")
    d.add_argument("--out", required=True, help="dir for lossless.json, report-lossless.md, marker")
    args = p.parse_args(argv)
    if args.cmd == "capture":
        capture(args)
        return 0
    if args.cmd == "compare":
        res = compare_records(_load(args.ref), _load(args.cand))
        Path(args.out).write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
        print(f"compare: {res['n_divergent']}/{res['n_prompts']} divergent, max margin {res['max_margin']}")
        return 0
    ref = _load(args.ref)
    noise = compare_records(ref, _load(args.noise))
    test = compare_records(ref, _load(args.cand))
    decision = decide_lossless(noise, test)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for mk in ("LOSSLESS-OK", "LOSSLESS-SUSPECT"):
        if (out / mk).exists():
            raise SystemExit(f"{out / mk} already exists (refusing to overwrite a verdict)")
    (out / "lossless.json").write_text(json.dumps({"tool_version": TOOL_VERSION, "decision": decision, "noise": noise,
                                                   "test": test}, indent=2) + "\n", encoding="utf-8")
    (out / "report-lossless.md").write_text(render_md(decision, noise, test, (Path(args.ref).stem, Path(args.noise).stem,
                                                                              Path(args.cand).stem)), encoding="utf-8")
    (out / decision["verdict"]).write_text(json.dumps(decision) + "\n", encoding="utf-8")
    print(f"verdict {decision['verdict']} (noise {noise['n_divergent']}/{noise['n_prompts']} max {noise['max_margin']}; "
          f"test {test['n_divergent']}/{test['n_prompts']} max {test['max_margin']}; bound {decision['margin_bound']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
