#!/usr/bin/env python3
"""Frozen 40-prompt Kearuga fidelity capture and comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import urllib.request
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_payload(model: str, prompt: str, max_tokens: int, *, logprobs: bool) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 1234,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if logprobs:
        payload.update(logprobs=True, top_logprobs=20)
    return payload


def resolve_model(base_url: str) -> str:
    with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=30) as response:
        return json.loads(response.read().decode())["data"][0]["id"]


def chat(base_url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read().decode())


def score_records(reference: list[dict], candidate: list[dict]) -> dict:
    if len(reference) != len(candidate):
        raise ValueError("record counts differ")
    rows = []
    margin_buckets = (("lt_0_5", 0.5), ("0_5_to_2", 2.0), ("2_to_5", 5.0), ("ge_5", math.inf))
    floor = -20.0
    for ref, cand in zip(reference, candidate, strict=True):
        if ref["prompt"] != cand["prompt"]:
            raise ValueError("prompt order differs")
        ref_lp = {item["token"]: item["logprob"] for item in ref["top20"]}
        cand_lp = {item["token"]: item["logprob"] for item in cand["top20"]}
        tokens = set(ref_lp) | set(cand_lp)
        p = {token: math.exp(ref_lp.get(token, floor)) for token in tokens}
        q = {token: math.exp(cand_lp.get(token, floor)) for token in tokens}
        p_sum, q_sum = sum(p.values()), sum(q.values())
        p = {token: value / p_sum for token, value in p.items()}
        q = {token: value / q_sum for token, value in q.items()}
        midpoint = {token: (p[token] + q[token]) / 2 for token in tokens}
        kl = sum(p[token] * math.log(max(p[token], 1e-30) / max(q[token], 1e-30)) for token in tokens)
        js = 0.5 * sum(
            p[token] * math.log(max(p[token], 1e-30) / max(midpoint[token], 1e-30))
            for token in tokens
        ) + 0.5 * sum(
            q[token] * math.log(max(q[token], 1e-30) / max(midpoint[token], 1e-30))
            for token in tokens
        )
        dot = sum(p[token] * q[token] for token in tokens)
        cosine = dot / math.sqrt(
            sum(value * value for value in p.values())
            * sum(value * value for value in q.values())
        )
        ref_margin = float(ref["top20"][0]["logprob"] - ref["top20"][1]["logprob"])
        margin_bucket = next(label for label, upper in margin_buckets if ref_margin < upper)
        rows.append({
            "prompt": ref["prompt"],
            "kl_top20_normalized": kl,
            "js_top20_normalized": js,
            "probability_cosine": cosine,
            "top20_overlap": len(set(ref_lp) & set(cand_lp)) / 20,
            "top1_agreement": ref["top20"][0]["token"] == cand["top20"][0]["token"],
            "bf16_top1_margin": ref_margin,
            "bf16_margin_bucket": margin_bucket,
            "continuation_exact": ref["continuation"] == cand["continuation"],
        })
    total = len(rows)
    margin_summary = {}
    for label, _ in margin_buckets:
        bucket_rows = [row for row in rows if row["bf16_margin_bucket"] == label]
        margin_summary[label] = {
            "total": len(bucket_rows),
            "top1_disagreements": sum(not row["top1_agreement"] for row in bucket_rows),
        }
    return {
        "total": total,
        "mean_kl": sum(row["kl_top20_normalized"] for row in rows) / total,
        "mean_js": sum(row["js_top20_normalized"] for row in rows) / total,
        "mean_probability_cosine": sum(row["probability_cosine"] for row in rows) / total,
        "mean_top20_overlap": sum(row["top20_overlap"] for row in rows) / total,
        "top1_agreement": {
            "passed": sum(row["top1_agreement"] for row in rows),
            "total": total,
        },
        "continuation_exact": {
            "passed": sum(row["continuation_exact"] for row in rows),
            "total": total,
        },
        "bf16_margin_buckets": margin_summary,
        "note": "KL/JS are normalized over the union of captured top-20 tokens with floor logprob -20; they are approximations, not full-vocabulary divergence.",
        "rows": rows,
    }


def capture(args) -> None:
    prompts = json.loads(args.prompts.read_text())
    if len(prompts) != 40 or len(set(prompts)) != 40:
        raise RuntimeError("expected 40 unique prompts")
    model = args.model or resolve_model(args.base_url)
    records = []
    for index, prompt in enumerate(prompts, start=1):
        first = chat(args.base_url, build_payload(model, prompt, 1, logprobs=True))
        continuation = chat(args.base_url, build_payload(model, prompt, 32, logprobs=False))
        top = first["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        records.append({
            "prompt": prompt,
            "top20": [
                {"token": item["token"], "logprob": item["logprob"]}
                for item in top
            ],
            "continuation": continuation["choices"][0]["message"]["content"] or "",
        })
        print(f"{index:02d}/40", flush=True)
    output = {
        "model": model,
        "base_url": args.base_url,
        "prompts_sha256": sha256_file(args.prompts),
        "protocol": {
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 1234,
            "enable_thinking": False,
            "top_logprobs": 20,
            "continuation_tokens": 32,
        },
        "records": records,
    }
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "records": len(records)}, indent=2))


def score(args) -> None:
    reference = json.loads(args.reference.read_text())
    candidate = json.loads(args.candidate.read_text())
    result = score_records(reference["records"], candidate["records"])
    result.update({
        "reference_sha256": sha256_file(args.reference),
        "candidate_sha256": sha256_file(args.candidate),
    })
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(required=True)
    capture_parser = sub.add_parser("capture")
    capture_parser.add_argument("--prompts", type=Path, required=True)
    capture_parser.add_argument("--base-url", required=True)
    capture_parser.add_argument("--model", default=None)
    capture_parser.add_argument("--output", type=Path, required=True)
    capture_parser.set_defaults(func=capture)
    score_parser = sub.add_parser("score")
    score_parser.add_argument("--reference", type=Path, required=True)
    score_parser.add_argument("--candidate", type=Path, required=True)
    score_parser.add_argument("--output", type=Path, required=True)
    score_parser.set_defaults(func=score)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
