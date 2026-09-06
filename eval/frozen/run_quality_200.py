#!/usr/bin/env python3
"""Prospective, identity-bound Quality-200 v2 response runner."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


TOKEN_CAPS = {
    "humaneval": 2048,
    "agentic_coding": 2048,
    "hard_reasoning": 2048,
    "gsm8k": 1024,
    "ifeval": 1024,
}


def max_tokens_for_family(family: str) -> int:
    if family not in TOKEN_CAPS:
        raise ValueError(f"unsupported family: {family}")
    return TOKEN_CAPS[family]


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def request_sha256(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def cap_hit(finish_reason: str | None, usage: dict, max_tokens: int) -> bool:
    return finish_reason == "length" or int(usage.get("completion_tokens") or 0) >= max_tokens


def build_payload(model: str, prompt: str, family: str) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 1234,
        "max_tokens": max_tokens_for_family(family),
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compute_run_fingerprint(
    *,
    rows: list[dict],
    dataset_sha256: str,
    runner_sha256: str,
    arm_receipt_sha256: str,
    model: str,
    model_path: str,
) -> str:
    item_contracts = []
    for row in rows:
        payload = build_payload(model, row["prompt"], row["family"])
        item_contracts.append({
            "id": row["id"],
            "family": row["family"],
            "prompt_sha256": prompt_sha256(row["prompt"]),
            "request_sha256": request_sha256(payload),
            "max_tokens": payload["max_tokens"],
        })
    contract = {
        "schema": "kearuga-quality-v2-run-fingerprint-v1",
        "dataset_sha256": dataset_sha256,
        "runner_sha256": runner_sha256,
        "arm_receipt_sha256": arm_receipt_sha256,
        "model": model,
        "model_path": model_path,
        "items": item_contracts,
    }
    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def read_jsonl_bytes(data: bytes, label: str) -> list[dict]:
    rows = []
    for index, line in enumerate(data.decode("utf-8").split("\n"), start=1):
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid {label} JSONL line {index}: {exc}") from exc
    return rows


def get_json(url: str, timeout: int = 30) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode())


def get_http_status(url: str, timeout: int = 30) -> int:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return int(response.status)
    except urllib.error.HTTPError as error:
        return int(error.code)


def canonical_json_sha256(value: object) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def inspect_live_container(container_name: str) -> dict:
    completed = subprocess.run(
        ["docker", "inspect", container_name],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"docker inspect failed: {(completed.stderr or completed.stdout)[:500]}")
    rows = json.loads(completed.stdout)
    if len(rows) != 1:
        raise RuntimeError("docker inspect did not return exactly one container")
    row = rows[0]
    command = row.get("Config", {}).get("Cmd") or row.get("Args") or []
    model_path = None
    if "--model-path" in command:
        index = command.index("--model-path")
        if index + 1 < len(command):
            model_path = command[index + 1]
    return {
        "container_id": row.get("Id"),
        "status": row.get("State", {}).get("Status"),
        "image_id": row.get("Image"),
        "command_sha256": canonical_json_sha256(command),
        "model_path": model_path,
    }


def validate_arm_identity(
    receipt: dict,
    model_info: dict,
    served_model: str,
    live_container: dict,
    live_model_info_sha256: str,
) -> None:
    required = {
        "schema", "model_path", "served_model_name", "checkpoint_tree_sha256",
        "checkpoint_validation_sha256", "checkpoint_all_files_passed",
        "checkpoint_tree_hash_passed", "image_id", "container_id",
        "container_status", "command_sha256", "health_http_status",
        "model_info_sha256",
    }
    missing = required - set(receipt)
    if missing:
        raise ValueError(f"arm receipt missing fields: {sorted(missing)}")
    if receipt["schema"] != "kearuga-live-arm-receipt-v2":
        raise ValueError("arm receipt schema mismatch")
    if receipt["health_http_status"] != 200:
        raise ValueError("arm receipt health status is not 200")
    if receipt["container_status"] != "running" or live_container.get("status") != "running":
        raise ValueError("container is not running")
    if receipt["checkpoint_all_files_passed"] is not True or receipt["checkpoint_tree_hash_passed"] is not True:
        raise ValueError("checkpoint validation did not pass")
    if model_info.get("model_path") != receipt["model_path"] or live_container.get("model_path") != receipt["model_path"]:
        raise ValueError("model path mismatch across receipt, API, and container")
    if served_model != receipt["served_model_name"]:
        raise ValueError("served model mismatch")
    for field in ("checkpoint_tree_sha256", "checkpoint_validation_sha256", "command_sha256", "model_info_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(receipt[field])):
            raise ValueError(f"{field} is malformed")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(receipt["image_id"])):
        raise ValueError("image ID is malformed")
    if not re.fullmatch(r"[0-9a-f]{64}", str(receipt["container_id"])):
        raise ValueError("container ID is malformed")
    comparisons = {
        "container_id": live_container.get("container_id"),
        "image_id": live_container.get("image_id"),
        "command_sha256": live_container.get("command_sha256"),
    }
    for field, live_value in comparisons.items():
        if receipt[field] != live_value:
            raise ValueError(f"container {field} mismatch")
    if receipt["model_info_sha256"] != live_model_info_sha256:
        raise ValueError("live model-info hash mismatch")


def resolve_model(base_url: str) -> str:
    body = get_json(base_url.rstrip("/") + "/models")
    rows = body.get("data") or []
    if len(rows) != 1 or not rows[0].get("id"):
        raise RuntimeError("expected exactly one served model")
    return rows[0]["id"]


def chat(base_url: str, payload: dict, attempts: int = 5) -> tuple[dict, float]:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                return json.loads(response.read().decode()), time.perf_counter() - started
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code != 429 and error.code < 500:
                raise
        except (urllib.error.URLError, TimeoutError, socket.timeout) as error:
            last_error = error
        if attempt < attempts:
            time.sleep(min(2 ** (attempt - 1), 16))
    raise RuntimeError(f"request failed after {attempts} attempts: {last_error}")


def validate_existing(
    existing: list[dict],
    dataset: dict[str, dict],
    model: str,
    arm_sha: str,
    run_fingerprint: str,
) -> set[str]:
    seen = set()
    for row in existing:
        item_id = row.get("id")
        if item_id in seen or item_id not in dataset:
            raise ValueError(f"duplicate or unknown existing ID: {item_id}")
        seen.add(item_id)
        source = dataset[item_id]
        payload = build_payload(model, source["prompt"], source["family"])
        if row.get("prompt_sha256") != prompt_sha256(source["prompt"]):
            raise ValueError(f"existing prompt hash mismatch: {item_id}")
        if row.get("request_sha256") != request_sha256(payload):
            raise ValueError(f"existing request hash mismatch: {item_id}")
        if row.get("arm_receipt_sha256") != arm_sha:
            raise ValueError(f"existing arm receipt mismatch: {item_id}")
        if row.get("run_fingerprint") != run_fingerprint:
            raise ValueError(f"existing run fingerprint mismatch: {item_id}")
    return {row["id"] for row in existing if row.get("status") == "ok"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--expected-dataset-sha256", required=True)
    parser.add_argument("--arm-receipt", type=Path, required=True)
    parser.add_argument("--expected-run-fingerprint", required=True)
    parser.add_argument("--container-name", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8888/v1")
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dataset_bytes = args.dataset.read_bytes()
    dataset_sha = sha256_bytes(dataset_bytes)
    if dataset_sha != args.expected_dataset_sha256:
        raise ValueError(f"dataset hash mismatch: {dataset_sha}")
    dataset_rows = read_jsonl_bytes(dataset_bytes, "dataset")
    if len(dataset_rows) != 200 or len({row["id"] for row in dataset_rows}) != 200:
        raise ValueError("dataset must contain exactly 200 unique IDs")
    dataset = {row["id"]: row for row in dataset_rows}

    arm_bytes = args.arm_receipt.read_bytes()
    arm_sha = sha256_bytes(arm_bytes)
    if not re.fullmatch(r"[0-9a-f]{64}", arm_sha):
        raise ValueError("arm receipt hash is malformed")
    arm_receipt = json.loads(arm_bytes)
    model = args.model or resolve_model(args.base_url)
    api_root = args.base_url.rstrip("/").removesuffix("/v1")
    model_info = get_json(api_root + "/model_info")
    model_info_sha = canonical_json_sha256(model_info)
    live_container = inspect_live_container(args.container_name)
    live_health = get_http_status(api_root + "/health")
    if live_health != 200:
        raise ValueError(f"live health status is {live_health}, expected 200")
    validate_arm_identity(arm_receipt, model_info, model, live_container, model_info_sha)
    runner_sha = sha256_bytes(Path(__file__).read_bytes())
    run_fingerprint = compute_run_fingerprint(
        rows=dataset_rows,
        dataset_sha256=dataset_sha,
        runner_sha256=runner_sha,
        arm_receipt_sha256=arm_sha,
        model=model,
        model_path=arm_receipt["model_path"],
    )
    if run_fingerprint != args.expected_run_fingerprint:
        raise ValueError(f"run fingerprint mismatch: computed={run_fingerprint}")

    existing_bytes = args.output.read_bytes() if args.output.exists() else b""
    existing = read_jsonl_bytes(existing_bytes, "existing output") if existing_bytes else []
    completed = validate_existing(existing, dataset, model, arm_sha, run_fingerprint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as handle:
        for index, row in enumerate(dataset_rows, start=1):
            if row["id"] in completed:
                continue
            current_info = get_json(api_root + "/model_info")
            current_info_sha = canonical_json_sha256(current_info)
            current_container = inspect_live_container(args.container_name)
            current_health = get_http_status(api_root + "/health")
            if current_health != 200:
                raise RuntimeError(f"arm health drift before {row['id']}: {current_health}")
            validate_arm_identity(arm_receipt, current_info, model, current_container, current_info_sha)
            payload = build_payload(model, row["prompt"], row["family"])
            response, wall = chat(args.base_url, payload)
            choices = response.get("choices")
            if not isinstance(choices, list) or len(choices) != 1:
                raise RuntimeError(f"malformed completion envelope for {row['id']}: choices must be a single-element list")
            choice = choices[0]
            message = choice.get("message") or {}
            text = message.get("content")
            finish_reason = choice.get("finish_reason")
            usage = response.get("usage") or {}
            completion_tokens = usage.get("completion_tokens")
            if not isinstance(text, str) or not isinstance(finish_reason, str) or not isinstance(completion_tokens, int):
                raise RuntimeError(
                    f"malformed completion envelope for {row['id']}: content/finish_reason/usage.completion_tokens required"
                )
            record = {
                "id": row["id"],
                "family": row["family"],
                "grade": row["grade"],
                "status": "ok",
                "text": text,
                "finish_reason": finish_reason,
                "cap_hit": cap_hit(finish_reason, usage, payload["max_tokens"]),
                "prompt_sha256": prompt_sha256(row["prompt"]),
                "request_sha256": request_sha256(payload),
                "dataset_sha256": dataset_sha,
                "arm_receipt_sha256": arm_sha,
                "run_fingerprint": run_fingerprint,
                "model_path": arm_receipt["model_path"],
                "usage": usage,
                "wall_s": round(wall, 6),
                "protocol": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "seed": 1234,
                    "enable_thinking": False,
                    "max_tokens": payload["max_tokens"],
                },
            }
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            print(f"{index:03d}/200 {row['id']} {row['family']} tokens={usage.get('completion_tokens')} wall={wall:.2f}s", flush=True)

    final_info = get_json(api_root + "/model_info")
    final_container = inspect_live_container(args.container_name)
    final_health = get_http_status(api_root + "/health")
    if final_health != 200:
        raise RuntimeError(f"arm health drift at finalization: {final_health}")
    validate_arm_identity(
        arm_receipt,
        final_info,
        model,
        final_container,
        canonical_json_sha256(final_info),
    )
    final_bytes = args.output.read_bytes()
    final = read_jsonl_bytes(final_bytes, "final output")
    if len(final) != 200 or len({row["id"] for row in final}) != 200 or any(row.get("status") != "ok" for row in final):
        raise RuntimeError("incomplete final output")
    family_counts = dict(sorted(collections.Counter(row["family"] for row in final).items()))
    cap_counts = dict(sorted(collections.Counter(row["family"] for row in final if row.get("cap_hit")).items()))
    manifest = {
        "schema": "kearuga-quality-200-v2-run-v1",
        "model": model,
        "model_path": arm_receipt["model_path"],
        "base_url": args.base_url,
        "count": len(final),
        "families": family_counts,
        "cap_hits": cap_counts,
        "dataset_sha256": dataset_sha,
        "runner_sha256": runner_sha,
        "arm_receipt_sha256": arm_sha,
        "run_fingerprint": run_fingerprint,
        "container_id": arm_receipt["container_id"],
        "output_sha256": sha256_bytes(final_bytes),
        "protocol": {
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 1234,
            "enable_thinking": False,
            "max_tokens": TOKEN_CAPS,
        },
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "generation_complete": not cap_counts,
    }
    manifest_path = Path(str(args.output) + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if cap_counts:
        raise RuntimeError(f"cap hits invalidate promotion comparison: {cap_counts}")


if __name__ == "__main__":
    main()
