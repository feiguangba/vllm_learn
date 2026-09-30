#!/usr/bin/env python3
"""Notebook hygiene checks for CI.

Every lesson notebook must:
1. be valid JSON (nbformat 4);
2. use the `python3` kernelspec (display name "Python 3 (vllm_learn)");
3. carry no oversized outputs (a single output blob > 200 KB or a notebook
   file > 1 MB usually means someone committed run results);
4. contain no machine-specific hardcoded paths (D:\\..., uv_envs, local
   vendor/vllm clones).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "exercises"
MAX_OUTPUT_BYTES = 200 * 1024
MAX_NOTEBOOK_BYTES = 1024 * 1024
BAD_SUBSTRINGS = ("D:\\", "D:/", "uv_envs", "vendor/vllm", "file:///D:/")

problems = []

for nb_path in sorted(ROOT.rglob("*.ipynb")):
    rel = nb_path.relative_to(ROOT.parent).as_posix()
    if nb_path.stat().st_size > MAX_NOTEBOOK_BYTES:
        problems.append(f"{rel}: file larger than 1 MB (committed outputs?)")
    try:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        problems.append(f"{rel}: invalid JSON ({exc})")
        continue
    ks = nb.get("metadata", {}).get("kernelspec", {})
    if ks.get("name") != "python3":
        problems.append(f"{rel}: kernelspec is {ks.get('name')!r}, expected 'python3'")
    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        for out in cell.get("outputs", []):
            if len(json.dumps(out, ensure_ascii=False)) > MAX_OUTPUT_BYTES:
                problems.append(f"{rel}: cell #{idx} has an output blob > 200 KB")
                break
        for line in cell.get("source", []):
            if any(bad in line for bad in BAD_SUBSTRINGS):
                problems.append(f"{rel}: cell #{idx} hardcodes a machine-specific path: {line.strip()[:80]}")
                break

if problems:
    print(f"{len(problems)} notebook hygiene problem(s):")
    for p in problems:
        print(" -", p)
    sys.exit(1)
print(f"all notebooks clean ({len(sorted(ROOT.rglob('*.ipynb')))} files)")
