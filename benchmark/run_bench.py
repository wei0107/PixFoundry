#!/usr/bin/env python3
"""PixFoundry micro-benchmarks: Single vs OpenMP backends.

Usage (inside venv, after `pip install -e .`):
  python benchmark/run_bench.py --outdir benchmark_output
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import importlib
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import numpy as np


@dataclass
class BenchCase:
    group: str
    name: str
    fn: Callable[..., Any]
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    kind: str  # rgb/gray/misc


def _now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _set_omp_env(threads: int | None):
    if threads is None:
        return
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["OMP_PROC_BIND"] = os.environ.get("OMP_PROC_BIND", "true")
    os.environ["OMP_PLACES"] = os.environ.get("OMP_PLACES", "cores")


def _import_pf():
    try:
        return importlib.import_module("pixfoundry")
    except Exception as e:
        raise SystemExit(
            "Failed to import pixfoundry. Run `pip install -e .` in this venv.\n"
            f"Import error: {e}"
        )


def _time_one(fn: Callable[..., Any], args: Tuple[Any, ...], kwargs: Dict[str, Any],
              warmup: int, repeat: int) -> Tuple[float, float, float]:
    for _ in range(warmup):
        fn(*args, **kwargs)

    times: List[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    med = statistics.median(times)
    mean = statistics.fmean(times)
    stdev = statistics.pstdev(times) if len(times) >= 2 else 0.0
    return med, mean, stdev


def _build_cases(pf, rng: np.random.Generator, sizes: List[Tuple[int, int]]):
    cases: List[BenchCase] = []

    def has(name: str) -> bool:
        return hasattr(pf, name)

    rgb_imgs = { (h,w): rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8) for (h,w) in sizes }
    gray_imgs = { (h,w): rng.integers(0, 256, size=(h, w), dtype=np.uint8) for (h,w) in sizes }

    # color
    if has("to_grayscale"):
        for (h,w), img in rgb_imgs.items():
            cases.append(BenchCase("color", f"to_grayscale_{h}x{w}", pf.to_grayscale, (img,), {}, "rgb"))
    if has("invert"):
        for (h,w), img in rgb_imgs.items():
            cases.append(BenchCase("color", f"invert_rgb_{h}x{w}", pf.invert, (img,), {}, "rgb"))
        for (h,w), img in gray_imgs.items():
            cases.append(BenchCase("color", f"invert_gray_{h}x{w}", pf.invert, (img,), {}, "gray"))
    if has("sepia"):
        for (h,w), img in rgb_imgs.items():
            cases.append(BenchCase("color", f"sepia_{h}x{w}", pf.sepia, (img,), {}, "rgb"))
    if has("adjust_brightness_contrast"):
        for (h,w), img in rgb_imgs.items():
            cases.append(BenchCase("color", f"brightness_contrast_{h}x{w}",
                                   pf.adjust_brightness_contrast, (img, 1.2, 15.0), {}, "rgb"))
    if has("gamma_correct"):
        for (h,w), img in rgb_imgs.items():
            cases.append(BenchCase("color", f"gamma_correct_{h}x{w}", pf.gamma_correct, (img, 1.8), {}, "rgb"))

    # filters
    if has("mean_filter"):
        for (h,w), img in rgb_imgs.items():
            cases.append(BenchCase("filters", f"mean_filter_k5_{h}x{w}", pf.mean_filter, (img, 5),
                                   {"border":"replicate", "border_value":0}, "rgb"))
    if has("gaussian_filter"):
        for (h,w), img in rgb_imgs.items():
            cases.append(BenchCase("filters", f"gaussian_sigma1.2_{h}x{w}", pf.gaussian_filter, (img, 1.2),
                                   {"border":"replicate", "border_value":0}, "rgb"))
    if has("median_filter"):
        for (h,w), img in rgb_imgs.items():
            cases.append(BenchCase("filters", f"median_filter_k5_{h}x{w}", pf.median_filter, (img, 5),
                                   {"border":"replicate", "border_value":0}, "rgb"))
    if has("bilateral_filter"):
        for (h,w), img in rgb_imgs.items():
            cases.append(BenchCase("filters", f"bilateral_k7_{h}x{w}", pf.bilateral_filter,
                                   (img, 7, 25.0, 7.0), {"border":"replicate", "border_value":0}, "rgb"))

    # effects
    if has("sharpen"):
        for (h,w), img in rgb_imgs.items():
            cases.append(BenchCase("effects", f"sharpen_{h}x{w}", pf.sharpen, (img,), {}, "rgb"))
    if has("emboss"):
        for (h,w), img in rgb_imgs.items():
            cases.append(BenchCase("effects", f"emboss_{h}x{w}", pf.emboss, (img,), {}, "rgb"))
    if has("cartoonize"):
        for (h,w), img in rgb_imgs.items():
            cases.append(BenchCase("effects", f"cartoonize_{h}x{w}", pf.cartoonize, (img,), {}, "rgb"))

    # geometry
    if has("resize"):
        for (h,w), img in rgb_imgs.items():
            cases.append(BenchCase("geometry", f"resize_{h}x{w}_to_{h*2}x{w*2}",
                                   pf.resize, (img,), {"height":h*2, "width":w*2}, "rgb"))
    if has("flip_horizontal"):
        for (h,w), img in rgb_imgs.items():
            cases.append(BenchCase("geometry", f"flip_horizontal_{h}x{w}", pf.flip_horizontal, (img,), {}, "rgb"))
    if has("flip_vertical"):
        for (h,w), img in rgb_imgs.items():
            cases.append(BenchCase("geometry", f"flip_vertical_{h}x{w}", pf.flip_vertical, (img,), {}, "rgb"))
    if has("crop"):
        for (h,w), img in rgb_imgs.items():
            y = h//4; x = w//4; ch = h//2; cw = w//2
            cases.append(BenchCase("geometry", f"crop_{h}x{w}_to_{ch}x{cw}",
                                   pf.crop, (img, y, x, ch, cw), {}, "rgb"))
    if has("rotate"):
        for (h,w), img in rgb_imgs.items():
            cases.append(BenchCase("geometry", f"rotate_15deg_{h}x{w}", pf.rotate, (img,), {"angle_deg":15.0}, "rgb"))

    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="benchmark_output")
    ap.add_argument("--sizes", default="256x256,512x512,1024x768")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--omp-threads", type=int, default=0,
                    help="set OMP_NUM_THREADS (0 means don't override)")
    args = ap.parse_args()

    sizes: List[Tuple[int,int]] = []
    for token in args.sizes.split(","):
        token = token.strip().lower()
        if not token:
            continue
        h, w = token.split("x", 1)
        sizes.append((int(h), int(w)))

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    omp_threads = None if args.omp_threads == 0 else args.omp_threads
    _set_omp_env(omp_threads)

    pf = _import_pf()
    backends = ["single", "openmp"]

    rng = np.random.default_rng(args.seed)
    cases = _build_cases(pf, rng, sizes)

    meta = {
        "timestamp": _now_iso(),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "omp_threads": os.environ.get("OMP_NUM_THREADS", "(default)"),
        "cases": len(cases),
        "warmup": args.warmup,
        "repeat": args.repeat,
        "sizes": sizes,
    }
    import json
    with open(os.path.join(outdir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    rows = []
    for case in cases:
        for backend in backends:
            kwargs = dict(case.kwargs)
            kwargs["backend"] = backend
            try:
                med, mean, stdev = _time_one(case.fn, case.args, kwargs, args.warmup, args.repeat)
                rows.append({
                    "group": case.group,
                    "case": case.name,
                    "backend": backend,
                    "median_s": med,
                    "mean_s": mean,
                    "stdev_s": stdev,
                })
            except Exception as e:
                rows.append({
                    "group": case.group,
                    "case": case.name,
                    "backend": backend,
                    "median_s": float("nan"),
                    "mean_s": float("nan"),
                    "stdev_s": float("nan"),
                    "error": repr(e),
                })

    csv_path = os.path.join(outdir, "results.csv")
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    by_case: Dict[Tuple[str,str], Dict[str, dict]] = {}
    for r in rows:
        by_case.setdefault((r["group"], r["case"]), {})[r["backend"]] = r

    report_lines = []
    report_lines.append("# PixFoundry Benchmark Report (Single vs OpenMP)\n")
    report_lines.append(f"- Generated: `{meta['timestamp']}`")
    report_lines.append(f"- Python: `{meta['python']}`")
    report_lines.append(f"- Platform: `{meta['platform']}`")
    report_lines.append(f"- OMP_NUM_THREADS: `{meta['omp_threads']}`")
    report_lines.append(f"- Warmup: `{meta['warmup']}`  Repeat: `{meta['repeat']}`")
    report_lines.append(f"- Sizes: `{', '.join([f'{h}x{w}' for h,w in sizes])}`\n")

    report_lines.append("## Summary (median time per call)\n")
    report_lines.append("| Group | Case | Single (ms) | OpenMP (ms) | Speedup | Notes |")
    report_lines.append("|---|---|---:|---:|---:|---|")

    for (group, name), rec in sorted(by_case.items()):
        s = rec.get("single")
        o = rec.get("openmp")
        if s is None or o is None:
            report_lines.append(f"| {group} | {name} | - | - | - | missing backend |")
            continue
        if "error" in s or "error" in o:
            note = (s.get("error") or o.get("error") or "")[:80]
            report_lines.append(f"| {group} | {name} | - | - | - | error: {note} |")
            continue
        s_ms = s["median_s"] * 1e3
        o_ms = o["median_s"] * 1e3
        spd = s["median_s"] / o["median_s"] if o["median_s"] else float("nan")
        report_lines.append(f"| {group} | {name} | {s_ms:.3f} | {o_ms:.3f} | {spd:.2f}× |  |")

    report_lines.append("\n## Raw data\n")
    report_lines.append("- `results.csv`")
    report_lines.append("- `meta.json`\n")

    md_path = os.path.join(outdir, "report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"[OK] wrote: {csv_path}")
    print(f"[OK] wrote: {md_path}")


if __name__ == "__main__":
    main()
