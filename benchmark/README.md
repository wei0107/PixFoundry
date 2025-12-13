# Benchmarks (Single vs OpenMP)

Run from repo root (after `pip install -e .`):

```bash
python benchmark/run_bench.py --outdir benchmark_output
# control sizes / repeats
python benchmark/run_bench.py --sizes 256x256,512x512 --warmup 2 --repeat 20 --omp-threads 8
```

Outputs:
- `benchmark_output/report.md`
- `benchmark_output/results.csv`
- `benchmark_output/meta.json`
