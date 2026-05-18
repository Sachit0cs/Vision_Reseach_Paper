"""Benchmark orchestrator (Phase 2 — STUB).

For each model, for each attack: build (or load) the appropriate poisoned set
per the protocol, run inference, compute all metrics, and write a structured
JSON result per (model, attack) to results/.
"""

from __future__ import annotations


class BenchmarkPipeline:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Phase 2: implement the benchmark pipeline.")

    def run(self):
        raise NotImplementedError("Phase 2: implement the benchmark pipeline.")
