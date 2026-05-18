"""CLI: build the benchmark datasets.

Currently implemented:
  --clean   Build the original (unmodified) ImageNet-val benchmark subset.

Not yet implemented (Phase 1):
  --shared-poisoned   patch / typographic / common-corruption sets.

The clean set is downloaded from the public, non-gated mirror
``evanarlian/imagenet_1k_resized_256`` (images pre-resized to 256px, the
standard ImageNet pre-crop size). Only the validation parquet shards are
fetched (~870 MB). One class-balanced image per ImageNet class is then sampled
with a fixed seed and written to ``data/clean/``.

Usage:
    python scripts/generate_datasets.py --clean
    python scripts/generate_datasets.py --clean --size 1000 --per-class 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

import pyarrow.parquet as pq
import yaml
from tqdm import tqdm

# Make the repo root importable so `datasets.sampler` resolves.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from datasets.sampler import ReservoirPerClass  # noqa: E402

_DATASETS_SERVER = "https://datasets-server.huggingface.co"


def load_config() -> dict:
    with open(os.path.join(_REPO_ROOT, "config.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _abs(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(_REPO_ROOT, path)


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def fetch_class_names(hf_repo: str) -> list[str]:
    """Fetch the 1000 index-aligned ImageNet class names from the dataset info."""
    info = _get_json(f"{_DATASETS_SERVER}/info?dataset={hf_repo}")["dataset_info"]
    config = info.get("default") or next(iter(info.values()))
    names = config["features"]["label"]["names"]
    if len(names) != 1000:
        print(f"  warning: expected 1000 class names, got {len(names)}")
    return names


def fetch_parquet_urls(hf_repo: str, split: str) -> list[dict]:
    """List the parquet shard URLs for a given split."""
    data = _get_json(f"{_DATASETS_SERVER}/parquet?dataset={hf_repo}")
    shards = [f for f in data["parquet_files"] if f["split"] == split]
    if not shards:
        raise RuntimeError(f"No parquet shards found for split '{split}' of {hf_repo}")
    return sorted(shards, key=lambda f: f["filename"])


def download_shard(url: str, dest: str, expected_size: int | None) -> None:
    """Download one parquet shard, skipping if already present and complete."""
    if os.path.exists(dest) and expected_size and os.path.getsize(dest) == expected_size:
        print(f"  cached: {os.path.basename(dest)}")
        return
    print(f"  downloading: {os.path.basename(dest)}")
    with urllib.request.urlopen(url, timeout=120) as r:
        total = int(r.headers.get("Content-Length", expected_size or 0))
        tmp = dest + ".part"
        with open(tmp, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, unit_divisor=1024, leave=False
        ) as bar:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                bar.update(len(chunk))
    os.replace(tmp, dest)


def build_clean_set(cfg: dict, size: int, per_class: int) -> None:
    ds_cfg = cfg["dataset"]
    hf_repo = ds_cfg["hf_repo"]
    split = ds_cfg["hf_split"]
    num_classes = ds_cfg["num_classes"]
    seed = cfg["seed"]

    cache_dir = _abs(cfg["paths"]["hf_cache"])
    clean_dir = _abs(cfg["paths"]["clean_set"])
    images_dir = os.path.join(clean_dir, "images")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    print(f"Source : {hf_repo} [{split}]")
    print(f"Target : {size} images, {per_class} per class, seed={seed}\n")

    # 1) Class names (index-aligned to labels) -> used by CLIP and the manifest.
    print("Fetching class names...")
    class_names = fetch_class_names(hf_repo)
    with open(os.path.join(clean_dir, "imagenet_classes.json"), "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2, ensure_ascii=False)

    # 2) Download the validation parquet shards.
    print("Resolving parquet shards...")
    shards = fetch_parquet_urls(hf_repo, split)
    local_shards = []
    for shard in shards:
        dest = os.path.join(cache_dir, f"{split}_{shard['filename']}")
        download_shard(shard["url"], dest, shard.get("size"))
        local_shards.append(dest)

    # 3) Single streaming pass: seeded class-balanced reservoir sampling.
    print("\nSampling class-balanced images...")
    reservoir = ReservoirPerClass(per_class=per_class, seed=seed)
    rows_seen = 0
    for shard_path in local_shards:
        pf = pq.ParquetFile(shard_path)
        for batch in pf.iter_batches(batch_size=512, columns=["image", "label"]):
            labels = batch.column("label").to_pylist()
            images = batch.column("image").to_pylist()
            for label, img in zip(labels, images):
                reservoir.offer(int(label), img["bytes"])
                rows_seen += 1
    print(f"  scanned {rows_seen} images across {len(local_shards)} shard(s)")

    selected = reservoir.result()

    # 4) Write images + manifest, ordered by (label, occurrence).
    print("\nWriting clean set...")
    records = []
    global_id = 0
    for label in sorted(selected):
        for img_bytes in selected[label]:
            filename = f"images/{label:04d}_{global_id:05d}.jpg"
            with open(os.path.join(clean_dir, filename), "wb") as f:
                f.write(img_bytes)
            records.append(
                {
                    "id": global_id,
                    "filename": filename,
                    "label": label,
                    "class_name": class_names[label],
                }
            )
            global_id += 1

    manifest = {
        "source": {"hf_repo": hf_repo, "split": split},
        "seed": seed,
        "per_class": per_class,
        "num_classes_covered": len(selected),
        "num_images": len(records),
        "images": records,
    }
    with open(os.path.join(clean_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # 5) Verify.
    print("\nVerification:")
    print(f"  images written     : {len(records)}")
    print(f"  classes covered    : {len(selected)} / {num_classes}")
    missing = num_classes - len(selected)
    if missing:
        print(f"  WARNING: {missing} class(es) had no image in the split")
    counts = {len(v) for v in selected.values()}
    print(f"  images per class   : {sorted(counts)}")
    print(f"\nClean set ready at: {clean_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build benchmark datasets.")
    parser.add_argument("--clean", action="store_true", help="Build the clean set.")
    parser.add_argument(
        "--shared-poisoned",
        action="store_true",
        help="Build patch/typographic/corruption sets (Phase 1 — not implemented).",
    )
    parser.add_argument("--size", type=int, default=None, help="Total clean images.")
    parser.add_argument("--per-class", type=int, default=None, help="Images per class.")
    args = parser.parse_args()

    cfg = load_config()
    size = args.size or cfg["dataset"]["benchmark_size"]
    per_class = args.per_class or cfg["dataset"]["per_class"]

    if not args.clean and not args.shared_poisoned:
        parser.error("Nothing to do. Pass --clean (or --shared-poisoned).")

    if args.clean:
        build_clean_set(cfg, size=size, per_class=per_class)

    if args.shared_poisoned:
        raise NotImplementedError(
            "Shared poisoned set (patch / typographic / corruptions) is Phase 1."
        )


if __name__ == "__main__":
    main()
