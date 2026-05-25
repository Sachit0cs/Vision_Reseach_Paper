"""CLI: build the benchmark datasets.

Implemented:
  --clean         Build the original (unmodified) ImageNet-val benchmark subset.
  --typographic   Build the typographic-overlay poisoned set from the clean set.

Not yet implemented (Phase 1):
  --shared-poisoned   patch / common-corruption sets (typographic is split out).

The clean set is downloaded from the public, non-gated mirror
``evanarlian/imagenet_1k_resized_256`` (images pre-resized to 256px, the
standard ImageNet pre-crop size). Only the validation parquet shards are
fetched (~870 MB). One class-balanced image per ImageNet class is then sampled
with a fixed seed and written to ``data/clean/``.

The typographic set is CPU-only — pure PIL drawing on top of the clean
images, no neural network involved. It generates one poisoned variant per
clean image and is fully deterministic given the global seed (config.yaml).

Usage:
    python scripts/generate_datasets.py --clean
    python scripts/generate_datasets.py --clean --size 1000 --per-class 1
    python scripts/generate_datasets.py --typographic
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import urllib.request

import yaml
from tqdm import tqdm

# Make the repo root importable so `datasets.sampler` resolves.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from datasets.sampler import ReservoirPerClass  # noqa: E402
from attacks.typographic import render_typographic  # noqa: E402

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
    # pyarrow is only needed for the clean-set parquet path. Lazy-import so the
    # typographic generator works without it installed.
    import pyarrow.parquet as pq

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


def _short_class_name(name: str) -> str:
    """ImageNet class names are comma-separated synonym lists; take the first.

    Matches the convention used by ``ClipZeroShotClassifier`` so the text we
    render here is the same string CLIP scores against in its text prompts.
    """
    return name.split(",")[0].strip().lower()


def build_typographic_set(
    cfg: dict,
    font_size_frac: float,
    position: str,
    padding_frac: float,
    opacity: float,
    jpeg_quality: int,
    preview_grid: bool,
) -> None:
    """Generate one typographic-poisoned variant per clean image.

    The attack is model-agnostic: PIL draws a misleading-class text sticker
    onto every clean image at native resolution, the result is JPEG-encoded
    to ``data/poisoned/typographic/images/``, and a manifest captures the
    seed, per-image target, and rendering config so the run is reproducible
    and committable to git.
    """
    from PIL import Image

    seed = int(cfg["seed"])
    clean_dir = _abs(cfg["paths"]["clean_set"])
    poisoned_root = _abs(os.path.join(cfg["paths"]["data_root"], "poisoned", "typographic"))
    images_dir = os.path.join(poisoned_root, "images")
    os.makedirs(images_dir, exist_ok=True)

    clean_manifest_path = os.path.join(clean_dir, "manifest.json")
    if not os.path.exists(clean_manifest_path):
        raise FileNotFoundError(
            f"No clean manifest at {clean_manifest_path}. "
            "Run: python scripts/generate_datasets.py --clean"
        )
    with open(clean_manifest_path, "r", encoding="utf-8") as f:
        clean_manifest = json.load(f)

    classes_path = os.path.join(clean_dir, "imagenet_classes.json")
    with open(classes_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)
    num_classes = len(class_names)

    print(f"Source : {clean_manifest_path}  ({clean_manifest['num_images']} images)")
    print(f"Target : {poisoned_root}")
    print(f"Config : font_size_frac={font_size_frac}  position={position}  "
          f"padding_frac={padding_frac}  opacity={opacity}  jpeg_quality={jpeg_quality}")
    print(f"Seed   : {seed}\n")

    rng = random.Random(seed)
    records = []
    for rec in tqdm(clean_manifest["images"], desc="rendering", leave=False):
        src = os.path.join(clean_dir, rec["filename"])
        true_label = int(rec["label"])
        # Pick a random target class != true label. Deterministic in image
        # order because the RNG was seeded above.
        target_label = rng.randrange(num_classes)
        if target_label == true_label:
            target_label = (target_label + 1) % num_classes
        overlay_text = _short_class_name(class_names[target_label])

        with Image.open(src) as im:
            adv = render_typographic(
                im,
                overlay_text,
                font_size_frac=font_size_frac,
                position=position,
                padding_frac=padding_frac,
                opacity=opacity,
            )
        # Mirror the clean-set filename so a poisoned image is trivially
        # paired with its clean counterpart.
        dest_rel = rec["filename"]
        dest_abs = os.path.join(poisoned_root, dest_rel)
        os.makedirs(os.path.dirname(dest_abs), exist_ok=True)
        adv.save(dest_abs, format="JPEG", quality=jpeg_quality, optimize=True)

        records.append(
            {
                "id": int(rec["id"]),
                "filename": dest_rel,
                "label": true_label,
                "true_class_name": rec["class_name"],
                "target_label": target_label,
                "target_class_name": class_names[target_label],
                "overlay_text": overlay_text,
            }
        )

    manifest = {
        "attack": "typographic",
        "seed": seed,
        "source_manifest": os.path.relpath(clean_manifest_path, _REPO_ROOT).replace("\\", "/"),
        "num_images": len(records),
        "config": {
            "font_size_frac": font_size_frac,
            "position": position,
            "padding_frac": padding_frac,
            "opacity": opacity,
            "jpeg_quality": jpeg_quality,
            "text_form": "lowercase_first_synonym",
        },
        "images": records,
    }
    with open(os.path.join(poisoned_root, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Disk-usage report — useful before committing to git.
    total_bytes = 0
    for r in records:
        total_bytes += os.path.getsize(os.path.join(poisoned_root, r["filename"]))
    print("\nVerification:")
    print(f"  images written : {len(records)}")
    print(f"  total size     : {total_bytes / (1024 * 1024):.1f} MiB  "
          f"({total_bytes / len(records) / 1024:.1f} KiB / image)")
    target_dist = {}
    for r in records:
        target_dist[r["target_label"]] = target_dist.get(r["target_label"], 0) + 1
    print(f"  distinct targets used: {len(target_dist)} / {num_classes}  "
          f"(max images / target: {max(target_dist.values())})")

    if preview_grid:
        _write_preview_grid(poisoned_root, records, clean_dir)

    print(f"\nTypographic set ready at: {poisoned_root}")


def _write_preview_grid(poisoned_root: str, records: list, clean_dir: str) -> None:
    """Write a 4x4 PNG grid pairing clean and poisoned images for quick eyeball."""
    from PIL import Image, ImageDraw

    sample = records[:16]
    if not sample:
        return
    cell_w, cell_h = 224, 224
    gap = 8
    cols, rows = 4, 4
    grid_w = cols * cell_w + (cols + 1) * gap
    grid_h = rows * cell_h + (rows + 1) * gap + 40 * rows  # +caption strip
    grid = Image.new("RGB", (grid_w, grid_h), (32, 32, 32))
    d = ImageDraw.Draw(grid)

    for i, r in enumerate(sample):
        row, col = divmod(i, cols)
        x = gap + col * (cell_w + gap)
        y = gap + row * (cell_h + 40 + gap)
        path = os.path.join(poisoned_root, r["filename"])
        im = Image.open(path).convert("RGB").resize((cell_w, cell_h))
        grid.paste(im, (x, y))
        caption = f"{r['true_class_name'].split(',')[0][:18]} -> {r['overlay_text'][:18]}"
        d.text((x + 4, y + cell_h + 4), caption, fill=(220, 220, 220))

    out = os.path.join(poisoned_root, "preview_grid.png")
    grid.save(out)
    print(f"  preview grid   : {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build benchmark datasets.")
    parser.add_argument("--clean", action="store_true", help="Build the clean set.")
    parser.add_argument(
        "--typographic",
        action="store_true",
        help="Build the typographic-overlay poisoned set (CPU-only, ~seconds).",
    )
    parser.add_argument(
        "--shared-poisoned",
        action="store_true",
        help="Build patch/corruption sets (Phase 1 — not implemented).",
    )
    parser.add_argument("--size", type=int, default=None, help="Total clean images.")
    parser.add_argument("--per-class", type=int, default=None, help="Images per class.")
    # Typographic styling overrides — defaults reproduce the classic
    # Goh-et-al. (2021) typographic-attack look.
    parser.add_argument("--font-size-frac", type=float, default=0.12)
    parser.add_argument("--position", type=str, default="top",
                        choices=("top", "center", "bottom"))
    parser.add_argument("--padding-frac", type=float, default=0.04)
    parser.add_argument("--opacity", type=float, default=1.0)
    parser.add_argument("--jpeg-quality", type=int, default=90,
                        help="JPEG quality for poisoned outputs (default 90).")
    parser.add_argument("--no-preview", action="store_true",
                        help="Skip writing the 4x4 preview grid PNG.")
    args = parser.parse_args()

    cfg = load_config()
    size = args.size or cfg["dataset"]["benchmark_size"]
    per_class = args.per_class or cfg["dataset"]["per_class"]

    if not (args.clean or args.typographic or args.shared_poisoned):
        parser.error("Nothing to do. Pass --clean, --typographic, or --shared-poisoned.")

    if args.clean:
        build_clean_set(cfg, size=size, per_class=per_class)

    if args.typographic:
        build_typographic_set(
            cfg,
            font_size_frac=args.font_size_frac,
            position=args.position,
            padding_frac=args.padding_frac,
            opacity=args.opacity,
            jpeg_quality=args.jpeg_quality,
            preview_grid=not args.no_preview,
        )

    if args.shared_poisoned:
        raise NotImplementedError(
            "Shared poisoned set (patch / corruptions) is Phase 1."
        )


if __name__ == "__main__":
    main()
