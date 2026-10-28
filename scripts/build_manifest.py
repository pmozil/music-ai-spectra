#!/usr/bin/env python3
"""Build a balanced one-real/one-synthetic MusicCaps manifest, split by caption.

FakeMusicCaps file stems/paths are matched to MusicCaps YouTube IDs. If several
synthetic variants exist for one ID, one is chosen deterministically so the
paired detection table is balanced 1:1. The chosen generator is recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".webm", ".wma"}
YOUTUBE_ID_RE = re.compile(r"[A-Za-z0-9_-]{11}")


def caption_group(caption: object) -> str:
    text = " ".join(str(caption or "").lower().split())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def find_ytid(stem: str, ids: set[str]) -> str | None:
    if stem in ids:
        return stem
    # Search 11-character windows, since some archives prefix model/seed tokens.
    for match in YOUTUBE_ID_RE.finditer(stem):
        candidate = match.group(0)
        if candidate in ids:
            return candidate
    for i in range(max(0, len(stem) - 10)):
        candidate = stem[i : i + 11]
        if candidate in ids and YOUTUBE_ID_RE.fullmatch(candidate):
            return candidate
    return None


def assign_splits(
    groups: list[str], seed: int, train: float, validation: float, test: float
) -> dict[str, str]:
    if abs(train + validation + test - 1.0) > 1e-6:
        raise ValueError("train + validation + test fractions must sum to 1")
    rng = np.random.default_rng(seed)
    groups = list(dict.fromkeys(groups))
    rng.shuffle(groups)
    n = len(groups)
    n_train = int(n * train)
    n_val = int(n * validation)
    # Ensure small datasets have non-empty test/validation where possible.
    if n >= 3:
        n_train = min(max(1, n_train), n - 2)
        n_val = min(max(1, n_val), n - n_train - 1)
    elif n == 2:
        n_train, n_val = 1, 0
    result = {}
    for g in groups[:n_train]:
        result[g] = "train"
    for g in groups[n_train : n_train + n_val]:
        result[g] = "validation"
    for g in groups[n_train + n_val :]:
        result[g] = "test"
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--metadata-csv", type=Path, default=Path("data/raw/musiccaps_metadata.csv")
    )
    ap.add_argument(
        "--human-audio-dir", type=Path, default=Path("data/raw/musiccaps_audio")
    )
    ap.add_argument(
        "--fake-audio-dir", type=Path, default=Path("data/raw/fakemusiccaps/extracted")
    )
    ap.add_argument("--output", type=Path, default=Path("data/processed/manifest.csv"))
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--validation", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    args = ap.parse_args()
    for path in (args.metadata_csv, args.human_audio_dir, args.fake_audio_dir):
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path.resolve()}")
    meta = pd.read_csv(args.metadata_csv, dtype={"ytid": str}).drop_duplicates(
        "ytid", keep="first"
    )
    meta["ytid"] = meta["ytid"].astype(str)
    ids = set(meta["ytid"])
    human_map = {}
    for path in sorted(args.human_audio_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTS and path.stem in ids:
            human_map.setdefault(path.stem, path)
    fake_map: dict[str, list[tuple[Path, str]]] = {}
    unmapped_fake = 0
    for path in sorted(args.fake_audio_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTS:
            continue
        ytid = find_ytid(path.stem, ids)
        if not ytid:
            unmapped_fake += 1
            continue
        rel = path.relative_to(args.fake_audio_dir)
        generator = rel.parts[-2] if len(rel.parts) > 1 else path.parent.name
        fake_map.setdefault(ytid, []).append((path, str(generator)))

    records = []
    paired_ids = sorted(set(human_map) & set(fake_map))
    no_human = len(set(fake_map) - set(human_map))
    no_fake = len(set(human_map) - set(fake_map))
    by_id = meta.set_index("ytid", drop=False)
    for ytid in paired_ids:
        row = by_id.loc[ytid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        caption = str(row["caption"])
        group = caption_group(caption)
        options = sorted(fake_map[ytid], key=lambda x: (x[1], str(x[0])))
        pick = int(
            hashlib.sha256(f"{args.seed}:{ytid}".encode()).hexdigest(), 16
        ) % len(options)
        fake_path, generator = options[pick]
        records.extend(
            [
                {
                    "sample_id": f"human_{ytid}",
                    "ytid": ytid,
                    "audio_path": str(human_map[ytid].resolve()),
                    "label": 0,
                    "class_name": "human",
                    "generator": "human_source",
                    "caption": caption,
                    "caption_group": group,
                    "start_s": float(row["start_s"]),
                    "end_s": float(row["end_s"]),
                    "is_audioset_eval": bool(row.get("is_audioset_eval", False)),
                },
                {
                    "sample_id": f"fake_{ytid}_{re.sub(r'[^A-Za-z0-9_-]+', '_', generator)}",
                    "ytid": ytid,
                    "audio_path": str(fake_path.resolve()),
                    "label": 1,
                    "class_name": "ai_generated",
                    "generator": generator,
                    "caption": caption,
                    "caption_group": group,
                    "start_s": float(row["start_s"]),
                    "end_s": float(row["end_s"]),
                    "is_audioset_eval": bool(row.get("is_audioset_eval", False)),
                },
            ]
        )
    if not records:
        raise RuntimeError(
            "No matching human/FakeMusicCaps caption pairs found. Check archive extraction and filename IDs."
        )
    manifest = pd.DataFrame(records)
    split_map = assign_splits(
        manifest["caption_group"].drop_duplicates().tolist(),
        args.seed,
        args.train,
        args.validation,
        args.test,
    )
    manifest["split"] = manifest["caption_group"].map(split_map)
    # Hard validation: no caption group crosses splits and every pair contributes one of each class.
    assert manifest.groupby("caption_group")["split"].nunique().max() == 1
    pair_counts = manifest.groupby("ytid")["label"].agg(["count", "sum"])
    assert ((pair_counts["count"] == 2) & (pair_counts["sum"] == 1)).all()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)
    print(
        f"Matched caption IDs: {len(paired_ids)}; clips in balanced manifest: {len(manifest)}"
    )
    print(
        f"Dropped synthetic IDs without downloaded human audio: {no_human}; human IDs without synthetic audio: {no_fake}; unmapped fake files: {unmapped_fake}"
    )
    print("Class counts by split:")
    print(
        manifest.groupby(["split", "class_name"])
        .size()
        .unstack(fill_value=0)
        .to_string()
    )
    print("Chosen generated variants by generator:")
    print(manifest.loc[manifest["label"] == 1, "generator"].value_counts().to_string())
    print(f"Manifest: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
