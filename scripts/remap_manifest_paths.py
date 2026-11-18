#!/usr/bin/env python3
"""Remap paths in the supplied manifest to audio files under this checkout.

The manifest's pairings, labels, caption groups and split assignments are preserved.
Human files are matched by exact YouTube-ID stem. Generated files are matched by
YouTube ID in the filename and generator directory/name.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import re

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".webm", ".wma"}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--human-dir", type=Path, default=Path("data/raw/musiccaps_audio"))
    ap.add_argument("--fake-dir", type=Path, default=Path("data/raw/fakemusiccaps/extracted"))
    ap.add_argument("--output", type=Path, default=Path("data/processed/manifest_local.csv"))
    ap.add_argument("--allow-missing", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest, dtype={"ytid": str})
    for p in (args.human_dir, args.fake_dir):
        if not p.is_dir():
            raise FileNotFoundError(f"Audio directory does not exist: {p}")

    human = {}
    for p in args.human_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            human.setdefault(p.stem, p.resolve())

    ids = set(df["ytid"].astype(str))
    fake_by_id = {ytid: [] for ytid in ids}
    id_pattern = re.compile(r"[A-Za-z0-9_-]{11}")
    for p in args.fake_dir.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in AUDIO_EXTS:
            continue
        candidates = [m.group(0) for m in id_pattern.finditer(p.stem)]
        for ytid in candidates:
            if ytid in fake_by_id:
                fake_by_id[ytid].append(p.resolve())

    new_paths, statuses = [], []
    for row in df.itertuples(index=False):
        ytid = str(row.ytid)
        if int(row.label) == 0:
            candidate = human.get(ytid)
        else:
            options = fake_by_id.get(ytid, [])
            generator = str(row.generator).lower()
            matching = [p for p in options if generator in str(p).lower()]
            candidate = sorted(matching or options)[0] if (matching or options) else None
        new_paths.append(str(candidate) if candidate else "")
        statuses.append("ok" if candidate else "missing")

    df["audio_path_original"] = df["audio_path"]
    df["audio_path"] = new_paths
    df["path_status"] = statuses
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"Rows: {len(df)}; resolved: {sum(s == 'ok' for s in statuses)}; missing: {sum(s != 'ok' for s in statuses)}")
    if any(s != "ok" for s in statuses):
        missing = df[df["path_status"].ne("ok")]
        missing[["sample_id", "ytid", "label", "generator", "audio_path_original"]].to_csv(
            args.output.with_name("manifest_missing_audio.csv"), index=False
        )
        if not args.allow_missing:
            raise SystemExit("Some paths were not resolved. See manifest_missing_audio.csv")
    print(f"Saved: {args.output}")

if __name__ == "__main__":
    main()
