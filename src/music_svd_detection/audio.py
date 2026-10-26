"""FFmpeg helpers for provenance measurement and common audio canonicalization."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def require_ffmpeg() -> None:
    for program in ("ffmpeg", "ffprobe"):
        if shutil.which(program) is None:
            raise RuntimeError(f"Required executable not found on PATH: {program}")


def probe_audio(path: str | Path) -> dict[str, object]:
    require_ffmpeg()
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_name,sample_rate,channels,duration",
        "-of",
        "json",
        str(path),
    ]
    p = subprocess.run(
        cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    streams = json.loads(p.stdout).get("streams", [])
    if not streams:
        raise ValueError(f"No audio stream found: {path}")
    s = streams[0]
    return {
        "raw_codec_name": str(s.get("codec_name", "unknown")),
        "raw_sample_rate_hz": int(s.get("sample_rate", 0) or 0),
        "raw_channels": int(s.get("channels", 0) or 0),
        "raw_duration_seconds": float(s.get("duration", 0.0) or 0.0),
    }


def decode_float_mono(
    path: str | Path, sample_rate: int | None = None
) -> tuple[np.ndarray, int]:
    """Decode to mono f32 PCM, preserving the input sample rate when omitted."""
    info = probe_audio(path)
    sr = int(sample_rate or info["raw_sample_rate_hz"])

    if sr <= 0:
        raise ValueError(f"Could not establish sample rate for {path}")

    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vn", "-ac", "1"]

    if sample_rate is not None:
        cmd += ["-ar", str(sample_rate)]

    cmd += ["-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"]

    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    audio = np.frombuffer(p.stdout, dtype="<f4").copy()

    if not audio.size:
        raise ValueError(f"Decoded audio is empty: {path}")

    return audio, sr


def canonicalize_audio(
    source: str | Path,
    destination: str | Path,
    sample_rate: int = 16_000,
    duration_seconds: float = 10.0,
    lowpass_hz: int = 7_000,
    mp3_bitrate: str = "96k",
    loudness_i_lufs: float = -23.0,
    loudness_tp_db: float = -1.5,
    loudness_lra: float = 7.0,
) -> dict[str, object]:
    """Resample, mono-mix, low-pass, loudness-normalize, encode common MP3, decode to WAV."""

    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    info = probe_audio(source)

    with tempfile.TemporaryDirectory(prefix="music-svd-") as td:
        interim = Path(td) / "common_codec.mp3"
        af = (
            f"aresample={sample_rate},lowpass=f={lowpass_hz},"
            f"loudnorm=I={loudness_i_lufs}:TP={loudness_tp_db}:LRA={loudness_lra},"
            f"apad=pad_dur={duration_seconds},atrim=duration={duration_seconds}"
        )
        encode = [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-af",
            af,
            "-ar",
            str(sample_rate),
            "-c:a",
            "libmp3lame",
            "-b:a",
            str(mp3_bitrate),
            "-t",
            str(duration_seconds),
            str(interim),
        ]

        subprocess.run(
            encode, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        decode = [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-i",
            str(interim),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-t",
            str(duration_seconds),
            "-c:a",
            "pcm_s16le",
            str(destination),
        ]

        subprocess.run(
            decode, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

    return info
