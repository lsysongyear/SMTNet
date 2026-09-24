"""
Extract voice/silence labels from SparKULee WAV files with Silero VAD.

Outputs use the same TSV columns as the existing Label_VAD files:
    file_id    start    end    label

Default outputs:
  output/segments_001.tsv ... segments_072.tsv
  output_by_story/segments_audio-<story>.tsv
  story_mapping.tsv
  summary.tsv

Run:
  python Label_VAD/extract_sparkulee_vad_segments.py
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIO_DIR = Path(
    "/dfs/share/chenjingLab/speech_tracking/dataset/SparKULee/raw_rename/audio"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "Label_VAD"
DEFAULT_SILERO_REPO = Path(
    "/gpfs/share/home/2201112028/lsycode/ICASSP_2027/silero-vad-master"
)
DEFAULT_FFMPEG = Path(
    "/gpfs/share/home/2201112028/miniconda3/envs/libribrain/bin/ffmpeg"
)


def parse_range(spec: str) -> set[int] | None:
    if not spec:
        return None
    result: set[int] = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            result.update(range(int(start), int(end) + 1))
        else:
            result.add(int(item))
    return result


def natural_key(path: Path) -> list[object]:
    text = path.stem
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def read_audio_ffmpeg(path: Path, ffmpeg: Path, sampling_rate: int) -> np.ndarray:
    cmd = [
        str(ffmpeg),
        "-v",
        "quiet",
        "-i",
        str(path),
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sampling_rate),
        "-ac",
        "1",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="replace"))
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def load_silero(repo_dir: Path):
    model, utils = torch.hub.load(
        repo_or_dir=str(repo_dir),
        model="silero_vad",
        source="local",
        trust_repo=True,
    )
    return model, utils[0]


def speech_to_full_segments(speech_ts, total_duration: float, sr: int):
    segments = []
    cursor = 0.0
    for ts in speech_ts:
        start = max(0.0, ts["start"] / sr)
        end = min(total_duration, ts["end"] / sr)
        if end <= start:
            continue
        if start > cursor + 0.005:
            segments.append((cursor, start, "silence"))
        segments.append((start, end, "voice"))
        cursor = max(cursor, end)
    if total_duration > cursor + 0.005:
        segments.append((cursor, total_duration, "silence"))
    if not segments:
        segments.append((0.0, total_duration, "silence"))
    return segments


def write_segments(path: Path, file_id: str, segments) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["file_id", "start", "end", "label"])
        for start, end, label in segments:
            writer.writerow([file_id, f"{start:.2f}", f"{end:.2f}", label])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--silero-repo", type=Path, default=DEFAULT_SILERO_REPO)
    parser.add_argument("--ffmpeg", type=Path, default=DEFAULT_FFMPEG)
    parser.add_argument("--sampling-rate", type=int, default=16000)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-speech-duration-ms", type=int, default=250)
    parser.add_argument("--min-silence-duration-ms", type=int, default=100)
    parser.add_argument("--max-speech-duration-s", type=float, default=30.0)
    parser.add_argument(
        "--indices",
        type=str,
        default="",
        help="Optional 1-based subset, e.g. '1', '1-10', or '1,3,5'.",
    )
    args = parser.parse_args()

    numeric_out = args.out_dir / "output"
    story_out = args.out_dir / "output_by_story"
    numeric_out.mkdir(parents=True, exist_ok=True)
    story_out.mkdir(parents=True, exist_ok=True)

    wavs = sorted(args.audio_dir.glob("*.wav"), key=natural_key)
    selected = parse_range(args.indices)
    if selected is not None:
        wavs = [wav for idx, wav in enumerate(wavs, start=1) if idx in selected]
    if not wavs:
        raise FileNotFoundError(f"No WAV files found in {args.audio_dir}")

    print(f"Loading Silero VAD from {args.silero_repo}")
    model, get_speech_ts = load_silero(args.silero_repo)
    print(f"Processing {len(wavs)} WAV file(s) from {args.audio_dir}")

    mapping_rows = []
    summary_rows = []

    for file_id, wav_path in enumerate(wavs, start=1):
        story_id = wav_path.stem
        audio = read_audio_ffmpeg(wav_path, args.ffmpeg, args.sampling_rate)
        total_duration = len(audio) / args.sampling_rate
        wav_tensor = torch.from_numpy(audio).float()

        speech_ts = get_speech_ts(
            wav_tensor,
            model,
            threshold=args.threshold,
            min_speech_duration_ms=args.min_speech_duration_ms,
            min_silence_duration_ms=args.min_silence_duration_ms,
            max_speech_duration_s=args.max_speech_duration_s,
            return_seconds=False,
        )
        segments = speech_to_full_segments(speech_ts, total_duration, args.sampling_rate)

        numeric_path = numeric_out / f"segments_{file_id:03d}.tsv"
        story_path = story_out / f"segments_{story_id}.tsv"
        write_segments(numeric_path, str(file_id), segments)
        write_segments(story_path, story_id, segments)

        voice_time = sum(end - start for start, end, label in segments if label == "voice")
        silence_time = sum(end - start for start, end, label in segments if label == "silence")
        mapping_rows.append([file_id, story_id, str(wav_path), str(numeric_path), str(story_path)])
        summary_rows.append(
            [
                file_id,
                story_id,
                f"{total_duration:.3f}",
                len(speech_ts),
                len(segments),
                f"{voice_time:.3f}",
                f"{silence_time:.3f}",
                f"{voice_time / total_duration:.6f}" if total_duration else "0.000000",
            ]
        )
        print(
            f"{file_id:03d} {story_id}: {len(speech_ts)} speech, "
            f"{len(segments)} segments, voice={voice_time:.1f}s/{total_duration:.1f}s"
        )

    with (args.out_dir / "story_mapping.tsv").open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["file_id", "story_id", "wav_path", "numeric_segments", "story_segments"])
        writer.writerows(mapping_rows)

    with (args.out_dir / "summary.tsv").open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "file_id",
                "story_id",
                "duration_sec",
                "speech_segments",
                "total_segments",
                "voice_sec",
                "silence_sec",
                "voice_ratio",
            ]
        )
        writer.writerows(summary_rows)

    print(f"Done. Results saved to {args.out_dir}")


if __name__ == "__main__":
    main()
