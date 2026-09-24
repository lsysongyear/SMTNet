"""
Extract voice/silence segments from MP3 audio using Silero VAD.
Output TSV format: file_id, start, end, label (same as VAD_v1/preprocess/output/segments_XX.tsv)

Usage:
  python extract_vad_segments.py                  # process all 50 trials
  python extract_vad_segments.py -t 1             # single trial
  python extract_vad_segments.py -t 1-10          # trial range
"""

import argparse
import os
import csv
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Audio loading via ffmpeg CLI (avoids torchcodec FFmpeg shared-library issues)
# ---------------------------------------------------------------------------

FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")


def read_audio_ffmpeg(path, sampling_rate=16000):
    """Decode MP3 to float32 mono numpy array using ffmpeg CLI."""
    cmd = [
        FFMPEG_BIN, "-v", "quiet", "-i", path,
        "-f", "s16le", "-acodec", "pcm_s16le",
        "-ar", str(sampling_rate), "-ac", "1", "-"
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode()}")
    raw = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return raw


# ---------------------------------------------------------------------------
# Silero VAD (raw torch hub — avoids silero-vad wrapper's torchcodec path)
# ---------------------------------------------------------------------------

SILERO_LOCAL = os.environ.get("SILERO_REPO", "silero-vad")


def get_silero_vad_model():
    model, utils = torch.hub.load(
        repo_or_dir=SILERO_LOCAL,
        model="silero_vad",
        source="local",
        trust_repo=True,
    )
    get_speech_ts = utils[0]
    return model, get_speech_ts


def vad_speech_timestamps(wav, model, get_speech_ts, sr=16000, threshold=0.5,
                          min_speech_duration_ms=250, min_silence_duration_ms=100,
                          max_speech_duration_s=30.0):
    """Run Silero VAD and return list of {'start': sec, 'end': sec}."""
    if isinstance(wav, np.ndarray):
        wav = torch.from_numpy(wav).float()

    raw_ts = get_speech_ts(
        wav, model,
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        max_speech_duration_s=max_speech_duration_s,
        return_seconds=False,  # get sample indices, convert manually for full precision
    )
    return [{"start": t["start"] / sr, "end": t["end"] / sr} for t in raw_ts]

AUDIO_DIR = os.environ.get("PKU_AUDIO_DIR", "audio")
OUT_DIR = os.environ.get("PKU_LABEL_DIR", str(Path(__file__).resolve().parent / "output"))

# Silero VAD parameters
SAMPLING_RATE = 16000
THRESHOLD = 0.5
MIN_SPEECH_DURATION_MS = 250
MIN_SILENCE_DURATION_MS = 100
MAX_SPEECH_DURATION_S = 30.0


def parse_trial_range(arg):
    trials = []
    for part in arg.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            trials.extend(range(int(lo), int(hi) + 1))
        else:
            trials.append(int(part))
    return trials


def speech_to_voice_silence(speech_timestamps, total_duration):
    """
    Convert silero speech timestamps to alternating silence/voice segments
    covering the full duration [0, total_duration].
    """
    segments = []
    cursor = 0.0

    for ts in speech_timestamps:
        s, e = ts["start"], ts["end"]
        # Silence before speech
        if s > cursor + 0.01:
            segments.append((cursor, s, "silence"))
        # Voice segment
        segments.append((s, e, "voice"))
        cursor = e

    # Trailing silence
    if total_duration > cursor + 0.01:
        segments.append((cursor, total_duration, "silence"))

    return segments


def main():
    global AUDIO_DIR, OUT_DIR, SILERO_LOCAL, FFMPEG_BIN
    parser = argparse.ArgumentParser(
        description="Extract VAD voice/silence segments from MP3 audio")
    parser.add_argument("-t", "--trials", type=str, default="1-50",
                        help="Trial range, e.g. '5', '1-10', '3,7,12' (default: 1-50)")
    parser.add_argument("--audio-dir", default=AUDIO_DIR)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--silero-repo", default=SILERO_LOCAL)
    parser.add_argument("--ffmpeg", default=FFMPEG_BIN)
    args = parser.parse_args()

    AUDIO_DIR, OUT_DIR = args.audio_dir, args.out_dir
    SILERO_LOCAL, FFMPEG_BIN = args.silero_repo, args.ffmpeg

    trials = parse_trial_range(args.trials)
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading Silero VAD model from local: {SILERO_LOCAL}")
    model, get_speech_ts = get_silero_vad_model()
    print(f"Processing {len(trials)} trial(s): {trials}\n")

    for trial in trials:
        audio_path = f"{AUDIO_DIR}/{trial}.mp3"
        if not os.path.exists(audio_path):
            alt_path = f"{AUDIO_DIR}/{trial:02d}.mp3"
            if os.path.exists(alt_path):
                audio_path = alt_path
            else:
                print(f"  Trial {trial}: SKIP — audio not found")
                continue

        # Load audio via ffmpeg CLI → float32 numpy array
        wav = read_audio_ffmpeg(audio_path, sampling_rate=SAMPLING_RATE)
        total_duration = len(wav) / SAMPLING_RATE

        speech_ts = vad_speech_timestamps(
            wav, model, get_speech_ts,
            sr=SAMPLING_RATE,
            threshold=THRESHOLD,
            min_speech_duration_ms=MIN_SPEECH_DURATION_MS,
            min_silence_duration_ms=MIN_SILENCE_DURATION_MS,
            max_speech_duration_s=MAX_SPEECH_DURATION_S,
        )

        segments = speech_to_voice_silence(speech_ts, total_duration)

        # Write TSV
        out_path = f"{OUT_DIR}/segments_{trial:02d}.tsv"
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["file_id", "start", "end", "label"])
            for start, end, label in segments:
                writer.writerow([str(trial), f"{start:.2f}", f"{end:.2f}", label])

        # Stats
        voice_time = sum(e - s for s, e, l in segments if l == "voice")
        sil_time = sum(e - s for s, e, l in segments if l == "silence")
        print(f"  Trial {trial:02d}: {len(speech_ts)} speech segments, "
              f"{len(segments)} total segments, "
              f"voice={voice_time:.1f}s ({voice_time/total_duration*100:.1f}%), "
              f"duration={total_duration:.1f}s -> {os.path.basename(out_path)}")

    print(f"\nDone. Results saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
