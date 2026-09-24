"""
Extract voice/silence labels from SEM4Lang WAV files with Silero VAD.

Output TSV format:
    file_id    start    end    label

Run:
  python Label_VAD/extract_vad_segments.py
"""

import argparse, os, csv, subprocess
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = Path("/dfs/share/chenjingLab/speech_tracking/dataset/SEM4Lang/SEM4Lang/raw_rename/audio")
OUT_DIR = PROJECT_ROOT / "Label_VAD" / "output"
SILERO_REPO = Path("/gpfs/share/home/2201112028/lsycode/ICASSP_2027/silero-vad-master")
FFMPEG = Path("/gpfs/share/home/2201112028/miniconda3/envs/libribrain/bin/ffmpeg")
SAMPLE_RATE = 16000


def load_audio_ffmpeg(wav_path):
    """Use ffmpeg to decode WAV to 16kHz mono float32 numpy array."""
    cmd = [
        str(FFMPEG), "-v", "error",
        "-i", str(wav_path),
        "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-f", "f32le", "-"
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode()}")
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    return audio


def extract_segments(audio_path, story_id, out_dir):
    """Extract voice/silence segments from one audio file, save as TSV."""
    print(f"  Processing story_{story_id:02d}.wav ...")

    audio = load_audio_ffmpeg(audio_path)
    audio_tensor = torch.from_numpy(audio)

    # Load Silero VAD
    model, utils = torch.hub.load(
        repo_or_dir=str(SILERO_REPO), model="silero_vad", source="local", trust_repo=True
    )
    (get_speech_timestamps, _, _, _, _) = utils

    # Get speech timestamps (in samples, NOT rounded seconds)
    speech_ts = get_speech_timestamps(audio_tensor, model, sampling_rate=SAMPLE_RATE,
                                       threshold=0.5, min_speech_duration_ms=250,
                                       min_silence_duration_ms=100, max_speech_duration_s=30)

    # Convert to time-based segments with labels
    total_samples = len(audio)
    total_sec = total_samples / SAMPLE_RATE
    segments = []
    last_end = 0.0

    for ts in speech_ts:
        start_sec = ts['start'] / SAMPLE_RATE
        end_sec = ts['end'] / SAMPLE_RATE

        # silence before this voice segment
        if last_end < start_sec:
            segments.append((story_id, round(last_end, 2), round(start_sec, 2), "silence"))
        # voice segment
        segments.append((story_id, round(start_sec, 2), round(end_sec, 2), "voice"))
        last_end = end_sec

    # trailing silence
    if last_end < total_sec:
        segments.append((story_id, round(last_end, 2), round(total_sec, 2), "silence"))

    # Save TSV
    os.makedirs(out_dir, exist_ok=True)
    tsv_path = os.path.join(out_dir, f"segments_{story_id:02d}.tsv")
    with open(tsv_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["file_id", "start", "end", "label"])
        for seg in segments:
            writer.writerow(seg)

    # Stats
    voice_sec = sum(s[2] - s[1] for s in segments if s[3] == "voice")
    print(f"    {len(segments)} segments, {voice_sec:.0f}s voice / {total_sec:.0f}s total ({100*voice_sec/total_sec:.1f}%)")
    return voice_sec, total_sec, len(segments)


def main():
    global AUDIO_DIR, SILERO_REPO, FFMPEG
    parser = argparse.ArgumentParser(description="Extract SEM4Lang speech labels")
    parser.add_argument("--audio-dir", type=Path, default=AUDIO_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--silero-repo", type=Path, default=SILERO_REPO)
    parser.add_argument("--ffmpeg", type=Path, default=FFMPEG)
    args = parser.parse_args()
    AUDIO_DIR, SILERO_REPO, FFMPEG = args.audio_dir, args.silero_repo, args.ffmpeg
    out_dir = str(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    wav_files = sorted(AUDIO_DIR.glob("story_*.wav"))
    print(f"Found {len(wav_files)} audio files in {AUDIO_DIR}")

    summary_rows = []
    for wav_path in wav_files:
        story_id = int(wav_path.stem.replace("story_", ""))
        voice_sec, total_sec, n_seg = extract_segments(str(wav_path), story_id, out_dir)
        summary_rows.append((story_id, total_sec, voice_sec, 100 * voice_sec / total_sec, n_seg))

    # Save summary
    summary_path = os.path.join(args.out_dir.parent, "summary.tsv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["story_id", "duration_sec", "voice_sec", "voice_pct", "n_segments"])
        for row in summary_rows:
            writer.writerow(row)

    print(f"\nDone. {len(wav_files)} files processed.")
    print(f"Segments: {out_dir}")
    print(f"Summary:  {summary_path}")


if __name__ == "__main__":
    main()
