"""
Preprocess raw MEG data from BIDS-format FIF files into story-segmented numpy arrays.

Input:  /dfs/share/chenjingLab/speech_tracking/dataset/SEM4Lang/SEM4Lang/raw/meg/sub-XX/MEG/
        /dfs/share/chenjingLab/speech_tracking/dataset/SEM4Lang/SEM4Lang/raw/audio/
Output: SEM4Lang/meg/sub-XX_story-YY_NF_100Hz.npy

Segment extraction (mirrors utils.py prepare_MEG trigger logic):
 1. Find the first audio trigger via mne.find_events(min_duration=0.002)
 2. Start = first_trigger_time + 40ms delay
 3. End   = start + audio_duration  (matches stimulus length exactly)

Each run (1-60) corresponds to one story. 204 gradiometer channels are selected,
no filtering applied, downsampled from 1000Hz to 100Hz.

Usage:
    python preprocess_meg.py                          # all 12 subjects
    python preprocess_meg.py --subjects sub-01        # single subject
    python preprocess_meg.py --subjects sub-01 sub-02 # specific subjects
"""

import os
import argparse
import numpy as np
import mne
from tqdm import tqdm

SRC_DIR = "/dfs/share/chenjingLab/speech_tracking/dataset/SEM4Lang/SEM4Lang/raw/meg"
AUDIO_DIR = "/dfs/share/chenjingLab/speech_tracking/dataset/SEM4Lang/SEM4Lang/raw/audio"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meg")

# MEG trigger → auditory cortex response delay (same as utils.py)
TRIGGER_DELAY_SEC = 0


def find_trigger_sample(raw):
    """Find the first audio trigger sample index (at original sfreq).

    Uses the same logic as utils.py SEM4LangData.prepare_MEG():
      mne.find_events(raw, min_duration=0.002) → first event.

    Returns sample index (int) at raw.info['sfreq'], or None if no trigger found.
    """
    try:
        events = mne.find_events(raw, min_duration=0.002, verbose=False)
        return events[0, 0]  # first trigger, sample index at original sfreq
    except (ValueError, RuntimeError):
        return None


def get_audio_duration_sec(story_id):
    """Return the duration (seconds) of story_{story_id}.wav, or None on failure."""
    audio_path = os.path.join(AUDIO_DIR, f"story_{story_id}.wav")
    if not os.path.exists(audio_path):
        return None
    try:
        import soundfile as sf
        return sf.info(audio_path).duration
    except Exception:
        try:
            import librosa
            return librosa.get_duration(path=audio_path)
        except Exception:
            try:
                from scipy.io import wavfile
                sr, data = wavfile.read(audio_path)
                return len(data) / sr
            except Exception:
                return None


def process_subject(subject):
    sub_src = os.path.join(SRC_DIR, subject, "MEG")
    os.makedirs(OUT_DIR, exist_ok=True)

    sub_id = subject.split("-")[1]

    for story_id in tqdm(range(1, 61), desc=subject, leave=False):
        fif_path = os.path.join(sub_src, f"{subject}_task-RDR_run-{story_id}_meg.fif")

        if not os.path.exists(fif_path):
            print(f"  SKIP: {fif_path} not found")
            continue

        raw = mne.io.read_raw_fif(fif_path, preload=False, verbose="warning")
        orig_sfreq = raw.info["sfreq"]

        picks = mne.pick_types(raw.info, meg="grad", eeg=False, eog=False, ecg=False)

        # ── Segment extraction (utils.py trigger logic) ──
        trigger_sample = find_trigger_sample(raw)
        audio_dur = get_audio_duration_sec(story_id)

        if trigger_sample is not None and audio_dur is not None:
            # start = first trigger + 40ms delay, offset by raw.first_time
            trigger_sec = trigger_sample / orig_sfreq
            start_sec = trigger_sec + TRIGGER_DELAY_SEC - raw.first_time
            start_sample = max(0, int(start_sec * orig_sfreq))

            # end = start + audio stimulus duration
            end_sample = start_sample + int(audio_dur * orig_sfreq)
            end_sample = min(end_sample, raw.n_times)

            data, _ = raw[picks, start_sample:end_sample]
        else:
            reason = []
            if trigger_sample is None:
                reason.append("no trigger")
            if audio_dur is None:
                reason.append("no audio file")
            print(f"  WARN: {', '.join(reason)} in {fif_path}, using full recording")
            data, _ = raw[picks, :]

        data = data.astype(np.float64)

        # downsample 1000Hz → 100Hz (no anti-aliasing filter, NF)
        data = mne.filter.resample(data, down=orig_sfreq / 100, verbose=False)

        out_path = os.path.join(OUT_DIR, f"sub-{sub_id}_story-{story_id:02d}_NF_100Hz.npy")
        np.save(out_path, data.astype(np.float32))


def main():
    global SRC_DIR, AUDIO_DIR, OUT_DIR
    parser = argparse.ArgumentParser(description="Segment MEG data by audio trigger + stimulus length")
    parser.add_argument("--subjects", nargs="+",
                        default=[f"sub-{i:02d}" for i in range(1, 13)],
                        help="Subject IDs to process (default: all 12)")
    parser.add_argument("--source-dir", default=SRC_DIR)
    parser.add_argument("--audio-dir", default=AUDIO_DIR)
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()
    SRC_DIR, AUDIO_DIR, OUT_DIR = args.source_dir, args.audio_dir, args.out_dir

    print(f"Subjects: {args.subjects}")
    print(f"Processing: NF (no filter) + 100Hz downsampling")
    print(f"Trigger delay: {TRIGGER_DELAY_SEC*1000:.0f}ms")
    print(f"Output dir: {OUT_DIR}")
    print()

    for subject in args.subjects:
        print(f"Processing {subject} ...")
        process_subject(subject)

    print("Done.")


if __name__ == "__main__":
    main()
