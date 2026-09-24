"""
Plot audio envelope with VAD voice/silence annotations for SEM4Lang stories.

Usage:
  python Label_VAD/plot_envelope_with_vad.py                       # story 01, full duration
  python Label_VAD/plot_envelope_with_vad.py -t 5                  # story 05 only
  python Label_VAD/plot_envelope_with_vad.py -t 1-10               # stories 1-10
  python Label_VAD/plot_envelope_with_vad.py -s 30 -d 15           # t=30s to t=45s
  python Label_VAD/plot_envelope_with_vad.py -c 14                 # use envelope channel 14
"""

import argparse, os, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ENV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "env")
SEG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")


def load_vad(story_id):
    """Load VAD segments from TSV."""
    seg_path = os.path.join(SEG_DIR, f"segments_{story_id:02d}.tsv")
    segments = []
    with open(seg_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            segments.append({
                'start': float(row['start']),
                'end': float(row['end']),
                'label': row['label'],
            })
    return segments


def plot_story(story_id, t_start, t_end):
    # Load single-channel envelope
    env_path = os.path.join(ENV_DIR, f"story-{story_id:02d}_envelope_64Hz.npy")
    env_1ch = np.load(env_path)  # (T,)
    n_samples = len(env_1ch)
    env_sr = 64.0
    env_duration = n_samples / env_sr

    # Load VAD
    segments = load_vad(story_id)
    vad_duration = segments[-1]['end'] if segments else 0

    print(f"  Story {story_id:02d}: env={env_duration:.1f}s ({n_samples}@64Hz), VAD={vad_duration:.1f}s")

    if t_end is None:
        t_end = min(env_duration, vad_duration)

    # Slice envelope to time window
    idx_start = int(t_start * env_sr)
    idx_end = int(t_end * env_sr)
    idx_start = max(0, idx_start)
    idx_end = min(n_samples, idx_end)

    env_slice = env_1ch[idx_start:idx_end]
    time_axis = np.arange(idx_start, idx_end) / env_sr

    # Plot
    fig_width = max(12, (t_end - t_start) * 0.6)
    fig, ax = plt.subplots(figsize=(fig_width, 4.5))

    # Shade voice/silence
    for seg in segments:
        x0, x1 = seg['start'], seg['end']
        if x1 < t_start or x0 > t_end:
            continue
        x0 = max(x0, t_start)
        x1 = min(x1, t_end)
        color = 'lightcoral' if seg['label'] == 'voice' else 'lightblue'
        alpha = 0.25 if seg['label'] == 'voice' else 0.15
        ax.axvspan(x0, x1, facecolor=color, alpha=alpha, edgecolor='none')

    ax.plot(time_axis, env_slice, color='black', linewidth=0.5, alpha=0.9)

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Envelope", fontsize=11)
    title = f"Story {story_id:02d} — Envelope + VAD"
    if t_start > 0 or t_end < env_duration:
        title += f"  [{t_start:.0f}s – {t_end:.0f}s]"
    ax.set_title(title, fontsize=13)
    ax.set_xlim(t_start, t_end)
    if len(env_slice) > 0:
        ax.set_ylim(env_slice.min() - 0.02, env_slice.max() + 0.02)

    legend_elements = [
        Patch(facecolor='lightcoral', alpha=0.25, label='Voice'),
        Patch(facecolor='lightblue', alpha=0.15, label='Silence'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Voice ratio in window
    voice_time = 0.0; total_time = t_end - t_start
    for seg in segments:
        x0, x1 = seg['start'], seg['end']
        if x1 < t_start or x0 > t_end: continue
        overlap = min(x1, t_end) - max(x0, t_start)
        if seg['label'] == 'voice': voice_time += overlap
    if total_time > 0:
        ax.text(0.99, 0.95, f"Voice: {100*voice_time/total_time:.1f}%",
                transform=ax.transAxes, fontsize=9, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    suffix = f"_{t_start:.0f}s-{t_end:.0f}s" if t_start > 0 or t_end < env_duration else ""
    out_path = os.path.join(OUT_DIR, f"story_{story_id:02d}{suffix}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {os.path.basename(out_path)}")


def main():
    parser = argparse.ArgumentParser(description="Plot audio envelope with VAD annotations")
    parser.add_argument("-s", "--start", type=float, default=0.0,
                        help="Start time in seconds (default: 0)")
    parser.add_argument("-d", "--duration", type=float, default=None,
                        help="Duration in seconds (default: full story)")
    args = parser.parse_args()

    stories = list(range(1, 61))
    t_start = args.start
    t_end = None if args.duration is None else t_start + args.duration

    print(f"Plotting {len(stories)} story(s): {stories}")
    print(f"Time window: {t_start:.1f}s – {'end' if t_end is None else f'{t_end:.1f}s'}")
    print()

    for sid in stories:
        plot_story(sid, t_start, t_end)

    print(f"\nDone. Plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
