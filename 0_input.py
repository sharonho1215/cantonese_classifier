"""Cantonese tone visualiser, v0.

Record the six "si" syllables, extract their pitch contours and plot them.

Usage:
    python tone_v0.py record   # record si1..si6 into recordings/
    python tone_v0.py plot     # analyse recordings/ and save tones_v0.png
    python tone_v0.py wave si2 # raw waveform, zoom and spectrogram for one word

You can also skip recording and drop your own files into recordings/
named si1.wav ... si6.wav (e.g. recorded on your phone and converted to WAV).
"""

import argparse
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import medfilt

# (file name, character, meaning)
WORDS = [
    ("si1", "詩", "poem"),
    ("si2", "史", "history"),
    ("si3", "試", "try"),
    ("si4", "時", "time"),
    ("si5", "市", "market"),
    ("si6", "事", "matter"),
]

SR = 22050            # sample rate (Hz)
HOP = 256             # pitch-tracking hop length (samples) -> ~11.6 ms per frame
REC_DIR = Path("recordings")
N_POINTS = 50         # points per contour after time-normalisation


def record(seconds: float = 2.0) -> None:
    """Record each syllable from the default microphone."""
    import sounddevice as sd  # imported here so plotting works without it

    REC_DIR.mkdir(exist_ok=True)
    for name, char, gloss in WORDS:
        input(f"Press Enter, then say  {name}  {char}  ({gloss}) ...")
        audio = sd.rec(int(seconds * SR), samplerate=SR, channels=1, dtype="float32")
        sd.wait()
        sf.write(REC_DIR / f"{name}.wav", audio[:, 0], SR)
        print(f"  saved {REC_DIR / (name + '.wav')}")


def pitch_contour(path: Path, fmin: float = 75, fmax: float = 500):
    """Return (times_s, f0_hz) for the voiced part of one recording."""
    y, sr = librosa.load(path, sr=SR)
    y, _ = librosa.effects.trim(y, top_db=30)          # cut leading/trailing silence
    f0, voiced, _ = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=sr, frame_length=1024, hop_length=HOP
    )
    keep = voiced & ~np.isnan(f0)                      # voiced frames only
    if keep.sum() < 5:
        return None
    times = librosa.times_like(f0, sr=sr, hop_length=HOP)[keep]
    f0 = medfilt(f0[keep], kernel_size=5)              # smooth out octave blips
    return times - times[0], f0


def time_normalise(f0: np.ndarray, n: int = N_POINTS) -> np.ndarray:
    """Stretch a contour onto n evenly spaced points (0-100% of duration)."""
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(f0)), f0)


def plot() -> None:
    contours = {}
    for name, char, gloss in WORDS:
        path = REC_DIR / f"{name}.wav"
        if not path.exists():
            print(f"missing {path}, skipping")
            continue
        result = pitch_contour(path)
        if result is None:
            print(f"{name}: too few voiced frames - try recording again, louder")
            continue
        contours[name] = result

    if not contours:
        raise SystemExit("No usable recordings found. Run: python tone_v0.py record")

    # Speaker reference = median pitch across all your syllables.
    ref = np.median(np.concatenate([f0 for _, f0 in contours.values()]))

    fig, (ax_raw, ax_norm) = plt.subplots(1, 2, figsize=(12, 4.5))
    pct = np.linspace(0, 100, N_POINTS)
    print(f"\nReference pitch: {ref:.0f} Hz\n")
    print(f"{'tone':<6}{'start (st)':>12}{'end (st)':>10}{'dur (ms)':>10}")

    for name, char, gloss in WORDS:
        if name not in contours:
            continue
        t, f0 = contours[name]
        label = f"{name} ({gloss})"
        ax_raw.plot(t * 1000, f0, label=label)

        semitones = 12 * np.log2(time_normalise(f0) / ref)
        ax_norm.plot(pct, semitones, label=label, linewidth=2)
        print(f"{name:<6}{semitones[0]:>12.1f}{semitones[-1]:>10.1f}{t[-1] * 1000:>10.0f}")

    ax_raw.set(title="Raw pitch", xlabel="Time (ms)", ylabel="F0 (Hz)")
    ax_norm.set(
        title="Normalised to your voice",
        xlabel="% of syllable duration",
        ylabel="Semitones relative to your median",
    )
    ax_norm.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    for ax in (ax_raw, ax_norm):
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("tones_v0.png", dpi=150)
    print("\nSaved tones_v0.png")
    plt.show()


def plot_wave(name: str = "si2") -> None:
    """Show one recording three ways: raw waveform, a zoomed slice, and a
    spectrogram with the pitch contour drawn on top."""
    path = REC_DIR / f"{name}.wav"
    if not path.exists():
        raise SystemExit(f"missing {path}")

    y, sr = librosa.load(path, sr=SR)
    t = np.arange(len(y)) / sr

    fig, (ax_wave, ax_zoom, ax_spec) = plt.subplots(3, 1, figsize=(10, 9))

    # 1. Raw waveform: microphone pressure vs time.
    ax_wave.plot(t, y, linewidth=0.5)
    ax_wave.set(title=f"{name}: raw waveform", xlabel="Time (s)", ylabel="Amplitude")

    # 2. Zoom: 30 ms from the loudest part, where individual vocal-fold cycles show.
    centre = np.argmax(np.abs(y))
    half = int(0.015 * sr)
    lo, hi = max(centre - half, 0), min(centre + half, len(y))
    ax_zoom.plot(t[lo:hi] * 1000, y[lo:hi])
    ax_zoom.set(title="30 ms zoom: each repeating cycle is one vocal-fold vibration",
                xlabel="Time (ms)", ylabel="Amplitude")

    # 3. Spectrogram: energy at each frequency over time, with F0 overlaid.
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=1024, hop_length=HOP)),
                                ref=np.max)
    librosa.display.specshow(S, sr=sr, hop_length=HOP, x_axis="time", y_axis="hz",
                             ax=ax_spec, cmap="magma")
    f0, voiced, _ = librosa.pyin(y, fmin=75, fmax=500, sr=sr,
                                 frame_length=1024, hop_length=HOP)
    ax_spec.plot(librosa.times_like(f0, sr=sr, hop_length=HOP), f0,
                 color="cyan", linewidth=2, label="pitch contour (F0)")
    ax_spec.set(title="Spectrogram with pitch contour", ylim=(0, 2000))
    ax_spec.legend(loc="upper right")

    fig.tight_layout()
    out = f"wave_{name}.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.show()


if __name__ == "__main__":
    import librosa.display  # noqa: F401  (needed for specshow)

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["record", "plot", "wave"])
    parser.add_argument("word", nargs="?", default="si2",
                        help="for wave mode: which recording, e.g. si4 (default si2)")
    parser.add_argument("--seconds", type=float, default=1.5,
                        help="recording length per syllable (default 1.5)")
    args = parser.parse_args()
    if args.mode == "record":
        record(args.seconds)
    elif args.mode == "plot":
        plot()
    else:
        plot_wave(args.word)