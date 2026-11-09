"""
Cantonese tone classifier trained on the Cantone dataset (Hugging Face: AlienKevin/cantone).

Cantone = ~34k isolated Cantonese syllables (e.g. "si1", "jyut6") spoken by 10 synthetic
TTS voices from Amazon, Apple, Google and Microsoft. The tone is the final digit of the label.

Pipeline
--------
  audio -> pyin pitch contour -> per-speaker normalisation (semitones re. speaker median)
        -> features (10-point contour, slope, curvature, duration)
        -> classifier, evaluated leave-one-speaker-out

Usage
-----
    pip install huggingface_hub datasets librosa scikit-learn pandas joblib matplotlib

    python tone_classifier.py download                  # ~1-2 GB into data/cantone/
    python tone_classifier.py inspect                   # check labels/speakers were parsed right
    python tone_classifier.py train --per-speaker 400   # quick run (~4k clips, a few minutes)
    python tone_classifier.py train                     # everything (pitch tracking takes a while,
                                                        #  features are cached after the first run)
    python tone_classifier.py diagnose                  # per-voice separation of tones 2/5, 1/3, 3/6
    python tone_classifier.py train --speakers 'google*' --test-rest
                                                        # train/evaluate on Google voices only,
                                                        #  then test that model on the other voices
    python tone_classifier.py train --speakers 'google*' --test-rest --norm range
    python tone_classifier.py predict recordings/       # test on your own si1_01.wav ... files
"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import joblib
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from joblib import Parallel, delayed
from scipy.signal import medfilt

REPO_ID = "AlienKevin/cantone"
DATA_DIR = Path("data/cantone")
CACHE = Path("data/cantone_features.csv")
MODEL_PATH = Path("tone_model.joblib")

SR = 16000            # Cantone is 16 kHz
HOP = 160             # 10 ms frames
FMIN, FMAX = 65, 450  # wide enough for male and female voices
N_POINTS = 10         # contour points per syllable used as features
NORM = "median"       # speaker normalisation: "median" (semitones) or "range" (1-5 scale)
TONES = [1, 2, 3, 4, 5, 6]
TONE_NAMES = {1: "1 high level", 2: "2 high rising", 3: "3 mid level",
              4: "4 low falling", 5: "5 low rising", 6: "6 low level"}

SYLLABLE_RE = re.compile(r"^([a-z]{1,6})([1-6])$")


# ==========================================================================
# 1. Download
# ==========================================================================
def download() -> None:
    """Download the dataset. Handles either raw WAV files or a parquet/'datasets' layout;
    either way it ends up as data/cantone/<speaker>/<syllable>.wav."""
    from huggingface_hub import snapshot_download

    raw = Path(snapshot_download(REPO_ID, repo_type="dataset", local_dir="data/cantone_raw"))
    wavs = list(raw.rglob("*.wav"))
    if wavs:
        print(f"Found {len(wavs)} WAV files in {raw}")
        if DATA_DIR.exists() or DATA_DIR.is_symlink():
            DATA_DIR.unlink() if DATA_DIR.is_symlink() else None
        if not DATA_DIR.exists():
            DATA_DIR.symlink_to(raw.resolve(), target_is_directory=True)
        print(f"Linked {DATA_DIR} -> {raw}")
        return

    # No WAVs: dataset is stored as parquet. Load via `datasets` and write WAVs out.
    print("No WAV files found; loading through the `datasets` library instead...")
    from datasets import load_dataset
    ds = load_dataset(str(raw))
    n = 0
    for split in ds.values():
        audio_col = next(c for c, f in split.features.items() if type(f).__name__ == "Audio")
        text_cols = [c for c, f in split.features.items()
                     if getattr(f, "dtype", None) == "string"]
        spk_col = next((c for c in text_cols if re.search("speaker|voice", c, re.I)), None)
        for row in split:
            syl = next((row[c] for c in text_cols if SYLLABLE_RE.match(str(row[c]))), None)
            if syl is None:
                continue
            spk = str(row[spk_col]) if spk_col else "unknown"
            out = DATA_DIR / spk / f"{syl}.wav"
            out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(out, row[audio_col]["array"], row[audio_col]["sampling_rate"])
            n += 1
    print(f"Wrote {n} WAV files to {DATA_DIR}")


# ==========================================================================
# 2. Index files: (path, speaker, syllable, tone)
# ==========================================================================
def parse_syllable(stem: str):
    """Find a Jyutping syllable like 'jyut6' anywhere in a file name."""
    for part in re.split(r"[_\-.\s]+", stem.lower()):
        if SYLLABLE_RE.match(part):
            return part
    m = re.search(r"([a-z]{1,6}[1-6])(?!\d)", stem.lower())
    return m.group(1) if m else None


def build_index(root: Path = DATA_DIR) -> pd.DataFrame:
    rows = []
    for p in sorted(root.rglob("*.wav")):
        syl = parse_syllable(p.stem)
        if syl is None:
            continue
        rel_parent = p.parent.relative_to(root)
        rows.append({"path": str(p), "speaker": str(rel_parent) if str(rel_parent) != "." else "",
                     "syllable": syl, "tone": int(syl[-1])})
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"No labelled WAV files found under {root}. Run `download` first.")
    if df["speaker"].nunique() <= 1:
        df["speaker"] = [re.sub(re.escape(s), "", Path(p).stem, flags=re.I).strip("_- .")
                         or "unknown" for p, s in zip(df["path"], df["syllable"])]
    return df


def select_speakers(df: pd.DataFrame, names: list[str] | None, invert: bool = False):
    """Keep voices whose name (or any folder in its path) matches one of `names`.
    Whole-name match, case-insensitive, with optional *:
      'googleA'  -> only googleA          'google*' -> googleA, googleB, googleC, googleD
      'A'        -> A, not Aasing         'google'  -> a folder called google/ (e.g. google/A)"""
    from fnmatch import fnmatch
    if not names:
        return df
    pats = [n.lower() for n in names]
    hit = df["speaker"].map(lambda s: any(fnmatch(c.lower(), p)
                                          for c in Path(s).parts for p in pats))
    out = df[~hit] if invert else df[hit]
    if not invert and out.empty:
        raise SystemExit(f"No voices matched {names}. Voices available: "
                         f"{sorted(df['speaker'].unique())}")
    return out


def inspect() -> None:
    df = build_index()
    print(f"{len(df)} labelled clips, {df['speaker'].nunique()} speakers\n")
    print("Clips per speaker:\n", df["speaker"].value_counts().to_string(), "\n")
    print("Clips per tone:\n", df["tone"].value_counts().sort_index().to_string(), "\n")
    print("Random examples (check the syllable/tone match the file name):")
    print(df.sample(min(10, len(df)), random_state=0)[["path", "speaker", "syllable", "tone"]]
          .to_string(index=False))
    if df["speaker"].nunique() < 3:
        print("\nWARNING: fewer than 3 speakers detected - speaker-split evaluation needs more.")


# ==========================================================================
# 3. Pitch features
# ==========================================================================
def extract_f0(path: str):
    """Longest unbroken voiced run of the pyin contour, in Hz (or None)."""
    try:
        y, sr = librosa.load(path, sr=SR)
    except Exception:
        return None
    if len(y) < 2048:
        y = np.pad(y, (0, 2048 - len(y)))
    f0, voiced, _ = librosa.pyin(y, fmin=FMIN, fmax=FMAX, sr=sr,
                                 frame_length=1024, hop_length=HOP)
    keep = voiced & ~np.isnan(f0)
    edges = np.diff(np.concatenate([[0], keep.astype(int), [0]]))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    if len(starts) == 0:
        return None
    i = np.argmax(ends - starts)
    if ends[i] - starts[i] < 5:            # < 50 ms of voicing
        return None
    return medfilt(f0[starts[i]:ends[i]], kernel_size=5)


def compute_features(df: pd.DataFrame, n_jobs: int = -1) -> pd.DataFrame:
    """Pitch-track every clip, cache raw contours to CSV."""
    cached = pd.read_csv(CACHE) if CACHE.exists() else pd.DataFrame(columns=["path"])
    todo = df[~df["path"].isin(set(cached["path"]))]
    if len(todo):
        print(f"Pitch-tracking {len(todo)} clips ({len(cached)} already cached)...")
        f0s = Parallel(n_jobs=n_jobs, verbose=5)(delayed(extract_f0)(p) for p in todo["path"])
        new = []
        for p, f0 in zip(todo["path"], f0s):
            if f0 is None:
                new.append({"path": p, "ok": False})
                continue
            pts = np.interp(np.linspace(0, 1, N_POINTS), np.linspace(0, 1, len(f0)), f0)
            rec = {"path": p, "ok": True, "dur_ms": len(f0) * HOP / SR * 1000,
                   "median_hz": float(np.median(f0))}
            rec.update({f"hz_{k}": v for k, v in enumerate(pts)})
            new.append(rec)
        cached = pd.concat([cached, pd.DataFrame(new)], ignore_index=True)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        cached.to_csv(CACHE, index=False)
    out = df.merge(cached, on="path", how="left")
    n_bad = (~out["ok"].astype(bool)).sum()
    if n_bad:
        print(f"Dropped {n_bad} clips with too little voicing")
    return out[out["ok"].astype(bool)].reset_index(drop=True)


def normalise_contours(feat: pd.DataFrame, method: str | None = None) -> np.ndarray:
    """Speaker-normalised contours, one row per syllable.

    method="median": semitones relative to the speaker's median pitch.
                     Removes differences in voice height (high vs low voice).
    method="range" : each speaker's own pitch range mapped onto a 1-5 scale, using the
                     5th-95th percentiles of all their pitch values (the T-value method).
                     Removes differences in voice height AND range (narrow vs wide voice).
    """
    method = method or NORM
    hz = feat[[f"hz_{k}" for k in range(N_POINTS)]].to_numpy(float)
    spk = feat["speaker"].to_numpy()
    if method == "median":
        ref = feat.groupby("speaker")["median_hz"].transform("median").to_numpy()[:, None]
        return 12 * np.log2(hz / ref)
    if method == "range":
        logc = np.log(hz)
        lo, hi = np.empty(len(feat)), np.empty(len(feat))
        for s in np.unique(spk):
            m = spk == s
            lo[m], hi[m] = np.percentile(logc[m], [5, 95])
        return 1 + 4 * (logc - lo[:, None]) / (hi - lo)[:, None]
    raise ValueError(f"unknown normalisation {method!r}")


def norm_label(method: str | None = None) -> str:
    return ("Semitones re. speaker median" if (method or NORM) == "median"
            else "Pitch on speaker's 1-5 range")


def make_X(feat: pd.DataFrame, normalise: bool = True, method: str | None = None) -> np.ndarray:
    """Turn raw Hz contours into model features.

    normalise=True : speaker-normalised (see normalise_contours; --norm median|range)
    normalise=False: raw Hz, as an ablation to show why normalisation matters
    """
    if normalise:
        c = normalise_contours(feat, method)
    else:
        c = feat[[f"hz_{k}" for k in range(N_POINTS)]].to_numpy(float)
    x = np.linspace(-1, 1, N_POINTS)
    curv, slope, _ = np.polyfit(x, c.T, 2)          # quadratic fit per syllable
    return np.column_stack([c, c.mean(1), slope, curv, c[:, -1] - c[:, 0],
                            np.log(feat["dur_ms"].to_numpy())])


# ==========================================================================
# 4. Train + evaluate (leave-one-speaker-out)
# ==========================================================================
def models():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return {
        "logistic regression": make_pipeline(StandardScaler(),
                                             LogisticRegression(max_iter=2000, C=1.0)),
        "gradient boosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                                            random_state=0),
    }


def train(per_speaker: int | None = None, n_jobs: int = -1,
          speakers: list[str] | None = None, test_rest: bool = False) -> None:
    from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                                 classification_report, confusion_matrix)
    from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

    full = build_index()
    if per_speaker:
        full = full.sample(frac=1, random_state=0).groupby("speaker").head(per_speaker)
    df = select_speakers(full, speakers)
    if speakers:
        print(f"Using voices: {sorted(df['speaker'].unique())}")
    feat = compute_features(df, n_jobs)
    y, groups = feat["tone"].to_numpy(), feat["speaker"].to_numpy()
    n_spk = len(set(groups))
    if n_spk < 3:
        raise SystemExit("Need at least 3 speakers for leave-one-speaker-out evaluation.")

    print(f"\nNormalisation: {NORM}")
    print(f"{len(feat)} syllables, {n_spk} speakers. Evaluation: leave-one-speaker-out "
          f"(train on {n_spk - 1} voices, test on the unseen one, repeat).")
    print(f"Chance (always predict the commonest tone): "
          f"{Counter(y).most_common(1)[0][1] / len(y):.1%}\n")

    logo = LeaveOneGroupOut()
    results, best = {}, None
    for norm in (False, True):
        X = make_X(feat, normalise=norm)
        for name, model in models().items():
            pred = cross_val_predict(model, X, y, groups=groups, cv=logo, n_jobs=n_jobs)
            acc = accuracy_score(y, pred)
            label = f"{name}, {'speaker-normalised' if norm else 'raw Hz'}"
            results[label] = acc
            print(f"  {label:<45} accuracy {acc:.1%}")
            if norm and (best is None or acc > best[1]):
                best = (name, acc, pred)

    name, acc, pred = best
    print(f"\nBest: {name} (speaker-normalised), {acc:.1%}\n")
    print(classification_report(y, pred, target_names=[TONE_NAMES[t] for t in TONES], digits=3))

    per_spk = (pd.DataFrame({"speaker": groups, "ok": y == pred})
               .groupby("speaker")["ok"].mean().sort_values())
    print("Accuracy on each held-out voice:\n" + per_spk.map("{:.1%}".format).to_string())

    # Figures: confusion matrix + mean contour per tone
    fig, (ax_cm, ax_ct) = plt.subplots(1, 2, figsize=(14, 5.5))
    cm = confusion_matrix(y, pred, labels=TONES, normalize="true")
    ConfusionMatrixDisplay(cm, display_labels=TONES).plot(ax=ax_cm, cmap="Blues",
                                                          values_format=".2f", colorbar=False)
    ax_cm.set(title=f"{name}, unseen voices ({acc:.1%})",
              xlabel="Predicted tone", ylabel="True tone")

    Xn = make_X(feat, normalise=True)[:, :N_POINTS]
    pct = np.linspace(0, 100, N_POINTS)
    for t in TONES:
        m, s = Xn[y == t].mean(0), Xn[y == t].std(0)
        ax_ct.plot(pct, m, linewidth=2, label=TONE_NAMES[t])
        ax_ct.fill_between(pct, m - s, m + s, alpha=0.12)
    if NORM == "median":
        ax_ct.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    voices = "all voices" if not speakers else ", ".join(speakers) + " voices"
    ax_ct.set(title=f"Mean contour per tone, {voices} (±1 SD)",
              xlabel="% of syllable", ylabel=norm_label())
    ax_ct.legend(fontsize=8)
    ax_ct.grid(alpha=0.3)
    fig.tight_layout()
    tag = ("_" + "-".join(speakers).replace("*", "") if speakers else "") + f"_{NORM}"
    fig.savefig(f"tone_classifier_results{tag}.png", dpi=150)
    print(f"\nSaved tone_classifier_results{tag}.png")

    # Refit the best model on everything and save it for `predict`
    final = models()[name].fit(make_X(feat, normalise=True), y)
    joblib.dump({"model": final, "name": name, "norm": NORM}, MODEL_PATH)
    print(f"Saved {MODEL_PATH}")

    # Optionally: test that model on all the voices that were NOT used for training
    if test_rest and speakers:
        rest = select_speakers(full, speakers, invert=True)
        if rest.empty:
            print("No other voices to test on.")
            return
        rf = compute_features(rest, n_jobs)
        ry, rpred = rf["tone"].to_numpy(), final.predict(make_X(rf, normalise=True))
        print(f"\n=== Model trained on {speakers}, tested on the other "
              f"{rf['speaker'].nunique()} voices: {accuracy_score(ry, rpred):.1%} ===")
        per = (pd.DataFrame({"speaker": rf["speaker"], "ok": ry == rpred})
               .groupby("speaker")["ok"].mean().sort_values())
        print(per.map("{:.1%}".format).to_string())
        fig2, ax2 = plt.subplots(figsize=(6.5, 5.5))
        cm2 = confusion_matrix(ry, rpred, labels=TONES, normalize="true")
        ConfusionMatrixDisplay(cm2, display_labels=TONES).plot(ax=ax2, cmap="Oranges",
                                                               values_format=".2f",
                                                               colorbar=False)
        ax2.set(title=f"Trained on {', '.join(speakers)}; tested on other voices ({NORM} norm)",
                xlabel="Predicted tone", ylabel="True tone")
        fig2.tight_layout()
        fig2.savefig(f"tone_classifier_testrest{tag}.png", dpi=150)
        print(f"Saved tone_classifier_testrest{tag}.png")


def diagnose() -> None:
    """Per voice: how clearly are tones 2 vs 5 and 1 vs 3 separated?

    Uses already-cached clips only (run `train` first). Separation is Cohen's d:
    difference between the two tones' means divided by their pooled SD.
    Rough guide: d < 0.5 essentially merged, 0.5-1.5 overlapping, > 1.5 clearly distinct."""
    if not CACHE.exists():
        raise SystemExit("No cached features yet - run `train` first.")
    df = build_index()
    feat = compute_features(df[df["path"].isin(set(pd.read_csv(CACHE)["path"]))])
    c = make_X(feat, normalise=True)[:, :N_POINTS]
    feat["end"] = c[:, -2]               # height near the end (90%), avoids end-of-voicing dropout
    feat["height"] = c.mean(axis=1)      # overall height

    def d(g, col, t1, t2):
        a, b = g.loc[g.tone == t1, col], g.loc[g.tone == t2, col]
        if len(a) < 2 or len(b) < 2:
            return np.nan
        return (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)

    rows = [{"voice": spk,
             "2 vs 5 (end height)": d(g, "end", 2, 5),
             "1 vs 3 (mean height)": d(g, "height", 1, 3),
             "3 vs 6 (mean height)": d(g, "height", 3, 6),
             "n clips": len(g)}
            for spk, g in feat.groupby("speaker")]
    out = pd.DataFrame(rows).sort_values("2 vs 5 (end height)")
    print("Tone separation per voice (Cohen's d; <0.5 merged, >1.5 clearly distinct)")
    print("(d is the same under median or range normalisation: both only shift/scale "
          "each voice)\n")
    print(out.to_string(index=False, float_format="{:.2f}".format))
    out.to_csv("diagnose.csv", index=False)
    print("\nSaved diagnose.csv")


# ==========================================================================
# 5. Predict your own recordings
# ==========================================================================
def predict(folder: str) -> None:
    """Classify your own recordings. File names like si2_01.wav give the true tone,
    so accuracy is reported. Normalisation uses the median pitch across ALL your files,
    so record several different tones before running this."""
    if not MODEL_PATH.exists():
        raise SystemExit("No trained model - run `train` first.")
    bundle = joblib.load(MODEL_PATH)
    paths = sorted(Path(folder).glob("*.wav"))
    rows = []
    for p in paths:
        f0 = extract_f0(str(p))
        if f0 is None:
            print(f"{p.name}: too little voicing, skipped")
            continue
        pts = np.interp(np.linspace(0, 1, N_POINTS), np.linspace(0, 1, len(f0)), f0)
        syl = parse_syllable(p.stem)
        rec = {"path": str(p), "speaker": "me", "tone": int(syl[-1]) if syl else None,
               "dur_ms": len(f0) * HOP / SR * 1000, "median_hz": float(np.median(f0))}
        rec.update({f"hz_{k}": v for k, v in enumerate(pts)})
        rows.append(rec)
    if not rows:
        raise SystemExit(f"No usable WAV files in {folder}")

    feat = pd.DataFrame(rows)
    X = make_X(feat, normalise=True, method=bundle.get("norm", "median"))
    pred = bundle["model"].predict(X)
    proba = bundle["model"].predict_proba(X)

    print(f"Model: {bundle['name']} ({bundle.get('norm', 'median')} normalisation)\n")
    print(f"{'file':<16}{'true':>6}{'pred':>6}{'confidence':>12}")
    for (_, r), p, pr in zip(feat.iterrows(), pred, proba):
        true = "?" if pd.isna(r["tone"]) else int(r["tone"])
        mark = "" if true == "?" else ("  ✓" if true == p else "  ✗")
        print(f"{Path(r['path']).name:<16}{true!s:>6}{p:>6}{pr.max():>11.0%}{mark}")
    known = feat["tone"].notna()
    if known.any():
        acc = (feat.loc[known, "tone"].to_numpy() == pred[known.to_numpy()]).mean()
        print(f"\nAccuracy on your recordings: {acc:.0%} ({known.sum()} files)")


# ==========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Cantonese tone classifier (Cantone dataset)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download", help="download Cantone from Hugging Face")
    sub.add_parser("inspect", help="show how files/labels/speakers were parsed")
    p_tr = sub.add_parser("train", help="extract features, evaluate, save model")
    p_tr.add_argument("--per-speaker", type=int, help="subsample N clips per voice (quick run)")
    p_tr.add_argument("--jobs", type=int, default=-1, help="CPU cores to use (default: all)")
    p_tr.add_argument("--speakers", nargs="+",
                      help="only these voices, e.g. --speakers 'google*'  or  googleA googleB")
    p_tr.add_argument("--test-rest", action="store_true",
                      help="also test the trained model on all the voices left out")
    p_tr.add_argument("--norm", choices=["median", "range"], default="median",
                      help="speaker normalisation: median (default) or range")
    sub.add_parser("diagnose", help="per-voice separation of tones 2/5, 1/3, 3/6")
    p_pr = sub.add_parser("predict", help="classify your own recordings")
    p_pr.add_argument("folder", help="folder of WAVs, e.g. recordings/")
    args = ap.parse_args()
    global NORM
    NORM = getattr(args, "norm", None) or "median"

    if args.cmd == "download":
        download()
    elif args.cmd == "inspect":
        inspect()
    elif args.cmd == "train":
        train(args.per_speaker, args.jobs, args.speakers, args.test_rest)
    elif args.cmd == "diagnose":
        diagnose()
    elif args.cmd == "predict":
        predict(args.folder)


if __name__ == "__main__":
    main()

