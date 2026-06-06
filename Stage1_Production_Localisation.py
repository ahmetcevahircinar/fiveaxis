# ============================================================
# Stage 1: Production Localisation from Raw CNC Streams
# Publishable implementation for hierarchical industrial signal
# contextualisation experiments.
#
# Pipeline used in the manuscript:
#   1. Session loading
#   2. Feature-group construction
#   3. One-time temporal feature preprocessing per session/experiment
#   4. Row-level probability estimation with logistic regression
#   5. Training-fold threshold selection from cached probabilities
#   6. Temporal post-processing of probabilities into intervals
#   7. Leave-one-session-out segment-level and row-level evaluation
# ============================================================

from __future__ import annotations

from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Keep warning handling narrow. A broad warnings.filterwarnings("ignore") can hide
# meaningful numerical or data-quality problems in publishable code.
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================
# 1) Configuration
# ============================================================

STAGE_NAME = "stage1_production_localisation"
# Notebook/script-friendly configuration. Edit these values directly when needed.
# DATA_DIR = None means: automatically search DATA_DIR_CANDIDATES for Data_from_*.csv files.
DATA_DIR = None
DATA_DIR_CANDIDATES = (Path("fiveaxis"), Path("."))
OUT_ROOT = Path("hierarchical_contextualisation_outputs") / "stage1_production_feature_groups"

LABEL_COL = "Production"
TIME_COL = "DateTime"
INPUT_FILE_PATTERN = "Data_from_*.csv"

STATIC_FEATURES = [
    "CabinDoorLockFront",
    "CabinDoorLockSide",
    "DNCMode",
    "ChipCleaningGunStatus",
    "OverrideSpindle",
    "SpindleApproval",
    "RapidTraverseKey",
    "Warmup",
]

PROGRAM_RELATED_FEATURES = [
    "ProgramStatus",
    "ProgramDetail",
    "ToolNumber",
    "DriveStatus",
    "DoorStatusMain",
]

PHYSICAL_PROCESS_FEATURES = [
    "CoolantFlow",
    "OverrideFeed",
    "FeedRate",
    "SpindleSpeed",
    "SpindleCleaning",
]

EXPERIMENTS = {
    "EXP01_all": STATIC_FEATURES + PROGRAM_RELATED_FEATURES + PHYSICAL_PROCESS_FEATURES,
    "EXP02_static_only": STATIC_FEATURES,
    "EXP03_program_only": PROGRAM_RELATED_FEATURES,
    "EXP04_physical_only": PHYSICAL_PROCESS_FEATURES,
    "EXP05_static_program": STATIC_FEATURES + PROGRAM_RELATED_FEATURES,
    "EXP06_static_physical": STATIC_FEATURES + PHYSICAL_PROCESS_FEATURES,
    "EXP07_program_physical": PROGRAM_RELATED_FEATURES + PHYSICAL_PROCESS_FEATURES,
}

ROLLING_WINDOWS = (25, 101)
SMOOTH_WINDOW = 301
MIN_SEG_LEN = 400
MERGE_GAP = 250
THRESH_GRID = np.round(np.linspace(0.20, 0.80, 25), 3)
THRESHOLD_OBJECTIVE_IOU_WEIGHT = 0.70
THRESHOLD_OBJECTIVE_F1_WEIGHT = 0.30

# A deliberately permissive segment matching threshold. Production intervals can be
# long and boundary offsets may be large after temporal smoothing; therefore this
# value is used only to decide whether a predicted interval is associated with a
# ground-truth interval for segment-level precision/recall. Boundary quality is
# reported separately through mean_best_gt_iou and boundary error metrics.
IOU_MATCH_THRESHOLD = 0.10

# Upper bound for per-session training rows after class-aware downsampling. This
# keeps LOO runs fast and reproducible while preserving positive/negative balance.
# Set to None to train on all available rows.
MAX_ROWS_PER_SESSION = 8000
BASE_RANDOM_SEED = 42
LR_MAX_ITER = 1000
LR_CLASS_WEIGHT = "balanced"
LR_SOLVER = "liblinear"

SAVE_FOLD_PLOTS = True
SAVE_SUMMARY_PLOTS = True
SAVE_TIMING_LOG = True

MAX_FEATURES_TO_PLOT = 8
FOLD_PLOT_WIDTH = 18
FOLD_PLOT_ROW_HEIGHT = 2.1
FOLD_PLOT_DPI = 150
FOLD_METRICS_FIGSIZE = (14, 5)
COMPARISON_FIGSIZE = (16, 6)
COUNT_ERROR_FIGSIZE = (16, 5)
SUMMARY_PLOT_DPI = 180


# ============================================================
# 2) Runtime Profiling Utilities
# ============================================================

TIMING_ROWS: list[dict] = []


def now() -> float:
    return time.perf_counter()


def log_timing(stage: str, fold: str | None, step: str, elapsed_sec: float, extra: dict | None = None) -> None:
    row = {
        "stage": stage,
        "fold": fold,
        "step": step,
        "elapsed_sec": float(elapsed_sec),
    }
    if extra:
        row.update(extra)
    TIMING_ROWS.append(row)


class Timer:
    def __init__(self, stage: str, fold: str | None, step: str, extra: dict | None = None):
        self.stage = stage
        self.fold = fold
        self.step = step
        self.extra = extra or {}
        self.t0: float | None = None

    def __enter__(self):
        self.t0 = now()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.t0 is not None:
            log_timing(self.stage, self.fold, self.step, now() - self.t0, self.extra)
        return False


# ============================================================
# 3) General Utility Functions
# ============================================================


def resolve_data_dir(data_dir: str | Path | None) -> Path:
    if data_dir is not None:
        path = Path(data_dir)
        if not path.exists():
            raise FileNotFoundError(f"DATA_DIR does not exist: {path}")
        if not list(path.glob(INPUT_FILE_PATTERN)):
            raise FileNotFoundError(f"No files matching {INPUT_FILE_PATTERN!r} were found in: {path}")
        return path

    for candidate in DATA_DIR_CANDIDATES:
        if candidate.exists() and list(candidate.glob(INPUT_FILE_PATTERN)):
            return candidate

    raise FileNotFoundError(
        f"No input files matching {INPUT_FILE_PATTERN!r} were found in: "
        f"{[str(p) for p in DATA_DIR_CANDIDATES]}. "
        "Set DATA_DIR explicitly if your dataset is stored elsewhere."
    )


def clean_numeric_series(series: pd.Series) -> pd.Series:
    """Convert a series to finite numeric values and impute short missing runs."""
    return (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .bfill()
        .fillna(0.0)
    )


def label_array(df: pd.DataFrame) -> np.ndarray:
    return clean_numeric_series(df[LABEL_COL]).astype(int).values


def contiguous_segments(mask: np.ndarray | pd.Series | list) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    padded = np.r_[False, mask, False]
    changes = np.diff(padded.astype(np.int8))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0] - 1
    return list(zip(starts.astype(int).tolist(), ends.astype(int).tolist()))


def merge_close_segments(segs: list[tuple[int, int]], gap: int = 0) -> list[tuple[int, int]]:
    if not segs:
        return []
    merged = [list(x) for x in sorted(segs)][0:1]
    for start, end in sorted(segs)[1:]:
        if start - merged[-1][1] - 1 <= gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [tuple(x) for x in merged]


def filter_short_segments(segs: list[tuple[int, int]], min_len: int = 1) -> list[tuple[int, int]]:
    return [(s, e) for s, e in segs if (e - s + 1) >= min_len]


def segments_from_mask(mask: np.ndarray | pd.Series | list, min_len: int = 1, merge_gap: int = 0) -> list[tuple[int, int]]:
    """Convert a boolean mask to merged and length-filtered inclusive intervals."""
    return filter_short_segments(merge_close_segments(contiguous_segments(mask), gap=merge_gap), min_len=min_len)


def segment_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    s1, e1 = a
    s2, e2 = b
    inter = max(0, min(e1, e2) - max(s1, s2) + 1)
    union = max(e1, e2) - min(s1, s2) + 1
    return inter / union if union > 0 else 0.0


def match_segments(
    pred_segs: list[tuple[int, int]],
    gt_segs: list[tuple[int, int]],
    iou_threshold: float = IOU_MATCH_THRESHOLD,
) -> list[dict]:
    pairs = [
        (segment_iou(pred, gt), pred_idx, gt_idx)
        for pred_idx, pred in enumerate(pred_segs)
        for gt_idx, gt in enumerate(gt_segs)
    ]
    pairs.sort(reverse=True)

    used_pred, used_gt, matches = set(), set(), []
    for iou, pred_idx, gt_idx in pairs:
        if iou < iou_threshold:
            break
        if pred_idx not in used_pred and gt_idx not in used_gt:
            used_pred.add(pred_idx)
            used_gt.add(gt_idx)
            matches.append({"pred_index": pred_idx, "gt_index": gt_idx, "iou": float(iou)})
    return matches


def segments_to_mask(segs: list[tuple[int, int]], n: int) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for start, end in segs:
        mask[max(0, start): min(n, end + 1)] = True
    return mask


def interval_text(segs: list[tuple[int, int]]) -> str:
    return "-" if not segs else " | ".join(f"{s}-{e}" for s, e in segs)


def smooth_prob(prob: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    return pd.Series(prob).rolling(window, center=True, min_periods=1).mean().values


def prob_to_segments(prob: np.ndarray, threshold: float) -> tuple[list[tuple[int, int]], np.ndarray]:
    prob_smooth = smooth_prob(prob, SMOOTH_WINDOW)
    pred_segs = segments_from_mask(prob_smooth >= threshold, min_len=MIN_SEG_LEN, merge_gap=MERGE_GAP)
    return pred_segs, prob_smooth


def best_iou_summary(pred_segs: list[tuple[int, int]], gt_segs: list[tuple[int, int]]) -> dict:
    if not gt_segs and not pred_segs:
        return {"mean_best_gt_iou": 1.0, "best_any_iou": 1.0}
    if not gt_segs or not pred_segs:
        return {"mean_best_gt_iou": 0.0, "best_any_iou": 0.0}

    best_per_gt = [max(segment_iou(pred, gt) for pred in pred_segs) for gt in gt_segs]
    best_any = max(segment_iou(pred, gt) for pred in pred_segs for gt in gt_segs)
    return {"mean_best_gt_iou": float(np.mean(best_per_gt)), "best_any_iou": float(best_any)}


def evaluate_segments(
    pred_segs: list[tuple[int, int]],
    gt_segs: list[tuple[int, int]],
    iou_thr: float = IOU_MATCH_THRESHOLD,
) -> tuple[dict, list[dict]]:
    matches = match_segments(pred_segs, gt_segs, iou_threshold=iou_thr)
    tp = len(matches)
    fp = len(pred_segs) - tp
    fn = len(gt_segs) - tp

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    matched_mean_iou = np.mean([m["iou"] for m in matches]) if matches else 0.0

    out = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched_mean_iou": float(matched_mean_iou),
        "n_pred": len(pred_segs),
        "n_gt": len(gt_segs),
    }
    out.update(best_iou_summary(pred_segs, gt_segs))
    return out, matches


def boundary_errors(pred_segs: list[tuple[int, int]], gt_segs: list[tuple[int, int]]) -> list[dict]:
    rows = []
    for gt_idx, gt in enumerate(gt_segs):
        if pred_segs:
            ious = [segment_iou(pred, gt) for pred in pred_segs]
            best_pred_idx = int(np.argmax(ious))
            pred = pred_segs[best_pred_idx]
            rows.append({
                "gt_index": gt_idx,
                "gt_start": gt[0],
                "gt_end": gt[1],
                "pred_index": best_pred_idx,
                "pred_start": pred[0],
                "pred_end": pred[1],
                "iou": float(ious[best_pred_idx]),
                "start_abs_error": abs(pred[0] - gt[0]),
                "end_abs_error": abs(pred[1] - gt[1]),
            })
        else:
            rows.append({
                "gt_index": gt_idx,
                "gt_start": gt[0],
                "gt_end": gt[1],
                "pred_index": np.nan,
                "pred_start": np.nan,
                "pred_end": np.nan,
                "iou": 0.0,
                "start_abs_error": np.nan,
                "end_abs_error": np.nan,
            })
    return rows


def safe_auc(y_true: np.ndarray, prob: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, prob))
    except ValueError:
        return np.nan


# ============================================================
# 4) Data Loading and Feature Preprocessing
# ============================================================


def load_sessions(data_dir: Path) -> dict[int, pd.DataFrame]:
    files = sorted(data_dir.glob(INPUT_FILE_PATTERN))
    sessions: dict[int, pd.DataFrame] = {}

    print("\nSTAGE 1 | Loading raw sessions")
    for sid, file_path in enumerate(files, start=1):
        df = pd.read_csv(file_path).copy()

        if LABEL_COL not in df.columns:
            raise ValueError(f"Required label column {LABEL_COL!r} was not found in {file_path.name}.")

        if TIME_COL in df.columns:
            df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")

        y = label_array(df)
        sessions[sid] = df
        print(
            f"  Session {sid:02d} | rows={len(df):6d} | "
            f"prod_ratio={np.mean(y == 1):.3f} | "
            f"gt_count={len(contiguous_segments(y == 1))} | file={file_path.name}"
        )

    if not sessions:
        raise FileNotFoundError(f"No CSV files matching {INPUT_FILE_PATTERN!r} were loaded from {data_dir}.")
    return sessions


def prepare_base_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for col in features:
        if col in df.columns:
            X[col] = clean_numeric_series(df[col])
        else:
            # Missing columns are explicitly represented as zero-valued features.
            # No forward/backward filling is needed for these synthetic columns.
            X[col] = np.zeros(len(df), dtype=np.float32)
    return X


def add_temporal_features(X: pd.DataFrame, rolling_windows: tuple[int, ...] = ROLLING_WINDOWS) -> pd.DataFrame:
    frames = [X.copy()]
    absdiff = X.diff().abs().fillna(0)

    for window in rolling_windows:
        means = X.rolling(window, center=True, min_periods=1).mean()
        means.columns = [f"{c}_mean_{window}" for c in X.columns]

        stds = X.rolling(window, center=True, min_periods=1).std().fillna(0)
        stds.columns = [f"{c}_std_{window}" for c in X.columns]

        diff_means = absdiff.rolling(window, center=True, min_periods=1).mean()
        diff_means.columns = [f"{c}_absdiff_mean_{window}" for c in X.columns]

        frames.extend([means, stds, diff_means])

    return pd.concat(frames, axis=1).replace([np.inf, -np.inf], np.nan).fillna(0)


def make_Xy(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, np.ndarray, list[tuple[int, int]]]:
    X_base = prepare_base_features(df, features)
    X = add_temporal_features(X_base)
    y = label_array(df)
    gt_segs = contiguous_segments(y == 1)
    return X, y, gt_segs


def precompute_experiment_features(
    sessions: dict[int, pd.DataFrame],
    experiments: dict[str, list[str]],
) -> dict[str, dict[int, dict]]:
    """Precompute X/y/GT once per experiment and session for all LOO folds."""
    cache: dict[str, dict[int, dict]] = {}
    print("\nSTAGE 1 | Precomputing temporal features")

    for exp_name, features in experiments.items():
        exp_cache: dict[int, dict] = {}
        t0 = now()
        for sid, df in sessions.items():
            X, y, gt_segs = make_Xy(df, features)
            exp_cache[sid] = {"X": X, "y": y, "gt_segs": gt_segs}
        cache[exp_name] = exp_cache
        log_timing(STAGE_NAME, exp_name, "precompute_features", now() - t0, {"experiment": exp_name})
        print(f"  {exp_name}: cached {len(exp_cache)} sessions")

    return cache


# ============================================================
# 5) Model Training, Prediction, and Threshold Selection
# ============================================================


def sample_training_rows(
    X: pd.DataFrame,
    y: np.ndarray,
    max_rows: int | None,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, np.ndarray]:
    if max_rows is None or len(y) <= max_rows:
        return X, y

    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]

    half = max_rows // 2
    take_pos = min(len(idx_pos), half)
    take_neg = min(len(idx_neg), max_rows - take_pos)

    sampled_idx = []
    if take_pos:
        sampled_idx.append(rng.choice(idx_pos, size=take_pos, replace=False))
    if take_neg:
        sampled_idx.append(rng.choice(idx_neg, size=take_neg, replace=False))

    idx = np.concatenate(sampled_idx)
    rng.shuffle(idx)
    return X.iloc[idx], y[idx]


def train_model(
    experiment_cache: dict[int, dict],
    train_sids: list[int],
    max_rows_per_session: int | None,
    fold_seed: int,
) -> Pipeline:
    Xs, ys = [], []
    rng = np.random.default_rng(fold_seed)

    for sid in train_sids:
        X = experiment_cache[sid]["X"]
        y = experiment_cache[sid]["y"]
        X_sampled, y_sampled = sample_training_rows(X, y, max_rows_per_session, rng)
        Xs.append(X_sampled)
        ys.append(y_sampled)

    X_train = pd.concat(Xs, axis=0, ignore_index=True)
    y_train = np.concatenate(ys)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=LR_MAX_ITER, class_weight=LR_CLASS_WEIGHT, solver=LR_SOLVER)),
    ])
    model.fit(X_train, y_train)
    return model


def predict_proba_cached(model: Pipeline, experiment_cache: dict[int, dict], sid: int) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    item = experiment_cache[sid]
    prob = model.predict_proba(item["X"])[:, 1]
    return prob, item["y"], item["gt_segs"]


def choose_threshold_on_train(
    model: Pipeline,
    experiment_cache: dict[int, dict],
    train_sids: list[int],
) -> tuple[float, pd.DataFrame]:
    # Heavy model predictions are computed once per training session. The threshold
    # grid then works only on cached probabilities.
    cached_predictions = []
    for sid in train_sids:
        prob, y, gt_segs = predict_proba_cached(model, experiment_cache, sid)
        cached_predictions.append({"sid": sid, "prob": prob, "y": y, "gt_segs": gt_segs})

    rows = []
    for threshold in THRESH_GRID:
        fold_scores = []
        for item in cached_predictions:
            pred_segs, _ = prob_to_segments(item["prob"], threshold=threshold)
            ev, _ = evaluate_segments(pred_segs, item["gt_segs"])
            score = (
                THRESHOLD_OBJECTIVE_IOU_WEIGHT * ev["mean_best_gt_iou"]
                + THRESHOLD_OBJECTIVE_F1_WEIGHT * ev["f1"]
            )
            fold_scores.append(score)

        rows.append({"threshold": threshold, "train_objective": float(np.mean(fold_scores))})

    threshold_df = pd.DataFrame(rows)
    best_threshold = float(threshold_df.sort_values("train_objective", ascending=False).iloc[0]["threshold"])
    return best_threshold, threshold_df


# ============================================================
# 6) Output Table Helpers
# ============================================================


def predicted_segments_rows(exp_name: str, test_sid: int, pred_segs: list[tuple[int, int]]) -> list[dict]:
    return [
        {
            "experiment": exp_name,
            "test_session": test_sid,
            "pred_no": idx,
            "pred_start": start,
            "pred_end": end,
            "length": end - start + 1,
        }
        for idx, (start, end) in enumerate(pred_segs, start=1)
    ]


def matched_segment_rows(
    exp_name: str,
    test_sid: int,
    matches: list[dict],
    pred_segs: list[tuple[int, int]],
    gt_segs: list[tuple[int, int]],
) -> list[dict]:
    rows = []
    for m in matches:
        pred = pred_segs[m["pred_index"]]
        gt = gt_segs[m["gt_index"]]
        rows.append({
            "experiment": exp_name,
            "test_session": test_sid,
            "gt_index": m["gt_index"],
            "gt_start": gt[0],
            "gt_end": gt[1],
            "gt_length": gt[1] - gt[0] + 1,
            "pred_index": m["pred_index"],
            "pred_start": pred[0],
            "pred_end": pred[1],
            "pred_length": pred[1] - pred[0] + 1,
            "iou": m["iou"],
        })
    return rows


def best_match_table_rows(
    exp_name: str,
    test_sid: int,
    pred_segs: list[tuple[int, int]],
    gt_segs: list[tuple[int, int]],
) -> list[dict]:
    rows = []
    for gt_idx, gt in enumerate(gt_segs):
        if pred_segs:
            ious = [segment_iou(pred, gt) for pred in pred_segs]
            best_pred_idx = int(np.argmax(ious))
            pred = pred_segs[best_pred_idx]
            rows.append({
                "experiment": exp_name,
                "test_session": test_sid,
                "gt_index": gt_idx,
                "gt_start": gt[0],
                "gt_end": gt[1],
                "best_pred_index": best_pred_idx,
                "best_pred_start": pred[0],
                "best_pred_end": pred[1],
                "best_iou_for_gt": float(ious[best_pred_idx]),
            })
        else:
            rows.append({
                "experiment": exp_name,
                "test_session": test_sid,
                "gt_index": gt_idx,
                "gt_start": gt[0],
                "gt_end": gt[1],
                "best_pred_index": np.nan,
                "best_pred_start": np.nan,
                "best_pred_end": np.nan,
                "best_iou_for_gt": 0.0,
            })
    return rows


# ============================================================
# 7) Plotting Utilities
# ============================================================


def draw_gt_pred_regions(ax, gt_mask: np.ndarray, pred_mask: np.ndarray) -> None:
    gt_only = gt_mask & ~pred_mask
    pred_only = pred_mask & ~gt_mask
    overlap = gt_mask & pred_mask

    for start, end in contiguous_segments(gt_only):
        ax.axvspan(start, end, color="limegreen", alpha=0.16)
    for start, end in contiguous_segments(pred_only):
        ax.axvspan(start, end, color="orangered", alpha=0.18)
    for start, end in contiguous_segments(overlap):
        ax.axvspan(start, end, color="mediumpurple", alpha=0.28)


def draw_boundaries(ax, gt_segs: list[tuple[int, int]], pred_segs: list[tuple[int, int]]) -> None:
    for start, end in gt_segs:
        ax.axvline(start, color="green", linewidth=2.4)
        ax.axvline(end, color="darkgreen", linewidth=2.4)
    for start, end in pred_segs:
        ax.axvline(start, color="red", linestyle="--", linewidth=1.7)
        ax.axvline(end, color="darkred", linestyle="--", linewidth=1.7)


def plot_fold(
    sessions: dict[int, pd.DataFrame],
    exp_name: str,
    sid: int,
    features: list[str],
    prob_smooth: np.ndarray,
    pred_segs: list[tuple[int, int]],
    gt_segs: list[tuple[int, int]],
    threshold: float,
    out_path: Path,
) -> None:
    df = sessions[sid]
    x = np.arange(len(df))
    gt_mask = segments_to_mask(gt_segs, len(df))
    pred_mask = segments_to_mask(pred_segs, len(df))
    plot_features = [c for c in features if c in df.columns][:MAX_FEATURES_TO_PLOT]

    nrows = len(plot_features) + 2
    fig, axes = plt.subplots(nrows, 1, figsize=(FOLD_PLOT_WIDTH, FOLD_PLOT_ROW_HEIGHT * nrows), sharex=True)

    ax = axes[0]
    draw_gt_pred_regions(ax, gt_mask, pred_mask)
    ax.plot(x, prob_smooth, lw=1.35, label="Smoothed P(production)")
    ax.axhline(threshold, color="black", ls="--", lw=1.2, label=f"thr={threshold:.2f}")
    draw_boundaries(ax, gt_segs, pred_segs)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("P(prod)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")

    for ax, col in zip(axes[1:-1], plot_features):
        draw_gt_pred_regions(ax, gt_mask, pred_mask)
        vals = clean_numeric_series(df[col]).values
        ax.plot(x, vals, lw=0.8)
        draw_boundaries(ax, gt_segs, pred_segs)
        ax.set_ylabel(col, rotation=0, labelpad=60, ha="right", va="center")
        ax.grid(alpha=0.25)

    ax = axes[-1]
    ax.set_ylim(0, 3)
    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_yticklabels(["GT", "Pred", "Overlap"])

    for start, end in gt_segs:
        ax.broken_barh([(start, end - start + 1)], (0.15, 0.7), facecolors="limegreen", alpha=0.40)
    for start, end in pred_segs:
        ax.broken_barh([(start, end - start + 1)], (1.15, 0.7), facecolors="orangered", alpha=0.38)
    for start, end in contiguous_segments(gt_mask & pred_mask):
        ax.broken_barh([(start, end - start + 1)], (2.15, 0.7), facecolors="mediumpurple", alpha=0.55)

    draw_boundaries(ax, gt_segs, pred_segs)
    ax.set_xlabel("Row index / time order")
    ax.grid(alpha=0.25)

    fig.suptitle(
        f"{exp_name} | Test session {sid} | GT={len(gt_segs)} | Pred={len(pred_segs)}",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=FOLD_PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def save_summary_plots(out_root: Path, comparison_df: pd.DataFrame, all_fold_summaries: list[pd.DataFrame]) -> None:
    plot_df = comparison_df[
        ["experiment", "mean_best_gt_iou", "mean_segment_f1", "mean_row_f1", "mean_abs_count_error"]
    ].copy()

    fig, ax = plt.subplots(figsize=COMPARISON_FIGSIZE)
    x = np.arange(len(plot_df))
    width = 0.22
    ax.bar(x - width, plot_df["mean_best_gt_iou"], width=width, label="Mean best GT IoU")
    ax.bar(x, plot_df["mean_segment_f1"], width=width, label="Segment F1")
    ax.bar(x + width, plot_df["mean_row_f1"], width=width, label="Row F1")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["experiment"], rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Feature group experiments — LOO comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_root / "experiment_comparison_barplot.png", dpi=SUMMARY_PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=COUNT_ERROR_FIGSIZE)
    ax.bar(plot_df["experiment"], plot_df["mean_abs_count_error"])
    ax.set_title("Mean absolute segment count error by experiment")
    ax.set_ylabel("Mean |Pred count - GT count|")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_root / "experiment_count_error_barplot.png", dpi=SUMMARY_PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    for summary_df in all_fold_summaries:
        exp_name = str(summary_df["experiment"].iloc[0])
        exp_dir = out_root / exp_name
        fig, ax = plt.subplots(figsize=FOLD_METRICS_FIGSIZE)
        summary_df[["test_session", "mean_best_gt_iou", "f1", "row_f1"]].plot(
            x="test_session", kind="bar", ax=ax
        )
        ax.set_title(f"{exp_name} | LOO fold-level metrics")
        ax.set_xlabel("Test session")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(exp_dir / "fold_metrics.png", dpi=SUMMARY_PLOT_DPI, bbox_inches="tight")
        plt.close(fig)


# ============================================================
# 8) Leave-One-Session-Out Evaluation
# ============================================================


def run_experiment(
    exp_name: str,
    features: list[str],
    sessions: dict[int, pd.DataFrame],
    experiment_cache: dict[int, dict],
    all_sids: list[int],
    out_root: Path,
    save_fold_plots: bool,
) -> pd.DataFrame:
    exp_t0 = now()
    print("\n" + "#" * 100)
    print(f"RUNNING {exp_name}")
    print("Features:", features)

    exp_dir = out_root / exp_name
    plot_dir = exp_dir / "plots"
    exp_dir.mkdir(parents=True, exist_ok=True)
    if save_fold_plots:
        plot_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    boundary_rows = []
    pred_rows = []
    match_rows = []
    best_match_rows = []
    threshold_rows = []

    for test_sid in all_sids:
        train_sids = [sid for sid in all_sids if sid != test_sid]
        fold_label = f"{exp_name}_S{test_sid:02d}"
        fold_t0 = now()
        print(f"  Fold | test={test_sid:02d} | train={train_sids}")

        fold_seed = BASE_RANDOM_SEED + 1000 + test_sid
        with Timer(STAGE_NAME, fold_label, "train_row_model", {"experiment": exp_name, "test_session": test_sid}):
            model = train_model(
                experiment_cache=experiment_cache,
                train_sids=train_sids,
                max_rows_per_session=MAX_ROWS_PER_SESSION,
                fold_seed=fold_seed,
            )

        with Timer(STAGE_NAME, fold_label, "select_threshold_on_cached_train_predictions", {"experiment": exp_name, "test_session": test_sid}):
            best_thr, threshold_df = choose_threshold_on_train(model, experiment_cache, train_sids)

        threshold_df["experiment"] = exp_name
        threshold_df["test_session"] = test_sid
        threshold_rows.append(threshold_df)

        with Timer(STAGE_NAME, fold_label, "predict_and_temporal_postprocess", {"experiment": exp_name, "test_session": test_sid}):
            prob, y, gt_segs = predict_proba_cached(model, experiment_cache, test_sid)
            pred_segs, prob_smooth = prob_to_segments(prob, threshold=best_thr)

        with Timer(STAGE_NAME, fold_label, "evaluate_segments", {"experiment": exp_name, "test_session": test_sid}):
            ev, matches = evaluate_segments(pred_segs, gt_segs)
            berrs = boundary_errors(pred_segs, gt_segs)

        # Row-level metrics are computed from the final post-processed segments, not
        # directly from the smoothed threshold mask. This keeps row-level and
        # segment-level outputs consistent after merge-gap and min-length filtering.
        y_pred_row = segments_to_mask(pred_segs, len(y)).astype(int)

        row_acc = accuracy_score(y, y_pred_row)
        row_f1 = f1_score(y, y_pred_row, zero_division=0)
        row_prec = precision_score(y, y_pred_row, zero_division=0)
        row_rec = recall_score(y, y_pred_row, zero_division=0)
        row_auc = safe_auc(y, prob)

        boundary_df = pd.DataFrame(berrs)
        summary_rows.append({
            "experiment": exp_name,
            "test_session": test_sid,
            "feature_count": len(features),
            "features": ", ".join(features),
            "threshold": best_thr,
            "rows": len(y),
            "gt_segments": len(gt_segs),
            "pred_segments": len(pred_segs),
            "count_error": len(pred_segs) - len(gt_segs),
            "abs_count_error": abs(len(pred_segs) - len(gt_segs)),
            "row_accuracy": row_acc,
            "row_precision": row_prec,
            "row_recall": row_rec,
            "row_f1": row_f1,
            "row_auc": row_auc,
            **ev,
            "gt_intervals": interval_text(gt_segs),
            "pred_intervals": interval_text(pred_segs),
            "mean_start_abs_error": float(boundary_df["start_abs_error"].mean()) if len(boundary_df) else np.nan,
            "mean_end_abs_error": float(boundary_df["end_abs_error"].mean()) if len(boundary_df) else np.nan,
        })

        for row in berrs:
            row.update({"experiment": exp_name, "test_session": test_sid})
            boundary_rows.append(row)

        pred_rows.extend(predicted_segments_rows(exp_name, test_sid, pred_segs))
        match_rows.extend(matched_segment_rows(exp_name, test_sid, matches, pred_segs, gt_segs))
        best_match_rows.extend(best_match_table_rows(exp_name, test_sid, pred_segs, gt_segs))

        if save_fold_plots:
            plot_path = plot_dir / f"{exp_name}_session_{test_sid:02d}.png"
            with Timer(STAGE_NAME, fold_label, "plot_fold", {"experiment": exp_name, "test_session": test_sid}):
                plot_fold(sessions, exp_name, test_sid, features, prob_smooth, pred_segs, gt_segs, best_thr, plot_path)

        log_timing(STAGE_NAME, fold_label, "fold_total", now() - fold_t0, {"experiment": exp_name, "test_session": test_sid})
        print(
            f"    thr={best_thr:.3f} | GT={len(gt_segs)} | Pred={len(pred_segs)} | "
            f"MeanBestIoU={ev['mean_best_gt_iou']:.3f} | SegF1={ev['f1']:.3f} | RowF1={row_f1:.3f}"
        )

    summary_df = pd.DataFrame(summary_rows)
    pd.DataFrame(boundary_rows).to_csv(exp_dir / "boundary_errors.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(exp_dir / "predicted_segments.csv", index=False)
    pd.DataFrame(match_rows).to_csv(exp_dir / "matched_segments.csv", index=False)
    pd.DataFrame(best_match_rows).to_csv(exp_dir / "best_match_per_gt_segment.csv", index=False)
    pd.concat(threshold_rows, ignore_index=True).to_csv(exp_dir / "threshold_selection.csv", index=False)
    summary_df.to_csv(exp_dir / "loo_summary.csv", index=False)

    metric_cols = [
        "row_accuracy",
        "row_precision",
        "row_recall",
        "row_f1",
        "row_auc",
        "precision",
        "recall",
        "f1",
        "matched_mean_iou",
        "mean_best_gt_iou",
        "best_any_iou",
        "mean_start_abs_error",
        "mean_end_abs_error",
        "abs_count_error",
    ]
    overall = summary_df[metric_cols].agg(["mean", "std", "min", "median", "max"]).T.reset_index()
    overall = overall.rename(columns={"index": "metric"})
    overall.to_csv(exp_dir / "overall_metrics.csv", index=False)

    log_timing(STAGE_NAME, exp_name, "experiment_total", now() - exp_t0, {"experiment": exp_name})
    print(f"  Saved experiment outputs to: {exp_dir.resolve()}")
    return summary_df


# ============================================================
# 9) Main Entry Point
# ============================================================


def main() -> None:
    data_dir = resolve_data_dir(DATA_DIR)
    out_root = Path(OUT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)

    print("STAGE 1 | Production localisation | Configuration")
    print("DATA_DIR:", data_dir.resolve())
    print("OUT_ROOT:", out_root.resolve())
    print("Stage:", STAGE_NAME)
    print("SAVE_FOLD_PLOTS:", SAVE_FOLD_PLOTS)
    print("SAVE_SUMMARY_PLOTS:", SAVE_SUMMARY_PLOTS)
    print("SAVE_TIMING_LOG:", SAVE_TIMING_LOG)
    print("Experiments:")
    for experiment_name, experiment_features in EXPERIMENTS.items():
        print(f"  {experiment_name}: {experiment_features}")

    sessions = load_sessions(data_dir)
    all_sids = sorted(sessions.keys())
    preprocessed_sessions = precompute_experiment_features(sessions, EXPERIMENTS)

    print("\nSTAGE 1 | Running production feature-group experiments")
    all_experiment_summaries = []
    all_fold_summaries = []

    for exp_name, features in EXPERIMENTS.items():
        summary_df = run_experiment(
            exp_name=exp_name,
            features=features,
            sessions=sessions,
            experiment_cache=preprocessed_sessions[exp_name],
            all_sids=all_sids,
            out_root=out_root,
            save_fold_plots=SAVE_FOLD_PLOTS,
        )

        all_fold_summaries.append(summary_df)
        all_experiment_summaries.append({
            "experiment": exp_name,
            "feature_count": len(features),
            "features": ", ".join(features),
            "mean_row_f1": summary_df["row_f1"].mean(),
            "mean_segment_f1": summary_df["f1"].mean(),
            "mean_best_gt_iou": summary_df["mean_best_gt_iou"].mean(),
            "mean_best_any_iou": summary_df["best_any_iou"].mean(),
            "mean_abs_count_error": summary_df["abs_count_error"].mean(),
            "mean_start_abs_error": summary_df["mean_start_abs_error"].mean(),
            "mean_end_abs_error": summary_df["mean_end_abs_error"].mean(),
        })

    print("\nSTAGE 1 | Preparing production comparison summary")
    comparison_df = pd.DataFrame(all_experiment_summaries)
    comparison_df = comparison_df.sort_values("mean_best_gt_iou", ascending=False).reset_index(drop=True)
    comparison_df.to_csv(out_root / "experiment_comparison_summary.csv", index=False)

    all_folds_df = pd.concat(all_fold_summaries, ignore_index=True)
    all_folds_df.to_csv(out_root / "all_experiments_fold_results.csv", index=False)

    if SAVE_SUMMARY_PLOTS:
        save_summary_plots(out_root, comparison_df, all_fold_summaries)

    if SAVE_TIMING_LOG:
        pd.DataFrame(TIMING_ROWS).to_csv(out_root / "runtime_timing_log.csv", index=False)

    print("\nEXPERIMENT COMPARISON SUMMARY:")
    print(comparison_df.round(4).to_string(index=False))

    print("\nALL FOLD RESULTS:")
    cols = [
        "experiment",
        "test_session",
        "gt_segments",
        "pred_segments",
        "mean_best_gt_iou",
        "f1",
        "row_f1",
        "abs_count_error",
        "gt_intervals",
        "pred_intervals",
    ]
    print(all_folds_df[cols].round(4).to_string(index=False))

    print("\nStage 1 completed.")
    print("Output directory:", out_root.resolve())
    print("Main output files:")
    print(" - experiment_comparison_summary.csv")
    print(" - all_experiments_fold_results.csv")
    if SAVE_SUMMARY_PLOTS:
        print(" - experiment_comparison_barplot.png")
        print(" - experiment_count_error_barplot.png")
    if SAVE_TIMING_LOG:
        print(" - runtime_timing_log.csv")
    print(" - Per-experiment outputs: loo_summary.csv, overall_metrics.csv, matched_segments.csv, best_match_per_gt_segment.csv")


if __name__ == "__main__":
    main()
