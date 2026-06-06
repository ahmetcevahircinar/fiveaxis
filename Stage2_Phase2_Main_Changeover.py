# ============================================================
# Stage 2 Phase 2 main changeover localisation after production-row removal
# Simplified hierarchical contextualisation pipeline.
# ============================================================

from pathlib import Path
import re
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

try:
    from IPython.display import display
except Exception:
    display = print


DATA_DIR_CANDIDATES = [
    Path("fiveaxis"),
    Path("."),
    Path("/mnt/data")
]

DATA_DIR = None
for p in DATA_DIR_CANDIDATES:
    if p.exists() and len(list(p.glob("Data_from_*.csv"))) > 0:
        DATA_DIR = p
        break

if DATA_DIR is None:
    raise FileNotFoundError("Data_from_*.csv files were not found. Please check DATA_DIR.")

OUT_ROOT = Path("hierarchical_contextualisation_outputs") / "stage2_phase2_main_changeover"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

TARGET_SESSIONS = [1, 2, 13]
LABEL_COL = "Phase_compressed"

TIME_COL_CANDIDATES = [
    "TIME_IN_MS", "DateTime", "Timestamp", "Time", "time", "Datetime", "timestamp"
]

PRODUCTION_PHASE = 5
TARGET_PHASE = 2
VALID_PHASES_AFTER_PRODUCTION_REMOVAL = [1, 2, 3, 4, 6]


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


BRANCHES = {
    "program": PROGRAM_RELATED_FEATURES,
    "physical": PHYSICAL_PROCESS_FEATURES,
    "program_physical": PROGRAM_RELATED_FEATURES + PHYSICAL_PROCESS_FEATURES,
}


COMMON_CANDIDATE_CONFIG = {
    "thresholds": [0.50, 0.35],
    "smooth_windows": [51],
    "merge_gaps": [40],
    "min_blob_lens": [20],
    "peak_quantiles": [0.88],
}



ROLL_WINDOWS = [15, 51]


MIN_SUBSEG_LEN_FALLBACK = 18
DURATION_TOLERANCE_LOW = 0.45
DURATION_TOLERANCE_HIGH = 1.85


CENTER_CLUSTER_FACTOR = 0.70
VOTE_RADIUS_FACTOR = 0.35
FINAL_NMS_IOU = 0.55


# First NMS removes near-duplicate intervals before clustering.
CANDIDATE_NMS_IOU = 0.88
# Final NMS selects non-overlapping output segments after cluster ranking.


TARGET_COUNT_LOW_MULTIPLIER = 0.75
TARGET_COUNT_HIGH_MULTIPLIER = 1.55


ENTROPY_GAIN_WEIGHT = 0.25
MIN_SPLIT_SIGNAL_SCORE = 0.40


# Lightweight local refinement for improving boundary placement.
BOUNDARY_REFINEMENT_RADIUS = 50
REFINE_W_ENTROPY_GAIN = 0.45
REFINE_W_BOUNDARY_SUPPORT = 0.35
REFINE_W_INTERNAL_PROB = 0.20


RANDOM_STATE = 42




print("\nSTAGE 2 | Phase 2 main changeover localisation | Configuration")
print("DATA_DIR:", DATA_DIR.resolve())
print("OUT_ROOT:", OUT_ROOT.resolve())
print("Target: Phase 2 Main changeover after removing Phase 5 production")
# PocketTable is intentionally excluded from the feature definitions.
print("Final publishable method with entropy-aware cluster scoring, threshold-expanded candidate generation, and lightweight boundary refinement")
print("Branches:", list(BRANCHES.keys()))


def detect_session_no(path: Path):
    name = path.name.lower()

    exact_map = {
        "data_from_2021-11-26.csv": 1,
        "data_from_2021-12-07.csv": 2,
        "data_from_2022-04.csv": 13,
    }

    if name in exact_map:
        return exact_map[name]

    m = re.search(r"(session|sess|s)[_\- ]*(1|2|13)\b", name)
    if m:
        return int(m.group(2))

    return None


def find_session_files(data_dir):
    csvs = sorted(data_dir.glob("Data_from_*.csv"))
    found = {}

    for p in csvs:
        sid = detect_session_no(p)
        if sid in TARGET_SESSIONS:
            found[sid] = p

    missing = [sid for sid in TARGET_SESSIONS if sid not in found]
    if missing:
        raise FileNotFoundError(
            f"Missing session files: {missing}\n"
            f"Search directory: {data_dir.resolve()}\n"
            f"Available files:\n" + "\n".join([p.name for p in csvs])
        )

    return found


def detect_time_col(df):
    for c in TIME_COL_CANDIDATES:
        if c in df.columns:
            return c
    return df.columns[0]


def contiguous_segments(mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    padded = np.r_[False, mask, False]
    changes = np.diff(padded.astype(np.int8))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0] - 1
    return list(zip(starts.astype(int).tolist(), ends.astype(int).tolist()))


def merge_close_segments(segs, gap=0):
    if not segs:
        return []

    segs = sorted(segs)
    merged = [list(segs[0])]

    for s, e in segs[1:]:
        if s - merged[-1][1] - 1 <= gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    return [tuple(x) for x in merged]


def filter_short_segments(segs, min_len=1):
    return [(s, e) for s, e in segs if (e - s + 1) >= min_len]


def segments_to_mask(segs, n):
    mask = np.zeros(n, dtype=bool)

    for s, e in segs:
        s = max(0, int(s))
        e = min(n - 1, int(e))
        if e >= s:
            mask[s:e+1] = True

    return mask


def segment_iou(a, b):
    """Compute the standard intersection-over-union for two inclusive 1D intervals."""
    s1, e1 = map(int, a)
    s2, e2 = map(int, b)

    if e1 < s1 or e2 < s2:
        return 0.0

    inter = max(0, min(e1, e2) - max(s1, s2) + 1)
    len_a = e1 - s1 + 1
    len_b = e2 - s2 + 1
    union = len_a + len_b - inter

    return float(inter / union) if union > 0 else 0.0


def interval_text(segs):
    if len(segs) == 0:
        return "-"
    return " | ".join([f"{s}-{e}" for s, e in segs])


def match_segments(pred_segs, gt_segs, iou_threshold=0.1):
    pairs = []

    for pi, p in enumerate(pred_segs):
        for gi, g in enumerate(gt_segs):
            pairs.append((segment_iou(p, g), pi, gi))

    pairs.sort(reverse=True)

    used_p = set()
    used_g = set()
    matches = []

    for iou, pi, gi in pairs:
        if iou < iou_threshold:
            break

        if pi not in used_p and gi not in used_g:
            used_p.add(pi)
            used_g.add(gi)
            matches.append({
                "pred_index": pi,
                "gt_index": gi,
                "iou": float(iou)
            })

    return matches


def best_iou_summary(pred_segs, gt_segs):
    if not gt_segs and not pred_segs:
        return {"mean_best_gt_iou": np.nan, "best_any_iou": np.nan}

    if not gt_segs or not pred_segs:
        return {"mean_best_gt_iou": 0.0, "best_any_iou": 0.0}

    best_per_gt = []
    for g in gt_segs:
        best_per_gt.append(max(segment_iou(p, g) for p in pred_segs))

    best_any = max((segment_iou(p, g) for p in pred_segs for g in gt_segs), default=0.0)

    return {
        "mean_best_gt_iou": float(np.mean(best_per_gt)),
        "best_any_iou": float(best_any)
    }


def evaluate_segments(pred_segs, gt_segs, iou_thr=0.10):
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
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "matched_mean_iou": float(matched_mean_iou),
        "n_pred": len(pred_segs),
        "n_gt": len(gt_segs)
    }

    out.update(best_iou_summary(pred_segs, gt_segs))

    return out, matches


def boundary_errors(pred_segs, gt_segs):
    rows = []

    for gi, g in enumerate(gt_segs):
        if pred_segs:
            ious = [segment_iou(p, g) for p in pred_segs]
            bi = int(np.argmax(ious))
            p = pred_segs[bi]

            rows.append({
                "gt_index": gi + 1,
                "gt_start_removed": g[0],
                "gt_end_removed": g[1],
                "pred_start_removed": p[0],
                "pred_end_removed": p[1],
                "iou": float(ious[bi]),
                "start_abs_error_removed_rows": abs(p[0] - g[0]),
                "end_abs_error_removed_rows": abs(p[1] - g[1])
            })
        else:
            rows.append({
                "gt_index": gi + 1,
                "gt_start_removed": g[0],
                "gt_end_removed": g[1],
                "pred_start_removed": np.nan,
                "pred_end_removed": np.nan,
                "iou": 0.0,
                "start_abs_error_removed_rows": np.nan,
                "end_abs_error_removed_rows": np.nan
            })

    return rows


def smooth_array(arr, window):
    if window <= 1:
        return np.asarray(arr)

    return (
        pd.Series(arr)
        .rolling(window, center=True, min_periods=1)
        .mean()
        .values
    )


def normalize_vector(v):
    v = np.asarray(v, dtype=float)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

    mn = np.min(v)
    mx = np.max(v)

    if mx <= mn:
        return np.zeros_like(v)

    return (v - mn) / (mx - mn)


print("\nSTAGE 2 | Loading non-production sessions")

files = find_session_files(DATA_DIR)

sessions = {}

for sid in TARGET_SESSIONS:
    f = files[sid]
    df = pd.read_csv(f).copy()
    df.columns = [c.strip() for c in df.columns]

    if LABEL_COL not in df.columns:
        raise ValueError(f"{f.name} does not contain the required column: {LABEL_COL}")

    time_col = detect_time_col(df)
    df["_time"] = pd.to_datetime(df[time_col], errors="coerce")
    df["_original_row"] = np.arange(len(df))

    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce")
    df = df.dropna(subset=[LABEL_COL]).copy()
    df[LABEL_COL] = df[LABEL_COL].astype(int)


    nonprod = df[df[LABEL_COL].isin(VALID_PHASES_AFTER_PRODUCTION_REMOVAL)].copy()
    nonprod = nonprod.reset_index(drop=True)
    nonprod["_removed_row"] = np.arange(len(nonprod))
    nonprod["target_phase2"] = (nonprod[LABEL_COL] == TARGET_PHASE).astype(int)

    sessions[sid] = nonprod

    gt_mask = nonprod["target_phase2"].values.astype(bool)
    gt_segs = contiguous_segments(gt_mask)

    print(
        f"  Session {sid:02d} | file={f.name} | raw={len(df)} | nonprod={len(nonprod)} | "
        f"phase2_segments={len(gt_segs)} | phase2_ratio={np.mean(gt_mask):.3f}"
    )


def prepare_base_features(df, features):
    cols = {}
    for c in features:
        if c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce")
        else:
            vals = pd.Series(0.0, index=df.index)
        cols[c] = vals.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
    return pd.DataFrame(cols, index=df.index)


def add_temporal_features(X):
    frames = [X.copy()]
    diff = X.diff().fillna(0)
    absdiff = X.diff().abs().fillna(0)

    diff1 = diff.copy()
    diff1.columns = [f"{c}_diff1" for c in X.columns]
    absdiff1 = absdiff.copy()
    absdiff1.columns = [f"{c}_absdiff1" for c in X.columns]
    frames.extend([diff1, absdiff1])

    for w in ROLL_WINDOWS:
        means = X.rolling(w, center=True, min_periods=1).mean()
        means.columns = [f"{c}_mean_{w}" for c in X.columns]

        stds = X.rolling(w, center=True, min_periods=1).std().fillna(0)
        stds.columns = [f"{c}_std_{w}" for c in X.columns]

        diff_means = absdiff.rolling(w, center=True, min_periods=1).mean()
        diff_means.columns = [f"{c}_absdiff_mean_{w}" for c in X.columns]

        frames.extend([means, stds, diff_means])

    out = pd.concat(frames, axis=1)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0)


def make_Xy(df, features):
    X0 = prepare_base_features(df, features)
    X = add_temporal_features(X0)
    y = df["target_phase2"].astype(int).values
    return X, y



def train_model(train_sids, features, random_state=42):
    Xs = []
    ys = []

    for sid in train_sids:
        X, y = make_Xy(sessions[sid], features)
        Xs.append(X)
        ys.append(y)

    X_train = pd.concat(Xs, axis=0, ignore_index=True)
    y_train = np.concatenate(ys)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            random_state=random_state
        ))
    ])

    model.fit(X_train, y_train)
    return model


def predict_proba_session(model, sid, features):
    X, y = make_Xy(sessions[sid], features)
    prob = model.predict_proba(X)[:, 1]
    return prob, y


def get_gt_phase2_segments_for_sid(sid):
    y = sessions[sid]["target_phase2"].values.astype(bool)
    return contiguous_segments(y)


def learn_duration_prior(train_sids):
    lengths = []

    for sid in train_sids:
        for s, e in get_gt_phase2_segments_for_sid(sid):
            lengths.append(e - s + 1)

    if len(lengths) == 0:
        lengths = [MIN_SUBSEG_LEN_FALLBACK, 80, 160]

    lengths = np.asarray(lengths, dtype=float)

    prior = {
        "n_train_segments": int(len(lengths)),
        "min_len": int(max(1, np.percentile(lengths, 5))),
        "p10_len": int(max(1, np.percentile(lengths, 10))),
        "p25_len": int(max(1, np.percentile(lengths, 25))),
        "median_len": int(max(1, np.percentile(lengths, 50))),
        "p75_len": int(max(1, np.percentile(lengths, 75))),
        "p90_len": int(max(1, np.percentile(lengths, 90))),
        "p95_len": int(max(1, np.percentile(lengths, 95))),
        "mean_len": float(np.mean(lengths)),
        "std_len": float(np.std(lengths)),
        "mean_segment_count": float(np.mean([len(get_gt_phase2_segments_for_sid(sid)) for sid in train_sids])),
    }

    prior["min_allowed_len"] = int(max(MIN_SUBSEG_LEN_FALLBACK, np.floor(prior["p10_len"] * DURATION_TOLERANCE_LOW)))
    prior["max_allowed_len"] = int(max(prior["median_len"] + 1, np.ceil(prior["p90_len"] * DURATION_TOLERANCE_HIGH)))

    return prior


def prob_to_initial_blobs(prob, threshold, smooth_window, merge_gap, min_blob_len):
    prob_s = smooth_array(prob, window=smooth_window)
    raw = contiguous_segments(prob_s >= threshold)
    merged = merge_close_segments(raw, gap=merge_gap)
    filtered = filter_short_segments(merged, min_len=min_blob_len)
    return filtered, prob_s


def build_branch_boundary_signal(df, branch_name):
    if branch_name == "program":
        feats = PROGRAM_RELATED_FEATURES
        jump_weight = 0.65
    elif branch_name == "physical":
        feats = PHYSICAL_PROCESS_FEATURES
        jump_weight = 0.25
    else:
        feats = PROGRAM_RELATED_FEATURES + PHYSICAL_PROCESS_FEATURES
        jump_weight = 0.45

    score = np.zeros(len(df), dtype=float)

    for c in feats:
        if c not in df.columns:
            continue

        v = (
            pd.to_numeric(df[c], errors="coerce")
            .ffill()
            .bfill()
            .fillna(0)
            .values
            .astype(float)
        )

        d = np.abs(np.diff(v, prepend=v[0]))
        d_s = smooth_array(d, 9)
        jumps = np.r_[0, (v[1:] != v[:-1]).astype(float)]

        score += (1.0 - jump_weight) * normalize_vector(d_s)
        score += jump_weight * normalize_vector(jumps)

    score = smooth_array(score, 7)
    score = normalize_vector(score)

    return score


def build_global_boundary_signals(df):
    program = build_branch_boundary_signal(df, "program")
    physical = build_branch_boundary_signal(df, "physical")
    combined = normalize_vector(0.55 * program + 0.45 * physical)
    return combined, program, physical


def candidate_split_points(blob, split_signal, duration_prior, peak_quantile):
    max_splits_per_blob = 30
    s, e = blob
    length = e - s + 1

    if length < duration_prior["max_allowed_len"]:
        return []

    local_score = split_signal[s:e+1]
    q = np.quantile(local_score, peak_quantile)
    local_candidates = np.where(local_score >= q)[0].tolist()

    expected_count = max(1, int(round(length / max(1, duration_prior["median_len"]))))
    expected_count = max(1, min(expected_count, max_splits_per_blob + 1))

    median_len = max(1, duration_prior["median_len"])
    anchors = [int(round(length * k / expected_count)) for k in range(1, expected_count)] if expected_count > 1 else []

    for a in anchors:
        lo = max(0, a - median_len // 2)
        hi = min(length - 1, a + median_len // 2)
        if hi > lo:
            local_best = lo + int(np.argmax(local_score[lo:hi+1]))
            local_candidates.append(local_best)

    local_candidates = sorted(set(local_candidates))

    min_allowed = duration_prior["min_allowed_len"]
    scored = []

    for loc in local_candidates:
        idx = s + loc

        if idx - s + 1 < min_allowed:
            continue
        if e - idx < min_allowed:
            continue

        scored.append((float(local_score[loc]), idx))

    scored.sort(reverse=True)
    return scored


def split_blob_duration_aware(blob, split_candidates, duration_prior):
    max_splits_per_blob = 30
    count_prior_tolerance = 0.35
    s, e = blob

    if e - s + 1 <= duration_prior["max_allowed_len"]:
        return [blob], []

    selected = []
    candidate_pool = split_candidates[:max_splits_per_blob * 3]

    def make_pieces(sorted_points):
        if not sorted_points:
            return [blob]

        out = []
        cur = s
        for sp in sorted_points:
            out.append((cur, sp))
            cur = sp + 1
        if cur <= e:
            out.append((cur, e))
        return out

    desired_count = max(1, int(round((e - s + 1) / max(1, duration_prior["median_len"]))))
    allowed_count = int(max(desired_count + 1, np.ceil(desired_count * (1 + count_prior_tolerance))))

    for cand_score, idx in candidate_pool:
        if len(selected) >= max_splits_per_blob:
            break

        if any(abs(idx - old) < duration_prior["min_allowed_len"] for old in selected):
            continue

        trial = selected + [idx]
        trial.sort()
        pieces = make_pieces(trial)
        lengths = [ee - ss + 1 for ss, ee in pieces]

        if min(lengths) < duration_prior["min_allowed_len"]:
            continue
        if len(pieces) > allowed_count:
            continue


        if cand_score >= MIN_SPLIT_SIGNAL_SCORE:
            selected = trial

    pieces = make_pieces(selected)
    pieces = merge_too_short_pieces(pieces, duration_prior["min_allowed_len"])
    return pieces, selected


def merge_too_short_pieces(pieces, min_len):
    if not pieces:
        return []

    pieces = [list(p) for p in sorted(pieces)]
    changed = True

    while changed and len(pieces) > 1:
        changed = False

        for i, (s, e) in enumerate(pieces):
            if e - s + 1 >= min_len:
                continue

            if i == 0:
                pieces[1][0] = s
                pieces.pop(i)
            elif i == len(pieces) - 1:
                pieces[i-1][1] = e
                pieces.pop(i)
            else:
                left_len = pieces[i-1][1] - pieces[i-1][0] + 1
                right_len = pieces[i+1][1] - pieces[i+1][0] + 1

                if left_len <= right_len:
                    pieces[i-1][1] = e
                    pieces.pop(i)
                else:
                    pieces[i+1][0] = s
                    pieces.pop(i)

            changed = True
            break

    return [tuple(p) for p in pieces]


def segment_confidence(seg, prob_s, boundary_signal, duration_prior):
    s, e = seg
    L = e - s + 1

    p_score = float(np.mean(prob_s[s:e+1])) if e >= s else 0.0
    b_start = boundary_signal[s] if 0 <= s < len(boundary_signal) else 0.0
    b_end = boundary_signal[e] if 0 <= e < len(boundary_signal) else 0.0
    b_score = 0.5 * b_start + 0.5 * b_end

    median_len = max(1, duration_prior["median_len"])
    length_penalty = min(1.0, abs(L - median_len) / max(1, median_len))

    conf = 0.58 * p_score + 0.32 * b_score + 0.10 * (1.0 - length_penalty)
    return float(conf)


def generate_branch_candidates(branch_name, prob, df, duration_prior):
    cfg = COMMON_CANDIDATE_CONFIG
    candidates = []

    boundary_signal = build_branch_boundary_signal(df, branch_name)

    for thr in cfg["thresholds"]:
        for sw in cfg["smooth_windows"]:
            for mg in cfg["merge_gaps"]:
                for min_blob in cfg["min_blob_lens"]:
                    init_blobs, prob_s = prob_to_initial_blobs(
                        prob,
                        threshold=thr,
                        smooth_window=sw,
                        merge_gap=mg,
                        min_blob_len=min_blob
                    )

                    for pq in cfg["peak_quantiles"]:
                        raw_segs = []

                        for blob in init_blobs:
                            split_candidates = candidate_split_points(
                                blob,
                                boundary_signal,
                                duration_prior,
                                pq
                            )
                            pieces, split_points = split_blob_duration_aware(
                                blob,
                                split_candidates,
                                duration_prior
                            )
                            raw_segs.extend(pieces)

                        for s, e in raw_segs:
                            conf = segment_confidence(
                                (s, e),
                                prob_s,
                                boundary_signal,
                                duration_prior
                            )

                            candidates.append({
                                "branch": branch_name,
                                "start": int(s),
                                "end": int(e),
                                "length": int(e - s + 1),
                                "center": float((s + e) / 2.0),
                                "confidence": conf,
                                "threshold": float(thr),
                                "smooth_window": int(sw),
                                "merge_gap": int(mg),
                                "min_blob_len": int(min_blob),
                                "peak_quantile": float(pq),
                            })

    return candidates


def add_candidate_nms_scores(candidates, duration_prior):
    if not candidates:
        return []

    confs = np.asarray([float(c.get("confidence", 0.0)) for c in candidates], dtype=float)
    conf_norm = normalize_vector(confs)
    med = max(1, duration_prior["median_len"])

    out = []

    for i, c in enumerate(candidates):
        item = dict(c)
        L = max(1, int(item["end"]) - int(item["start"]) + 1)
        duration_fit = 1.0 / (1.0 + abs(L - med) / med)

        item["_nms_conf_norm"] = float(conf_norm[i])
        item["_nms_duration_fit"] = float(duration_fit)
        item["_nms_score"] = float(0.78 * conf_norm[i] + 0.22 * duration_fit)
        out.append(item)

    return out


def nms_candidates(items, iou_thr, max_keep=None):
    if not items:
        return []

    ordered = sorted(items, key=lambda x: float(x.get("_nms_score", x.get("confidence", 0.0))), reverse=True)
    kept = []

    for item in ordered:
        seg = (int(item["start"]), int(item["end"]))

        if any(segment_iou(seg, (int(k["start"]), int(k["end"]))) >= iou_thr for k in kept):
            continue

        kept.append(item)

        if max_keep is not None and len(kept) >= max_keep:
            break

    return kept


def remove_near_duplicate_candidates(candidates, duration_prior):
    if not candidates:
        return candidates, {
            "candidate_count_before_nms": len(candidates),
            "candidate_count_after_nms": len(candidates),
            "candidate_nms_keep_ratio": 1.0,
        }

    scored = add_candidate_nms_scores(candidates, duration_prior)
    before = len(scored)
    kept = nms_candidates(scored, CANDIDATE_NMS_IOU)
    kept = sorted(kept, key=lambda x: int(x["start"]))

    return kept, {
        "candidate_count_before_nms": int(before),
        "candidate_count_after_nms": int(len(kept)),
        "candidate_nms_keep_ratio": float(len(kept) / max(1, before)),
    }


def gaussian_vote(arr, center, weight, radius):
    n = len(arr)
    radius = max(1, int(radius))
    lo = max(0, int(round(center)) - radius)
    hi = min(n - 1, int(round(center)) + radius)

    xs = np.arange(lo, hi + 1)
    sigma = max(1.0, radius / 2.0)
    vals = weight * np.exp(-0.5 * ((xs - center) / sigma) ** 2)
    arr[lo:hi+1] += vals


def cluster_candidates(candidates, duration_prior):
    if not candidates:
        return []

    cands = sorted(candidates, key=lambda x: x["center"])
    radius = max(8, int(duration_prior["median_len"] * CENTER_CLUSTER_FACTOR))

    clusters = []
    current = [cands[0]]

    for c in cands[1:]:
        cur_center = np.mean([x["center"] for x in current])

        if abs(c["center"] - cur_center) <= radius:
            current.append(c)
        else:
            clusters.append(current)
            current = [c]

    if current:
        clusters.append(current)

    cluster_rows = []

    for idx, members in enumerate(clusters, start=1):
        starts = np.array([m["start"] for m in members])
        ends = np.array([m["end"] for m in members])
        centers = np.array([m["center"] for m in members])
        confs = np.array([m["confidence"] for m in members])

        branches = sorted(set(m["branch"] for m in members))
        diversity = len(branches)

        total_conf = float(np.sum(confs))
        mean_conf = float(np.mean(confs))


        score = total_conf * (1.0 + 0.12 * (diversity - 1)) + 0.15 * mean_conf

        cluster_rows.append({
            "cluster_id": idx,
            "members": members,
            "n_members": len(members),
            "branches": ",".join(branches),
            "branch_diversity": diversity,
            "center_mean": float(np.average(centers, weights=np.maximum(confs, 1e-6))),
            "start_median": int(np.median(starts)),
            "end_median": int(np.median(ends)),
            "score": float(score),
            "mean_confidence": mean_conf,
            "total_confidence": total_conf,
        })

    return cluster_rows


def dense_boundary_voting_for_cluster(cluster, n, duration_prior, global_boundary_signal):
    members = cluster["members"]

    start_votes = np.zeros(n, dtype=float)
    end_votes = np.zeros(n, dtype=float)

    radius = max(5, int(duration_prior["median_len"] * VOTE_RADIUS_FACTOR))

    for m in members:
        w = max(1e-6, float(m["confidence"]) ** 2)
        gaussian_vote(start_votes, m["start"], w, radius)
        gaussian_vote(end_votes, m["end"], w, radius)


    boundary_norm = np.clip(np.asarray(global_boundary_signal, dtype=float), 0.0, 1.0)
    start_votes = normalize_vector(start_votes) * 0.72 + boundary_norm * 0.28
    end_votes = normalize_vector(end_votes) * 0.72 + boundary_norm * 0.28

    s_med = int(cluster["start_median"])
    e_med = int(cluster["end_median"])

    search_radius = max(radius, int(duration_prior["median_len"] * 0.75))

    s_lo = max(0, s_med - search_radius)
    s_hi = min(n - 1, s_med + search_radius)

    e_lo = max(0, e_med - search_radius)
    e_hi = min(n - 1, e_med + search_radius)

    s_new = int(s_lo + np.argmax(start_votes[s_lo:s_hi+1]))
    e_new = int(e_lo + np.argmax(end_votes[e_lo:e_hi+1]))

    if e_new <= s_new:
        s_new = s_med
        e_new = e_med

    min_len = duration_prior["min_allowed_len"]
    max_len = duration_prior["max_allowed_len"]

    L = e_new - s_new + 1

    if L < min_len:
        center = int(round(cluster["center_mean"]))
        half = min_len // 2
        s_new = max(0, center - half)
        e_new = min(n - 1, s_new + min_len - 1)

    L = e_new - s_new + 1

    if L > max_len:
        center = int(round(cluster["center_mean"]))
        half = max_len // 2
        s_new = max(0, center - half)
        e_new = min(n - 1, s_new + max_len - 1)

    return int(s_new), int(e_new), start_votes, end_votes


def probability_support_track(probs):
    branch_order = ["program", "physical", "program_physical"]
    weights = np.asarray([0.40, 0.25, 0.35], dtype=float)
    weights = weights / weights.sum()

    tracks = [normalize_vector(np.asarray(probs[b], dtype=float)) for b in branch_order]
    out = np.zeros_like(tracks[0], dtype=float)

    for w, t in zip(weights, tracks):
        out += w * t

    return normalize_vector(smooth_array(out, 21))


def cluster_to_candidate_segment(cluster, n, duration_prior, global_boundary_signal):
    s, e, sv, ev = dense_boundary_voting_for_cluster(cluster, n=n, duration_prior=duration_prior, global_boundary_signal=global_boundary_signal)
    return (int(s), int(e)), sv, ev



def binary_entropy_from_mean(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0
    p = float(np.clip(np.nanmean(values), 1e-9, 1.0 - 1e-9))
    return float(-(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p)))


def segment_entropy_gain(seg, prob_track, duration_prior):
    """Estimate how strongly a candidate interval separates its local context.

    The score follows the information-gain idea used in entropy-based time-series
    segmentation: a useful candidate should reduce uncertainty when the local
    context is split into inside-candidate and outside-candidate regions.
    """
    if prob_track is None:
        return 0.0

    n = len(prob_track)
    if n == 0:
        return 0.0

    s, e = seg
    s = max(0, min(n - 1, int(s)))
    e = max(s, min(n - 1, int(e)))

    context_radius = max(10, int(round(0.50 * max(1, duration_prior["median_len"]))))
    lo = max(0, s - context_radius)
    hi = min(n - 1, e + context_radius)

    local = np.asarray(prob_track[lo:hi + 1], dtype=float)
    if local.size < 3:
        return 0.0

    inside = np.asarray(prob_track[s:e + 1], dtype=float)
    outside_parts = []
    if lo < s:
        outside_parts.append(np.asarray(prob_track[lo:s], dtype=float))
    if e + 1 <= hi:
        outside_parts.append(np.asarray(prob_track[e + 1:hi + 1], dtype=float))

    if not outside_parts:
        return 0.0

    outside = np.concatenate(outside_parts)
    if inside.size == 0 or outside.size == 0:
        return 0.0

    h_all = binary_entropy_from_mean(local)
    h_inside = binary_entropy_from_mean(inside)
    h_outside = binary_entropy_from_mean(outside)

    weighted_after = (
        (inside.size / local.size) * h_inside
        + (outside.size / local.size) * h_outside
    )

    gain = h_all - weighted_after
    return float(max(0.0, gain))

def cluster_feature_row(cluster, seg, duration_prior, global_boundary_signal, prob_track, cached_metrics=None):
    cached_metrics = cached_metrics or {}
    s, e = seg
    n = len(global_boundary_signal)
    s = max(0, min(n - 1, int(s)))
    e = max(s, min(n - 1, int(e)))
    L = e - s + 1
    med = max(1, duration_prior["median_len"])
    duration_fit = 1.0 / (1.0 + abs(L - med) / med)
    boundary_support = float(cached_metrics.get(
        "boundary_support",
        0.50 * global_boundary_signal[s] + 0.50 * global_boundary_signal[e]
    ))
    internal_boundary = float(np.mean(global_boundary_signal[s:e+1])) if e >= s else 0.0

    if prob_track is None:
        prob_support = 0.0
        internal_prob = 0.0
    else:
        prob_support = float(0.50 * prob_track[s] + 0.50 * prob_track[e])
        internal_prob = float(cached_metrics.get(
            "internal_prob",
            np.mean(prob_track[s:e+1]) if e >= s else 0.0
        ))

    entropy_gain = float(cached_metrics.get(
        "entropy_gain",
        segment_entropy_gain((s, e), prob_track, duration_prior)
    ))

    return {
        "cluster_id": int(cluster["cluster_id"]),
        "start": s,
        "end": e,
        "length": L,
        "center_mean": float(cluster.get("center_mean", (s + e) / 2)),
        "cluster_score": float(cluster.get("score", 0.0)),
        "n_members": float(cluster.get("n_members", 1)),
        "branch_diversity": float(cluster.get("branch_diversity", 1)),
        "mean_confidence": float(cluster.get("mean_confidence", 0.0)),
        "total_confidence": float(cluster.get("total_confidence", 0.0)),
        "duration_fit": float(duration_fit),
        "boundary_support": boundary_support,
        "internal_boundary": internal_boundary,
        "prob_support": prob_support,
        "internal_prob": internal_prob,
        "entropy_gain": entropy_gain,
        "branches": cluster.get("branches", ""),
    }


def refine_segment_boundaries(seg, duration_prior, global_boundary_signal, prob_track):
    """Refine interval boundaries using a small local search.

    The refinement keeps the compact pipeline intact: the initial interval is
    still produced by cluster-level boundary voting. Around this interval, a
    limited search is performed and the best pair is selected using entropy
    gain, boundary evidence, and internal probability support.
    """
    n = len(global_boundary_signal)
    if n == 0:
        return tuple(map(int, seg)), {}

    s0, e0 = int(seg[0]), int(seg[1])
    s0 = max(0, min(n - 1, s0))
    e0 = max(s0, min(n - 1, e0))

    radius = max(5, int(BOUNDARY_REFINEMENT_RADIUS))
    s_candidates = range(max(0, s0 - radius), min(n - 1, s0 + radius) + 1)
    e_candidates = range(max(0, e0 - radius), min(n - 1, e0 + radius) + 1)

    min_len = max(1, int(duration_prior["min_allowed_len"]))
    max_len = max(min_len, int(duration_prior["max_allowed_len"]))

    def score_interval(ss, ee):
        if ee < ss:
            return -np.inf, {}
        L = ee - ss + 1
        if L < min_len or L > max_len:
            return -np.inf, {}

        entropy_gain = segment_entropy_gain((ss, ee), prob_track, duration_prior)
        boundary_support = float(0.5 * global_boundary_signal[ss] + 0.5 * global_boundary_signal[ee])
        if prob_track is None:
            internal_prob = 0.0
        else:
            internal_prob = float(np.mean(prob_track[ss:ee + 1])) if ee >= ss else 0.0

        score = float(
            REFINE_W_ENTROPY_GAIN * entropy_gain
            + REFINE_W_BOUNDARY_SUPPORT * boundary_support
            + REFINE_W_INTERNAL_PROB * internal_prob
        )
        return score, {
            "entropy_gain": float(entropy_gain),
            "boundary_support": boundary_support,
            "internal_prob": internal_prob,
            "refinement_score": score,
        }

    best_s, best_e = s0, e0
    best_score, best_metrics = score_interval(s0, e0)

    for ss in s_candidates:
        for ee in e_candidates:
            score, metrics = score_interval(ss, ee)
            if score > best_score:
                best_score = score
                best_metrics = metrics
                best_s, best_e = int(ss), int(ee)

    return (int(best_s), int(best_e)), best_metrics


def build_cluster_candidate_table(clusters, duration_prior, n, global_boundary_signal, prob_track):
    rows = []
    start_vote_total = np.zeros(n, dtype=float)
    end_vote_total = np.zeros(n, dtype=float)
    for c in clusters:
        seg, sv, ev = cluster_to_candidate_segment(c, n, duration_prior, global_boundary_signal)
        seg, refined_metrics = refine_segment_boundaries(seg, duration_prior, global_boundary_signal, prob_track)
        row = cluster_feature_row(c, seg, duration_prior, global_boundary_signal, prob_track, refined_metrics)
        rows.append(row)
        start_vote_total += sv * max(1e-6, row["cluster_score"])
        end_vote_total += ev * max(1e-6, row["cluster_score"])
    df = pd.DataFrame(rows)
    return df, normalize_vector(start_vote_total), normalize_vector(end_vote_total)


def prepare_cluster_ranking(cand_df):
    """Rank clusters using native cluster evidence and entropy-based separation gain."""
    if cand_df is None or len(cand_df) == 0:
        return cand_df

    out = cand_df.copy()
    cluster_norm = normalize_vector(out["cluster_score"].values)
    entropy_norm = normalize_vector(out["entropy_gain"].values) if "entropy_gain" in out.columns else np.zeros(len(out))

    out["cluster_score_norm"] = cluster_norm
    out["entropy_gain_norm"] = entropy_norm
    out["rank_score"] = (
        (1.0 - ENTROPY_GAIN_WEIGHT) * cluster_norm
        + ENTROPY_GAIN_WEIGHT * entropy_norm
    )
    return out


def nms_select_by_score(cand_df, score_col, train_sids, duration_prior, low_mult=TARGET_COUNT_LOW_MULTIPLIER, high_mult=TARGET_COUNT_HIGH_MULTIPLIER):
    if cand_df is None or len(cand_df) == 0:
        return [], pd.DataFrame()

    train_counts = np.asarray([len(get_gt_phase2_segments_for_sid(sid)) for sid in train_sids], dtype=float)
    train_count = int(round(np.mean(train_counts)))

    low = max(1, int(np.floor(train_count * low_mult)))
    high = max(low, int(np.ceil(train_count * high_mult)))

    ranked = cand_df.sort_values(score_col, ascending=False).copy()
    items = ranked.to_dict("records")

    selected = []
    for item in items:
        seg = (int(item["start"]), int(item["end"]))

        too_overlap = False
        for sel in selected:
            sel_seg = (int(sel["start"]), int(sel["end"]))
            if segment_iou(seg, sel_seg) > FINAL_NMS_IOU:
                too_overlap = True
                break

        if too_overlap:
            continue

        selected.append(item)
        if len(selected) >= high:
            break

    if len(selected) < low:
        selected_keys = set((int(x["start"]), int(x["end"])) for x in selected)
        for item in items:
            key = (int(item["start"]), int(item["end"]))
            if key in selected_keys:
                continue

            max_ov = 0.0
            for sel in selected:
                max_ov = max(max_ov, segment_iou(key, (int(sel["start"]), int(sel["end"]))))

            if max_ov <= 0.75:
                selected.append(item)
                selected_keys.add(key)

            if len(selected) >= low:
                break

    selected = sorted(selected, key=lambda x: int(x["start"]))
    df = pd.DataFrame(selected)
    segs = [(int(r["start"]), int(r["end"])) for _, r in df.iterrows()] if len(df) else []
    return segs, df



def select_final_candidates(cand_df, train_sids, duration_prior):
    """Select final candidate intervals using the cluster-based rank score and NMS."""
    segs, selected_df = nms_select_by_score(
        cand_df,
        score_col="rank_score",
        train_sids=train_sids,
        duration_prior=duration_prior
    )

    diagnostics = {
        "selected_count": len(selected_df),
    }

    return segs, selected_df, diagnostics


def run_fold(test_sid, train_sids):
    duration_prior = learn_duration_prior(train_sids)

    models = {}
    probs = {}
    y_ref = None

    for branch_name, features in BRANCHES.items():
        model = train_model(
            train_sids,
            features,
            random_state=RANDOM_STATE + test_sid + len(branch_name)
        )
        prob, y = predict_proba_session(model, test_sid, features)
        models[branch_name] = model
        probs[branch_name] = prob
        if y_ref is None:
            y_ref = y

    all_candidates = []
    branch_candidate_counts = {}
    for branch_name, prob in probs.items():
        candidates = generate_branch_candidates(
            branch_name=branch_name,
            prob=prob,
            df=sessions[test_sid],
            duration_prior=duration_prior
        )
        all_candidates.extend(candidates)
        branch_candidate_counts[branch_name] = len(candidates)

    all_candidates, candidate_nms_diag = remove_near_duplicate_candidates(all_candidates, duration_prior)
    branch_candidate_counts.update(candidate_nms_diag)

    clusters = cluster_candidates(all_candidates, duration_prior)

    global_boundary, program_boundary, physical_boundary = build_global_boundary_signals(sessions[test_sid])
    prob_track = probability_support_track(probs)

    cluster_candidate_df, start_votes, end_votes = build_cluster_candidate_table(
        clusters,
        duration_prior,
        len(sessions[test_sid]),
        global_boundary,
        prob_track
    )

    ranked_cluster_candidate_df = prepare_cluster_ranking(cluster_candidate_df)

    final_segs, final_df, selection_diagnostics = select_final_candidates(
        ranked_cluster_candidate_df,
        train_sids=train_sids,
        duration_prior=duration_prior
    )


    candidate_df = pd.DataFrame(all_candidates)

    cluster_export_rows = []
    for cluster in clusters:
        row = {k: v for k, v in cluster.items() if k != "members"}
        cluster_export_rows.append(row)
    cluster_df = pd.DataFrame(cluster_export_rows)

    return {
        "duration_prior": duration_prior,
        "models": models,
        "probs": probs,
        "y": y_ref,
        "candidate_df": candidate_df,
        "cluster_df": cluster_df,
        "cluster_candidate_df": cluster_candidate_df,
        "ranked_cluster_candidate_df": ranked_cluster_candidate_df,
        "selection_diagnostics": selection_diagnostics,
        "final_df": final_df,
        "final_segs": final_segs,
        "global_boundary": global_boundary,
        "program_boundary": program_boundary,
        "physical_boundary": physical_boundary,
        "start_votes": start_votes,
        "end_votes": end_votes,
        "branch_candidate_counts": branch_candidate_counts,
    }

def normalized_values(df, col):
    if col not in df.columns:
        return np.zeros(len(df))

    v = (
        pd.to_numeric(df[col], errors="coerce")
        .ffill()
        .bfill()
        .fillna(0)
        .values
        .astype(float)
    )

    mn = np.nanmin(v)
    mx = np.nanmax(v)

    if not np.isfinite(mn) or not np.isfinite(mx) or mx == mn:
        return np.zeros(len(v))

    return (v - mn) / (mx - mn)


def phase_color_map(phases):
    cmap = {
        1: "#7B68EE",
        2: "#66CDAA",
        3: "#FFA500",
        4: "#87CEFA",
        6: "#FF69B4",
    }
    return [cmap.get(int(p), "#CCCCCC") for p in phases]


def draw_regions(ax, gt_mask, pred_mask):
    gt_only = gt_mask & ~pred_mask
    pred_only = pred_mask & ~gt_mask
    overlap = gt_mask & pred_mask

    for s, e in contiguous_segments(gt_only):
        ax.axvspan(s, e, color="limegreen", alpha=0.14)

    for s, e in contiguous_segments(pred_only):
        ax.axvspan(s, e, color="orangered", alpha=0.16)

    for s, e in contiguous_segments(overlap):
        ax.axvspan(s, e, color="mediumpurple", alpha=0.25)


def draw_boundaries(ax, gt_segs, pred_segs):
    for s, e in gt_segs:
        ax.axvline(s, color="green", linewidth=2.1)
        ax.axvline(e, color="darkgreen", linewidth=2.1)

    for s, e in pred_segs:
        ax.axvline(s, color="red", linestyle="--", linewidth=1.5)
        ax.axvline(e, color="darkred", linestyle="--", linewidth=1.5)


def plot_fold(sid, result, out_path):
    df = sessions[sid].copy()
    n = len(df)
    x = np.arange(n)

    gt_segs = get_gt_phase2_segments_for_sid(sid)
    pred_segs = result["final_segs"]

    gt_mask = segments_to_mask(gt_segs, n)
    pred_mask = segments_to_mask(pred_segs, n)

    plot_features = [c for c in PROGRAM_RELATED_FEATURES + PHYSICAL_PROCESS_FEATURES if c in df.columns][:18]

    nrows = len(plot_features) + 8

    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(19, max(12, 1.05 * nrows)),
        sharex=True
    )


    rows = []
    for b in ["program", "physical", "program_physical"]:
        if b in result["probs"]:
            rows.append((f"P_{b}", smooth_array(result["probs"][b], 51)))

    rows.extend([
        ("Boundary", result["global_boundary"]),
        ("ProgB", result["program_boundary"]),
        ("PhysB", result["physical_boundary"]),
        ("StartVote", result["start_votes"]),
        ("EndVote", result["end_votes"]),
    ])

    for i, (lab, arr) in enumerate(rows):
        ax = axes[i]
        draw_regions(ax, gt_mask, pred_mask)
        ax.plot(x, arr, lw=1.0, label=lab)
        draw_boundaries(ax, gt_segs, pred_segs)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(lab, rotation=0, labelpad=55, ha="right", va="center", fontsize=8)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right")

    ax = axes[len(rows)]
    phases = df[LABEL_COL].values
    colors = phase_color_map(phases)

    for i in range(n):
        ax.axvspan(i, i + 1, color=colors[i], alpha=0.85, linewidth=0)

    draw_boundaries(ax, gt_segs, pred_segs)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.5])
    ax.set_yticklabels(["5-phase"])
    ax.grid(alpha=0.10)

    start_idx = len(rows) + 1

    for ax, c in zip(axes[start_idx:-1], plot_features):
        draw_regions(ax, gt_mask, pred_mask)
        vals = normalized_values(df, c)
        ax.plot(x, vals, lw=0.75)
        draw_boundaries(ax, gt_segs, pred_segs)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(c, rotation=0, labelpad=70, ha="right", va="center", fontsize=8)
        ax.grid(alpha=0.22)

    ax = axes[-1]
    ax.set_ylim(0, 3)
    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_yticklabels(["GT", "Pred", "Overlap"])

    for s, e in gt_segs:
        ax.broken_barh([(s, e - s + 1)], (0.15, 0.7), facecolors="limegreen", alpha=0.45)

    for s, e in pred_segs:
        ax.broken_barh([(s, e - s + 1)], (1.15, 0.7), facecolors="orangered", alpha=0.45)

    for s, e in contiguous_segments(gt_mask & pred_mask):
        ax.broken_barh([(s, e - s + 1)], (2.15, 0.7), facecolors="mediumpurple", alpha=0.60)

    draw_boundaries(ax, gt_segs, pred_segs)
    ax.set_xlabel("Row index after production removal")
    ax.grid(alpha=0.22)

    fig.suptitle(
        f"Publishable Phase 2 Pipeline | Session {sid:02d} | GT={len(gt_segs)} | Pred={len(pred_segs)}",
        fontsize=13,
        y=0.996
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


print("\nSTAGE 2 | Running Phase 2 leave-one-session-out evaluation")

pipeline_start_time = time.perf_counter()

fold_summary_rows = []
boundary_rows = []
segment_rows = []
duration_rows = []
for test_sid in TARGET_SESSIONS:
    train_sids = [s for s in TARGET_SESSIONS if s != test_sid]

    print(f"\nFold | test={test_sid:02d} | train={train_sids}")

    fold_dir = OUT_ROOT / f"session_{test_sid:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    fold_start_time = time.perf_counter()
    result = run_fold(test_sid, train_sids)
    fold_runtime_sec = time.perf_counter() - fold_start_time

    gt_segs = get_gt_phase2_segments_for_sid(test_sid)
    pred_segs = result["final_segs"]

    y = result["y"]
    pred_mask = segments_to_mask(pred_segs, len(y)).astype(int)

    row_acc = accuracy_score(y, pred_mask)
    row_f1 = f1_score(y, pred_mask, zero_division=0)
    row_prec = precision_score(y, pred_mask, zero_division=0)
    row_rec = recall_score(y, pred_mask, zero_division=0)

    try:

        auc_prob = result["probs"].get("program_physical", result["probs"].get("program"))
        row_auc = roc_auc_score(y, auc_prob)
    except Exception:
        row_auc = np.nan

    ev, matches = evaluate_segments(pred_segs, gt_segs)
    bdf = pd.DataFrame(boundary_errors(pred_segs, gt_segs))
    mean_start_err = float(bdf["start_abs_error_removed_rows"].mean()) if len(bdf) else np.nan
    mean_end_err = float(bdf["end_abs_error_removed_rows"].mean()) if len(bdf) else np.nan
    mean_boundary_iou = float(bdf["iou"].mean()) if len(bdf) else 0.0

    candidate_df = result["candidate_df"]
    cluster_df = result["cluster_df"]
    final_df = result["final_df"]
    duration_prior = result["duration_prior"]

    candidate_df.to_csv(fold_dir / "candidate_pool.csv", index=False)
    cluster_df.to_csv(fold_dir / "candidate_clusters.csv", index=False)
    final_df.to_csv(fold_dir / "final_selected_segments.csv", index=False)
    result["cluster_candidate_df"].to_csv(fold_dir / "cluster_candidate_features.csv", index=False)
    result["ranked_cluster_candidate_df"].to_csv(fold_dir / "ranked_cluster_candidates.csv", index=False)
    bdf.to_csv(fold_dir / "boundary_errors.csv", index=False)
    row_pred_df = sessions[test_sid][[
        "_removed_row",
        "_original_row",
        "_time",
        LABEL_COL,
        "target_phase2"
    ]].copy()

    for b, p in result["probs"].items():
        row_pred_df[f"prob_{b}_raw"] = p
        row_pred_df[f"prob_{b}_smooth51"] = smooth_array(p, 51)

    row_pred_df["global_boundary_signal"] = result["global_boundary"]
    row_pred_df["program_boundary_signal"] = result["program_boundary"]
    row_pred_df["physical_boundary_signal"] = result["physical_boundary"]
    row_pred_df["start_vote"] = result["start_votes"]
    row_pred_df["end_vote"] = result["end_votes"]
    row_pred_df["pred_phase2"] = pred_mask

    row_pred_df.to_csv(fold_dir / "row_predictions.csv", index=False)

    for kind, segs in [("GT", gt_segs), ("PRED", pred_segs)]:
        for k, (s, e) in enumerate(segs, start=1):
            segment_rows.append({
                "test_session": test_sid,
                "type": kind,
                "segment_no": k,
                "start_removed_row": s,
                "end_removed_row": e,
                "length": e - s + 1
            })

    duration_row = duration_prior.copy()
    duration_row["test_session"] = test_sid
    duration_row["train_sessions"] = ",".join(map(str, train_sids))
    duration_rows.append(duration_row)

    summary = {
        "method": "publishable_phase2_pipeline",
        "test_session": test_sid,
        "train_sessions": ",".join(map(str, train_sids)),
        "fold_runtime_sec": fold_runtime_sec,
        "gt_segments": len(gt_segs),
        "pred_segments": len(pred_segs),
        "abs_count_error": abs(len(pred_segs) - len(gt_segs)),
        "candidate_count": len(candidate_df),
        "cluster_count": len(cluster_df),
        "cluster_candidate_count": len(result["cluster_candidate_df"]),
        "selected_count": result["selection_diagnostics"].get("selected_count", np.nan),
        "program_candidates": result["branch_candidate_counts"].get("program", 0),
        "physical_candidates": result["branch_candidate_counts"].get("physical", 0),
        "program_physical_candidates": result["branch_candidate_counts"].get("program_physical", 0),
        "duration_median_train": duration_prior["median_len"],
        "duration_min_allowed_train": duration_prior["min_allowed_len"],
        "duration_max_allowed_train": duration_prior["max_allowed_len"],
        "row_accuracy": row_acc,
        "row_precision": row_prec,
        "row_recall": row_rec,
        "row_f1": row_f1,
        "row_auc": row_auc,
        **ev,
        "mean_boundary_iou_to_best_pred": mean_boundary_iou,
        "mean_start_abs_error_removed_rows": mean_start_err,
        "mean_end_abs_error_removed_rows": mean_end_err,
        "gt_intervals_removed_rows": interval_text(gt_segs),
        "pred_intervals_removed_rows": interval_text(pred_segs),
    }

    fold_summary_rows.append(summary)

    for br in bdf.to_dict("records"):
        br["test_session"] = test_sid
        boundary_rows.append(br)

    plot_fold(
        sid=test_sid,
        result=result,
        out_path=fold_dir / f"session_{test_sid:02d}_stage2_localisation.png"
    )

    rawcand_count = int(result["branch_candidate_counts"].get(
        "candidate_count_before_nms",
        len(candidate_df)
    ))

    print(
        f"  GT={len(gt_segs):2d} | Pred={len(pred_segs):2d} | "
        f"RawCand={rawcand_count:5d} | AfterNMS={len(candidate_df):5d} | Clusters={len(cluster_df):3d} | "
        f"SegF1={ev['f1']:.3f} | MeanBestIoU={ev['mean_best_gt_iou']:.3f} | "
        f"BestAnyIoU={ev['best_any_iou']:.3f} | CountErr={abs(len(pred_segs)-len(gt_segs)):2d} | "
        f"RowF1={row_f1:.3f} | Runtime={fold_runtime_sec:.2f}s"
    )


print("\nSTAGE 2 | Preparing Phase 2 final summary")

summary_df = pd.DataFrame(fold_summary_rows)
boundary_df = pd.DataFrame(boundary_rows)
segment_df = pd.DataFrame(segment_rows)
duration_df = pd.DataFrame(duration_rows)

summary_df.to_csv(OUT_ROOT / "stage2_loo_summary.csv", index=False)
boundary_df.to_csv(OUT_ROOT / "stage2_boundary_errors.csv", index=False)
segment_df.to_csv(OUT_ROOT / "stage2_gt_pred_segments.csv", index=False)
duration_df.to_csv(OUT_ROOT / "stage2_duration_prior_by_fold.csv", index=False)

total_runtime_sec = time.perf_counter() - pipeline_start_time

runtime_summary = pd.DataFrame([
    {
        "metric": "total_runtime_sec",
        "value": float(total_runtime_sec),
    },
    {
        "metric": "mean_fold_runtime_sec",
        "value": float(summary_df["fold_runtime_sec"].mean()) if "fold_runtime_sec" in summary_df.columns else np.nan,
    },
    {
        "metric": "min_fold_runtime_sec",
        "value": float(summary_df["fold_runtime_sec"].min()) if "fold_runtime_sec" in summary_df.columns else np.nan,
    },
    {
        "metric": "max_fold_runtime_sec",
        "value": float(summary_df["fold_runtime_sec"].max()) if "fold_runtime_sec" in summary_df.columns else np.nan,
    },
])
runtime_summary.to_csv(OUT_ROOT / "runtime_summary.csv", index=False)

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
    "mean_boundary_iou_to_best_pred",
    "mean_start_abs_error_removed_rows",
    "mean_end_abs_error_removed_rows",
    "abs_count_error"
]

overall = (
    summary_df[metric_cols]
    .agg(["mean", "std", "min", "median", "max"])
    .T
    .reset_index()
    .rename(columns={"index": "metric"})
)

overall.to_csv(OUT_ROOT / "stage2_overall_metrics.csv", index=False)

print("\nPUBLISHABLE PHASE 2 PIPELINE SUMMARY:")
display(summary_df.round(4))

print("\nOVERALL METRICS:")
display(overall.round(4))

print("\nRUNTIME SUMMARY:")
display(runtime_summary.round(4))


fig, ax = plt.subplots(figsize=(12, 5))

summary_df[[
    "test_session",
    "mean_best_gt_iou",
    "best_any_iou",
    "f1",
    "row_f1"
]].plot(
    x="test_session",
    kind="bar",
    ax=ax
)

ax.set_title("Publishable Phase 2 Pipeline | LOO metrics")
ax.set_xlabel("Test session")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()
fig.savefig(OUT_ROOT / "stage2_fold_metrics.png", dpi=180, bbox_inches="tight")
plt.close(fig)


fig, ax = plt.subplots(figsize=(10, 5))

x = np.arange(len(summary_df))
w = 0.35

ax.bar(x - w/2, summary_df["gt_segments"], width=w, label="GT segments")
ax.bar(x + w/2, summary_df["pred_segments"], width=w, label="Pred segments")

ax.set_xticks(x)
ax.set_xticklabels([f"S{int(s):02d}" for s in summary_df["test_session"]])
ax.set_ylabel("Segment count")
ax.set_title("Publishable Phase 2 Pipeline | Segment Count Comparison")
ax.grid(axis="y", alpha=0.25)
ax.legend()

fig.tight_layout()
fig.savefig(OUT_ROOT / "stage2_count_comparison.png", dpi=180, bbox_inches="tight")
plt.close(fig)

print("\nPUBLISHABLE STAGE 2 PIPELINE COMPLETED.")
print("Output directory:", OUT_ROOT.resolve())
print("\nKey output files:")
print(" - stage2_loo_summary.csv")
print(" - stage2_overall_metrics.csv")
print(" - stage2_boundary_errors.csv")
print(" - stage2_gt_pred_segments.csv")
print(" - stage2_fold_metrics.png")
print(" - stage2_count_comparison.png")
print(" - runtime_summary.csv")
print(" - In each session folder:")
print("   - candidate_pool.csv")
print("   - candidate_clusters.csv")
print("   - final_selected_segments.csv")
print("   - boundary_errors.csv")
print("   - row_predictions.csv")
print("   - session_XX_stage2_localisation.png")
