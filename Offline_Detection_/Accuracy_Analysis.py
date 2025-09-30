from M201_1min_analysis import get_screw_idle
from M201_variation import idle_by_variation
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from typing import List, Tuple
import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, labels=("Pred Non-Idle","Pred Idle"),
                          ylabels=("Ref Non-Idle","Ref Idle"),
                          title="Confusion Matrix"):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=ylabels,
                cbar=False, square=True)
    plt.ylabel("Reference")
    plt.xlabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def get_consecutive_fp(ref, pred, threshold=15):
    """Return Panda dataframe of consecutive false positives longer than threshold."""
    result_df = pd.DataFrame({'ref': ref, 'pred': pred})
    result_df['fp'] = ((result_df['ref'] == 0) & (result_df['pred'] == 1))

    blocks = (result_df['fp'] != result_df['fp'].shift()).astype(int).cumsum()

    consecutive_fp = (
        result_df.groupby(blocks)['fp']
        .agg(start=lambda x: x.index[0],
             end = lambda x: x.index[-1],
             length='size',
             is_fp='first'
        )
    )

    valid_fp = consecutive_fp.loc[consecutive_fp.is_fp & (consecutive_fp.length >= threshold),
                                                                                ['start', 'end']]
    return valid_fp
    # fp = np.where(pred == 1 & ref ==0)[0]
    # consecutive_fp = []
    # for i in range(len(fp)):
    #     if i == 0:
    #         if fp[i] - fp[i+1] > threshold:
    #             consecutive_fp.append((fp[i], fp[i+1]))
    #     else:
    #         if fp[i] - fp[i-1] > threshold:
    #             consecutive_fp.append((fp[i], fp[i+1]))

Interval = Tuple[int, int]

def get_idx_len(folder_path: str) -> int:
    """Return length of data file."""
    data = pd.read_csv(folder_path)
    return len(data)

def _normalize_and_clip(intervals: List[Interval], N: int) -> List[Interval]:
    """Ensure start < end, clip to [0, N], sort & merge overlaps."""
    ints = []
    for s, e in intervals:
        s, e = int(max(0, s)), int(min(N, e))
        if e > s:
            ints.append((s, e))
    if not ints:
        return []
    # sort & merge
    ints.sort()
    merged = [ints[0]]
    for s, e in ints[1:]:
        ls, le = merged[-1]
        if s <= le:  # overlap or touching
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged

def intervals_to_mask(N: int, intervals: List[Interval]) -> np.ndarray:
    """Return boolean mask of length N where True = idle."""
    mask = np.zeros(N, dtype=bool)
    for s, e in intervals:
        mask[s:e] = True
    return mask

def time_sample_metrics(N: int,
                        ref_intervals: List[Interval],
                        pred_intervals: List[Interval]):
    """This function computes the metrics in an element-wise fashion."""
    ref = _normalize_and_clip(ref_intervals, N)
    pred = _normalize_and_clip(pred_intervals, N)

    y_true = intervals_to_mask(N, ref)
    y_pred = intervals_to_mask(N, pred)

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm)

    tp = np.sum(y_true & y_pred)
    fp = np.sum(~y_true & y_pred)
    fn = np.sum(y_true & ~y_pred)
    tn = np.sum(~y_true & ~y_pred)

    true_fp = get_consecutive_fp(y_true, y_pred, 15)
    print(len(true_fp))
    print(true_fp.head(8))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2*precision*recall / (precision+recall) if (precision+recall) else 0.0
    accuracy  = (tp + tn) / N if N else 0.0
    jaccard   = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0  # IoU over time

    # Symmetric difference in samples = misclassified samples
    sym_diff = fp + fn

    return {
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precision": precision, "recall": recall, "f1": f1,
        "accuracy": accuracy, "jaccard_time": jaccard,
        "sym_diff_samples": int(sym_diff)
    }
# The following functions are not used for now
def _pairwise_iou(a: Interval, b: Interval) -> float:
    s1, e1 = a; s2, e2 = b
    inter = max(0, min(e1, e2) - max(s1, s2))
    union = (e1 - s1) + (e2 - s2) - inter
    return inter / union if union > 0 else 0.0

def segment_level_metrics(N: int,
                          ref_intervals: List[Interval],
                          pred_intervals: List[Interval],
                          iou_match_thresh: float = 0.3,
                          boundary_tol: int = 0):
    """Greedy IoU matching of segments; reports over/under-seg and boundary errors.
       boundary_tol is a tolerance (in samples) when counting boundary errors.
    """
    ref = _normalize_and_clip(ref_intervals, N)
    pred = _normalize_and_clip(pred_intervals, N)

    if not ref and not pred:
        return {
            "matched": 0, "ref_segments": 0, "pred_segments": 0,
            "recall_seg": 1.0, "precision_seg": 1.0, "f1_seg": 1.0,
            "mean_iou_matched": 1.0, "mean_abs_start_err": 0.0, "mean_abs_end_err": 0.0,
            "oversegmentation": 0, "undersegmentation": 0
        }

    # Build IoU matrix
    iou = np.zeros((len(ref), len(pred)), dtype=float)
    for i, r in enumerate(ref):
        for j, p in enumerate(pred):
            iou[i, j] = _pairwise_iou(r, p)

    # Greedy matching by IoU
    matched_pairs = []
    used_ref = set(); used_pred = set()
    while True:
        i, j = np.unravel_index(np.argmax(iou), iou.shape) if iou.size else (None, None)
        if iou.size == 0 or iou[i, j] < iou_match_thresh:
            break
        matched_pairs.append((i, j, iou[i, j]))
        used_ref.add(i); used_pred.add(j)
        iou[i, :] = -1.0
        iou[:, j] = -1.0

    # Segment-level PR/F1
    tp_seg = len(matched_pairs)
    fn_seg = len(ref) - tp_seg
    fp_seg = len(pred) - tp_seg
    precision_seg = tp_seg / (tp_seg + fp_seg) if (tp_seg + fp_seg) else 0.0
    recall_seg    = tp_seg / (tp_seg + fn_seg) if (tp_seg + fn_seg) else 0.0
    f1_seg        = 2*precision_seg*recall_seg / (precision_seg + recall_seg) if (precision_seg + recall_seg) else 0.0

    # Boundary errors on matched pairs
    start_errs, end_errs, ious = [], [], []
    for i, j, ij_iou in matched_pairs:
        rs, re = ref[i]; ps, pe = pred[j]
        start_errs.append(abs(ps - rs))
        end_errs.append(abs(pe - re))
        ious.append(ij_iou)

    mean_abs_start_err = float(np.mean(start_errs)) if start_errs else None
    mean_abs_end_err   = float(np.mean(end_errs)) if end_errs else None
    mean_iou_matched   = float(np.mean(ious)) if ious else None

    # Count over/under-segmentation:
    # overseg: multiple predicted segments matched (by IoU≥thresh) to the same reference region;
    # underseg: one predicted covers multiple refs
    # We approximate by counting overlaps ignoring 1-to-1 greedy matches.
    def _count_multi_overlaps(A: List[Interval], B: List[Interval]):
        # For each interval in A, count how many in B have IoU≥thresh; sum extras beyond 1
        total_extra = 0
        for a in A:
            cnt = sum(_pairwise_iou(a, b) >= iou_match_thresh for b in B)
            if cnt < 1 or cnt > 1:
                # print(cnt)
                pass
            total_extra += max(0, cnt - 1)
        return total_extra

    overseg = _count_multi_overlaps(ref, pred)   # preds split a ref
    underseg = _count_multi_overlaps(pred, ref)  # preds merge refs

    # Optional: boundary-tolerance hit-rate (how often both ends are within tol)
    if boundary_tol > 0 and start_errs:
        boundary_ok = [ (se <= boundary_tol) and (ee <= boundary_tol) for se, ee in zip(start_errs, end_errs) ]
        boundary_within_tol_rate = float(np.mean(boundary_ok))
    else:
        boundary_within_tol_rate = None

    return {
        "ref_segments": len(ref),
        "pred_segments": len(pred),
        "matched": tp_seg,
        "precision_seg": precision_seg,
        "recall_seg": recall_seg,
        "f1_seg": f1_seg,
        "mean_iou_matched": mean_iou_matched,
        "mean_abs_start_err": mean_abs_start_err,
        "mean_abs_end_err": mean_abs_end_err,
        "boundary_within_tol_rate": boundary_within_tol_rate,
        "oversegmentation": int(overseg),
        "undersegmentation": int(underseg),
    }

if __name__ == "__main__":
    folder_path = 'C:\\Users\\tresp\\Desktop\\Calculation\\Magna\\merged_data\\201_1min.csv'

    # Idle by Variation: ibv (prediction); Idle by Screw: ibs(reference)
    idle_by_var = idle_by_variation(folder_path, 'by_run',
                                    hampel_win=4, var_win=6, var_threshold=5, n_sigmas=3)
    idle_by_var = idle_by_var[:, :2]
    idle_by_screw = get_screw_idle(folder_path, 15)


    # Transform the 2*N dimension array to a list of tuples: [(start, end), ...]
    ibv = [(idle_by_var[i, 0], idle_by_var[i, 1]) for i in range(idle_by_var.shape[0])]
    ibs = [(idle_by_screw[i, 0], idle_by_screw[i, 1]) for i in range(idle_by_screw.shape[0])]

    idx_len = get_idx_len(folder_path)
    ts_metrics = time_sample_metrics(idx_len, ibs, ibv)
    # seg_metrics = segment_level_metrics(idx_len, ibs, ibv, 0.4, 10)

    print(ts_metrics)
# print(seg_metrics)