from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional, List, Tuple, Any
from collections import deque
import numpy as np
import pandas as pd

# =========================
# Helpers & core utilities
# =========================

def _to_series(x, value_col: str | None = None) -> pd.Series:
    """
    Accept a Series or DataFrame. Return a Series with a DatetimeIndex if possible,
    else a Range/Int index. De-duplicates and sorts index.
    """
    if isinstance(x, pd.Series):
        s = x.copy()
        if not isinstance(s.index, (pd.DatetimeIndex, pd.RangeIndex)):
            s.index = pd.RangeIndex(len(s))
        s = s[~s.index.duplicated(keep="last")].sort_index()
        return s

    if not isinstance(x, pd.DataFrame):
        raise TypeError("batch must be a pandas Series or DataFrame")

    df = x.copy()
    time_candidates = ["ts", "time", "timestamp", "date", "datetime"]
    if not isinstance(df.index, pd.DatetimeIndex):
        for c in time_candidates:
            if c in df.columns:
                idx = pd.to_datetime(df[c], utc=True, errors="coerce")
                if idx.notna().any():
                    df = df.set_index(idx)
                    break

    if value_col is None:
        non_time_cols = [c for c in df.columns if c not in time_candidates]
        if not non_time_cols:
            raise ValueError("No value column found. Pass a Series or provide 'value_col'.")
        value_col = non_time_cols[0]

    s = df[value_col].copy()
    if not isinstance(s.index, (pd.DatetimeIndex, pd.RangeIndex)):
        s.index = pd.RangeIndex(len(s))
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s

def _ema_fallback(values: pd.Series, last_ema: float | None, alpha: float = 0.25):
    """Causal EMA for tiny batches."""
    clean = values.copy()
    ema = last_ema
    out = []
    for v in clean.values:
        if pd.isna(v):
            out.append(ema if ema is not None else v)
            continue
        if ema is None:
            ema = v
        else:
            ema = alpha * v + (1 - alpha) * ema
        out.append(ema)
    return pd.Series(out, index=clean.index), ema

def _rolling_median(s: pd.Series, win: int) -> pd.Series:
    win = max(1, min(win, len(s)))
    return s.rolling(win, min_periods=1).median()

def _rolling_mad(s: pd.Series, win: int) -> pd.Series:
    """Rolling MAD with min_periods=1 (robust & causal over the provided slice)."""
    win = max(1, min(win, len(s)))
    def mad_fn(x):
        m = np.median(x)
        return np.median(np.abs(x - m))
    return s.rolling(win, min_periods=1).apply(mad_fn, raw=True)

def hampel_clean(s: pd.Series, win: int = 8, n_sigmas: float = 3.0) -> pd.Series:
    """
    Hampel-like cleaning with rolling median+MAD. Causal within the provided series; we ffill only.
    """
    if len(s) == 0:
        return s.copy()
    med = _rolling_median(s, win)
    mad = _rolling_mad(s, win)
    sigma = 1.4826 * mad
    diff = (s - med).abs()
    mask = diff > (n_sigmas * sigma)
    out = s.copy()
    out[mask] = med[mask]
    return out.ffill()  # online-safe (no bfill)

# =========================
# State & detector classes
# =========================

@dataclass
class OnlineIdleDetectorState:
    # tails (to bridge windows across batches)
    tail_idx: deque = field(default_factory=lambda: deque(maxlen=512))
    tail_raw: deque = field(default_factory=lambda: deque(maxlen=512))
    tail_clean: deque = field(default_factory=lambda: deque(maxlen=512))

    # OFF state (raw-based)
    in_off: bool = False
    zero_run_start_idx: Any | None = None
    zero_run_len: int = 0

    # LOWVAR state (clean-based; true across batches)
    lowvar_in_run: bool = False
    lowvar_open_start: Any | None = None      # TRUE start (may be many batches ago)
    lowvar_len_so_far: int = 0                # cumulative samples across batches
    lowvar_open_id: int | None = None
    lowvar_start_notified: bool = False

    # EMA seed for tiny batches
    ema_last: float | None = None

    # Boundary bookkeeping (for correct end indices at batch start)
    prev_batch_last_idx: Any | None = None
    prev_batch_last_lowvar: bool = False
    prev_batch_last_off: bool = False

class OnlineIdleDetector:
    def __init__(
        self,
        # cleaning & robust low-var
        hampel_win: int = 8,
        n_sigmas: float = 3.0,
        tiny_threshold: int = 5,
        ema_alpha: float = 0.25,

        # MAD-based low-variance settings
        mad_win: int = 8,
        mad_threshold: float | None = None,     # threshold on MAD (same units as signal)
        mad_threshold_in: str = "sigma",          # "mad" or "sigma" (σ ≈ 1.4826*MAD)
        mad_auto_factor: float = 0.5,           # when auto: thr_mad = factor * median(MAD)

        # OFF logic (raw)
        off_level: float = 1.0,
        off_min_len: int = 15,

        # Low-var min length
        min_len: int = 15,

        # Callbacks (any may be None)
        on_lowvar_start: Optional[Callable[..., None]] = None,
        on_lowvar_end:   Optional[Callable[..., None]] = None,
        on_off_start:    Optional[Callable[..., None]] = None,
        on_off_end:      Optional[Callable[..., None]] = None,

        # Optional user context passed to callbacks
        context: Any = None,
    ):
        self.hampel_win = hampel_win
        self.n_sigmas = n_sigmas
        self.tiny_threshold = tiny_threshold
        self.ema_alpha = ema_alpha

        self.mad_win = mad_win
        self.mad_threshold = mad_threshold
        self.mad_threshold_in = mad_threshold_in
        self.mad_auto_factor = mad_auto_factor

        self.off_level = off_level
        self.off_min_len = off_min_len
        self.min_len = min_len

        self.on_lowvar_start = on_lowvar_start
        self.on_lowvar_end   = on_lowvar_end
        self.on_off_start    = on_off_start
        self.on_off_end      = on_off_end
        self.context         = context

        # Keep enough tail to cover the longest dependency
        self.tail_len = max(self.mad_win, self.hampel_win, self.off_min_len, self.min_len) - 1

        # run id for lowvar (unique across batches)
        self._next_run_id = 1

    def _new_run_id(self) -> int:
        rid = self._next_run_id
        self._next_run_id += 1
        return rid

    # ---- Tail management ----
    def _concat_with_tail(self, s: pd.Series, st: OnlineIdleDetectorState) -> pd.Series:
        if len(st.tail_idx) == 0:
            return s
        prev_idx = pd.Index(list(st.tail_idx))
        prev_raw = pd.Series(list(st.tail_raw), index=prev_idx)
        return pd.concat([prev_raw, s]).sort_index()

    def _concat_clean_with_tail(self, s_clean_batch: pd.Series, st: OnlineIdleDetectorState) -> pd.Series:
        prev_idx = pd.Index(list(st.tail_idx))
        prev_clean = pd.Series(list(st.tail_clean), index=prev_idx)
        full = pd.concat([prev_clean, s_clean_batch]).sort_index()
        return full[~full.index.duplicated(keep="last")]

    def _update_tail(self, s_raw_full: pd.Series, s_clean_full: pd.Series, st: OnlineIdleDetectorState):
        if len(s_raw_full) == 0:
            return
        keep = min(self.tail_len, len(s_raw_full))
        tail_raw = s_raw_full.iloc[-keep:]
        tail_clean = s_clean_full.iloc[-keep:]
        st.tail_idx.clear(); st.tail_raw.clear(); st.tail_clean.clear()
        for i, v in zip(tail_raw.index, tail_raw.values):
            st.tail_idx.append(i); st.tail_raw.append(v)
        for v in tail_clean.values:
            st.tail_clean.append(v)

    # ---- Cleaning ----
    def _clean_batch(self, s_raw_batch: pd.Series, st: OnlineIdleDetectorState) -> pd.Series:
        if len(s_raw_batch) == 0:
            return s_raw_batch
        if len(s_raw_batch) < self.tiny_threshold:
            clean, st.ema_last = _ema_fallback(s_raw_batch, st.ema_last, alpha=self.ema_alpha)
            return clean.ffill()
        # Hampel on (tail + batch), then slice to current batch
        s_raw_full = self._concat_with_tail(s_raw_batch, st)
        s_clean_full = hampel_clean(s_raw_full, win=self.hampel_win, n_sigmas=self.n_sigmas)
        st.ema_last = float(s_clean_full.iloc[-1]) if len(s_clean_full) else st.ema_last
        return s_clean_full.iloc[-len(s_raw_batch):]

    # ---- OFF mask over [tail|current] ----
    def _off_mask_on_full(self, s_raw_full: pd.Series) -> pd.Series:
        if len(s_raw_full) == 0:
            return s_raw_full.astype(bool)
        zeroish = (s_raw_full <= self.off_level).astype(int)
        roll = zeroish.rolling(self.off_min_len, min_periods=1).sum()
        seed = roll >= self.off_min_len
        off_mask = pd.Series(False, index=s_raw_full.index)
        in_block = False
        block_start = None
        vals = zeroish.values
        idx = s_raw_full.index
        for i, z in enumerate(vals):
            if z == 1 and not in_block:
                in_block = True; block_start = i
            if (z == 0 or i == len(vals)-1) and in_block:
                block_end = i if z == 1 and i == len(vals)-1 else i-1
                block_idx = idx[block_start:block_end+1]
                if seed.loc[block_idx].any():
                    off_mask.loc[block_idx] = True
                in_block = False
        return off_mask

    # =========================
    # Main entry: process batch
    # =========================
    def process_batch(
        self,
        batch: pd.Series | pd.DataFrame,
        value_col: str | None = None,
        state: OnlineIdleDetectorState | None = None,
    ):
        """
        Returns:
            dict with keys:
              clean: Series (cleaned values for current batch)
              off_periods: list[(start_idx, end_idx, length_samples)] that END in this batch
              lowvar_periods: list[(start_idx, end_idx, length_samples)] that END in this batch
              state: OnlineIdleDetectorState
        """
        print("Batch Processing...")
        st = state or OnlineIdleDetectorState()
        s_raw_batch = _to_series(batch, value_col=value_col)
        s_raw_batch = s_raw_batch[~s_raw_batch.index.duplicated(keep="last")].sort_index()

        if len(s_raw_batch) == 0:
            return dict(clean=s_raw_batch, off_periods=[], lowvar_periods=[], state=st)

        # 1) Clean (causal)
        s_clean_batch = self._clean_batch(s_raw_batch, st)

        # 2) Build FULL series
        s_raw_full  = self._concat_with_tail(s_raw_batch, st)
        s_clean_full = self._concat_clean_with_tail(s_clean_batch, st)

        # 3) OFF detection: mask for exclusion + per-batch state machine for true starts/ends
        off_mask_full = self._off_mask_on_full(s_raw_full)

        off_periods: List[Tuple[Any, Any, int]] = []
        idx_values = list(s_raw_batch.index)
        raw_values = s_raw_batch.values
        prev_idx = None
        for i, (idx_i, v) in enumerate(zip(idx_values, raw_values)):
            if v <= self.off_level:
                if not st.in_off:
                    st.in_off = True
                    if st.zero_run_start_idx is None:
                        st.zero_run_start_idx = idx_i
                    st.zero_run_len = 1
                    if self.off_min_len == 1 and self.on_off_start:
                        self.on_off_start(start_idx=st.zero_run_start_idx, first_valid_idx=idx_i,
                                          min_len=self.off_min_len, context=self.context)
                else:
                    st.zero_run_len += 1
                    if st.zero_run_len == self.off_min_len and self.on_off_start:
                        self.on_off_start(start_idx=st.zero_run_start_idx, first_valid_idx=idx_i,
                                          min_len=self.off_min_len, context=self.context)
            else:
                if st.in_off:
                    # Close using the *previous* sample; if we're at batch start use last idx from previous batch
                    end_idx = prev_idx if prev_idx is not None else (st.prev_batch_last_idx if
                                                                     st.prev_batch_last_idx is not
                                                                     None else st.zero_run_start_idx)
                    total_len = st.zero_run_len
                    if total_len >= self.off_min_len:
                        if self.on_off_end:
                            self.on_off_end(start_idx=st.zero_run_start_idx, end_idx=end_idx,
                                            length=total_len, context=self.context)
                        off_periods.append((st.zero_run_start_idx, end_idx, total_len))
                    st.in_off = False
                    st.zero_run_start_idx = None
                    st.zero_run_len = 0
            prev_idx = idx_i

        # 4) Low-variance (MAD) on CLEAN, excluding OFF
        mad_full = _rolling_mad(s_clean_full, self.mad_win)

        if self.mad_threshold is not None:
            thr_mad = float(self.mad_threshold) / 1.4826 if (
                self.mad_threshold_in.lower().startswith("sig")) else float(self.mad_threshold)
        else:
            med_mad = float(np.nanmedian(mad_full)) if len(mad_full) else 0.0
            thr_mad = max(0.0, self.mad_auto_factor * med_mad)

        lowvar_mask_full = (mad_full <= thr_mad) & (~off_mask_full)

        # Restrict to current batch for the *event* logic
        batch_first_idx = s_raw_batch.index[0]
        batch_last_idx  = s_raw_batch.index[-1]
        lowvar_mask_cur = lowvar_mask_full.loc[batch_first_idx:batch_last_idx]

        lowvar_periods: List[Tuple[Any, Any, int]] = []
        prev_idx = None
        for idx_i, is_low in lowvar_mask_cur.items():
            if is_low:
                if not st.lowvar_in_run:
                    st.lowvar_in_run = True
                    if st.lowvar_open_start is None:
                        st.lowvar_open_start = idx_i  # TRUE start (or preserved from earlier batch)
                    # If continuing from previous batch, len_so_far already > 0
                    if st.lowvar_len_so_far == 0:
                        st.lowvar_len_so_far = 1
                else:
                    st.lowvar_len_so_far += 1

                # Fire START once on first reach of min_len (in this or previous batch)
                if (not st.lowvar_start_notified) and (st.lowvar_len_so_far >= self.min_len):
                    if st.lowvar_open_id is None:
                        st.lowvar_open_id = self._new_run_id()
                    st.lowvar_start_notified = True
                    if self.on_lowvar_start:
                        sigma_eq = 1.4826 * thr_mad
                        self.on_lowvar_start(
                            run_id=st.lowvar_open_id,
                            start_idx=st.lowvar_open_start,   # TRUE start
                            first_valid_idx=idx_i,            # where min_len was achieved (this batch)
                            min_len=self.min_len,
                            threshold=f"MAD<= {thr_mad:.6g} (≈ σ<= {sigma_eq:.6g})",
                            context=self.context,
                        )
            else:
                if st.lowvar_in_run:
                    # Close on the last True; at batch start use the previous batch's last index
                    end_idx = prev_idx if prev_idx is not None else (st.prev_batch_last_idx if
                                                                     st.prev_batch_last_idx is not None else
                                                                     st.lowvar_open_start)
                    total_len = st.lowvar_len_so_far
                    if total_len >= self.min_len:
                        if self.on_lowvar_end:
                            run_id = st.lowvar_open_id if st.lowvar_open_id is not None else self._new_run_id()
                            self.on_lowvar_end(
                                run_id=run_id,
                                start_idx=st.lowvar_open_start,
                                end_idx=end_idx,
                                length=total_len,             # TRUE length across batches
                                context=self.context,
                            )
                        lowvar_periods.append((st.lowvar_open_start, end_idx, total_len))
                    # reset low-var state
                    st.lowvar_in_run = False
                    st.lowvar_open_start = None
                    st.lowvar_len_so_far = 0
                    st.lowvar_open_id = None
                    st.lowvar_start_notified = False
            prev_idx = idx_i

        # 5) Update tails & boundary bookkeeping
        self._update_tail(s_raw_full, s_clean_full, st)
        st.prev_batch_last_idx = batch_last_idx
        st.prev_batch_last_lowvar = bool(lowvar_mask_cur.iloc[-1])
        st.prev_batch_last_off = bool((s_raw_batch <= self.off_level).iloc[-1])

        return dict(
            clean=s_clean_batch,
            off_periods=off_periods,
            lowvar_periods=lowvar_periods,
            state=st
        )


if __name__ == "__main__":
    #  Console notifiers
    def notify_lowvar_start(run_id, start_idx, first_valid_idx, min_len, threshold, context=None):
        print(f"[LOWVAR-START] id={run_id} start={start_idx} first_valid={first_valid_idx} "
              f"(min_len={min_len}, {threshold}) ctx={context}")

    def notify_lowvar_end(run_id, start_idx, end_idx, length, context=None):
        print(f"[LOWVAR-END]   id={run_id} start={start_idx} end={end_idx} len={length} ctx={context}")

    def notify_off_start(start_idx, first_valid_idx, min_len, context=None):
        print(f"[OFF-START]    start={start_idx} first_valid={first_valid_idx} (min_len={min_len}) ctx={context}")

    def notify_off_end(start_idx, end_idx, length, context=None):
        print(f"[OFF-END]      start={start_idx} end={end_idx} len={length} ctx={context}")

    det = OnlineIdleDetector(
        hampel_win=8, n_sigmas=3.0,
        tiny_threshold=5, ema_alpha=0.25,
        mad_win=8, mad_threshold=None, mad_auto_factor=0.5,
        off_level=1.0, off_min_len=15,
        min_len=15,
        on_lowvar_start=notify_lowvar_start,
        on_lowvar_end=notify_lowvar_end,
        on_off_start=notify_off_start,
        on_off_end=notify_off_end,
        context={"device":"Motor Sum"},
    )
