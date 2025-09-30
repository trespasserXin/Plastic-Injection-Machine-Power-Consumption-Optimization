# event_logging.py
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import pandas as pd

@dataclass
class EventLogger:
    log_path: Path
    machine_id: str
    # integer period_id that you wanted; we keep incrementing per machine
    _next_period_id: int = 1
    # keep partially-open periods by detector-run_id
    _open_lowvar: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def _write_jsonl(self, obj: dict):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # -------- LOW-VAR HOOKS (wired to OnlineIdleDetector callbacks) --------
    # --- inside EventLogger ---

    # use a stable key across batches for the same signal
    def _lowvar_key(self) -> str:
        return f"{self.machine_id}::lowvar"

    def on_lowvar_start(
            self,
            *,
            run_id: int,  # kept for compatibility; not used as the key
            start_idx,
            first_valid_idx,
            min_len: int,
            threshold: str,
            batch_series: pd.Series,
            det_params: dict,
            run_id_str: str,
            source_file: Optional[str] = None
    ):
        k = self._lowvar_key()
        batch_start_ts = batch_series.index.min()
        batch_end_ts = batch_series.index.max()

        self._open_lowvar[k] = dict(
            period_id=self._next_period_id,
            event_type="low_var",
            run_id=run_id_str,
            machine_id=self.machine_id,
            actual_start_ts=pd.to_datetime(start_idx).isoformat(),
            first_cross_ts=pd.to_datetime(first_valid_idx).isoformat(),
            confirmed_ts=pd.to_datetime(first_valid_idx).isoformat(),
            opened_in_this_run=(pd.to_datetime(start_idx) >= batch_start_ts),
            spans_batches=(pd.to_datetime(start_idx) < batch_start_ts),
            mad_win_s=int(det_params.get("mad_win_s", 0)),
            mad_threshold_kw=float(det_params.get("mad_threshold_kw", 0.0)) if det_params.get(
                "mad_threshold_kw") is not None else None,
            off_min_len_s=int(det_params.get("off_min_len_s", 0)),
            low_var_min_len_s=int(det_params.get("low_var_min_len_s", 0)),
            source=dict(source_file=source_file,
                        file_start_ts=batch_start_ts.isoformat(),
                        file_end_ts=batch_end_ts.isoformat()),
            _batch_start_ts=batch_start_ts,
            _batch_end_ts=batch_end_ts,
            notes=f"Start threshold: {threshold}",
            labels=["review_candidate"]
        )
        self._next_period_id += 1

    def on_lowvar_end(
            self,
            *,
            run_id: int,  # kept for compatibility
            start_idx,
            end_idx,
            length: int,
            batch_series: pd.Series
    ):
        k = self._lowvar_key()
        if k not in self._open_lowvar:
            # start was in a previous process or missed; create a minimal one
            batch_start_ts = batch_series.index.min()
            batch_end_ts = batch_series.index.max()
            self._open_lowvar[k] = dict(
                period_id=self._next_period_id,
                event_type="low_var",
                run_id="unknown",
                machine_id=self.machine_id,
                actual_start_ts=pd.to_datetime(start_idx).isoformat(),
                first_cross_ts=pd.to_datetime(start_idx).isoformat(),
                confirmed_ts=pd.to_datetime(start_idx).isoformat(),
                opened_in_this_run=True,
                spans_batches=False,
                mad_win_s=None, mad_threshold_kw=None,
                off_min_len_s=None, low_var_min_len_s=None,
                source=dict(source_file=None,
                            file_start_ts=batch_start_ts.isoformat(),
                            file_end_ts=batch_end_ts.isoformat()),
                _batch_start_ts=batch_start_ts,
                _batch_end_ts=batch_end_ts,
                notes="(auto-created at end; start callback missing)",
                labels=["review_candidate"]
            )
            self._next_period_id += 1

        rec = self._open_lowvar.pop(k)

        curr_batch_start = batch_series.index.min()
        curr_batch_end = batch_series.index.max()
        end_ts = pd.to_datetime(end_idx)

        duration_s = int(length)  # adjust if your sampling period differs

        t0 = max(pd.to_datetime(rec["actual_start_ts"]), curr_batch_start)
        t1 = end_ts
        seg = batch_series.loc[(batch_series.index >= t0) & (batch_series.index <= t1)]
        mean_kw = float(seg.mean()) if len(seg) else None
        std_kw = float(seg.std(ddof=0)) if len(seg) else None
        mad_kw = float((seg.sub(seg.median()).abs().median())) if len(seg) else None

        event = {
            "event_type": rec["event_type"],
            "period_id": rec["period_id"],
            "run_id": rec["run_id"],
            "machine_id": rec["machine_id"],

            "actual_start_ts": rec["actual_start_ts"],
            "first_cross_ts": rec["first_cross_ts"],
            "confirmed_ts": rec["confirmed_ts"],
            "end_ts": end_ts.isoformat(),
            "duration_s": duration_s,

            "spans_batches": pd.to_datetime(rec["actual_start_ts"]) < curr_batch_start,
            "opened_in_this_run": curr_batch_start <= pd.to_datetime(rec["actual_start_ts"]) <= curr_batch_end,
            "closed_in_this_run": curr_batch_start <= end_ts <= curr_batch_end,
            "status": "closed",

            "mad_win_s": rec.get("mad_win_s"),
            "mad_threshold_kw": rec.get("mad_threshold_kw"),
            "off_min_len_s": rec.get("off_min_len_s"),
            "low_var_min_len_s": rec.get("low_var_min_len_s"),

            "features_snapshot": {
                "mad_kw": mad_kw,
                "mean_kw": mean_kw,
                "std_kw": std_kw
            },

            "exclusions": {"within_off_mask": False, "maintenance_window": False},
            "source": {
                **(rec.get("source") or {}),
                "file_start_ts": curr_batch_start.isoformat(),
                "file_end_ts": curr_batch_end.isoformat()
            },
            "notes": (rec.get("notes") or "") + (" | features partial (spans batches)"
                                                 if pd.to_datetime(rec["actual_start_ts"]) < curr_batch_start else ""),
            "labels": rec.get("labels", []),
        }

        self._write_jsonl(event)

    def emit_truncated_lowvar_if_any(self, *, batch_series: pd.Series):
        k = self._lowvar_key()
        if k not in self._open_lowvar:
            return
        rec = self._open_lowvar[k]
        batch_end_ts = batch_series.index.max()
        event = {
            "event_type": rec["event_type"],
            "period_id": rec["period_id"],
            "run_id": rec["run_id"],
            "machine_id": rec["machine_id"],
            "actual_start_ts": rec["actual_start_ts"],
            "first_cross_ts": rec["first_cross_ts"],
            "confirmed_ts": rec["confirmed_ts"],
            "end_ts": batch_end_ts.isoformat(),
            "duration_s": None,
            "spans_batches": True,
            "opened_in_this_run": rec["opened_in_this_run"],
            "closed_in_this_run": False,
            "status": "truncated",
            "mad_win_s": rec.get("mad_win_s"),
            "mad_threshold_kw": rec.get("mad_threshold_kw"),
            "off_min_len_s": rec.get("off_min_len_s"),
            "low_var_min_len_s": rec.get("low_var_min_len_s"),
            "features_snapshot": {"mad_kw": None, "mean_kw": None, "std_kw": None},
            "exclusions": {"within_off_mask": False, "maintenance_window": False},
            "source": rec.get("source"),
            "notes": (rec.get("notes") or "") + " | truncated at batch end",
            "labels": rec.get("labels", [])
        }
        self._write_jsonl(event)
        # keep it open for the next batch

