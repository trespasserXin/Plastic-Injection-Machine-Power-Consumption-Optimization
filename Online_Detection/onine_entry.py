from online_detection_v3 import OnlineIdleDetector
from sftp_get import get_file
import pandas as pd
from event_logging import EventLogger
from pathlib import Path

prev_file = None
end_ts = None
state = None


def write_log(msg):
    msg = str(msg)
    with open('var_log.txt', 'a') as f:
        print(msg)
        f.write(msg + '\n')

def notify_lowvar_start(run_id, start_idx, first_valid_idx, min_len, threshold, context=None):
    msg = (f"[LOWVAR-START] id={run_id} start={start_idx} first_valid={first_valid_idx} "
          f"(min_len={min_len}, {threshold}) ctx={context}")
    write_log(msg)

def notify_lowvar_end(run_id, start_idx, end_idx, length, context=None):
    msg = f"[LOWVAR-END]   id={run_id} start={start_idx} end={end_idx} len={length} ctx={context}"
    write_log(msg)

def notify_off_start(start_idx, first_valid_idx, min_len, context=None):
    msg = f"[OFF-START]    start={start_idx} first_valid={first_valid_idx} (min_len={min_len}) ctx={context}"
    write_log(msg)

def notify_off_end(start_idx, end_idx, length, context=None):
    msg = f"[OFF-END]      start={start_idx} end={end_idx} len={length} ctx={context}"
    write_log(msg)

def make_callbacks(batch_series, det, run_id_str, source_file, evt_logger):
    det_params = dict(
        mad_win_s=int(det.mad_win),     # if your window is in samples (1/min), convert to seconds
        mad_threshold_kw=det.mad_threshold,
        off_min_len_s=int(det.off_min_len),
        low_var_min_len_s=int(det.min_len)
    )

    def lowvar_start(run_id, start_idx, first_valid_idx, min_len, threshold, context=None):
        evt_logger.on_lowvar_start(
            run_id=run_id,
            start_idx=start_idx,
            first_valid_idx=first_valid_idx,
            min_len=min_len,
            threshold=threshold,
            batch_series=batch_series,
            det_params=det_params,
            run_id_str=run_id_str,
            source_file=source_file
        )
        notify_lowvar_start(run_id, start_idx, first_valid_idx, min_len, threshold, context=None)

    def lowvar_end(run_id, start_idx, end_idx, length, context=None):
        evt_logger.on_lowvar_end(
            run_id=run_id,
            start_idx=start_idx,
            end_idx=end_idx,
            length=length,
            batch_series=batch_series
        )
        notify_lowvar_end(run_id, start_idx, end_idx, length, context=None)

    return lowvar_start, lowvar_end


evt_logger = EventLogger(
            log_path=Path("event_log.jsonl"),
            machine_id="Machine201_motor_sum")

for i in range(1405, 320, -1):
    current_info = get_file(prev_file, end_ts, i)
    prev_file, end_ts, function_call, motor_sum, file = (current_info[0], current_info[1],
                                                   current_info[2], current_info[3], current_info[4])

    if function_call:
        print('Data Received, Starting Detection')

        det = OnlineIdleDetector(
            mad_win=6, hampel_win=4, n_sigmas=3.0,
            mad_threshold=5, min_len=15,
            off_level=1.0, off_min_len=10,
            tiny_threshold=5, ema_alpha=0.25,
            on_off_start=notify_off_start,
            on_off_end=notify_off_end,
            context={"device": "Motor Sum"},
        )

        # off_msg= "Batch off_periods:", out["off_periods"]
        # write_log(off_msg)
        # low_var_msg = "Batch lowvar_periods:", out["lowvar_periods"]
        # write_log(low_var_msg)

        batch_start = motor_sum.index.min().strftime("%Y-%m-%dT%H:%M:%S%z") if (
            motor_sum.index.tzinfo) else motor_sum.index.min().strftime("%Y-%m-%dT%H:%M:%S")
        run_id_str = f"{batch_start}__MOTOR1"
        source_file = file

        cb_lowvar_start, cb_lowvar_end = make_callbacks(motor_sum, det, run_id_str, source_file,
                                                        evt_logger)
        det.on_lowvar_start = cb_lowvar_start
        det.on_lowvar_end = cb_lowvar_end

        out = det.process_batch(motor_sum, state=state)
        state = out["state"]

        # det.on_off_start = cb_off_start
        # det.on_off_end = cb_off_end
        # time.sleep(10)

